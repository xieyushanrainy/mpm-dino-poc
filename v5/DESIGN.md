# V5 causal deformation design

Date: 2026-08-05

## Status and material passport

- Origin: V5 `grill-me` design interview plus verified V4.2/V4.3 evidence
- Status: approved specification; initial implementation complete, experiment training not started
- Dataset: `v41/dataset`
- Split: `v41/manifests/v41_uid_splits.json`
- Split manifest SHA-256: `2f4add1aa6f6f68b965d031cf12bf60f981a10f628a09bdc7f75a574cd42949b`
- Test data used: no
- Neural initialization: random for every V5 component; no V4 weights

## 1. Goal

V5 is a from-scratch, inference-causal deformation system. Its primary success
criterion is a three-seed mean validation event-amplitude-normalized canonical
deformation MSE of at most `0.7025`, matching or beating the non-deployable
V4.3 oracle compact-memory mean.

The final threshold is intentionally aspirational rather than
information-matched: V5 may not receive ground-truth query contact, stage or
event time at inference. Per-seed and per-UID results remain mandatory
disclosures, but neither is a hard promotion rule. Physical metrics, parameter
count and latency are diagnostic only. Simplicity is the tie-breaker between
models with comparable primary validation performance.

Before memory is allowed, the standalone causal base three-seed mean must
strictly beat `0.91098`, the prior V4.3 causal-contact two-seed mean chosen as
the qualification reference. If the base already reaches `0.7025`, V5 stops
without memory.

## 2. Non-negotiable contracts

### 2.1 Data and causality

- Reuse the 40-train, 10-validation and sealed 10-test UID split.
- Observe frames 0 and 1; predict frames 2--60.
- Family may control sampling and reporting but is not a model input.
- Soft samples train deformation. Rigid samples train COM and rotation; they do
  not initially provide an explicit zero-deformation loss.
- Future query positions, masks, contact, stage, event time and canonical
  targets are forbidden from every inference input path.
- Ground-truth contact and event annotations may be training-only auxiliary
  labels.
- The sealed test set is not used for architecture choice, stopping, threshold
  choice or fallback selection.

### 2.2 From scratch

All V5 neural weights start from random initialization. V4 checkpoints,
adapters, readers and learned memory tokens are not loaded. Reusing audited
data loaders, Kabsch target construction, analytic ballistic motion, loss
implementations, metrics and leakage checks is allowed.

### 2.3 Motion factorization

V5 preserves

```text
x_hat[t,i] = c_hat[t] + R_hat[t] (q_i + d_hat[t,i])
```

where `c_hat` is COM, `R_hat` is a proper rotation, `q` is centered reference
geometry and `d_hat` is zero-mean canonical deformation. COM and rotation have
separate heads and metrics but share one randomly initialized graph/temporal
physical trunk, matching the V4.2 global-motion design. The deformation path
may not hide global translation.

## 3. Architecture

### 3.1 COM stage

COM retains the analytic ballistic trajectory plus a learned residual:

```text
c_hat[t] = c_ballistic[t] + delta_c[t]
```

The residual head is trained from scratch using causal observations. The
ballistic component is not learned. During the global-motion stage, the COM
loss and rotation loss jointly update the shared physical trunk.

### 3.2 Rotation stage

A proper-SO(3) rotation head is trained jointly with COM from scratch and selected by validation
geodesic error. Identity is the required baseline. Learned rotation is adopted
only if its three-seed mean beats identity; the learned-versus-identity choice
is global across seeds. Otherwise V5 falls back to identity and deformation
work continues. Global checkpoints contain the shared trunk plus both heads.

### 3.3 Learned pointwise interaction stage

The previous timing-only interface is rejected because it was null in V4.3.
The hand-engineered gap/proximity/vertical-velocity interface is also not
carried forward. Instead, a small pointwise interaction encoder learns a causal
latent from centered reference points and the frozen predicted rigid future:

```text
x_rigid[t,i] = c_hat[t] + R_hat[t] q_i
z[t,i] = interaction_encoder(q_i, x_rigid[t,i], global_rigid_context[t])
```

The representation has two training-only auxiliary heads:

- pointwise contact prediction;
- learned event-relative time prediction.

The latent `z`, rather than either auxiliary prediction, conditions the causal
deformation decoder. The timing prediction remains available for memory phase
addressing if memory is later admitted. The interaction checkpoint is selected
using its own validation auxiliary metrics and then frozen.

The encoder must not receive DINO, material ID, family, future query truth, or
the old analytic contact features. Observed motion may influence it only
through the already frozen causal COM/rotation prediction. This keeps the local
deformation representation restricted to reference geometry and predicted
rigid interaction.

### 3.4 Standalone causal deformation stage

For soft samples, a randomly initialized decoder predicts

```text
d_base[t,i] = zero_mean(deformation_decoder(q_i, z[t,i]))
```

The primary loss and checkpoint metric are event-amplitude-normalized canonical
deformation MSE. The default arm freezes the shared physical trunk, COM and
rotation heads, and interaction encoder. A separately named controlled arm may
set `trunk_gradient_scale` in `(0, 1]`, using the V4 local-gradient construction
`h.detach() + alpha * (h - h.detach())`. This lets deformation fine-tune the
shared trunk while COM and rotation head parameters remain frozen. Such an arm
must report COM and rotation drift and cannot silently replace the frozen arm.

Promotion rules over seeds 42, 123 and 456:

- `mean < 0.91098`: causal base qualifies; proceed only if needed;
- `mean <= 0.7025`: final V5 success; omit memory;
- `mean >= 0.91098`: stop this architecture; memory may not rescue it.

### 3.5 Optional compact-memory stage

Memory is admitted only for a qualified base that misses `0.7025`.

The permitted bank is limited to the same population and scope as the V4.3
oracle deformation bank: the 20 soft training UIDs, with training-target
mechanics organized at the existing three event phases (contact onset,
compression and peak deformation). Rebuilding it is allowed and required for
new learned tokenization; it may not be enlarged with validation/test UIDs,
additional objects or an auxiliary bank.

Bank construction may use ground-truth source canonical deformation and source
event annotations. It learns 32 compact tokens per source entry from scratch.
Training retrieval is leave-one-UID-out; validation and test retrieve from
training UIDs only.

DINO is restricted to object selection:

- select top-3 source objects;
- do not pass pointwise DINO into the causal base or memory reader;
- do not claim or require source/query point correspondence.

The causally predicted query event time linearly interpolates the three stored
source phases. Out-of-range predictions clamp to the nearest stored phase. A
compact reader cross-attends from query geometry/interaction features to the
three sets of 32 compact tokens. It predicts a bounded zero-mean residual:

```text
d_hat = zero_mean(d_base + sigmoid(g) * residual_bound * tanh(r_memory))
```

`g` is one global scalar initialized so its sigmoid is near zero. The frozen
base must remain a valid predictor when memory is zeroed. The numerical bound
is fixed before the three-seed memory run; it is not selected from a broad
validation sweep.

## 4. Stagewise freezing and selection

Training is explicitly staged. Protected heads remain immutable. The shared
trunk is frozen by default downstream, with only the declared deformation
fine-tuning arm allowed to update it:

1. Shared global motion: joint COM and rotation training; report validation COM
   trajectory error and SO(3) geodesic error separately.
2. Rotation decision: global learned-versus-identity fallback.
3. Interaction: validation contact and event-time auxiliary metrics.
4. Causal deformation: validation event-normalized canonical MSE.
5. Optional memory: validation event-normalized canonical MSE.

Earlier checkpoints must not be retrospectively selected for downstream
deformation performance. Protected-head parameter hashes are checked before
and after every later stage. A trunk-fine-tuned deformation checkpoint is a new
named branch and must additionally measure global-motion output drift.

## 5. Evaluation and reporting

The primary result is the arithmetic mean of the three independently trained
seed results. Required reporting includes:

- primary objective per seed, mean and per validation UID;
- zero-memory ablation for any memory model;
- causal-label audit proving query future labels are absent at inference;
- COM error and ballistic-only comparison;
- learned rotation and identity geodesic error;
- pointwise contact metrics and event-time error;
- canonical NRMSE, strain RMSE, magnitude correlation and peak ratio;
- full-trajectory error, parameter count, 96-token maximum reader load and
  inference latency;
- checkpoint, bank and split hashes.

These secondary results are diagnostic and cannot override the formal primary
criterion. Material physical regressions must nevertheless be stated plainly.

## 6. Planning defaults, not evidence claims

The following are implementation defaults to fix before execution, not frozen
scientific conclusions:

- hidden width and block count;
- contact/time auxiliary-loss weights;
- precise event-time coordinate and masking outside valid events;
- residual magnitude bound and near-zero gate logit;
- optimizer, learning rate, epoch limit and early-stopping patience;
- the tolerance used to call two models comparable for the simplicity
  tie-breaker.

They must be recorded once in an immutable run manifest. Changing one after
viewing validation outcomes creates a new named experiment rather than a silent
continuation.
