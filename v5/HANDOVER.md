# V5 context handover: evidence before design

## Purpose

This document transfers the verified V4.2/V4.3 context into a new V5 planning
conversation. It deliberately does **not** choose the V5 architecture, training
stages, controlled matrix, success thresholds or memory-phase interface.

The next conversation should run in Plan mode, read the evidence below, and use
the `grill-me` skill to interview the user before freezing a V5 design.

No V5 implementation or experiment has been authorized by this handover.

## Material Passport

- Origin: V4.2/V4.3 experiment artifacts
- Origin Date: 2026-08-04
- Verification Status: CONTEXT_ONLY
- Test data used: no
- V5 design status: open

## Repository and dataset

Repository root:

```text
/Users/yushanxie/workspace/cgvi/term3/mpm-DINO-poc
```

Dataset:

```text
v41/dataset
```

Authoritative split:

```text
v41/manifests/v41_uid_splits.json
SHA-256: 2f4add1aa6f6f68b965d031cf12bf60f981a10f628a09bdc7f75a574cd42949b
```

Dataset facts:

- 61 frames and 2,048 points per episode;
- frames 0 and 1 are observed and frames 2--60 are the 59 targets;
- static tensors include reference geometry, 384-dimensional point DINO and
  DINO validity;
- trajectory tensors include positions, velocities, activity and time;
- split is object-UID-disjoint: 40 train, 10 validation and 10 sealed test UIDs;
- rigid and soft families are present; fluid was excluded from these studies;
- family is permitted for sampling/reporting but was forbidden as model input.

Read:

1. `v41/dataset/README.md`
2. `v41/dataset/collection.json`
3. `v41/manifests/v41_uid_splits.json`
4. `v4/src/mpm_dino_v4/v41_data.py`

## Shared motion decomposition

The working V4.2/V4.3 models separate global translation, global rotation and
local deformation:

```text
x_hat[t,i] = c_hat[t] + R_hat[t] (q_i + d_hat[t,i])
```

- `c_hat`: predicted centre of mass;
- `R_hat`: predicted global rotation;
- `q_i`: centred reference point;
- `d_hat`: pointwise zero-mean canonical deformation.

The zero-mean constraint prevents the deformation head from hiding COM
translation. Canonical targets remove ground-truth COM and proper Kabsch
rotation, so deformation is evaluated separately from rigid-motion error.

This decomposition is one of the strongest architectural conclusions to carry
into V5 planning, but the exact implementation remains open.

Read:

1. `v42/V42_EXPERIMENT_SUMMARY.md`
2. `v42/ROTATION_EXPERIMENT_RESULTS.md`
3. `v4/src/mpm_dino_v4/v42_model.py`
4. `v4/src/mpm_dino_v4/v42_geometry.py`

## Current oracle-conditioned deformation architecture

### Working architecture

The strongest oracle-conditioned deformation result combines:

1. the frozen V4.2 full-condition base;
2. aligned-DINO top-3 retrieval from a train-only soft-object bank;
3. 32 compact tokens from each retrieved object;
4. a compact cross-attention reader;
5. a bounded gated residual added to the base deformation.

```text
d_hat = zero_mean(d_base + sigmoid(gate) * bounded_memory_residual)
```

The base receives future-ground-truth point contact, stage and event time. This
is a research/sufficiency model, not deployable inference.

The memory bank contains 20 soft training UIDs and 60 stage entries. Training
retrieval excludes the query UID; validation retrieves from training UIDs only;
test remained sealed.

### Checkpoints and metrics

Frozen base:

```text
v42/checkpoints/v42_adapter_full_seed42_best.pt
SHA-256: 85dd4bb3024268ec1b2dedbf1a6ed09d1175628a8f1cdb995060fcf5430c4dfb
validation event-normalized MSE: 0.7771909118
```

Best compact reader:

```text
v43/run/bad_neighbour_rejection_seed42_123_456/compact_baseline/seed123/best.pt
SHA-256: 5709b2ec8e2dbd34a32007ad06c7ce0d42457682a0101017582d4fc8e5aa70ce
validation event-normalized MSE: 0.6629586637
```

Three-seed compact-baseline results:

| Seed | Objective |
|---:|---:|
| 42 | 0.7681 |
| 123 | 0.6630 |
| 456 | 0.6765 |
| Mean | 0.7025 |

### What is supported

- Compact latent memory is more effective than explicit source-field attention.
- DINO appears useful primarily for sparse object/prototype selection.
- A bounded residual on a competent base is safer than copied trajectories.
- Fixed-weight memory/DINO removal changes model behaviour.

### What is not supported

- Universal or exact pointwise DINO-mechanics correspondence.
- Reliable rejection of harmful plausible neighbours.
- Explicit pointwise deformation-field transfer.
- Deployability: the query path uses oracle contact/stage/time.
- UID-balanced improvement: gains were concentrated in selected objects.

Read:

1. `v43/V43_DEFORMATION_CLOSEOUT.md`
2. `v43/DESIGN.md`
3. `v43/FIELD_ATTENTION_DESIGN.md`
4. `v43/REJECTION_DESIGN.md`
5. `v43/run/bad_neighbour_rejection_seed42_123_456/BANK_MANIFEST.json`
6. `v43/run/bad_neighbour_rejection_seed42_123_456/compact_baseline/seed123/config.json`
7. `v4/src/mpm_dino_v4/v43_retrieval.py`
8. `v4/src/mpm_dino_v4/v43_train.py`

## Current causal deformation architecture

### Working architecture

The best causal candidate keeps a frozen Gate-1E physical/COM/rotation model.
It predicts a zero-deformation rigid future trajectory:

```text
x_tilde[t,i] = c_hat[t] + R_hat[t] q_i
```

Fixed analytic calculations then produce:

- normalized signed floor gap;
- smooth one-sided floor proximity;
- normalized vertical velocity;
- time relative to predicted contact onset.

These are combined with four static geometry/curvature proxies. Seven oracle
stage slots remain zero, giving the same 15-channel adapter interface as the
oracle experiment. The condition is projected into detached physical features;
the geometry-region adapter and canonical deformation head are trainable while
the physical trunk, COM and rotation remain protected.

The contact calculator itself is not learned. The rigid predictor is a frozen
learned model; the downstream condition projection and deformation adapter are
learned.

### Checkpoint and metrics

Best causal candidate:

```text
v43/run/causal_contact_seed42_456/causal_continuous/seed42/best.pt
SHA-256: 119b37214c8d9d78f3868f690e579b58c5d836f297eba1e4912e42c852817008
best epoch: 46
validation event-normalized MSE: 0.8373412609
```

Frozen global source:

```text
v42/run/gate1e_seed42_456/seed42/best_total.pt
SHA-256: 4ff20f1896819c81a059a90b3ad0a63e079f75881dea0e71c28fad0eaeb9873a
```

Matched results:

| Condition | Seed 42 | Seed 456 | Mean |
|---|---:|---:|---:|
| Static control | 0.99091 | 0.98801 | 0.98946 |
| Timing only | 0.99092 | 0.98881 | 0.98986 |
| Causal continuous | 0.83734 | 0.98462 | 0.91098 |
| Oracle ceiling | 0.77719 | 0.77891 | 0.77805 |

### What is supported

- Continuous pointwise contact features can contain useful deformation signal.
- Timing alone is insufficient.
- The model-input path is genuinely causal: future targets/contact/stages are
  not used to build query conditions.
- Protected COM and rotation remain bit-identical.

### What is not supported

- Robustness: the causal benefit did not replicate in seed 456.
- UID balance: the seed-42 gain was dominated by one validation object.
- Reliable deformation amplitude or strain coherence.
- A claim that Gate-1E rotation is solved; it is an operational placeholder.

Read:

1. `v43/CAUSAL_CONTACT_DESIGN.md`
2. `v43/run/causal_contact_seed42_456/RESULTS.md`
3. `v43/run/causal_contact_seed42_456/MATRIX_CONFIG.json`
4. `v43/run/causal_contact_seed42_456/causal_continuous/seed42/config.json`
5. `v4/src/mpm_dino_v4/v43_causal_contact.py`
6. `v43/run_causal_contact_replacement.py`

## COM and rotation status

COM uses a physics-informed ballistic trajectory plus a learned residual. The
first predicted horizon is anchored to the ballistic result. This separation is
worth preserving as evidence, while its exact V5 training arrangement is open.

No learned rotation method was promoted. Identity remains the safest default.
The completed V4.3 rotation-memory matrix found geometry memory better than
aligned DINO, but the improvement was modest and diagnostic only.

Best experimental rotation checkpoint:

```text
v43/run/rotation_memory/geometry/seed42/best.pt
family-balanced validation error: 2.370135774 degrees
identity/zero-memory error: 2.508700456 degrees
```

Do not assume that V5 must copy Gate-1E rotation, identity-only rotation, or the
geometry-memory rotation experiment. That is a planning decision.

Read:

1. `v42/ROTATION_EXPERIMENT_RESULTS.md`
2. `v43/run/rotation_memory/RESULTS.md`
3. `v43/ROTATION_MEMORY_DESIGN.md`

## Main mechanisms keeping the current systems working

These are evidence-backed mechanisms, not a frozen V5 specification:

1. **Motion factorization:** COM, rotation and deformation are represented and
   supervised separately.
2. **Physics anchor:** ballistic COM supplies a stable base; learning predicts a
   correction rather than the entire translation from nothing.
3. **Contact localization:** pointwise gap/proximity/velocity is more useful
   than a global timing scalar.
4. **Canonical supervision:** target translation and rotation are removed before
   deformation loss, preventing rigid motion from masquerading as deformation.
5. **Event-normalized objective:** training focuses on contact/compression/peak
   frames and normalizes by each episode's deformation amplitude; zero predicts
   approximately one.
6. **Competent base plus bounded correction:** compact memory corrects a base
   prediction instead of copying a retrieved trajectory.
7. **DINO as retrieval/prototype signal:** evidence is stronger for choosing
   useful objects than for direct material regression or exact correspondence.
8. **Leakage-safe memory:** training-only sources, leave-one-UID-out training
   retrieval and sealed test data.
9. **Protected-path attribution:** local experiments cannot silently improve or
   damage COM/rotation and then claim a deformation gain.

## Important incompatibility between the two current architectures

The current oracle memory reader uses ground-truth query stage/event information
when building/selecting stage-specific memories. The current causal model has no
memory reader. Therefore the two checkpoints cannot simply be connected and
called causal.

A V5 plan must decide how memory phase is handled without query target leakage.
Possibilities include exposing all source phases, fixed mapping from predicted
causal time, continuous phase tokens, or another design. This handover does not
choose among them.

Similarly, “training from scratch” needs an explicit definition: random model
weights may coexist with reused tested data/target utilities and a bank made
from training targets, but the next planning conversation must decide the exact
boundary.

## Questions intentionally left open for Plan mode and `grill-me`

The next conversation should challenge at least these decisions:

1. What is the primary V5 goal: deployable causal performance, oracle-quality
   ceiling, simplicity, inference cost, or a prioritized combination?
2. What exactly must be trained from scratch: all neural weights, memory tokens,
   retrieval embeddings, COM/rotation, or only deformation?
3. Which current mechanisms are mandatory and which are merely historical?
4. Should V5 retain a base-plus-memory residual or use one unified deformation
   decoder?
5. Is an external bank acceptable at inference? If not, is distillation,
   learned prototypes or bank packaging acceptable?
6. How should a causal query address source deformation phase without oracle
   stage/time?
7. Should DINO perform only object retrieval, also enter query tokens, or be
   excluded until a causal base works?
8. How should different source/query shapes be aligned, given that exact
   pointwise correspondence was not proven necessary?
9. How should COM and rotation be trained/evaluated, and is identity the V5
   rotation default?
10. Should rigid examples supervise only COM/rotation, also enforce zero local
    deformation, or be handled another way?
11. What does “match/beat” mean across seeds, UIDs, causal/oracle information
    differences, physical metrics, latency and complexity?
12. What is the smallest controlled experiment that can distinguish the chosen
    architectural claims?

Do not answer these questions on behalf of the user in this handover.

## Expected next-conversation behavior

1. Enter Plan mode.
2. Read this handover and the essential evidence files.
3. Summarize the oracle and causal architectures and their proven mechanisms.
4. Invoke the `grill-me` skill and interview the user one consequential question
   at a time.
5. Keep an explicit decision log: decided, tentative and open.
6. Only after the user approves the direction, draft `v5/DESIGN.md`, an
   experiment matrix and implementation plan.
7. Do not implement, run training or freeze controlled experiments during the
   initial context-transfer/grilling phase.

