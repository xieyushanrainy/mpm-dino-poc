# V4.2

V4.2 separates COM/rigid motion from local deformation, corrects
rotation-contaminated supervision, and tests stage-aware deformation learning
with controlled gradient routing. Gate 0 and the Gate-1 rotation experiments
are complete; Gate 2 is the next protected architecture screen.

Read in this order:

1. [`CONTEXT.md`](CONTEXT.md)
2. [`PLAN.md`](PLAN.md)
3. [`ROTATION_EXPERIMENT_RESULTS.md`](ROTATION_EXPERIMENT_RESULTS.md)
4. [`GATE2_HANDOVER_PROMPT.md`](GATE2_HANDOVER_PROMPT.md)

Implementation scaffolding now lives in `v4/src/mpm_dino_v4/v42_*.py`.
It provides rotation-aware targets, protected global/local gradient routing,
stage metadata and separated losses. Experimental gates remain manual.

The learned local conditions are architecture-identical: geometry-only is the
zero-DINO control using the real validity mask, while real and point-shuffled
DINO use the same projection, four geometry-aware region tokens,
cross-attention adapter and canonical-displacement head.

Gate-0 contact geometry uses the mean floor gap of the lowest four active
surface points with an episode-adaptive tolerance, corroborated by a raw excess
vertical COM acceleration of `0.2g`. The same stage logic applies to all
families. This replaces the rejected 2% point-height quantile.

Run the training/validation-only Gate-0 audit from the repository root:

```bash
PYTHONPATH=v2/src:v4/src python v42/audit_targets_and_stages.py
```

The original Gate 1 found useful post-contact COM learning, but damaged the
near-exact ballistic H1 prediction and did not materially improve rotation over
an identity baseline. Gate 1B is the approved minimal repair:

- retain the existing ballistic-plus-residual COM parameterization;
- force the learned COM residual to exactly zero at H1;
- optimize rotation with smooth squared chordal distance while reporting
  geodesic radians;
- add a `0.25` key-horizon chordal rotation term at
  H1/H8/H16/H30/H40/H59;
- write `VALIDATION_BASELINES.json` from `best.pt`, stratified by family and
  Panel Z/V, comparing learned COM with ballistic COM and learned rotation
  with identity.

Run Gate 1B on the lab server:

```bash
PYTHONPATH=v2/src:v3/src:v4/src python -u v42/run_gate1.py \
  --device cuda \
  --seeds 42 456 \
  --epochs 120 \
  --draws 40 \
  --patience 20 \
  --plateau-patience 5 \
  --no-amp \
  --runs v42/runs/gate1b_seed42_456
```

This command does not launch Gate 2. Gate-1 checkpoints must be reviewed
before the protected local screen starts.

After Gate 1B, run the validation-only rotation audit without loading a trained
model or touching test data:

```bash
PYTHONPATH=v4/src python v42/audit_rotation_baselines.py
```

It compares identity with a constant-angular-velocity baseline inferred by
proper Kabsch alignment from `x0` to `x1`, reports target rotation vectors and
impact-time angular changes, and checks whether Kabsch conditioning or
deformation residuals explain rotation instability.

The audit found no useful observed angular-velocity baseline. Gate 1C therefore
predicts an axis-angle rotation around identity, anchors H1 to exact identity,
and adds detached event-emphasized rotation supervision. Stage labels remain
training metadata and are never model inputs. A checkpoint is eligible only if
its mean H8/H16/H30/H40/H59 rotation error improves identity by at least 1% and
its H59 error is no more than 10% worse than identity.

Run Gate 1C on the lab server:

```bash
PYTHONPATH=v2/src:v3/src:v4/src python -u v42/run_gate1c.py \
  --device cuda \
  --seeds 42 456 \
  --epochs 120 \
  --draws 40 \
  --patience 20 \
  --plateau-patience 5 \
  --no-amp \
  --runs v42/runs/gate1c_seed42_456
```

This command runs Gate 1C only. It does not start Gate 2.

Gate 1C showed that joint COM/rotation optimization destabilizes rotation.
Gate 1D is therefore a protected attention-based rotation screen. It starts
from each seed's Gate-1B `best.pt`, freezes the physical trunk and COM head in
evaluation mode, and trains only a geometry/contact attention adapter plus an
axis-angle rotation head. Before training, every validation COM output must be
bit-identical to the source Gate-1B model.

```bash
PYTHONPATH=v2/src:v3/src:v4/src python -u v42/run_gate1d.py \
  --device cuda \
  --seeds 42 456 \
  --epochs 120 \
  --draws 40 \
  --patience 20 \
  --plateau-patience 5 \
  --min-eligible-epoch 60 \
  --gate1b-root v42/runs/gate1b_seed42_456 \
  --runs v42/runs/gate1d_seed42_456
```

An eligible checkpoint must improve identity by at least 1% overall across
H8/H16/H30/H40/H59, must not regress in rigid-Z, rigid-V or soft-Z separately,
and must keep each stratum's H59 error within 1.10 times identity. Eligibility
does not begin before epoch 60. If no epoch passes, the run retains only
`best_total.pt` for diagnosis. Gate 1D does not launch Gate 2.

Gate 1D learned some H8-H40 impact rotation but developed severe H59 drift.
Gate 1E keeps the same protected attention inputs and instead predicts changes
in angular velocity. These are accumulated into angular velocity and integrated
through time into the rotation trajectory:

```text
delta_omega_t = attention(features_t)
omega_t = omega_(t-1) + delta_omega_t
R_t = R_(t-1) Exp(dt * omega_t)
```

H1 remains exact identity. Detached target angular velocity and angular
acceleration add loss weights `0.50` and `0.25`, respectively. The Gate-1D
identity screen and COM/trunk protection remain unchanged.

```bash
PYTHONPATH=v2/src:v3/src:v4/src python -u v42/run_gate1e.py \
  --device cuda \
  --seeds 42 456 \
  --epochs 120 \
  --draws 40 \
  --patience 20 \
  --plateau-patience 5 \
  --min-eligible-epoch 60 \
  --gate1b-root v42/runs/gate1b_seed42_456 \
  --runs v42/runs/gate1e_seed42_456
```

Gate 1E does not launch Gate 2.

Gate 1E produced a small overall improvement but did not pass the required
per-stratum identity screen. Gate 1F keeps the Gate-1B COM/trunk protected and
compares two contact-driven rotation formulations:

- `absolute`: predicts an absolute axis-angle residual multiplied by a
  cumulative, inference-time contact hazard. This tests whether Gate-1E's
  recurrent integration drift is the main problem.
- `impulse`: predicts contact-gated angular-velocity impulses, applies fixed
  damping `rho=0.95`, and integrates them through time.

Both variants infer contact probability and the contacting reference region
from detached physical point features, normalized reference geometry,
ballistic floor gap and the observed finite-difference velocity. Ground-truth
contact stages and lowest-four contact patches are detached auxiliary targets
for training only; they are never model inputs.

Gate 1F retains full-trajectory chordal, key-horizon, event and rigid-fit
losses. It adds class-balanced contact-onset BCE (`0.25`) and a normalized
contact-point Huber loss (`0.10`). The impulse variant also retains the
angular-velocity (`0.50`) and angular-acceleration (`0.25`) objectives.

Checkpoint selection uses rigid rotation-active frames, defined before
training as target rotation of at least `0.05 rad`. Eligibility requires both
rigid Panel Z and Panel V to improve identity on active frames, at least `1%`
pooled active-frame improvement, predicted rotation no greater than `0.01 rad`
on inactive frames, and the existing H59 `1.10x` identity guard. Soft-body
results remain a reported safety diagnostic rather than a Gate-1F promotion
criterion.

Run both Gate 1F variants on the lab server:

```bash
PYTHONPATH=v2/src:v3/src:v4/src python -u v42/run_gate1f.py \
  --device cuda \
  --variants absolute impulse \
  --seeds 42 456 \
  --epochs 120 \
  --draws 40 \
  --patience 20 \
  --plateau-patience 5 \
  --min-eligible-epoch 60 \
  --gate1b-root v42/runs/gate1b_seed42_456 \
  --runs v42/runs/gate1f_seed42_456
```

This runner starts Gate 1F only. It does not start Gate 2.

## Gate 2: protected geometry-only canonical local screen

Gate 2 is now implemented but has not been trained locally. It compares:

1. the frozen zero-local-output baseline from each seed's Gate-1E
   `best_total.pt`; and
2. the architecture-matched geometry-only local model, which supplies zero
   DINO features with the real `dino_valid` mask.

`v42/run_gate2.py` has no real-DINO condition and no Gate-3 dispatch. It
strictly loads the matching Gate-1E checkpoint, records its SHA-256, freezes
the physical trunk, COM head, and complete rotation branch, and trains only
the DINO projection retained for capacity matching, geometry-aware region
tokens, cross-attention adapter, and canonical displacement head. The local
branch reads stopped-gradient physical features (`alpha=0`).

The objective is exactly:

```text
1.00 canonical displacement
+ 0.50 normalized strain
+ 0.25 edge length
+ 0.25 local velocity
+ 0.25 rigid zero-local
```

Canonical vector loss excludes Kabsch-degenerate frames; invariant losses
remain. Gate-0 stage labels and weights, and Kabsch targets, are detached
preprocessing and are never model inputs. World reconstruction and penetration
are diagnostics only and are not backpropagated.

The historical Gate-1E configs omitted model width, block count, heads, and
dropout. The Gate-2 loader therefore verifies width and block count directly
from checkpoint tensors and uses the fixed `gate1e_v1` contract values
(`heads=4`, `dropout=0.1`). This reconstruction is recorded in every Gate-2
config; the loader never substitutes another checkpoint.

From a csh/tcsh lab-server shell, run:

```csh
mkdir -p v42/runs/gate2_seed42_456
setenv PYTHONPATH "v2/src:v3/src:v4/src"
nohup .venv-v41/bin/python -u v42/run_gate2.py \
  --device cuda \
  --seeds 42 456 \
  --epochs 120 \
  --draws 40 \
  --patience 20 \
  --plateau-patience 5 \
  --gate1e-root v42/runs/gate1e_seed42_456 \
  --runs v42/runs/gate2_seed42_456 \
  >& v42/runs/gate2_seed42_456/console.log &
echo $! > v42/runs/gate2_seed42_456/launcher.pid
```

If the reviewed checkpoints are instead downloaded under `v42/run`, pass
`--gate1e-root v42/run/gate1e_seed42_456` explicitly. Do not rename or copy a
different checkpoint into that location.

Every seed writes a frozen zero-local validation report, a geometry-only
validation report, checkpoint/source hashes, and bit-identity checks. Panel Z
and Panel V and rigid/soft strata remain separate. The two-seed decision must
apply the full frozen screen: at least 10% stage-weighted canonical-NRMSE and
strain improvement, compression/peak improvement in both seeds, UID-balanced
magnitude correlation at least 0.5, identifiable onset/peak median errors no
greater than two frames, rigid local RMS below 0.1% radius, and exact frozen
COM/rotation identity.

Timing is diagnostic and uses a predeclared fixed rule: an event is
identifiable when Gate 0 finds contact and target peak canonical RMS exceeds
`1e-4` of reference radius; onset is the first post-contact frame at 20% of
the target peak, and peak is the first maximum. No test data set this rule.

This command starts Gate 2 only. It cannot launch Gate 3 and cannot train a
real-DINO condition.

### Gate-2 result

The retrieved two-seed Gate-2 matrix under `v42/run/gate2_seed42_456` is
complete and integrity-valid, but it fails the frozen screen. Two-seed soft
Panel-Z improvement is only `0.435%` for stage-weighted canonical NRMSE and
`0.074%` for strain RMSE, versus the required `10%`. Magnitude correlations
are `0.299` and `0.287`; learned onset is not detected and median peak timing
error is 52 frames for both seeds. Rigid local RMS and frozen COM/rotation
identity pass their safety checks.

The full validation-only report is
[`run/gate2_seed42_456/analysis_20260731/RESULTS.md`](run/gate2_seed42_456/analysis_20260731/RESULTS.md).
Gate 3 / real-DINO training remains unauthorized.

## Oracle temporal/material controlled-group diagnostic

The single-frame overfit experiment is set aside. This experiment is explicitly
not a gate: it is a matched 2x2 controlled comparison intended to localize the
remaining failure at a high level.

| Variant | Ground-truth event stage/time | Simulator-source material |
|---|---:|---:|
| `geometry_control` | no | no |
| `oracle_temporal` | yes | no |
| `oracle_material` | no | yes |
| `oracle_both` | yes | yes |

Every arm uses the same 15-channel conditioning-capable decoder, initialization,
Gate-1E source, balanced sampling, composite loss, total-mass stage weighting,
checkpoint selection and validation split. Disabled inputs are zeros. Temporal
conditioning contains the seven ground-truth stage indicators plus normalized
time relative to ground-truth contact. Material conditioning contains family and
fixed-scale simulator-source density, friction, Young's modulus and damping
features. These are deliberately non-deployable oracle inputs; they diagnose
missing information rather than propose the final model.

Interpret the main effects descriptively: temporal improvement implicates event
state/timing ambiguity; material improvement implicates material ambiguity; an
improvement only in `oracle_both` suggests interaction; no meaningful improvement
shifts attention to representation, decoder optimization, target quality or
deformation signal-to-noise. The runner writes `CONTROLLED_EFFECTS.json` with
per-seed cell means, temporal/material main effects and interaction. It writes no
pass/fail decision and uses no test data, real DINO, or Gate 3.

From csh/tcsh on the lab server:

```csh
mkdir -p v42/runs/oracle_controls_seed42_456
setenv PYTHONPATH "v2/src:v3/src:v4/src"
nohup .venv-v41/bin/python -u v42/run_oracle_controls.py \
  --device cuda \
  --seeds 42 456 \
  --epochs 120 \
  --draws 40 \
  --patience 20 \
  --plateau-patience 5 \
  --gate1e-root v42/runs/gate1e_seed42_456 \
  --runs v42/runs/oracle_controls_seed42_456 \
  >& v42/runs/oracle_controls_seed42_456/console.log &
echo $! > v42/runs/oracle_controls_seed42_456/launcher.pid
```

Use singular `v42/run/gate1e_seed42_456` if that is where the downloaded
Gate-1E checkpoints live.

## Diagnostic 1: decoder single-example overfit

The next diagnostic follows the frozen order established after Gate 2C. It asks
whether the existing protected local branch can represent and optimize an
unambiguously nonzero deformation before adding temporal or material inputs.

`run_decoder_overfit.py` deterministically selects the Panel-Z soft training
episode with the largest canonical deformation, using the training split only.
It then runs two independent fits from the reviewed Gate-1E source:

1. `single_frame`: fit only that episode's peak-deformation frame;
2. `single_episode`: fit all 59 target frames of the same episode with the
   Gate-2C total-mass stage weights.

COM, rotation and the physical trunk remain frozen. DINO is zero, and neither
validation nor test data is loaded. A low scalar loss is insufficient to pass:
the single-frame fit must reduce canonical error by at least 95% and recover
90–110% of target magnitude. The episode fit must additionally reach at least
0.95 magnitude correlation, recover 90–110% of target peak magnitude and place
the peak within one frame. Failure of the single-frame check stops the matrix,
because a temporal-conditioning experiment would then be premature.

From a csh/tcsh lab-server shell:

```csh
mkdir -p v42/runs/decoder_overfit_seed42
setenv PYTHONPATH "v2/src:v3/src:v4/src"
nohup .venv-v41/bin/python -u v42/run_decoder_overfit.py \
  --device cuda \
  --seed 42 \
  --steps 2000 \
  --lr 1e-3 \
  --log-every 25 \
  --gate1e-root v42/runs/gate1e_seed42_456 \
  --runs v42/runs/decoder_overfit_seed42 \
  >& v42/runs/decoder_overfit_seed42/console.log &
echo $! > v42/runs/decoder_overfit_seed42/launcher.pid
```

Use singular `v42/run/gate1e_seed42_456` when that is the lab-server source
location. This was the original gated diagnostic order; it has now been
superseded by the exploratory controlled-group experiment above. Learned DINO
remains outside this experiment.

### Decoder-overfit result

The downloaded seed-42 run is complete and integrity-valid, but `single_frame`
fails the frozen gate and correctly stops before `single_episode`. It recovers
`97.76%` of target aggregate magnitude and a meaningful spatial pattern
(`0.956` component correlation), but canonical error falls only `70.55%`
instead of the required `95%`; residual RMS remains `29.45%` of target RMS.
The composite loss continues decreasing after the best canonical checkpoint
while pointwise reconstruction worsens, so more identical training is not
supported as the next diagnostic.

The analysis report is
[`run/decoder_overfit_seed42/analysis_20260801/RESULTS.md`](run/decoder_overfit_seed42/analysis_20260801/RESULTS.md).
At that point the original gated plan called for a canonical-only one-frame
audit. That path is now set aside in favour of the non-gated oracle controlled
comparison above; learned DINO remains unauthorized.

### Diagnostic 1B: canonical-only single-frame overfit

Diagnostic 1B preserves the selected training UID/frame, Gate-1E initialization,
zero-DINO geometry pathway, trainable prefixes, optimizer, step budget and
frozen thresholds from Diagnostic 1. Its only experimental change is the scalar
used for backpropagation:

```text
Diagnostic 1:  canonical + 0.50 strain + 0.25 edge + 0.25 local velocity
Diagnostic 1B: canonical only
```

The canonical term is the existing radius-normalized Smooth-L1/Huber loss with
`beta=0.01`; it is not a new target or metric. Auxiliary terms are still
computed and logged but do not contribute gradients. A pass still requires at
least 95% canonical-NRMSE reduction and 90–110% recovered magnitude.

From a csh/tcsh lab-server shell:

```csh
mkdir -p v42/runs/canonical_overfit_seed42
setenv PYTHONPATH "v2/src:v3/src:v4/src"
nohup .venv-v41/bin/python -u v42/run_canonical_overfit.py \
  --device cuda \
  --seed 42 \
  --steps 2000 \
  --lr 1e-3 \
  --log-every 25 \
  --gate1e-root v42/runs/gate1e_seed42_456 \
  --runs v42/runs/canonical_overfit_seed42 \
  >& v42/runs/canonical_overfit_seed42/console.log &
echo $! > v42/runs/canonical_overfit_seed42/launcher.pid
```

Use singular `v42/run/gate1e_seed42_456` if appropriate. If 1B passes while
Diagnostic 1 failed, basic output capacity is supported and composite-objective
conflict becomes the leading explanation. If 1B also fails, the next audit must
target spatial representation, decoder capacity or conditioning. Neither result
directly authorizes temporal, material or learned-DINO experiments.

#### Diagnostic-1B result

The downloaded canonical-only run is complete and integrity-valid but fails the
unchanged one-frame gate. Canonical NRMSE falls `71.36%` versus `70.55%` under
the composite objective, only a `0.81` percentage-point gain and far below the
required `95%`. Predicted magnitude is correct (`95.89%` of target), while
pointwise residual remains `28.64%` of target RMS. Removing auxiliary-loss
gradients therefore does not remove the spatial reconstruction ceiling.

The analysis report is
[`run/canonical_overfit_seed42/analysis_20260801/RESULTS.md`](run/canonical_overfit_seed42/analysis_20260801/RESULTS.md).
The smallest next isolation is a matched normalized-MSE one-frame fit; temporal,
material and learned-DINO stages remain unauthorized.

### Diagnostic 1C: restored composite with normalized canonical MSE

Diagnostic 1C restores the auxiliary objective from Diagnostic 1 and changes
only the canonical error geometry:

```text
canonical_mse = sum(mask * ||prediction - target||^2 / radius^2) / sum(mask)
objective = canonical_mse + 0.50 strain + 0.25 edge + 0.25 local_velocity
```

The square root of `canonical_mse` is exactly the reported canonical NRMSE, so
large pointwise residuals now receive gradients proportional to their size.
The selected UID/frame, initialization, zero-DINO pathway, optimizer, 2,000-step
budget and auxiliary coefficients are unchanged. The original 95% reduction
criterion is retained as a strict near-exact diagnostic for direct comparison;
analysis should also report non-collapse capability separately rather than use
that strict threshold as the sole progression criterion.

From a csh/tcsh lab-server shell:

```csh
mkdir -p v42/runs/mse_overfit_seed42
setenv PYTHONPATH "v2/src:v3/src:v4/src"
nohup .venv-v41/bin/python -u v42/run_mse_overfit.py \
  --device cuda \
  --seed 42 \
  --steps 2000 \
  --lr 1e-3 \
  --log-every 25 \
  --gate1e-root v42/runs/gate1e_seed42_456 \
  --runs v42/runs/mse_overfit_seed42 \
  >& v42/runs/mse_overfit_seed42/console.log &
echo $! > v42/runs/mse_overfit_seed42/launcher.pid
```

Use singular `v42/run/gate1e_seed42_456` if appropriate. This experiment does
not load validation/test data or train DINO, and it does not automatically
authorize later stages.

#### Diagnostic-1C result

The downloaded MSE run is complete and integrity-valid but does not improve the
fit. Best canonical NRMSE is `0.123721`, versus `0.109833` for composite Huber
and `0.106817` for canonical-only Huber. Error reduction falls to `66.82%` and
relative residual rises to `33.18%`, although approximate magnitude and broad
direction remain learned. The MSE run also shows a transient optimization spike
near step 1850 before recovering.

The analysis report is
[`run/mse_overfit_seed42/analysis_20260801/RESULTS.md`](run/mse_overfit_seed42/analysis_20260801/RESULTS.md).
Across the three matched objectives, loss formulation no longer appears to be
the primary spatial bottleneck; pointwise conditioning and decoder
representation are the next diagnostic focus.

## Gate 2B: family-balanced deformation-signal diagnostic

Gate 2B is an isolated follow-up to the failed Gate-2 screen. It does not
overwrite Gate 2, authorize Gate 3, use test data, or train real DINO. It tests
whether the near-zero solution was encouraged by duplicated rigid supervision
and weak soft-deformation scaling while keeping the model, data split, stage
weights, checkpoint selection and frozen evaluation screen unchanged.

The reviewed variants are:

| Variant | Explicit rigid-family coefficient | Extra rigid-zero term | Maximum soft amplification |
|---|---:|---:|---:|
| `balanced_x1` | 0.25 | 0 | 1x |
| `balanced_x5` | 0.25 | 0 | 5x |
| `balanced_x20` | 0.25 | 0 | 20x |

Training batches remain size one and the UID-balanced sampler alternates soft
and rigid families exactly; the runner rejects an odd draw count. For each soft
episode, it computes one detached scale

```text
max(q95 framewise target canonical RMS, 0.005 * reference radius)
```

and applies `min(radius / scale, variant cap)` to canonical-displacement and
local-velocity residuals. A single episode-level scale preserves relative
frame magnitudes and timing. Strain and edge-length terms retain their existing
normalization. Rigid canonical, strain, edge and velocity supervision remains,
but its aggregate contribution is multiplied by 0.25 and the additional
rigid-zero term is removed. Rigid safety is still enforced by the unchanged
`<0.1%` reference-radius validation threshold.

The existing Gate-2 run is the matched legacy control, so the Gate-2B runner
starts only the three new variants. From a csh/tcsh lab-server shell:

```csh
mkdir -p v42/runs/gate2b_seed42_456
setenv PYTHONPATH "v2/src:v3/src:v4/src"
nohup .venv-v41/bin/python -u v42/run_gate2b.py \
  --device cuda \
  --seeds 42 456 \
  --variants balanced_x1 balanced_x5 balanced_x20 \
  --epochs 120 \
  --draws 40 \
  --patience 20 \
  --plateau-patience 5 \
  --gate1e-root v42/runs/gate1e_seed42_456 \
  --runs v42/runs/gate2b_seed42_456 \
  >& v42/runs/gate2b_seed42_456/console.log &
echo $! > v42/runs/gate2b_seed42_456/launcher.pid
```

As with Gate 2, use `--gate1e-root v42/run/gate1e_seed42_456` if the reviewed
source checkpoints are under the downloaded singular `run` directory. Each
variant/seed writes to its own directory and records its complete loss contract,
source hashes, frozen-path identity checks and unchanged Gate-2 screen outcome.

### Gate-2B result

The retrieved six-run Gate-2B matrix under `v42/run/gate2b_seed42_456` is
complete and integrity-valid, but every variant fails the unchanged Gate-2
screen. `balanced_x1` is numerically best with only `0.584%` canonical-NRMSE
and `0.080%` strain-RMSE improvement. The 5x and 20x variants do not amplify
predicted deformation behavior: magnitude correlation remains below `0.31`, no
onset is detected, and median peak timing error remains 51–52 frames. Rigid
safety and frozen COM/rotation identity pass in all runs.

The validation-only report is
[`run/gate2b_seed42_456/analysis_20260731/RESULTS.md`](run/gate2b_seed42_456/analysis_20260731/RESULTS.md).
Gate 3 / real-DINO training remains unauthorized.

## Gate 2C: total-mass stage-balanced diagnostic

Gate 2C isolates temporal frame imbalance while retaining the numerically best
Gate-2B condition, `balanced_x1`: family-balanced batches, rigid-family
coefficient `0.25`, no extra rigid-zero term, and no soft deformation
amplification. Only the training-frame weights change.

For each stage present in an episode, frame weights are computed as

```text
unnormalized_weight[t] = stage_importance[stage[t]] / frames_in_stage[stage[t]]
weight = unnormalized_weight / mean(unnormalized_weight)
```

Consequently, the total weight of a stage is proportional to its declared
importance rather than its saved-frame count. The existing importance values
remain `1/2/3/4/3/1` for free flight, contact onset, compression, peak,
recovery and post-event. Weights are computed over the 59 target frames only,
are detached target-derived preprocessing and are not model inputs. There is no
weight cap: retaining the historical cap of 4 would specifically suppress rare
one-frame stages and undo total-mass balancing. Mean episode weight remains one.

Validation reporting, checkpoint selection and the complete frozen Gate-2
screen continue to use the historical evaluation definition. Gate 2C therefore
tests training supervision only and remains directly comparable with
`balanced_x1`.

From a csh/tcsh lab-server shell:

```csh
mkdir -p v42/runs/gate2c_seed42_456
setenv PYTHONPATH "v2/src:v3/src:v4/src"
nohup .venv-v41/bin/python -u v42/run_gate2c.py \
  --device cuda \
  --seeds 42 456 \
  --epochs 120 \
  --draws 40 \
  --patience 20 \
  --plateau-patience 5 \
  --gate1e-root v42/runs/gate1e_seed42_456 \
  --runs v42/runs/gate2c_seed42_456 \
  >& v42/runs/gate2c_seed42_456/console.log &
echo $! > v42/runs/gate2c_seed42_456/launcher.pid
```

Use `--gate1e-root v42/run/gate1e_seed42_456` if the reviewed sources are under
the downloaded singular `run` directory. The runner trains Gate 2C only: it
does not use test data, train real DINO or dispatch Gate 3.

### Gate-2C result

The retrieved two-seed run under `v42/run/gate2c_seed42_456` is complete and
integrity-valid, but it fails the unchanged Gate-2 screen. Total-mass balancing
was active and strongly upweighted rare frames, yet two-seed improvement over
zero-local is only `0.665%` canonical NRMSE and `0.073%` strain RMSE. Predicted
peaks remain only `3.2–3.4%` of target magnitude, onset is never detected, and
median peak error is 52 frames. Rigid safety passes narrowly, reaching `0.0986%`
of reference radius against the `0.1%` maximum.

The validation-only report is
[`run/gate2c_seed42_456/analysis_20260801/RESULTS.md`](run/gate2c_seed42_456/analysis_20260801/RESULTS.md).
Gate 3 / real-DINO training remains unauthorized.
