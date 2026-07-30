# V4.2

V4.2 is currently a design proposal for separating COM/rigid motion from local
deformation, correcting rotation-contaminated supervision, and testing
stage-aware deformation learning with controlled gradient routing.

Read in this order:

1. [`CONTEXT.md`](CONTEXT.md)
2. [`PLAN.md`](PLAN.md)

No V4.2 training is authorized or recorded in this directory yet.

Implementation scaffolding now lives in `v4/src/mpm_dino_v4/v42_*.py`.
It provides rotation-aware targets, protected global/local gradient routing,
stage metadata and separated losses. This does not record or authorize a
training run; experimental gates remain manual.

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
