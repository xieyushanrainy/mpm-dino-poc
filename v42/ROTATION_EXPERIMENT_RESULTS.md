# V4.2 rotation experiment results and Gate 2 decision

## Scope

This memo closes the current V4.2 rotation investigation and records the
decision governing Gate 2. It covers Gate 1B through Gate 1F. Retrieved run
configurations, checkpoints, histories and `RUN_COMPLETE.json` files are
authoritative for what was actually trained.

No rotation experiment passed its predeclared promotion screen. Gate 1E is
retained only as the least damaging learned rotation placeholder for protected
local-architecture experiments. Rotation is not considered solved.

## Fixed decomposition

V4.2 reconstructs future points as

\[
\hat x_{t,i}
=
\hat c_t+\hat R_t(q_i+\hat d_{t,i}),
\qquad
q_i=x_{1,i}-c_1.
\]

The COM path is physical-only. Rotation is explicit. The local head predicts
canonical non-rigid displacement, for which zero is the correct rigid-object
solution.

## Experiment progression

### Gate 1B: corrected COM and chordal rotation

Gate 1B anchored the learned COM residual to zero at H1, retained the
finite-difference ballistic baseline, trained rotation with a smooth chordal
loss and reported geodesic radians. COM improved without sacrificing H1, but
rotation did not materially improve over identity.

Gate 1B remains the authoritative protected COM source.

### Rotation audit and Gate 1C

The validation-only audit tested identity and a constant-angular-velocity
baseline inferred by Kabsch alignment from `x0` to `x1`. The observed angular
motion did not provide a useful baseline. Gate 1C therefore used an
identity-anchored axis-angle residual, but joint COM/rotation optimization
remained unstable.

### Gate 1D: protected attention rotation

Gate 1D froze the Gate-1B physical trunk and COM head and trained a point
attention rotation adapter. It learned some impact-time rotation but developed
large H59 drift. The experiment established that rotation could be isolated
without changing COM, but it did not pass.

### Gate 1E: protected angular dynamics

Gate 1E retained the protected attention inputs and predicted per-frame angular
velocity changes that were accumulated and integrated:

\[
\Delta\omega_t=f_\theta(h_t),\qquad
\omega_t=\omega_{t-1}+\Delta\omega_t,\qquad
R_t=R_{t-1}\exp(\Delta t[\omega_t]_\times).
\]

Both seeds completed 120 epochs and failed the identity screen. The reported
diagnostic checkpoints were:

- `v42/run/gate1e_seed42_456/seed42/best_total.pt`, epoch 20;
- `v42/run/gate1e_seed42_456/seed456/best_total.pt`, epoch 27.

Gate 1E post-contact identity-relative results were:

| Seed | Overall | Rigid Panel Z | Rigid Panel V | Soft Panel Z |
|---:|---:|---:|---:|---:|
| 42 | +1.38% | -1.23% | +4.35% | +0.66% |
| 456 | -0.85% | -6.09% | +1.00% | +0.51% |

Positive values mean lower error than identity. Panel V showed a small useful
signal; Panel Z did not. Gate 1E nevertheless had substantially better
late-horizon behaviour than Gate 1F.

### Gate 1F: contact-driven rotation

Gate 1F compared:

- contact-gated absolute residual rotation;
- contact-impulse rotation with damped angular integration.

Both variants inferred contact from deployable physical/geometry inputs.
Ground-truth contact stages and lowest-four contact patches were auxiliary
training targets only.

All four jobs completed 120 epochs. None produced an eligible checkpoint.

| Variant | Seed | Reported epoch | Pooled active improvement | Rigid Z | Rigid V |
|---|---:|---:|---:|---:|---:|
| Absolute | 42 | 1 | +4.55% | -4.2% | +9.9% |
| Absolute | 456 | 1 | -0.18% | -1.6% | +0.7% |
| Impulse | 42 | 16 | +15.48% | -2.2% | +26.3% |
| Impulse | 456 | 1 | -1.14% | -1.5% | -0.9% |

The large impulse seed-42 gain came entirely from Panel V and did not
reproduce. Rigid Panel Z regressed in every Gate-1F job. Contact auxiliary
losses improved while active rotation, inactive false rotation and H59
stability worsened. This showed that learning contact onset/region did not
provide a reliable mapping to rotation axis, sign and magnitude.

H59 validation geodesic error:

| Model | Seed | Rigid Z | Rigid V |
|---|---:|---:|---:|
| Identity | — | 1.201° | 0.979° |
| Gate 1E | 42 | 1.233° | 0.685° |
| Gate 1E | 456 | 1.599° | 1.053° |
| Gate 1F absolute | 42 | 1.609° | 1.060° |
| Gate 1F absolute | 456 | 1.825° | 1.433° |
| Gate 1F impulse | 42 | 2.903° | 2.454° |
| Gate 1F impulse | 456 | 1.747° | 1.293° |

The full Gate-1F report is
`v42/run/gate1f_seed42_456_v3/RESULTS.md`.

## Interpretation

The current data and objectives contain a weak, seed-dependent Panel-V
rotation signal. They do not support reliable Panel-Z rotation. Additional
epochs are not justified: Gate-1F learning rates reached their floor, contact
losses continued to improve, and rotation metrics deteriorated.

The rotation investigation therefore has diminishing value for the immediate
architecture question. Continuing to elaborate rotation would delay testing
the canonical local-deformation pathway without evidence that another small
rotation change will resolve the underlying identifiability problem.

## Decision for Gate 2

Proceed to the protected geometry-only canonical-deformation screen while
freezing Gate 1E rotation as an operational placeholder.

This decision does **not** promote Gate 1E and does **not** claim rotation is
solved. It permits an independent test of whether the canonical local head can
learn deformation.

For each seed:

1. Start from that seed's Gate-1E `best_total.pt`.
2. Freeze the physical trunk, COM head and complete rotation branch.
3. Allow local losses to update only the geometry adapter and canonical local
   head (`alpha=0`).
4. Do not backpropagate reconstructed world losses.
5. Use canonical displacement and strain as the primary selection metrics.
6. Report world error only as a diagnostic because frozen rotation error can
   contaminate it.
7. Continue reporting identity rotation beside Gate 1E.

Gate 2 must first compare zero-local output with the geometry-only/zero-DINO
local model. Geometry-only means the matched zero-DINO control with the same
adapter architecture and real validity mask; it is not an ablated architecture.

Real DINO is not part of the initial Gate-2 learnability screen. It may be
tested only after the geometry-only local pathway demonstrates nonzero,
correctly timed deformation learning. The later matched visual attribution
matrix must use geometry/zero-DINO, aligned real DINO and point-shuffled DINO
with identical frozen physical checkpoints, local initialization, sampling and
stage weights.

Gate 2 does not authorize gradients into the physical trunk, COM head or
rotation head. Coupling experiments remain a later gate.

