# V4.3 causal-contact oracle-replacement experiment

## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: plan/run preparation
- Origin Date: 2026-08-03
- Verification Status: IMPLEMENTED_AND_SMOKE_VERIFIED
- Test data authorized: no

## Question

How much of the V4.2 oracle-conditioned deformation improvement survives when
future ground-truth contact and event timing are replaced with features computed
only from the frozen predicted COM/rotation rigid trajectory?

## Frozen ceiling

The existing `adapter_full` checkpoints are not retrained. The runner validates
and hashes them as the oracle ceiling:

| Seed | Oracle objective | Best epoch |
|---:|---:|---:|
| 42 | 0.7771909 | 13 |
| 456 | 0.7789131 | 8 |

They receive future-ground-truth contact, curvature, stage, and event time.
They are reference ceilings, not causal models.

## Causal feature construction

For each batch, a second frozen Gate-1E model predicts COM and rotation from
`x0`, `x1`, graph, gravity, timestep, and floor only. Canonical deformation is
forced to zero:

\[
\tilde x_{t,i}=\hat c_t+q_i\hat R_t.
\]

The causal continuous condition contains:

1. normalized signed floor gap, clipped after division by `0.1R`;
2. one-sided smooth proximity `sigmoid(-gap/(0.02R))`;
3. normalized vertical velocity;
4. normalized time relative to the first predicted point within `0.01R` of the
   floor.

Discrete oracle-stage channels remain zero. The builder never reads future
target positions, target activity, target-derived contact, or stage labels.
Ground-truth stages remain permitted only for selecting event frames in the
training objective.

## Matched arms

All trained arms have the same 15-channel adapter architecture, Gate-1E source,
four static curvature channels, seven zero discrete-stage channels, soft-only
event-frame objective, sampler, optimizer, and checkpoint selection.

| Arm | Point contact | Event-relative time | Curvature |
|---|---|---|---|
| `static_control` | zero | zero | fixed causal proxies |
| `causal_timing_only` | zero | rigid-proxy onset | fixed causal proxies |
| `causal_continuous` | signed gap, proximity, vertical velocity | rigid-proxy onset | fixed causal proxies |
| Oracle ceiling | ground-truth future contact | ground-truth stages/time | fixed proxies |

Curvature is held constant rather than interpreted as a positive feature. This
matrix isolates causal timing and the incremental pointwise continuous signal.

## Primary analysis

For each seed, report validation event-normalized MSE and compute oracle-benefit
recovery relative to the matched static control:

\[
\text{recovery}
=
\frac{L_{static}-L_{causal}}
     {L_{static}-L_{oracle}}.
\]

Also report canonical NRMSE, strain RMSE, predicted/target peak ratio,
magnitude correlation, onset/peak timing, and every validation UID. Do not
promote a causal method from the scalar objective alone.

## Interpretation

- Timing-only helps: reliable onset is a useful major component.
- Continuous beats timing-only: noisy pointwise gap/proximity retains useful
  localization information.
- Continuous approaches oracle: analytic preprocessing may be sufficient.
- Continuous helps but leaves a large gap: add a learned residual contact/gap
  correction.
- Causal arms remain near static: oracle benefit depends on contact information
  the rigid proxy does not recover.

## Verification

- 43 relevant unit tests pass.
- A regression test poisons future targets, target masks, and target stages and
  verifies bit-identical causal conditions.
- One-batch CPU training, validation, checkpointing, protected-path checks, and
  completion reporting pass for `causal_continuous`.

## Targeted third optimization replicate

The optional `--source-seed` runner mode holds the frozen Gate-1E physical/COM/
rotation model and oracle ceiling fixed while changing the adapter training
seed. The reviewed third replicate uses training seed 123 with source seed 42
and trains only `static_control` and `causal_continuous`. This isolates whether
the successful causal seed-42 result depends on adapter initialization/sampling.
It is not a fully independent end-to-end seed, which would require training new
Gate-1E and oracle seed-123 checkpoints first.
