# V4 Run Log and Current Conclusions

The subsequent correspondence-preserving V4.1 architecture screen is reported
separately in [`V41_RESULTS.md`](V41_RESULTS.md); the V4 results below are
preserved as prior evidence and are not architecture-identical controls for
V4.1.

## Dataset audit

The saved audit is `FLUID_AUDIT.json`. The balanced package contains 30 objects per family. Median frame-0-to-frame-1 COM drops are 5.72 mm rigid, 3.81 mm soft body, and 104.05 mm fluid. Fluid points cross the declared floor in 27/30 objects, and the median minimum active fraction is zero.

The fluid route is therefore gated out of primary training. Its rest-surface tracers, absent collision floor, and unresolved Mantaflow velocity units must be fixed and regenerated before a three-family generalization claim.

## Verification runs

- Unit suite: five tests passed.
- Cached all 90 fixed reference graphs and validated reciprocal adjacency.
- CPU smoke training: real-DINO, seed 42, hidden size 32, one graph layer, one epoch, two train and validation batches.
- Smoke checkpoint reload and validation recurrence completed through H4.

The natural-distribution constant-velocity test baseline on the 10 held-out rigid+soft objects produced object-weighted aggregate RMSE of 2.05 mm at H1, 14.54 mm at H4, 37.90 mm at H8, and 93.21 mm at H16. Per-family and per-object values are saved in `runs/constant_velocity_test.json`.

The smoke checkpoint is deliberately under-trained and cannot support a DINO conclusion.

## Real DINOv2 versus zero DINO

The initial matched experiment used real point-aligned DINOv2 and zero DINO with seeds 42, 123, and 456. A subsequent matched experiment added deterministic scene-shuffled DINO for the same seeds. Point-shuffled DINO remains deferred. All other architecture, data, sampling, optimization, and evaluation settings were identical.

The initial benchmark predicted about 20–25 seconds per epoch and 30–70 minutes for six plateau-driven runs. Actual model training took approximately 50 minutes. Exhaustive arbitrary-start test rollouts took another 24 minutes, largely because rigid Kabsch SVD falls back from MPS to CPU.

| Mode | Seed | Stop epoch | Final LR | Best epoch | Best validation objective |
|---|---:|---:|---:|---:|---:|
| zero | 42 | 16 | 5e-5 | 11 | 0.00124066 |
| zero | 123 | 15 | 5e-5 | 7 | 0.00124163 |
| zero | 456 | 10 | 1e-4 | 10 | 0.00125218 |
| DINOv2 | 42 | 24 | 5e-5 | 23 | 0.00123503 |
| DINOv2 | 123 | 23 | 2.5e-5 | 15 | 0.00123874 |
| DINOv2 | 456 | 14 | 1e-4 | 12 | 0.00124240 |

Object-weighted aggregate test RMSE, mean ± sample standard deviation across three seeds:

| Horizon | Constant velocity | Zero DINO | Real DINOv2 | DINO vs zero |
|---|---:|---:|---:|---:|
| H1 | 2.05 mm | 2.113 ± 0.011 mm | 2.113 ± 0.017 mm | -0.01% |
| H4 | 14.54 mm | 14.928 ± 0.175 mm | 14.866 ± 0.090 mm | +0.41% |
| H8 | 37.90 mm | 38.612 ± 0.759 mm | 38.218 ± 0.153 mm | +1.02% |
| H16 | 93.21 mm | 93.523 ± 3.094 mm | 91.295 ± 0.391 mm | +2.38% |

At H8, the real-versus-zero margin is 0.394 mm, smaller than the zero-DINO seed standard deviation of 0.759 mm. At H16, the 2.228 mm margin is also smaller than the 3.094 mm zero-DINO seed variation. Real DINO therefore does **not** beat zero DINO beyond seed variation under the predeclared criterion. It is also worse than constant velocity through H8, although it improves H16 over constant velocity by about 2.05%.

The same qualitative result holds when rigid and soft body are reported separately. Real DINO is slightly lower-error at longer horizons, but the improvements are small relative to seed variation.

Raw per-seed metrics are in `runs/{zero,real}/seed*/test_metrics.json`; the initial combined summary is in `runs/real_vs_zero_aggregate.json`. Each run directory retains its resolved configuration, complete `history.jsonl`, best and last checkpoints, and raw object/window rollout rows.

## Scene-shuffled DINO follow-up

The scene-shuffled control completed at epochs 9, 16, and 14 for seeds 42, 123, and 456. Final learning rates were `1e-4`, `1e-4`, and `5e-5`; best validation objectives were 0.00125119, 0.00124378, and 0.00124240.

| Horizon | Zero DINO | Scene-shuffled DINO | Real DINOv2 |
|---|---:|---:|---:|
| H1 | 2.113 ± 0.011 mm | 2.122 ± 0.043 mm | 2.113 ± 0.017 mm |
| H4 | 14.928 ± 0.175 mm | 15.017 ± 0.359 mm | 14.866 ± 0.090 mm |
| H8 | 38.612 ± 0.759 mm | 38.908 ± 1.164 mm | 38.218 ± 0.153 mm |
| H16 | 93.523 ± 3.094 mm | 94.334 ± 4.436 mm | 91.295 ± 0.391 mm |

Real DINO has the best mean at H4/H8/H16 and substantially lower seed variance at longer horizons. Relative to scene-shuffled DINO, its mean error is lower by 0.151 mm at H4, 0.690 mm at H8, and 3.039 mm at H16. Relative to zero DINO, the corresponding margins are 0.061 mm, 0.394 mm, and 2.228 mm.

This is a consistent weak-positive pattern, but it does not pass the predeclared strict threshold: at H8 and H16, the real-DINO margin over each control is smaller than the largest control seed standard deviation. Paired seeds are also not unanimous—shuffled seed 456 is better than real seed 456 at H8/H16, while real wins for seeds 42 and 123. The correct conclusion is therefore **promising but not confirmed**: real point-aligned DINO improves the mean and stability, yet the evidence is insufficient to separate that gain confidently from small-sample seed variability.

The complete three-control summary and verdict are saved in `runs/dino_controls_aggregate.json`.

## Track B full-trajectory experiment

The implementation and nine-test suite passed before full training. A
production-width MPS benchmark completed forward/backward without an
out-of-memory error. Five initial production epochs took 224.86 seconds,
approximately 45 seconds per epoch including validation. The initial four runs
required about five hours of MPS training. A later matched seed-456 run used a
100-epoch cap: zero DINO reached the cap with a 30.72 mm best validation
objective at epoch 88, while real DINO plateau-stopped at epoch 80 with a
27.73 mm best at epoch 50.

Seed 42 used the plateau rule. Zero stopped at epoch 128 with a best validation
H8/H16 objective of 30.79 mm at epoch 126; real DINO stopped at epoch 98 with
34.84 mm at epoch 89. Both seed-123 conditions were capped at epoch 96. Zero's
best was 30.48 mm at epoch 95 and real DINO's was 26.39 mm at epoch 96. The
seed-123 cap is matched, but neither condition should be described as
plateau-complete.

The frame-0 analytic baselines demonstrate why ballistic extrapolation is a
reference rather than a competitive contact model:

| Model | H1 | H4 | H8 | H16 | H59 |
|---|---:|---:|---:|---:|---:|
| Ballistic | 3.62 mm | 62.32 mm | 347.58 mm | 1442.66 mm | 19443.32 mm |
| Constant velocity | 7.28 mm | 49.23 mm | 81.72 mm | 60.05 mm | 158.21 mm |
| Track A1 real DINO, three-seed mean | 7.29 mm | 49.32 mm | 82.05 mm | 64.81 mm | 254.34 mm |
| Track A1 zero DINO, three-seed mean | 7.30 mm | 49.42 mm | 82.44 mm | 65.07 mm | 249.74 mm |
| Track B zero DINO, three-seed mean | 23.81 mm | 28.04 mm | 57.40 mm | 51.35 mm | 77.88 mm |
| Track B real DINO, three-seed mean | 29.76 mm | 34.36 mm | 62.90 mm | 57.79 mm | 64.76 mm |

These figures use only the rollout beginning at `X[0],X[1]` on the same ten
test objects. Track B zero DINO improves over both frame-0 Track A1 and the
best analytic baseline at H8 and H16. It also greatly improves H59, showing
that simultaneous trajectory prediction avoids Track A1's long-rollout drift.
However, it materially degrades H1: 23.81 mm versus 7.28 mm constant velocity.
Track B therefore meets the long-horizon part of the architectural objective
but fails the requirement not to materially degrade H1.

Object-weighted test RMSE by family, mean ± sample standard deviation over
seeds 42, 123, and 456:

| Condition | Family | H1 | H4 | H8 | H16 | H59 |
|---|---|---:|---:|---:|---:|---:|
| Zero DINO | aggregate | 23.81 ± 15.29 | 28.04 ± 0.48 | 57.40 ± 0.61 | 51.35 ± 0.28 | 77.88 ± 7.58 |
| Real DINO | aggregate | 29.76 ± 15.79 | 34.36 ± 4.73 | 62.90 ± 1.15 | 57.79 ± 2.98 | 64.76 ± 6.60 |
| Zero DINO | rigid | 24.49 ± 15.92 | 24.39 ± 2.25 | 90.26 ± 0.54 | 69.16 ± 0.66 | 118.57 ± 17.91 |
| Real DINO | rigid | 31.08 ± 15.80 | 28.36 ± 6.44 | 98.28 ± 2.93 | 81.83 ± 3.45 | 88.22 ± 6.91 |
| Zero DINO | soft body | 23.13 ± 14.65 | 31.70 ± 2.47 | 24.55 ± 0.88 | 33.54 ± 0.86 | 37.19 ± 2.79 |
| Real DINO | soft body | 28.45 ± 15.84 | 40.37 ± 6.55 | 27.52 ± 1.50 | 33.75 ± 4.14 | 41.29 ± 11.15 |

All table entries are millimetres. Real DINO is worse than zero DINO for all
three paired seeds at aggregate H1/H4/H8/H16. It improves aggregate H59 for all
three seeds, mainly through rigid trajectories, but worsens the three-seed
soft-body H59 mean. Real DINO also has more floor penetration than zero at H8
and H16. Thus pooled DINO utility is **not supported at the co-primary H8/H16
horizons** in this run. The contrary seed-123 and seed-456 validation results
did not generalize to the test objects.

The dominant Track B H8/H16 error is rigid-object COM motion, while local
rigidity residual remains sub-millimetre. This suggests the next architectural
work should target time-dependent global/contact dynamics and H1 anchoring,
rather than adding stronger shape constraints. A useful modification would
predict corrections relative to constant velocity or blend the ballistic and
constant-velocity references, with explicit early-horizon weighting.

Raw metrics are saved beside each checkpoint. The stopped seed-123 zero run is
preserved as `runs/track_b/zero/seed123_interrupted_epoch96`; its cap is matched
by `runs/track_b/real/seed123`. The earlier two-seed architecture comparison is
`runs/track_b/comparison_two_seed.json`; the completed three-seed comparison is
`runs/track_b/comparison_three_seed.json`.
