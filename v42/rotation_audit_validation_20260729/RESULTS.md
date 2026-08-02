# V4.2 validation-only rotation audit

No training or test data was used.

## Baseline comparison

| Group | Horizon | Identity (deg) | Constant angular (deg) | Improvement | Episode wins |
|---|---:|---:|---:|---:|---:|
| rigid/panel_V | H1 | 0.000 | 0.000 | +0.0% | 0% |
| rigid/panel_V | H8 | 5.609 | 5.609 | +0.0% | 50% |
| rigid/panel_V | H16 | 1.212 | 1.212 | +0.0% | 67% |
| rigid/panel_V | H30 | 3.259 | 3.259 | +0.0% | 50% |
| rigid/panel_V | H40 | 2.327 | 2.327 | +0.0% | 67% |
| rigid/panel_V | H59 | 0.979 | 0.979 | +0.0% | 50% |
| rigid/panel_Z | H1 | 0.000 | 0.000 | +0.0% | 0% |
| rigid/panel_Z | H8 | 3.960 | 3.960 | +0.0% | 60% |
| rigid/panel_Z | H16 | 1.039 | 1.039 | +0.0% | 60% |
| rigid/panel_Z | H30 | 1.770 | 1.770 | +0.0% | 60% |
| rigid/panel_Z | H40 | 1.883 | 1.883 | +0.0% | 60% |
| rigid/panel_Z | H59 | 1.201 | 1.201 | +0.0% | 60% |
| soft_body/panel_Z | H1 | 0.000 | 0.000 | -34.2% | 0% |
| soft_body/panel_Z | H8 | 0.467 | 0.467 | -0.0% | 20% |
| soft_body/panel_Z | H16 | 1.072 | 1.072 | +0.0% | 40% |
| soft_body/panel_Z | H30 | 2.750 | 2.749 | +0.0% | 60% |
| soft_body/panel_Z | H40 | 3.531 | 3.531 | +0.0% | 60% |
| soft_body/panel_Z | H59 | 11.774 | 11.774 | +0.0% | 60% |

## Diagnostics

- Constant-angular baseline overall improvement: +0.0%.
- Frames below the Kabsch `1e-3` conditioning threshold: 0.
- Correlation between minimum singular ratio and maximum angular-step change: -0.2962555079414257.
- Correlation between deformation residual and maximum angular-step change: -0.48073308328978037.
- Maximum observed x0-to-x1 rotation: 0.000056 degrees.
- Median impact angular-velocity change: 0.856 rad/s.

## Decision

The observed-step rotation is effectively zero, and constant-angular extrapolation does not improve identity. Do not adopt a residual-over-observed-angular-velocity rotation baseline for this dataset.

See `rotation_audit.json` for per-episode rotation vectors, angular velocities, conditioning values and horizon errors.
