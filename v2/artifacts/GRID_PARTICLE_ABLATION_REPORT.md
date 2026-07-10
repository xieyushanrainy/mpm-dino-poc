# Grid–Particle Fusion Ablation Report

Date: 2026-07-08

## Architectures

- **Fused:** existing 3D U-Net grid pathway plus direct particle-held features.
- **Grid-only:** displacement head receives only sampled U-Net decoder features,
  predicted occupancy, and predicted grid velocity.
- **Particle-only:** no grid or U-Net. A particle MLP receives persistent/local
  particle state and a 64D permutation-invariant controller context produced by
  a shared particle–controller pair MLP with masked mean/max pooling.

All models use final-layer DINO. Checkpoints were selected by validation
particle distance without rollout fine-tuning.

## Seed-42 screen

| Variant | Parameters | H1 | H4 | H8 | H16 | H4 calls/s |
|---|---:|---:|---:|---:|---:|---:|
| Fused | 937,759 | 0.003804 | 0.006630 | 0.010314 | 0.017065 | 67 |
| Grid-only | 933,407 | 0.003900 | 0.007091 | 0.011288 | 0.019151 | 66 |
| **Particle-only** | **82,675** | **0.003643** | **0.006166** | **0.009521** | **0.015839** | **123** |

Particle-only was selected for replication. It uses 91.2% fewer parameters
than fused and achieved approximately 1.8 times the measured H4 forward-call
throughput. Observed MPS allocation was 6.79 MB versus 10.21 MB for fused.

## Three-seed validation

Mean ± sample standard deviation:

| Horizon | Fused | Particle-only | Mean improvement | Particle wins |
|---:|---:|---:|---:|---:|
| 1 | 0.003822 ± 0.000094 | **0.003676 ± 0.000033** | 3.83% | 3/3 |
| 4 | 0.006574 ± 0.000172 | **0.006266 ± 0.000112** | 4.68% | 3/3 |
| 8 | 0.010150 ± 0.000268 | **0.009732 ± 0.000237** | 4.12% | 3/3 |
| 16 | 0.016664 ± 0.000434 | **0.016371 ± 0.000566** | 1.76% | 1/3 |

Particle-only consistently improves horizons 1, 4, and 8. At horizon 16 its
mean remains lower because of a strong seed-42 result, but fused narrowly wins
seeds 123 and 456. The long-horizon validation advantage is therefore not
seed-consistent.

## Untouched test results

| Horizon | Fused mean | Particle-only mean | Improvement | Particle wins |
|---:|---:|---:|---:|---:|
| 1 | 0.005458 | **0.004978** | 8.79% | 3/3 |
| 4 | 0.010606 | **0.009845** | 7.18% | 3/3 |
| 8 | 0.017743 | **0.016667** | 6.07% | 3/3 |
| 16 | 0.031528 | **0.030206** | 4.19% | 3/3 |

Particle-only beats fused for every seed and every measured test horizon.

## Motion-stratified observation

At seed-42 validation H4, model/persistence ratios from lowest to highest
motion quartile were:

```text
fused:        1.171, 1.003, 0.862, 0.842
grid-only:    1.254, 1.104, 0.938, 0.878
particle-only:1.093, 0.928, 0.842, 0.762
```

Direct particle–controller interaction improves both stationary and active
windows, with the largest useful margin in the highest-motion quartile.

## Conclusion

Fusion does not satisfy the acceptance criterion. The fused model fails to
beat particle-only at horizons 1, 4, and 8 across all seeds, and loses at every
held-out test horizon. Grid-only is the weakest seed-42 variant, showing that
voxel aggregation does not preserve enough particle-specific information by
itself.

For this dataset, a compact particle MLP with explicit reference/deformation
state and direct permutation-invariant controller interaction is more accurate,
smaller, and faster than the 3D U-Net architecture. The U-Net/grid pathway
should be reconsidered rather than retained by default.

This does not prove grids are generally unhelpful for deformable dynamics. The
tracked surface dataset is sparse, controller points are directly observed,
and the current U-Net receives lossy voxel averages at resolution 32. A future
grid model would need a stronger justification such as contact between many
objects, volumetric state, or grid-native physical fields.

## Artifacts

```text
v2/runs/grid_particle_ablation/seed*/
v2/runs/grid_particle_ablation/selected_alternative.txt
```
