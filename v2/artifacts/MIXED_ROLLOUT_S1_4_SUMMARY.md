# Mixed-Horizon 1-4 Rollout Experiment

Date: 2026-07-08

## Hypothesis

Test whether uniformly mixed rollout horizons, stronger teacher-forced
supervision, smaller training slices, and frequent validation avoid the
regression caused by fixed-horizon full-epoch fine-tuning.

## Configuration

```text
initial checkpoint:      v2/runs/v2b_mps/one_step/best.pt
training horizons:       uniform integer sample from 1 through 4 per batch
validation horizon:      fixed 4 steps
training slice:          0.25 of a sampled full epoch
teacher weight:          1.0
initial learning rate:   2.5e-6
plateau learning rates:  1.25e-6, then 1e-6
early-stop patience:     6 validation intervals
device/environment:      MPS / mpm-dino-poc
```

The sampled horizons were approximately uniform. Across the six intervals,
each horizon appeared between 556 and 635 times.

## Result

The guarded epoch-0 incumbent was:

```text
four-step recurrent particle mean: 0.0065195928
teacher-forced particle mean:      0.00386395
```

Training results:

| Interval | Four-step recurrent | Teacher-forced | Guard |
|---:|---:|---:|:---:|
| 1 | 0.00705903 | 0.00402773 | pass |
| 2 | 0.00702060 | 0.00402753 | pass |
| 3 | 0.00732283 | 0.00413651 | pass |
| 4 | 0.00743242 | 0.00418028 | pass |
| 5 | 0.00758505 | 0.00423387 | pass |
| 6 | 0.00779235 | 0.00431300 | fail |

The experiment early-stopped with no guarded improvement. `best.pt` therefore
retains epoch 0 and has model tensors identical to the one-step checkpoint.

## Interpretation

The changes reduced damage but did not improve recurrence. The best trained
mixed-horizon interval (`0.00702060`) was substantially better than the first
fixed step-4 fine-tuning epoch (`0.00839682`) and preserved the teacher guard
for five intervals. It was still 7.7% worse than the inherited four-step
incumbent.

Thus mixed 1-4 horizons and stronger teacher supervision help prevent
catastrophic forgetting, but they do not solve the recurrent optimization
problem in this form. Increasing the horizon to 8 is not supported by this
result.
