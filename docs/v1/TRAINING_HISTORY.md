# V1 Consolidated Training History

This file preserves the meaningful training lineage so the canonical `rollout_s4` run can be retained without relying on every intermediate run directory.

## Dataset and split

```text
scenes:       22 PhysTwin scenarios
train/val/test: 16 / 3 / 3 complete scenarios
particles:    up to 2,048, padded with masks
grid:         32 x 32 x 32
DINO:         frozen DINOv3 ViT-B/16, 768D -> learned 16D
frame rate:   normally 30 FPS
action:       prescribed controller position/velocity, 30 points
```

Split manifests remain under `data/shared/splits/`; they resolve to frozen V1
caches under `data/v1/cache/`.

## Stage 0: single-scene smoke training

```text
run:        archive/v1_runs/one_step
scene:      single_lift_zebra
epochs:     20
checkpoint: last.pt
```

Purpose: validate the complete DINO extraction, cache, particle-grid transfer, MPS training and visualization pipeline. This was not a generalization experiment.

## Stage 1: initial multi-scene continuation

```text
run:    archive/v1_runs/multiscene_resume
epochs: 21-25
```

This stage still selected by aggregate composite loss. Validation composite loss fell from `0.00317325` to `0.00186423`. Held-out one-step evaluation nevertheless showed:

```text
model normalized particle mean:       0.00812116
persistence normalized particle mean: 0.00498154
constant-velocity mean:               0.01592957
```

This exposed loss-scale and checkpoint-selection misalignment.

## Stage 2: particle-focused one-step training

The particle objective changed to normalized Smooth-L1 with beta `0.01`. Checkpoint selection, scheduling and early stopping moved to validation particle mean.

Best meaningful checkpoint:

```text
source run:             archive/v1_runs/multiscene_particle
epoch:                  32
validation particle:    0.0040866636
validation persistence: 0.0038641035
ratio:                  1.057597
```

Continuation through epoch 40 reduced LR from `1e-4` to `2.5e-5` but did not improve epoch 32. The lowest composite objective occurred at epoch 38:

```text
validation particle: 0.0041005282
objective:           0.0020028832
```

The epoch-38 objective checkpoint seeded the subsequent rollout experiment. Its particle error differed from epoch 32 by only about 0.34%.

Motion stratification showed why aggregate persistence was difficult:

```text
lowest-motion quarter model/persistence:  1.291
second quarter:                           1.184
third quarter:                            1.038
highest-motion quarter:                   0.909
moving half:                              0.957
```

## Stage 3: two-step rollout fine-tuning

```text
run:                     archive/v1_runs/rollout_s2
best epoch:              5
validation recurrent:    0.0051012548
recurrent persistence:   0.0049138445
ratio:                    1.038139
teacher-forced:           0.0040969337
teacher guardrail:        passed
```

The starting recurrent baseline was `0.00518432`, so epoch 5 improved it by about 1.6%. Continuation through epoch 10 and LR reduction to `6.25e-6` did not improve epoch 5.

Motion-stratified two-step validation:

```text
lowest-motion quarter ratio:  1.251
second quarter:               1.169
third quarter:                1.009
highest-motion quarter:       0.909
moving half:                  0.944
```

## Stage 4: four-step rollout fine-tuning — canonical V1

```text
run:                     v1/artifacts/rollout_s4
checkpoint:              best.pt
best epoch:              3
validation recurrent:    0.0071929494
recurrent persistence:   0.0070527965
ratio:                    1.019872
teacher-forced:           0.0041393263
teacher guardrail:        passed
```

Epochs 4-5 worsened validation. A separate continuation through effective epoch 8 reduced LR to `5e-6` but never beat epoch 3; it is recorded as `rollout_s4_2_no_good` and is not canonical.

Evaluation of the canonical checkpoint over the three untouched test scenarios:

```text
model particle mean:       0.0061416072
persistence:               0.0049815430
constant velocity:         0.015929565
model / persistence:       1.2329
```

## Velocity-feedback diagnostic

Validation over rolling windows using the canonical step-4 checkpoint:

| Alpha | Horizon 4 | Horizon 8 | Horizon 16 |
|---:|---:|---:|---:|
| 0.25 | 1.110 | 1.078 | 1.013 |
| 0.50 | 1.080 | 1.047 | 0.998 |
| 0.75 | 1.059 | 1.036 | 0.994 |
| 1.00 | **1.048** | **1.031** | **0.992** |

Values are model/persistence ratios; lower is better. Full feedback is best at relevant short horizons.

Long single-origin trajectories:

```text
double_stretch_zebra, frame 192:
  alpha 1.00: 343.20 mm
  alpha 0.25: 303.33 mm

single_push_sloth, frame 64:
  alpha 1.00: 278.92 mm
  alpha 0.25: 277.96 mm

single_lift_cloth_4, frame 128:
  alpha 1.00: 108.49 mm
  alpha 0.25: 108.21 mm
```

Damping modestly delays very late zebra failure but does not cure divergence. Sloth and cloth show negligible sensitivity and no out-of-grid particles.

## Canonical files to retain

```text
v1/artifacts/rollout_s4/best.pt
v1/artifacts/rollout_s4/history.jsonl
v1/artifacts/rollout_s4/velocity_feedback_val.csv
v1/artifacts/rollout_s4/velocity_feedback_unseen_zebra.csv
v1/artifacts/rollout_s4/velocity_feedback_push_sloth_long.csv
v1/artifacts/rollout_s4/velocity_feedback_lift_cloth_long.csv
v1/artifacts/rollout_s4/unseen_push_sloth.mp4
v1/artifacts/rollout_s4/unseen_lift_cloth4.mp4
```

All numerical values above were transcribed from the saved histories and diagnostic CSVs on 2026-07-04.
