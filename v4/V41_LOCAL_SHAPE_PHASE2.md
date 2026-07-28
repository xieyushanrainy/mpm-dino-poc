# V4.1 Phase 2: COM-normalized local-shape experiment

Date prepared: 2026-07-28

## Question

When global translation is removed from both training and selection, can DINO
improve centre-relative deformation beyond physical geometry, geometry-token,
and point-shuffled controls?

This is a local-shape experiment, not a world-trajectory replacement.

## Prediction target

Ground-truth future COM is used only to construct the supervised target:

```text
target_shape[t, i] = target_position[t, i] - target_COM[t]
```

It is never supplied to the model. The prediction is:

```text
predicted_shape =
    centred_ballistic_shape + predicted_zero_mean_local_residual
```

There is no world-position loss and no COM loss. Model selection uses only
centre-relative soft-body shape NRMSE at H16/H30/H40.

## Conditions

| Condition | Physical trunk | Region tokens | DINO |
|---|---|---|---|
| `physical_only` | yes | no | none |
| `geometry_tokens` | yes | four | zero |
| `real_dino` | yes | four | aligned real |
| `point_shuffled` | yes | four | point-shuffled |

The three token conditions are architecture-identical and must have
byte-identical complete initial weights per seed. All four conditions must have
byte-identical initial physical-trunk weights.

## Soft/rigid mixture

A soft-only experiment could learn an unconditional deformation prior. Phase 2
therefore uses:

- 30 soft-body draws per 40-draw epoch;
- 10 rigid draws per 40-draw epoch;
- UID-balanced sampling within each family;
- no family label in the model input.

Soft bodies provide the primary deformation signal. Rigid objects are negative
controls. A radius-normalized rigid penalty encourages the predicted local
correction to remain zero for rigid episodes.

Validation and test results must be reported separately:

- soft shape, edge-vector and normalized strain errors;
- rigid false local-deformation magnitude and rigid shape/strain errors.

## Loss

```text
1.00 * radius-normalized centre-relative shape
0.25 * radius-normalized edge-vector error
0.50 * normalized edge-strain error
0.25 * shape at H16/H30/H40/H59
0.25 * rigid zero-local-residual penalty
```

All-frame terms are weighted toward frames with larger ground-truth
centre-relative deformation. Per-frame weights are computed within each
training episode:

```text
weight[t] = 0.25 + deformation[t] / mean_deformation
```

and renormalized to mean one. This weighting uses training targets only and
does not expose future information to the model.

## Initial prototype

- Seeds: 42 and 456
- Conditions: four
- Scientific runs: eight
- CUDA FP32, AMP disabled
- Width 128, four blocks, four heads
- 40 draws per epoch
- Cap 80 epochs
- Early-stopping patience 15
- ReduceLROnPlateau patience 5
- Frozen V4.1 UID split

Based on the retrieved lab timing of approximately 14.6 seconds per epoch, the
eight-run matrix has an upper-bound training estimate of about 2.6 GPU-hours
if every run reaches epoch 80. Early stopping should usually reduce this.

## Predeclared screening criteria

Before expanding to a third seed, real DINO should:

1. beat `geometry_tokens` and `point_shuffled` on soft H30 or H40 shape NRMSE;
2. win both initial seeds at the chosen horizon;
3. improve the two-seed soft shape mean by at least 5%;
4. improve or remain consistent on edge-vector and strain errors;
5. keep soft H1 within 10% of `geometry_tokens`;
6. avoid material H59 instability;
7. not increase rigid predicted-local RMS by more than 10% relative to
   `geometry_tokens`;
8. show measurable fixed-weight degradation when real DINO is zeroed or
   point-shuffled.

Failure means DINO has not demonstrated useful local-shape information under
the present data and architecture.

## Lab command

Run from the repository root:

```bash
bash -lc 'mkdir -p v41/runs/local_shape_phase2_seed42_456; nohup env PYTHONPATH=v2/src:v3/src:v4/src .venv-v41/bin/python -u v41/run_local_shape_phase2_matrix.py --device cuda --seeds 42 456 --epochs 80 --patience 15 --draws 40 --no-amp --runs v41/runs/local_shape_phase2_seed42_456 > v41/runs/local_shape_phase2_seed42_456/console.log 2>&1 & echo $! > v41/runs/local_shape_phase2_seed42_456/launcher.pid'
```

The runner is resumable and refuses to mix changed settings or manifest
content into the same matrix.
