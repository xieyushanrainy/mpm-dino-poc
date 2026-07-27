# Pooled Track B shape-balanced loss experiment

Status: implementation tested; production training not launched.

## Purpose

Repeat the exact V4 Track B pooled-DINO/FiLM architecture on V4.1 while changing
only the loss composition. This tests whether the previous failure to capture
subtle shape change was caused partly by an objective dominated by duplicate
world-coordinate and COM terms.

The experiment retains:

- the frozen V4.1 UID split and Panel Z/V episodes;
- real pooled DINO versus architecture-identical zero DINO;
- seeds 42, 123 and 456;
- width 128, four blocks and four heads;
- 40 UID-balanced draws per epoch;
- CUDA FP32 without AMP;
- 150-epoch cap, early-stopping patience 30 and plateau patience 5;
- validation selection by normalized H16/H30/H40 RMSE with the matched-zero H1
  guard.

## Loss profile

Select `shape_balanced_v1`:

```text
1.00 radius-normalized world loss
0.50 radius-normalized COM loss
1.00 radius-normalized centre-relative shape loss
0.50 normalized edge-strain loss
0.25 radius-normalized H16/H30/H40 key loss
```

The fixed normalization scale for each object is its frame-0 reference radius.
It is never recomputed per future frame.

Centre-relative shape is:

```text
(predicted_position - predicted_COM)
-
(target_position - target_COM)
```

Normalized strain for reference edge `ij` is:

```text
(current_edge_length - rest_edge_length) / rest_edge_length
```

The loss compares predicted and target normalized strain. All normalized terms
use Smooth L1 with beta `0.01`.

This profile removes the legacy duplicate residual/position supervision and
aligns key-horizon training with the H16/H30/H40 checkpoint selector. H59 is
evaluation-only.

## Output isolation

Use a new matrix directory:

```text
v41/runs/track_b_pooled_shape_loss_cap150_p30_fp32/
```

Do not point the new command at the completed legacy pooled Track B directory.
The matrix entry point records `loss_profile` in both matrix and run
configurations and refuses to resume with a changed profile.

## One-line launch

Run from the repository root:

```bash
bash -lc 'mkdir -p v41/runs/track_b_pooled_shape_loss_cap150_p30_fp32; nohup env PYTHONPATH=v2/src:v3/src:v4/src .venv-v41/bin/python -u v41/run_track_b_pooled_matrix.py --loss-profile shape_balanced_v1 --runs v41/runs/track_b_pooled_shape_loss_cap150_p30_fp32 > v41/runs/track_b_pooled_shape_loss_cap150_p30_fp32/console.log 2>&1 & echo $! > v41/runs/track_b_pooled_shape_loss_cap150_p30_fp32/launcher.pid'
```

Monitor:

```bash
tail -f v41/runs/track_b_pooled_shape_loss_cap150_p30_fp32/console.log
```

Rerun the identical launch command to resume.

## Evaluation order

1. Validate all run markers, hashes, histories and resolved loss profiles.
2. Never substitute `last.pt` for an absent guarded `best.pt`.
3. Evaluate eligible checkpoints on H1/H8/H16/H30/H40/H59 and every frame.
4. Keep Panel Z and Panel V separate.
5. Compare world, COM and centre-relative shape errors explicitly.
6. Report normalized strain/edge behavior, penetration and rigid Kabsch error.
7. Apply the same frozen promotion rule; H59 remains diagnostic only.

The first scientific question is whether the zero-DINO physical backbone
improves centre-relative shape without unacceptable COM, H1 or penetration
regression. DINO conclusions remain secondary to that backbone check.
