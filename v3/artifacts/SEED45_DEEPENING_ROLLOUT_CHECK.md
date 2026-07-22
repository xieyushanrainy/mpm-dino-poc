# Seed 45 Deepening and Rollout Check

Date: 2026-07-12

## Goal

Improve the seen-family/unseen-object seed 45 final-DINO `latent_graph` model, then test whether rollout training at step 2 and step 4 helps without degrading recurrent validation.

Primary metric: mean of H4 and H8 particle error.

History file:

- `v3/runs/seed45_final_deep/stage_history.jsonl`

Visualized unseen/test object:

- `data/v2/cache/single_push_sloth.pt`

## Result Summary

| Stage | Status | H1 | H4 | H8 | H16 | H4/H8 mean |
|---|---|---:|---:|---:|---:|---:|
| Original seed45 final-DINO | baseline | 0.00380729 | 0.00713225 | 0.01274740 | 0.02628970 | 0.00993983 |
| One-step continuation, `lr=5e-5` | rejected | 0.00387324 | 0.00762271 | 0.01408259 | 0.02993074 | 0.01085265 |
| One-step continuation, `lr=1e-5` | accepted | 0.00371593 | 0.00686701 | 0.01220368 | 0.02507803 | 0.00953534 |
| Rollout-S2 from accepted one-step, `last.pt` | rejected | 0.00376596 | 0.00728159 | 0.01342699 | 0.02840060 | 0.01035429 |

Rollout-S4 was skipped because rollout-S2 failed the no-degradation gate.

## Accepted Final Model

Use:

```text
v3/runs/seed45_final_deep/one_step_very_low_lr/best.pt
```

This checkpoint improves the original seed45 model:

- H4/H8 mean: `0.00993983 -> 0.00953534`, a `4.07%` reduction.
- H16: `0.02628970 -> 0.02507803`, a `4.61%` reduction.
- H1: `0.00380729 -> 0.00371593`, a `2.40%` reduction.

## Rollout Finding

Rollout-S2 training did not help in this run.

The rollout trainer's own initial S2 validation recurrent error was `0.00459914`. The best trained S2 epoch only reached `0.00465969`, so the rollout objective did not improve over the starting checkpoint. Horizon evaluation of the trained `last.pt` confirmed degradation:

- H4/H8 mean worsened from `0.00953534` to `0.01035429`.
- H16 worsened from `0.02507803` to `0.02840060`.

Because S2 degraded, rollout-S4 was not run.

## Visualizations

All videos use `single_push_sloth` with `--max-frames 60`.

| Stage | Video | Frame 61 recurrent error | Frame 61 teacher-forced error |
|---|---|---:|---:|
| Original seed45 | `v3/runs/seed45_final_deep/visualizations/00_seed45_start_single_push_sloth.mp4` | 247.970 mm | 12.390 mm |
| One-step `lr=5e-5` | `v3/runs/seed45_final_deep/visualizations/01_one_step_low_lr_single_push_sloth.mp4` | 246.140 mm | 12.341 mm |
| One-step `lr=1e-5` | `v3/runs/seed45_final_deep/visualizations/02_one_step_very_low_lr_single_push_sloth.mp4` | 241.728 mm | 12.314 mm |
| Rollout-S2 `last.pt` | `v3/runs/seed45_final_deep/visualizations/03_rollout_s2_last_single_push_sloth.mp4` | 247.052 mm | 12.410 mm |
| Final accepted model | `v3/runs/seed45_final_deep/visualizations/04_final_accepted_single_push_sloth.mp4` | 241.728 mm | 12.314 mm |

## Commands

The exact training/evaluation/visualization commands and results are recorded in:

```text
v3/runs/seed45_final_deep/stage_history.jsonl
```

Scripts touched for this run:

- `v3/scripts/train.py`: added checkpoint initialization for one-step continuation.
- `v3/scripts/visualize_rollout.py`: added V3 rollout visualization.

Verification:

```bash
conda run -n mpm-dino-poc env PYTHONPATH=v3/src:v2/src python -m py_compile v3/scripts/train.py v3/scripts/train_rollout.py v3/scripts/evaluate_horizons.py v3/scripts/visualize_rollout.py
conda run -n mpm-dino-poc env PYTHONPATH=v3/src:v2/src pytest -q v3/tests
```

Result: `4 passed`.

## Conclusion

The useful improvement came from very-low-LR one-step continuation, not rollout loss.

For seed 45, the best next model is `one_step_very_low_lr/best.pt`. Rollout-S2 currently appears too easy to overfit or perturb: it slightly improves some geometry-like behavior in related checks, but here it worsened the particle rollout metrics that matter. Do not proceed to rollout-S4 from this run.
