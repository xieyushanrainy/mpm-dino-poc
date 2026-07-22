# Seed 45 Final-DINO Rollout-S2 Check

Date: 2026-07-12

## Question

Using the winning seen-family/unseen-object seed 45 final-DINO model, does adding a short rollout loss help recurrent validation behavior?

This is a depth check, not a new DINO-vs-zero sweep.

## Setup

- Architecture: `latent_graph`
- DINO mode: `final`
- Geometry: default full geometry, no bottleneck
- Seed: `45`
- Base checkpoint: `v3/runs/seen_family_unseen_object_final_vs_zero/final/seed45/one_step/best.pt`
- Rollout checkpoint: `v3/runs/seed45_final_deep/rollout_s2/best.pt`
- Train split: `data/shared/splits/seen_family_unseen_object_train.txt`
- Validation split:
  - `double_stretch_zebra`
  - `single_lift_cloth_4`
  - `single_push_rope_1`
  - `single_push_sloth`

## Commands

Rollout fine-tuning:

```bash
conda run -n mpm-dino-poc env PYTHONPATH=v3/src:v2/src python v3/scripts/train_rollout.py $(python -c 'from pathlib import Path; cache=Path("data/v2/cache"); manifest=Path("data/shared/splits/seen_family_unseen_object_train.txt"); print(" ".join(str(cache / (Path(line).stem + ".pt")) for line in manifest.read_text().splitlines() if line.strip()))') --val-caches $(python -c 'from pathlib import Path; cache=Path("data/v2/cache"); manifest=Path("data/shared/splits/seen_family_unseen_object_val.txt"); print(" ".join(str(cache / (Path(line).stem + ".pt")) for line in manifest.read_text().splitlines() if line.strip()))') --checkpoint v3/runs/seen_family_unseen_object_final_vs_zero/final/seed45/one_step/best.pt --steps 2 --epochs 20 --lr 2.5e-5 --lr-patience 2 --early-stop-patience 6 --teacher-weight 0.5 --no-regression-ratio 1.1 --random-min-steps 1 --seed 45 --device mps --output v3/runs/seed45_final_deep/rollout_s2
```

Horizon evaluation:

```bash
conda run -n mpm-dino-poc env PYTHONPATH=v3/src:v2/src python v3/scripts/evaluate_horizons.py v3/runs/seed45_final_deep/rollout_s2/best.pt data/v2/cache/double_stretch_zebra.pt data/v2/cache/single_lift_cloth_4.pt data/v2/cache/single_push_rope_1.pt data/v2/cache/single_push_sloth.pt --horizons 1 4 8 16 --device mps --output v3/runs/seed45_final_deep/rollout_s2/validation_horizons.json
```

Verification after adding the rollout trainer:

```bash
conda run -n mpm-dino-poc env PYTHONPATH=v3/src:v2/src python -m py_compile v3/scripts/train_rollout.py v3/scripts/evaluate_horizons.py v3/scripts/train.py
conda run -n mpm-dino-poc env PYTHONPATH=v3/src:v2/src pytest -q v3/tests
```

Result: `4 passed in 0.70s`.

## Rollout Training Behavior

The rollout objective was trainable and stable.

- Initial recurrent validation particle error: `0.00474012`
- Best recurrent validation particle error: `0.00466032`
- Relative improvement on the rollout-S2 validation objective: about `1.7%`
- Teacher-forced guardrail remained satisfied for all epochs.
- Early stopped after 8 epochs because there was no further meaningful guarded improvement.

This says rollout loss works mechanically: it backpropagates, stays finite, and can improve the direct 2-step validation objective a little.

## Horizon Results

Particle error, lower is better.

| Model | H1 | H4 | H8 | H16 | H4/H8 mean |
|---|---:|---:|---:|---:|---:|
| one-step base | 0.00380729 | 0.00713225 | 0.01274740 | 0.02628970 | 0.00993983 |
| rollout-S2 | 0.00373766 | 0.00708202 | 0.01298320 | 0.02768839 | 0.01003261 |
| rollout-S2 delta | -1.83% | -0.70% | +1.85% | +5.32% | +0.93% |

Edge-vector error:

| Model | H1 | H4 | H8 | H16 |
|---|---:|---:|---:|---:|
| one-step base | 0.00033341 | 0.00050723 | 0.00073377 | 0.00129389 |
| rollout-S2 | 0.00032889 | 0.00049888 | 0.00072095 | 0.00127061 |
| rollout-S2 delta | -1.35% | -1.65% | -1.75% | -1.80% |

Edge-length error:

| Model | H1 | H4 | H8 | H16 |
|---|---:|---:|---:|---:|
| one-step base | 0.00038572 | 0.00059265 | 0.00086348 | 0.00156666 |
| rollout-S2 | 0.00037934 | 0.00058134 | 0.00084443 | 0.00152250 |
| rollout-S2 delta | -1.65% | -1.91% | -2.21% | -2.82% |

## Interpretation

Rollout loss passes the basic implementation/usefulness sanity check, but it does not yet improve the main metric.

The signal is split:

- Positive: H1 and H4 particle error improve slightly.
- Positive: edge-vector and edge-length errors improve at every horizon.
- Negative: H8 and H16 particle error regress.
- Negative: the primary H4/H8 particle mean worsens from `0.00993983` to `0.01003261`, a `0.93%` regression.

So this rollout-S2 run is not a clear performance win. It looks like the model learned a slightly more geometry-consistent short-horizon update, but that did not translate into better medium/long particle rollout. The H16 regression is the main warning sign.

## Conclusion

Rollout loss works as an optimization mechanism, but this first seed-45 rollout-S2 fine-tune does not justify claiming deeper training improves the model.

Recommended next step, if continuing: try one conservative follow-up from this checkpoint with lower learning rate and/or stronger teacher guardrail before going to longer rollout depth. For example, keep rollout depth at 2 and use `lr=1e-5`, or move to rollout-S4 only with a very small `lr` and require no H8/H16 regression before treating it as useful.
