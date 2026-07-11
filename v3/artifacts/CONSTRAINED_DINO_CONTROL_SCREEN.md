# Constrained DINO Control Screen

This screen tests whether bottlenecking the object-latent geometry path forces the V3 `latent_graph` model to use DINO in a causally meaningful way.

## Setup

- Variant: `latent_graph`
- Latent geometry constraint: `--latent-geometry-mode bottleneck --latent-geometry-dim 1`
- Seeds: `42`, `123`, `456`
- DINO modes:
  - `final`
  - `shuffled_particles`
  - `scene_shuffled`
- Training: one-step V3 training with edge losses
- Evaluation: recurrent validation horizons H1/H4/H8/H16
- Run root: `v3/runs/constrained_dino_controls/bottleneck1`
- Summary JSON: `v3/runs/constrained_dino_controls/bottleneck1/summary.json`

## Aggregate Result

| Mode | H1 mean | H4 mean | H8 mean | H16 mean | H4/H8 mean | Delta vs final H4/H8 |
|---|---:|---:|---:|---:|---:|---:|
| final-DINO | 0.00331876 | 0.00569011 | 0.00905247 | 0.01568554 | 0.00737129 | 0.00% |
| particle-shuffled DINO | 0.00352483 | 0.00619473 | 0.00985446 | 0.01688902 | 0.00802460 | +8.86% |
| scene-shuffled DINO | 0.00336303 | 0.00592778 | 0.00967759 | 0.01727962 | 0.00780269 | +5.85% |

Lower is better. On the three-seed mean, final-DINO is better than both controls.

## Per-Seed H4/H8 Mean

| Seed | final-DINO | particle-shuffled | scene-shuffled |
|---:|---:|---:|---:|
| 42 | 0.00724207 | 0.00712953 | 0.00748363 |
| 123 | 0.00727895 | 0.00732759 | 0.00754591 |
| 456 | 0.00759286 | 0.00961668 | 0.00837851 |

Final-DINO is not a clean per-seed winner against particle-shuffle:

- Seed 42: particle-shuffle is better than final-DINO.
- Seed 123: particle-shuffle is almost tied with final-DINO.
- Seed 456: particle-shuffle is much worse than final-DINO.

Final-DINO does beat scene-shuffle on all three seeds.

## Interpretation

The geometry bottleneck makes DINO look more useful on aggregate, especially against scene-shuffled DINO. This suggests some object-level DINO identity may help the constrained latent model.

However, the particle-shuffle control does not cleanly fail across seeds. Since particle-shuffle can beat or nearly tie final-DINO at H4/H8 for two of three seeds, this does not validate strong per-particle DINO semantics. The model may benefit from DINO as a weak object-level or distributional signal, but the exact particle-to-DINO alignment is still not reliably important.

The test therefore validates the usefulness of the controls themselves: without particle-shuffle and scene-shuffle, the aggregate final-DINO win would be easy to overinterpret.

## Recommendation

Do not treat this as decisive evidence that DINO is worth keeping in the main V3 architecture. The fair conclusion is:

1. Bottlenecking geometry can make final-DINO outperform shuffled controls on average.
2. Scene-level DINO identity appears more meaningful than per-particle DINO placement.
3. The effect is still seed-sensitive and not strong enough to justify architectural complexity by itself.
4. If DINO remains in V3, use it as an optional object/material latent signal with explicit controls, not as a required per-particle dynamics input.

The next decisive test would be a matched zero-DINO run under the same bottleneck constraint. If final-DINO beats zero-DINO and scene-shuffle consistently, then DINO may be useful as object-level conditioning. If zero-DINO matches final-DINO, the aggregate shuffle result is likely a training/noise artifact rather than real visual grounding.
