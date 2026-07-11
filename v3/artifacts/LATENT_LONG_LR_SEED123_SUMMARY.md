# Latent Graph Long / Low-LR DINO Check, Seed 123

This run tests whether the winning V3 `latent_graph` architecture benefits from training longer with a lower learning rate, and whether final-DINO then beats an equally extended zero-DINO control.

## Setup

- Architecture: V3 `latent_graph`
- Seed: `123`
- Device: `mps`
- Training: one-step objective with V2 edge losses
- Epoch budget: `100`
- Initial LR: `1e-4`
- LR patience: `5`
- Early-stop patience: `14`
- Minimum relative improvement: `0.005`
- Train split: `data/shared/splits/poc_train.txt`
- Validation split: `data/shared/splits/poc_val.txt`

Checkpoints:

- final-DINO: `v3/runs/latent_long_lr/seed123/final/one_step/best.pt`
- zero-DINO: `v3/runs/latent_long_lr/seed123/zero/one_step/best.pt`

## Training Outcome

| Mode | Epochs | Best epoch | Best val objective | Best val particle mean | Best LR |
|---|---:|---:|---:|---:|---:|
| final-DINO | 49 | 41 | 0.00038518 | 0.00324375 | 0.0000125 |
| zero-DINO | 53 | 49 | 0.00038516 | 0.00327223 | 0.000003125 |

The one-step objective is effectively tied. The final-DINO run has slightly better one-step particle mean, but the difference is very small.

## Recurrent Validation Horizons

| Horizon | final-DINO particle | zero-DINO particle | final delta vs zero | final edge vector | zero edge vector |
|---:|---:|---:|---:|---:|---:|
| H1 | 0.00321964 | 0.00325074 | -0.96% | 0.00029051 | 0.00029406 |
| H4 | 0.00541210 | 0.00544023 | -0.52% | 0.00045622 | 0.00046133 |
| H8 | 0.00857075 | 0.00861752 | -0.54% | 0.00069480 | 0.00070308 |
| H16 | 0.01470560 | 0.01488734 | -1.22% | 0.00127639 | 0.00129041 |

Final-DINO is slightly better at every recurrent horizon in this seed, including edge-vector error. The margin is small: about 0.5% at H4/H8 and 1.2% at H16. This is not yet a decisive DINO result unless repeated across seeds and compared to shuffled-DINO.

## Latent Visualization

Learned `z_object` PCA outputs:

- final-DINO SVG: `v3/artifacts/latent_viz/latent_seed123_final_pca.svg`
- final-DINO CSV: `v3/artifacts/latent_viz/latent_seed123_final_pca.csv`
- final-DINO summary: `v3/artifacts/latent_viz/latent_seed123_final_summary.json`
- zero-DINO SVG: `v3/artifacts/latent_viz/latent_seed123_zero_pca.svg`
- zero-DINO CSV: `v3/artifacts/latent_viz/latent_seed123_zero_pca.csv`
- zero-DINO summary: `v3/artifacts/latent_viz/latent_seed123_zero_summary.json`

| Embedding | PC1 variance | PC2 variance | 2D nearest-family accuracy |
|---|---:|---:|---:|
| raw pooled DINO | 0.3694 | 0.2331 | 0.8182 |
| learned `z_object`, final-DINO | 0.5713 | 0.2206 | 0.2727 |
| learned `z_object`, zero-DINO | 0.9907 | 0.0075 | 0.4545 |

Raw pooled DINO still clusters object families strongly in this dataset. The learned final-DINO latent does not preserve that family structure in 2D PCA, despite using final-DINO as input. The zero-DINO latent mostly collapses to one dominant component, but its simple 2D nearest-family score is higher than the final-DINO learned latent.

## Interpretation

Longer, lower-LR training does not explain away DINO underperformance. In this seed, final-DINO gives a tiny recurrent improvement over zero-DINO, but the gain is much smaller than the likely seed-to-seed variation seen in earlier screens.

The latent visualization suggests that the raw DINO signal contains scene/object-family information, but the current `latent_graph` training objective does not force the object latent to retain or use that information in a clearly separable way. The dynamics model may be learning mostly from geometry, action metadata, and recurrence state, with DINO contributing only weak regularization or a small bias.

## Recommended Next Checks

1. Repeat this exact long/low-LR comparison for at least two more seeds.
2. Add a matched shuffled-DINO long/low-LR run for seed 123.
3. If final-DINO does not beat zero and shuffled by more than seed variance, treat DINO as non-essential for this architecture.
4. If DINO remains visually meaningful only before training, test an auxiliary latent objective that preserves raw-DINO family/material structure, but keep it separate from the main dynamics selection.
