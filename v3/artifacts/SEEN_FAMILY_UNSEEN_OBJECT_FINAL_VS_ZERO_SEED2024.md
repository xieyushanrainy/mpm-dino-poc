# Seen-Family Unseen-Object Final-DINO vs Zero-DINO Seed 2024

## Setup

- Architecture: `latent_graph`
- Geometry: default full geometry, no bottleneck
- DINO modes: `final`, `zero`
- Seeds: `2024`
- Train manifest: `data/shared/splits/seen_family_unseen_object_train.txt`
- Validation manifest: `data/shared/splits/seen_family_unseen_object_val.txt`
- Evaluation: recurrent validation horizons H1/H4/H8/H16
- Primary metric: mean of H4 and H8 particle error

## Per-Seed Validation Particle Error

| Seed | Mode | H1 | H4 | H8 | H16 | H4/H8 mean | H16 edge vector | H16 edge length |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 2024 | final | 0.00398765 | 0.00815180 | 0.01511893 | 0.03108456 | 0.01163536 | 0.00128291 | 0.00157290 |
| 2024 | zero | 0.00417928 | 0.00838119 | 0.01499098 | 0.03016933 | 0.01168609 | 0.00138692 | 0.00178150 |

## H4/H8 Final vs Zero

| Seed | final-DINO | zero-DINO | Final delta | Final delta % |
|---:|---:|---:|---:|---:|
| 2024 | 0.01163536 | 0.01168609 | -0.00005072 | -0.43% |

## Aggregate

| Mode | H1 mean | H4 mean | H8 mean | H16 mean | H4/H8 mean |
|---|---:|---:|---:|---:|---:|
| final | 0.00398765 | 0.00815180 | 0.01511893 | 0.03108456 | 0.01163536 |
| zero | 0.00417928 | 0.00838119 | 0.01499098 | 0.03016933 | 0.01168609 |

Aggregate final delta vs zero: -0.00005072 (-0.43%).

## Interpretation

INCONCLUSIVE: results are mixed or margins are tiny, so DINO remains weak and scene-shuffle can wait.
