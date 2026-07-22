# Seen-Family Unseen-Object Final-DINO vs Zero-DINO Seed 999

## Setup

- Architecture: `latent_graph`
- Geometry: default full geometry, no bottleneck
- DINO modes: `final`, `zero`
- Seeds: `999`
- Train manifest: `data/shared/splits/seen_family_unseen_object_train.txt`
- Validation manifest: `data/shared/splits/seen_family_unseen_object_val.txt`
- Evaluation: recurrent validation horizons H1/H4/H8/H16
- Primary metric: mean of H4 and H8 particle error

## Per-Seed Validation Particle Error

| Seed | Mode | H1 | H4 | H8 | H16 | H4/H8 mean | H16 edge vector | H16 edge length |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 999 | final | 0.00385424 | 0.00740985 | 0.01348293 | 0.02801676 | 0.01044639 | 0.00126076 | 0.00157094 |
| 999 | zero | 0.00384459 | 0.00736629 | 0.01357657 | 0.02881442 | 0.01047143 | 0.00132289 | 0.00161728 |

## H4/H8 Final vs Zero

| Seed | final-DINO | zero-DINO | Final delta | Final delta % |
|---:|---:|---:|---:|---:|
| 999 | 0.01044639 | 0.01047143 | -0.00002504 | -0.24% |

## Aggregate

| Mode | H1 mean | H4 mean | H8 mean | H16 mean | H4/H8 mean |
|---|---:|---:|---:|---:|---:|
| final | 0.00385424 | 0.00740985 | 0.01348293 | 0.02801676 | 0.01044639 |
| zero | 0.00384459 | 0.00736629 | 0.01357657 | 0.02881442 | 0.01047143 |

Aggregate final delta vs zero: -0.00002504 (-0.24%).

## Interpretation

INCONCLUSIVE: results are mixed or margins are tiny, so DINO remains weak and scene-shuffle can wait.
