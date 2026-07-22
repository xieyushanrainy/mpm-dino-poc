# Seen-Family Unseen-Object Final-DINO vs Zero-DINO Seed 789

## Setup

- Architecture: `latent_graph`
- Geometry: default full geometry, no bottleneck
- DINO modes: `final`, `zero`
- Seeds: `789`
- Train manifest: `data/shared/splits/seen_family_unseen_object_train.txt`
- Validation manifest: `data/shared/splits/seen_family_unseen_object_val.txt`
- Evaluation: recurrent validation horizons H1/H4/H8/H16
- Primary metric: mean of H4 and H8 particle error

## Per-Seed Validation Particle Error

| Seed | Mode | H1 | H4 | H8 | H16 | H4/H8 mean | H16 edge vector | H16 edge length |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 789 | final | 0.00391911 | 0.00776157 | 0.01388604 | 0.02753261 | 0.01082381 | 0.00122388 | 0.00149759 |
| 789 | zero | 0.00401699 | 0.00789770 | 0.01399532 | 0.02724766 | 0.01094651 | 0.00125726 | 0.00156145 |

## H4/H8 Final vs Zero

| Seed | final-DINO | zero-DINO | Final delta | Final delta % |
|---:|---:|---:|---:|---:|
| 789 | 0.01082381 | 0.01094651 | -0.00012270 | -1.12% |

## Aggregate

| Mode | H1 mean | H4 mean | H8 mean | H16 mean | H4/H8 mean |
|---|---:|---:|---:|---:|---:|
| final | 0.00391911 | 0.00776157 | 0.01388604 | 0.02753261 | 0.01082381 |
| zero | 0.00401699 | 0.00789770 | 0.01399532 | 0.02724766 | 0.01094651 |

Aggregate final delta vs zero: -0.00012270 (-1.12%).

## Interpretation

PASS: final-DINO beats zero-DINO on every seed by more than a tiny margin.
