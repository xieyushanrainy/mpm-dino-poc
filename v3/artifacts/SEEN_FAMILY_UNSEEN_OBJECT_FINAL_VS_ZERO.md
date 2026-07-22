# Seen-Family Unseen-Object Final-DINO vs Zero-DINO

## Setup

- Architecture: `latent_graph`
- Geometry: default full geometry, no bottleneck
- DINO modes: `final`, `zero`
- Seeds: `45, 123, 456`
- Train manifest: `data/shared/splits/seen_family_unseen_object_train.txt`
- Validation manifest: `data/shared/splits/seen_family_unseen_object_val.txt`
- Evaluation: recurrent validation horizons H1/H4/H8/H16
- Primary metric: mean of H4 and H8 particle error

## Per-Seed Validation Particle Error

| Seed | Mode | H1 | H4 | H8 | H16 | H4/H8 mean | H16 edge vector | H16 edge length |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 45 | final | 0.00380729 | 0.00713225 | 0.01274740 | 0.02628970 | 0.00993983 | 0.00129389 | 0.00156666 |
| 45 | zero | 0.00394408 | 0.00769386 | 0.01368198 | 0.02700766 | 0.01068792 | 0.00125598 | 0.00154411 |
| 123 | final | 0.00476073 | 0.01017760 | 0.01814145 | 0.03494735 | 0.01415952 | 0.00127827 | 0.00155129 |
| 123 | zero | 0.00392549 | 0.00763196 | 0.01363324 | 0.02733606 | 0.01063260 | 0.00127577 | 0.00158293 |
| 456 | final | 0.00378073 | 0.00721849 | 0.01360928 | 0.03038289 | 0.01041388 | 0.00136854 | 0.00168581 |
| 456 | zero | 0.00396108 | 0.00771585 | 0.01400861 | 0.03013741 | 0.01086223 | 0.00133610 | 0.00164380 |

## H4/H8 Final vs Zero

| Seed | final-DINO | zero-DINO | Final delta | Final delta % |
|---:|---:|---:|---:|---:|
| 45 | 0.00993983 | 0.01068792 | -0.00074809 | -7.00% |
| 123 | 0.01415952 | 0.01063260 | 0.00352692 | 33.17% |
| 456 | 0.01041388 | 0.01086223 | -0.00044835 | -4.13% |

## Aggregate

| Mode | H1 mean | H4 mean | H8 mean | H16 mean | H4/H8 mean |
|---|---:|---:|---:|---:|---:|
| final | 0.00411625 | 0.00817611 | 0.01483271 | 0.03053998 | 0.01150441 |
| zero | 0.00394355 | 0.00768056 | 0.01377461 | 0.02816038 | 0.01072758 |

Aggregate final delta vs zero: 0.00077683 (7.24%).

## Interpretation

FAIL: final-DINO ties or loses to zero-DINO, so DINO should not be required in the main V3 architecture.
