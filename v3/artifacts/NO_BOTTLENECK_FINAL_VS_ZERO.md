# No-Bottleneck Final-DINO vs Zero-DINO

## Setup

- Architecture: `latent_graph`
- Geometry: default full geometry, no bottleneck
- DINO modes: `final`, `zero`
- Seeds: `45, 123, 456`
- Evaluation: recurrent validation horizons H1/H4/H8/H16
- Primary metric: mean of H4 and H8 particle error

## Per-Seed Validation Particle Error

| Seed | Mode | H1 | H4 | H8 | H16 | H4/H8 mean | H16 edge vector | H16 edge length |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 45 | final | 0.00323263 | 0.00543802 | 0.00865824 | 0.01480331 | 0.00704813 | 0.00128056 | 0.00172161 |
| 45 | zero | 0.00313485 | 0.00514056 | 0.00812513 | 0.01414033 | 0.00663285 | 0.00127112 | 0.00170608 |
| 123 | final | 0.00328898 | 0.00567163 | 0.00920707 | 0.01637044 | 0.00743935 | 0.00130111 | 0.00172239 |
| 123 | zero | 0.00320570 | 0.00523811 | 0.00810962 | 0.01368752 | 0.00667387 | 0.00127886 | 0.00171175 |
| 456 | final | 0.00333593 | 0.00575949 | 0.00928788 | 0.01615564 | 0.00752368 | 0.00128601 | 0.00170674 |
| 456 | zero | 0.00327530 | 0.00546155 | 0.00854979 | 0.01465134 | 0.00700567 | 0.00126486 | 0.00169847 |

## H4/H8 Final vs Zero

| Seed | final-DINO | zero-DINO | Final delta | Final delta % |
|---:|---:|---:|---:|---:|
| 45 | 0.00704813 | 0.00663285 | 0.00041528 | 6.26% |
| 123 | 0.00743935 | 0.00667387 | 0.00076548 | 11.47% |
| 456 | 0.00752368 | 0.00700567 | 0.00051801 | 7.39% |

## Aggregate

| Mode | H1 mean | H4 mean | H8 mean | H16 mean | H4/H8 mean |
|---|---:|---:|---:|---:|---:|
| final | 0.00328585 | 0.00562305 | 0.00905106 | 0.01577647 | 0.00733705 |
| zero | 0.00320528 | 0.00528008 | 0.00826152 | 0.01415973 | 0.00677080 |

Aggregate final delta vs zero: 0.00056626 (8.36%).

## Interpretation

FAIL: final-DINO ties or loses to zero-DINO, so DINO should not be required in the main V3 architecture.
