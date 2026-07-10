# V3 Seed-42 Architecture Screen Summary

Date: 2026-07-10

## Question

Test the V3 DINO-centric, particle-native architecture screen on a single seed
before committing to the full three-seed experiment.

The screen compared:

- **V2 particle-only baseline:** the previous best compact particle-only model.
- **V3 graph-direct:** DINO-conditioned reference-graph message passing.
- **V3 latent-graph:** object-level geometry/DINO latent with FiLM-conditioned
  graph dynamics.
- **V3 action-token graph:** action-token cross-attention followed by graph
  message passing.

All V3 models avoid continuous controller trajectory dependence. On the current
POC dataset, the scripts derive a fixed initial action vector and contact point
from the window-initial controller state.

## Configuration

```text
environment:       conda mpm-dino-poc
device:            mps
seed:              42
train split:        data/shared/splits/poc_train.txt
validation split:   data/shared/splits/poc_val.txt
test split:         data/shared/splits/poc_test.txt
selection metric:   validation mean of H4 and H8 recurrent particle error
output root:        v3/runs/architecture_screen/
```

The run used the seed-42-only screen configuration. The current script supports
this directly with `--seeds 42`.

## Architecture Screen

Validation recurrent particle error:

| Model | H1 | H4 | H8 | H16 | H4/H8 mean |
|---|---:|---:|---:|---:|---:|
| V2 particle-only baseline | 0.00358268 | 0.00590707 | 0.00896080 | 0.01468931 | 0.00743393 |
| V3 graph-direct final-DINO | 0.00350644 | 0.00621000 | 0.00991183 | 0.01700149 | 0.00806091 |
| **V3 latent-graph final-DINO** | **0.00323423** | **0.00536425** | **0.00844489** | **0.01463774** | **0.00690457** |
| V3 action-token graph final-DINO | 0.00345361 | 0.00585750 | 0.00916143 | 0.01567336 | 0.00750947 |

The selected seed-42 V3 candidate is:

```text
latent_graph:final
```

`latent_graph` improved validation H4/H8 over both the V2 particle-only
baseline and the other V3 candidates for this seed. The advantage is strongest
at H1/H4/H8 and small at H16.

## DINO Controls

Winner-focused validation controls for `latent_graph`:

| DINO mode | H1 | H4 | H8 | H16 | H4/H8 mean |
|---|---:|---:|---:|---:|---:|
| final-DINO | 0.00323423 | 0.00536425 | 0.00844489 | 0.01463774 | 0.00690457 |
| zero-DINO | 0.00322514 | 0.00529421 | 0.00822066 | 0.01381700 | 0.00675744 |
| shuffled-particle DINO | 0.00322707 | 0.00531634 | 0.00829027 | 0.01411683 | 0.00680330 |
| geometry-only | **0.00322576** | **0.00528672** | **0.00821013** | **0.01381085** | **0.00674842** |

For this seed, DINO does **not** earn its role. The geometry-only and zero-DINO
controls are slightly better than final-DINO at the selection horizons.
Shuffled-DINO is also close to real DINO, which argues against a meaningful
particle-level DINO signal in this run.

## Test Evaluation

Only the V2 particle-only baseline and selected V3 `latent_graph:final` winner
were evaluated on the untouched test split.

| Model | H1 | H4 | H8 | H16 |
|---|---:|---:|---:|---:|
| V2 particle-only baseline | 0.00474549 | 0.00917112 | **0.01537201** | **0.02768838** |
| V3 latent-graph final-DINO | **0.00419973** | **0.00882601** | 0.01605798 | 0.03172051 |

The test result is mixed:

- V3 `latent_graph:final` improves H1 and H4.
- The V2 particle-only baseline remains better at H8 and H16.
- This suggests the latent graph improves short recurrent accuracy but may not
  yet stabilize longer recurrent rollouts.

## Interpretation

Seed 42 supports the architectural idea that an object-level latent plus
reference-graph dynamics is a promising V3 direction. It does **not** support
keeping DINO as a necessary component. The strongest control in this run is
geometry-only latent graph dynamics, not final-DINO latent graph dynamics.

Important caveats:

- This is a one-seed screen, not a robust conclusion.
- The current POC action metadata is derived from controller tracks, not true
  future impulse labels.
- Validation selection favors H4/H8; test H16 still regresses for the selected
  V3 final-DINO model.
- Because final-DINO loses to zero/geometry controls on validation, future
  three-seed runs should include geometry-only or zero-DINO as first-class
  candidates, not only as post-hoc controls.

## Artifacts

This summary preserves the relevant metrics. The original local run outputs
were generated under `v3/runs/architecture_screen/`, but run directories are
ignored and may be removed during housekeeping. Regenerate them with:

```bash
PYTHONPATH=v3/src:v2/src python v3/scripts/run_architecture_screen.py --device mps --seeds 42
```

## Recommended Next Step

Before running the full three-seed screen, promote `latent_graph:geometry_only`
or `latent_graph:zero` from a post-selection control into the main candidate
set. The current seed-42 result says the useful mechanism is likely
reference-graph latent dynamics, not DINO conditioning.
