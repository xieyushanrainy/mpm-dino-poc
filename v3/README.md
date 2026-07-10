# V3 Workspace

V3 tests particle-native, future-compatible architecture candidates that avoid
continuous controller-trajectory dependence. It reuses V2 cache/data/loss/graph
utilities, but the V3 model package and scripts live under `v3/`.

Read [PLAN.md](PLAN.md) before changing the architecture.

## Layout

```text
v3/src/mpm_dino_v3/   V3 action adapter and candidate models
v3/scripts/           training, recurrent evaluation and architecture screen
v3/tests/             V3 model/interface tests
v3/runs/              ignored local training outputs
v3/artifacts/         documented run summaries
```

## Environment

From the repository root:

```bash
conda activate mpm-dino-poc
export PYTHONPATH="$PWD/v3/src:$PWD/v2/src"
pytest -q v3/tests
```

V3 needs both `v3/src` and `v2/src` on `PYTHONPATH` because it imports V2 cache
and loss helpers.

## Run The Architecture Screen

The full three-seed screen is expensive. For a seed-42 smoke screen, run:

```bash
PYTHONPATH=v3/src:v2/src python v3/scripts/run_architecture_screen.py --device mps --seeds 42
```

For the default three-seed screen:

```bash
PYTHONPATH=v3/src:v2/src python v3/scripts/run_architecture_screen.py --device mps
```

Outputs are written under:

```text
v3/runs/architecture_screen/
```

The seed-42 pilot findings are documented in:

```text
v3/artifacts/SEED42_ARCHITECTURE_SCREEN_SUMMARY.md
```

## Current Finding

The seed-42 pilot selected `latent_graph`, but DINO controls showed that
geometry-only and zero-DINO variants slightly beat final-DINO at the selection
horizons. Treat DINO as unproven until the full multi-seed screen says
otherwise.

## Script And Test Notes

V3 scripts:

- `v3/scripts/train.py`: train one V3 candidate.
- `v3/scripts/evaluate_horizons.py`: recurrent H1/H4/H8/H16 evaluation.
- `v3/scripts/run_architecture_screen.py`: run the baseline, V3 candidates,
  winner-focused DINO controls and test evaluation.

Local training outputs belong under `v3/runs/` and are ignored by git. Promote
only concise reports or selected small artifacts to `v3/artifacts/`.

The V3 tests cover action-summary extraction, candidate output shapes,
backpropagation and recurrent-gradient finiteness.
