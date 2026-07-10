# V2 Workspace

V2 adds effective reference geometry and local surface-deformation state to the
particle-grid surrogate. Read `../docs/v2/CONTEXT.md` and
`../docs/v2/PLAN.md` before implementation.

Intended layout:

```text
v2/src/mpm_dino_v2/   versioned package
v2/scripts/           preprocessing, training, evaluation and diagnostics
v2/tests/             unit and integration tests
v2/runs/              ignored local training outputs
v2/artifacts/         documented summaries and retained small artifacts
```

V2 may read `data/shared/` and `data/v1/cache/`, but must write its versioned
cache schema only under `data/v2/` so V1 caches are not silently overwritten.

Schema version 1 adds normalized frame-0 `x0`, a non-padding `particle_mask`,
and fixed reciprocal neighbour tensors (`neighbour_indices`,
`neighbour_mask`, `rest_edge_vectors`, and `rest_edge_lengths`). Large generated
cache directories such as `data/v2/cache/`, `data/v2/cache_dino_zero/`,
`data/v2/cache_dino_block06/`, `data/v2/cache_dino_block09/` and
`data/v2/dino_layers/` are ignored by git and should be regenerated locally.

## Environment

From the repository root:

```bash
conda activate mpm-dino-poc
export PYTHONPATH="$PWD/v2/src"
pytest -q v2/tests
```

## Common Commands

Prepare V2 caches after V1 caches are available:

```bash
PYTHONPATH=v2/src python v2/scripts/prepare_v2_cache.py \
  --input data/v1/cache \
  --output-dir data/v2/cache \
  --report data/v2/graph_report_all.json
```

Train the V2 one-step model:

```bash
PYTHONPATH=v2/src python v2/scripts/train.py \
  data/v2/cache/*.pt \
  --val-caches data/v2/cache/<val-scene>.pt \
  --variant particle_only \
  --device mps \
  --output v2/runs/example_particle_only
```

For the completed V2 findings, start with:

- `v2/artifacts/V2B_TRAINING_SUMMARY.md`
- `v2/artifacts/DINO_LAYER_ABLATION_REPORT.md`
- `v2/artifacts/GRID_PARTICLE_ABLATION_REPORT.md`
- `v2/artifacts/MIXED_ROLLOUT_S1_4_SUMMARY.md`

Local training outputs belong under `v2/runs/` and are ignored by git. Promote
only concise reports or selected small artifacts to `v2/artifacts/`.
