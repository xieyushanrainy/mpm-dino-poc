# MPM-DINO POC

This repository is split into a frozen V1 implementation and a clean V2
workspace.

## Layout

```text
mpm-DINO-poc/
├── v1/
│   ├── src/mpm_dino/          frozen V1 package
│   ├── scripts/               V1 preprocessing, training and diagnostics
│   ├── tests/                 V1 tests
│   └── artifacts/rollout_s4/  canonical V1 checkpoint and results
├── v2/
│   ├── src/mpm_dino_v2/       V2 implementation workspace
│   ├── scripts/               V2 entry-point workspace
│   └── tests/                 V2 test workspace
├── data/
│   ├── shared/                DINO features and split manifests
│   ├── v1/cache/              frozen V1 scene caches
│   └── v2/                    V2 versioned-cache workspace
├── docs/v1/                    V1 conclusion and consolidated history
├── docs/v2/                    V2 context and implementation plan
├── docs/reference/             shared implementation references
└── archive/v1_runs/            superseded V1 runs, retained but inactive
```

## Start V2

Read in order:

1. [V2_CONTEXT.md](V2_CONTEXT.md)
2. [docs/v2/CONTEXT.md](docs/v2/CONTEXT.md)
3. [docs/v2/PLAN.md](docs/v2/PLAN.md)

Canonical V1 baseline:

```text
v1/artifacts/rollout_s4/best.pt
```

## Run frozen V1

Activate the shared environment, then run commands from the repository root:

```bash
conda activate mpm-dino-poc
export PYTHONPATH="$PWD/v1/src"
pytest -q v1/tests
```

See [v1/README.md](v1/README.md) for V1 commands. Superseded checkpoints are
archived only to preserve provenance; new development belongs under `v2/`.
