# MPM-DINO POC

This repository is split into frozen V1 provenance, a documented V2 baseline
workspace, and the current V3 architecture-screen workspace.

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
│   ├── tests/                 V2 test workspace
│   └── artifacts/             V2 reports and retained summaries
├── v3/
│   ├── src/mpm_dino_v3/       V3 particle-native candidate models
│   ├── scripts/               V3 training/evaluation/screen entry points
│   ├── tests/                 V3 tests
│   └── artifacts/             V3 run summaries
├── data/
│   ├── shared/                DINO features and split manifests
│   ├── v1/cache/              frozen V1 scene caches
│   └── v2/                    V2 versioned-cache workspace
├── docs/v1/                    V1 conclusion and consolidated history
├── docs/v2/                    V2 context and implementation plan
├── docs/reference/             shared implementation references
└── archive/v1_runs/            superseded V1 runs, retained but inactive
```

## Environment

Create or update the shared conda environment from
[environment.yml](environment.yml):

```bash
conda env create -f environment.yml
conda activate mpm-dino-poc
```

If the environment already exists:

```bash
conda env update -n mpm-dino-poc -f environment.yml --prune
```

## Version Entry Points

- [v1/README.md](v1/README.md): frozen V1 commands and canonical baseline.
- [v2/README.md](v2/README.md): V2 reference/deformation-aware baseline,
  ablations and scripts.
- [v3/README.md](v3/README.md): current V3 DINO-centric architecture screen.

## Start V2/V3

Read in order:

1. [V2_CONTEXT.md](V2_CONTEXT.md)
2. [docs/v2/CONTEXT.md](docs/v2/CONTEXT.md)
3. [docs/v2/PLAN.md](docs/v2/PLAN.md)
4. [v3/PLAN.md](v3/PLAN.md), if working on V3 architecture screens.

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
archived only to preserve provenance; new development belongs under `v2/` or
`v3/`.

## Generated Data And Runs

Large local caches and training outputs are intentionally ignored:

```text
data/v2/cache*/
data/v2/dino_layers/
v2/runs/
v3/runs/
```

Regenerate them with the version-specific scripts. Promote only concise reports
and selected artifacts into `v2/artifacts/` or `v3/artifacts/`.
