# V4.1 lab-server training

The single matrix entry point is `v41/run_matrix.py`. Its frozen defaults are:

- mechanisms: M1, M2 and staged M6;
- conditions: real and architecture-identical zero DINO;
- seeds: 42, 123 and 456;
- width 128, four blocks, four heads;
- 100-epoch cap;
- early-stopping patience 15;
- `ReduceLROnPlateau` patience 5, factor 0.5 and minimum LR `1e-6`;
- 40 UID-balanced draws per epoch;
- CUDA FP16 automatic mixed precision;
- M6 adapter LR `2e-4` and trunk LR `2e-5`.

## Environment and launch without Conda

From the repository root:

```bash
python3 --version
nvidia-smi

python3 -m venv .venv-v41
source .venv-v41/bin/activate
python -m pip install --upgrade pip wheel setuptools
python -m pip install -r v41/requirements-lab.txt

python - <<'PY'
import torch
print("PyTorch:", torch.__version__)
print("CUDA runtime:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable; do not launch training")
print("GPU:", torch.cuda.get_device_name(0))
PY

PYTHONPATH=v2/src:v3/src:v4/src \
python -m pytest v4/tests -q

mkdir -p v41/runs/lab_100ep
nohup bash -c '
  set -o pipefail
  PYTHONPATH=v2/src:v3/src:v4/src \
  .venv-v41/bin/python -u v41/run_matrix.py 2>&1 |
  tee -a v41/runs/lab_100ep/console.log
' > v41/runs/lab_100ep/nohup.log 2>&1 &
echo $! > v41/runs/lab_100ep/launcher.pid
```

Re-running the identical command is the supported resume operation. A run is
skipped only when it has `RUN_COMPLETE.json`; an interrupted run resumes from
`last.pt`, including model, optimizer, LR scheduler, AMP scaler, sampling epoch,
and Python/NumPy/PyTorch/CUDA RNG state.

Do not change seeds, epoch cap, draws, patience, learning rates, widths, losses,
or manifests inside an existing matrix directory. The entry point rejects
changed budget settings recorded in `MATRIX_CONFIG.json`.

## Saved artifacts

Each run directory contains:

```text
config.json          fully resolved run configuration
history.jsonl        one flushed record per completed epoch
last.pt              atomic resumable checkpoint
best.pt              atomic guarded-selection checkpoint
RUNNING.json         present only while the run is active
RUN_FAILED.json      interruption/failure record and resume instruction
RUN_COMPLETE.json    terminal status plus SHA-256 hashes
```

Checkpoints contain model weights, optimizer state, plateau scheduler state,
AMP scaler state, current/best validation values, early-stopping state,
sampler epoch, and RNG states. `best.pt` is selected by validation normalized
mean RMSE across H16/H30/H40 subject to the matched-zero H1 guard.

At matrix completion, `MATRIX_COMPLETE.json` indexes all completed runs and
their artifact hashes. `MATRIX_CONFIG.json` records the host, GPU, Python,
PyTorch/CUDA versions, git commit, manifest hash, and resolved command options.

## Monitoring and retrieval

```bash
tail -f v41/runs/lab_100ep/console.log
find v41/runs/lab_100ep -name RUN_COMPLETE.json | wc -l
nvidia-smi
```

There are 21 training stages in total: 12 M1/M2 runs, three M6 stage-1 trunks,
and six M6 stage-2 runs.

Retrieve the complete directory, preserving partial checkpoints and metadata:

```bash
rsync -av --partial --checksum \
  LAB_HOST:/path/to/mpm-DINO-poc/v41/runs/lab_100ep/ \
  v41/runs/lab_100ep/
```

Do not retrieve only `best.pt`: later evaluation needs its sibling
`config.json`, `history.jsonl`, completion record, and the matrix-level
provenance files. Dataset and split identity can be checked against
`v41/manifests/v41_uid_splits.json` and its recorded hashes.
