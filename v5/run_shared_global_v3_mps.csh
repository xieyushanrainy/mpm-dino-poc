#!/bin/csh -f

# Local Apple Silicon launcher. Activate conda env mpm-dino-poc first.

if (! $?CONDA_PREFIX || ! -x "$CONDA_PREFIX/bin/python") then
  echo "ERROR: activate the conda environment first: conda activate mpm-dino-poc"
  exit 1
endif

set PYTHON = "$CONDA_PREFIX/bin/python"
set DATASET = v41/dataset
set MANIFEST = v41/manifests/v41_uid_splits.json
set CONFIG = v5/config.default.json
set RUNS = v5/run/shared_v3/global_mps

if (! -d "$DATASET") then
  echo "ERROR: Dataset not found: $DATASET"
  exit 1
endif
if (! -f "$MANIFEST") then
  echo "ERROR: Manifest not found: $MANIFEST"
  exit 1
endif
if (! -f "$CONFIG") then
  echo "ERROR: V5 config not found: $CONFIG"
  exit 1
endif

mkdir -p "$RUNS"
if ($status != 0) exit 1
setenv PYTHONPATH ".:v2/src:v4/src:v5/src"

$PYTHON -c "import torch; assert torch.backends.mps.is_available(), 'PyTorch MPS is unavailable'"
if ($status != 0) then
  echo "ERROR: the active Conda environment does not provide working MPS support"
  exit 1
endif

$PYTHON -m mpm_dino_v5.cli audit "$MANIFEST"
if ($status != 0) then
  echo "ERROR: V5 integrity audit failed"
  exit 1
endif

foreach seed (42 123 456)
  echo "[`whoami` `date`] Training shared physical v3 on MPS, seed $seed"
  $PYTHON -m mpm_dino_v5.cli train shared-global \
    --dataset "$DATASET" \
    --manifest "$MANIFEST" \
    --output "$RUNS/seed${seed}" \
    --seed "$seed" \
    --config "$CONFIG" \
    --device mps \
    --epochs 120 \
    --draws 40 \
    --lr 0.0002 \
    --patience 20 \
    --accumulation 4
  if ($status != 0) then
    echo "ERROR: shared physical v3 MPS seed $seed failed"
    exit 1
  endif
end

echo "[`whoami` `date`] V5 shared physical v3 MPS complete"
exit 0
