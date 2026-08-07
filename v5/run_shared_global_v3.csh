#!/bin/csh -f

# V5 shared physical v3: jointly train the COM and rotation heads and their
# V4.2-class physical trunk. All neural weights start from random initialization.

if ($?CONDA_PREFIX && -x "$CONDA_PREFIX/bin/python") then
  set PYTHON = "$CONDA_PREFIX/bin/python"
else
  set PYTHON = .venv-v41/bin/python
endif
set DATASET = v41/dataset
set MANIFEST = v41/manifests/v41_uid_splits.json
set CONFIG = v5/config.default.json
set RUNS = v5/run/shared_v3/global

if (! -x "$PYTHON") then
  echo "ERROR: activate conda env mpm-dino-poc or provide .venv-v41/bin/python"
  exit 1
endif
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
setenv PYTORCH_CUDA_ALLOC_CONF "expandable_segments:True"

# This 2,048-point, 59-frame model needs several GiB free at peak. A GPU with
# only hundreds of MiB available will fail even though the model parameters fit.
which nvidia-smi >& /dev/null
if ($status == 0) then
  set FREE_MIB = `nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 0 | head -1`
  if ($FREE_MIB < 3072) then
    echo "ERROR: GPU 0 has only ${FREE_MIB} MiB free; shared-global requires at least 3072 MiB before launch."
    echo "Use an unoccupied GPU or stop the process holding its memory."
    nvidia-smi
    exit 1
  endif
endif

$PYTHON -m mpm_dino_v5.cli audit "$MANIFEST"
if ($status != 0) then
  echo "ERROR: V5 integrity audit failed"
  exit 1
endif

foreach seed (42 123 456)
  echo "[`whoami` `date`] Training shared physical v3 seed $seed"
  $PYTHON -m mpm_dino_v5.cli train shared-global \
    --dataset "$DATASET" \
    --manifest "$MANIFEST" \
    --output "$RUNS/seed${seed}" \
    --seed "$seed" \
    --config "$CONFIG" \
    --device cuda \
    --epochs 120 \
    --draws 40 \
    --lr 0.0002 \
    --patience 20 \
    --accumulation 4
  if ($status != 0) then
    echo "ERROR: shared physical v3 seed $seed failed"
    exit 1
  endif
end

echo "[`whoami` `date`] V5 shared physical v3 complete"
exit 0
