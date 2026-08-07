#!/bin/csh -f

# Lab CUDA launcher for the standalone V5 causal deformation stage.
# Global motion and interaction are frozen; the selected rotation is identity.

set PYTHON = .venv-v41/bin/python
set DATASET = v41/dataset
set MANIFEST = v41/manifests/v41_uid_splits.json
set CONFIG = v5/config.default.json
set GLOBAL_RUNS = v5/run/shared_v3/global
set INTERACTION_RUNS = v5/run/shared_v3/interaction_cuda
set RUNS = v5/run/shared_v3/deformation_cuda

if (! -x "$PYTHON") then
  echo "ERROR: Python environment not found: $PYTHON"
  exit 1
endif
if (! -d "$DATASET" || ! -f "$MANIFEST" || ! -f "$CONFIG") then
  echo "ERROR: dataset, manifest, or V5 config is missing"
  exit 1
endif

mkdir -p "$RUNS"
if ($status != 0) exit 1
setenv PYTHONPATH ".:v2/src:v4/src:v5/src"
setenv PYTORCH_CUDA_ALLOC_CONF "expandable_segments:True"

which nvidia-smi >& /dev/null
if ($status == 0) then
  set FREE_MIB = `nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 0 | head -1`
  if ($FREE_MIB < 8192) then
    echo "ERROR: GPU 0 has only ${FREE_MIB} MiB free; deformation training requires 8192 MiB free before launch."
    nvidia-smi
    exit 1
  endif
endif

foreach seed (42 123 456)
  set GLOBAL = "$GLOBAL_RUNS/seed${seed}/best.pt"
  set INTERACTION = "$INTERACTION_RUNS/seed${seed}/best.pt"
  if (! -f "$GLOBAL") then
    echo "ERROR: missing shared-global checkpoint: $GLOBAL"
    exit 1
  endif
  if (! -f "$INTERACTION") then
    echo "ERROR: missing shared-interaction checkpoint: $INTERACTION"
    exit 1
  endif

  echo "[`whoami` `date`] Training frozen-trunk causal deformation seed $seed"
  $PYTHON -m mpm_dino_v5.cli train shared-deformation \
    --dataset "$DATASET" \
    --manifest "$MANIFEST" \
    --output "$RUNS/seed${seed}" \
    --seed "$seed" \
    --config "$CONFIG" \
    --device cuda \
    --global-checkpoint "$GLOBAL" \
    --interaction-checkpoint "$INTERACTION" \
    --identity-rotation \
    --trunk-gradient-scale 0 \
    --epochs 120 \
    --draws 40 \
    --lr 0.0002 \
    --patience 20 \
    --accumulation 4
  if ($status != 0) then
    echo "ERROR: CUDA shared deformation seed $seed failed"
    exit 1
  endif
end

echo "[`whoami` `date`] V5 shared deformation CUDA complete"
exit 0
