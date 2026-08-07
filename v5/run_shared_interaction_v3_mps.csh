#!/bin/csh -f

# Train the causal pointwise interaction encoder from the three completed MPS
# shared-global checkpoints. The three-seed rotation decision is identity.

if (! $?CONDA_PREFIX || ! -x "$CONDA_PREFIX/bin/python") then
  echo "ERROR: activate the conda environment first: conda activate mpm-dino-poc"
  exit 1
endif

set PYTHON = "$CONDA_PREFIX/bin/python"
set DATASET = v41/dataset
set MANIFEST = v41/manifests/v41_uid_splits.json
set CONFIG = v5/config.default.json
set GLOBAL_RUNS = v5/run/shared_v3/global_mps
set RUNS = v5/run/shared_v3/interaction_mps

mkdir -p "$RUNS"
if ($status != 0) exit 1
setenv PYTHONPATH ".:v2/src:v4/src:v5/src"

$PYTHON -c "import torch; assert torch.backends.mps.is_available(), 'PyTorch MPS is unavailable'"
if ($status != 0) exit 1

foreach seed (42 123 456)
  set GLOBAL = "$GLOBAL_RUNS/seed${seed}/best.pt"
  if (! -f "$GLOBAL") then
    echo "ERROR: missing shared-global checkpoint: $GLOBAL"
    exit 1
  endif
  echo "[`whoami` `date`] Training identity-conditioned interaction seed $seed"
  $PYTHON -m mpm_dino_v5.cli train shared-interaction \
    --dataset "$DATASET" \
    --manifest "$MANIFEST" \
    --output "$RUNS/seed${seed}" \
    --seed "$seed" \
    --config "$CONFIG" \
    --device mps \
    --global-checkpoint "$GLOBAL" \
    --identity-rotation \
    --epochs 120 \
    --draws 40 \
    --lr 0.0002 \
    --patience 20 \
    --accumulation 4
  if ($status != 0) then
    echo "ERROR: shared interaction seed $seed failed"
    exit 1
  endif
end

echo "[`whoami` `date`] V5 shared interaction MPS complete"
exit 0
