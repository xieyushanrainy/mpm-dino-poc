#!/bin/csh -f

# V5 Gate 4: train the standalone causal deformation decoder for all seeds.
# COM, learned rotation, and pointwise interaction checkpoints remain frozen.

set PYTHON = .venv-v41/bin/python
set DATASET = v41/dataset
set MANIFEST = v41/manifests/v41_uid_splits.json
set CONFIG = v5/config.default.json
set GLOBAL_RUNS = v5/run/global_motion
set INTERACTION_RUNS = v5/run/interaction
set DEFORMATION_RUNS = v5/run/deformation

if (! -x "$PYTHON") then
  echo "ERROR: Python environment not found or not executable: $PYTHON"
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

foreach seed (42 123 456)
  if (! -f "$GLOBAL_RUNS/com/seed${seed}/best.pt") then
    echo "ERROR: Missing COM checkpoint for seed $seed"
    exit 1
  endif
  if (! -f "$GLOBAL_RUNS/rotation/seed${seed}/best.pt") then
    echo "ERROR: Missing rotation checkpoint for seed $seed"
    exit 1
  endif
  if (! -f "$INTERACTION_RUNS/seed${seed}/best.pt") then
    echo "ERROR: Missing interaction checkpoint for seed $seed"
    exit 1
  endif
end

mkdir -p "$DEFORMATION_RUNS"
if ($status != 0) exit 1

setenv PYTHONPATH ".:v2/src:v4/src:v5/src"

echo "[`whoami` `date`] Starting V5 standalone deformation stage"

foreach seed (42 123 456)
  echo "[`whoami` `date`] Training deformation seed $seed"

  $PYTHON -m mpm_dino_v5.cli train deformation \
    --dataset "$DATASET" \
    --manifest "$MANIFEST" \
    --output "$DEFORMATION_RUNS/seed${seed}" \
    --seed "$seed" \
    --config "$CONFIG" \
    --device cuda \
    --epochs 120 \
    --draws 40 \
    --lr 0.0002 \
    --patience 20 \
    --accumulation 4 \
    --com-checkpoint "$GLOBAL_RUNS/com/seed${seed}/best.pt" \
    --rotation-checkpoint "$GLOBAL_RUNS/rotation/seed${seed}/best.pt" \
    --interaction-checkpoint "$INTERACTION_RUNS/seed${seed}/best.pt"

  if ($status != 0) then
    echo "ERROR: deformation seed $seed failed"
    exit 1
  endif
end

echo "[`whoami` `date`] V5 standalone deformation stage complete"
exit 0
