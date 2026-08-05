#!/bin/csh -f

# V5 Gates 0--2: integrity audit, COM training, and rotation training.
# This script intentionally stops before interaction/deformation training so
# the global three-seed learned-versus-identity rotation decision can be made.

set PYTHON = .venv-v41/bin/python
set DATASET = v41/dataset
set MANIFEST = v41/manifests/v41_uid_splits.json
set CONFIG = v5/config.default.json
set RUNS = v5/run/global_motion

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

mkdir -p "$RUNS"
if ($status != 0) exit 1

setenv PYTHONPATH ".:v2/src:v4/src:v5/src"

echo "[`whoami` `date`] Starting V5 integrity audit"
$PYTHON -m mpm_dino_v5.cli audit "$MANIFEST"
if ($status != 0) then
  echo "ERROR: V5 integrity audit failed"
  exit 1
endif

foreach seed (42 123 456)
  echo "[`whoami` `date`] Training COM seed $seed"
  $PYTHON -m mpm_dino_v5.cli train com \
    --dataset "$DATASET" \
    --manifest "$MANIFEST" \
    --output "$RUNS/com/seed${seed}" \
    --seed "$seed" \
    --config "$CONFIG" \
    --device cuda \
    --epochs 120 \
    --draws 40 \
    --lr 0.0002 \
    --patience 20 \
    --accumulation 4

  if ($status != 0) then
    echo "ERROR: COM seed $seed failed"
    exit 1
  endif
end

foreach seed (42 123 456)
  echo "[`whoami` `date`] Training rotation seed $seed"
  $PYTHON -m mpm_dino_v5.cli train rotation \
    --dataset "$DATASET" \
    --manifest "$MANIFEST" \
    --output "$RUNS/rotation/seed${seed}" \
    --seed "$seed" \
    --config "$CONFIG" \
    --device cuda \
    --epochs 120 \
    --draws 40 \
    --lr 0.0002 \
    --patience 20 \
    --accumulation 4

  if ($status != 0) then
    echo "ERROR: rotation seed $seed failed"
    exit 1
  endif
end

echo "[`whoami` `date`] V5 global-motion job complete"
exit 0
