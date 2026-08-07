#!/bin/csh -f

# V5 COM v2: restore the V4.2-class pointwise graph/temporal physical trunk,
# initialize every neural weight from scratch, and train the multi-term COM loss.

set PYTHON = .venv-v41/bin/python
set DATASET = v41/dataset
set MANIFEST = v41/manifests/v41_uid_splits.json
set CONFIG = v5/config.default.json
set RUNS = v5/run/com_physical_v2

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

echo "[`whoami` `date`] Starting V5 physical COM v2"

$PYTHON -m mpm_dino_v5.cli audit "$MANIFEST"
if ($status != 0) then
  echo "ERROR: V5 integrity audit failed"
  exit 1
endif

foreach seed (42 123 456)
  echo "[`whoami` `date`] Training physical COM v2 seed $seed"

  $PYTHON -m mpm_dino_v5.cli train com \
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
    echo "ERROR: physical COM v2 seed $seed failed"
    exit 1
  endif
end

echo "[`whoami` `date`] V5 physical COM v2 complete"
exit 0
