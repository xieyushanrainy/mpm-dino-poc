#!/bin/csh -f

# V5 shared physical v3: jointly train the COM and rotation heads and their
# V4.2-class physical trunk. All neural weights start from random initialization.

set PYTHON = .venv-v41/bin/python
set DATASET = v41/dataset
set MANIFEST = v41/manifests/v41_uid_splits.json
set CONFIG = v5/config.default.json
set RUNS = v5/run/shared_v3/global

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
