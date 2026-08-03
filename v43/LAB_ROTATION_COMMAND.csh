#!/bin/tcsh
mkdir -p v43/run/rotation_memory
nohup .venv-v41/bin/python v43/run_rotation_memory.py \
  --matrix v43/ROTATION_MATRIX.json \
  --runs v43/run/rotation_memory \
  --full --no-smoke \
  --seeds 42 123 456 \
  --arms zero_memory geometry aligned_dino scene_shuffled \
  --epochs 120 --draws 40 --patience 20 \
  --device cuda >& v43/run/rotation_memory/console.log &
echo $! >! v43/run/rotation_memory/launcher.pid
