#!/bin/tcsh
mkdir -p v43/run/rotation_memory
nohup .venv-v41/bin/python v43/run_rotation_memory.py \
  --matrix v43/ROTATION_MATRIX.json \
  --runs v43/run/rotation_memory \
  --device cuda >& v43/run/rotation_memory/console.log &
echo $! >! v43/run/rotation_memory/launcher.pid
