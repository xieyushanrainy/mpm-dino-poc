#!/bin/tcsh
mkdir -p v43/runs/cpu_smoke
setenv PYTHONPATH "v2/src:v3/src:v4/src"
nohup .venv-v41/bin/python -u v43/smoke_retrieval.py \
  --output v43/runs/cpu_smoke/RUN_COMPLETE.json \
  >& v43/runs/cpu_smoke/console.log &
echo $! > v43/runs/cpu_smoke/launcher.pid
