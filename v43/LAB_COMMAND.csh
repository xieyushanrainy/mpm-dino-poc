#!/bin/tcsh
setenv PYTHONPATH "v2/src:v3/src:v4/src"
mkdir -p v43/runs/deformation_retrieval_seed42_456
nohup .venv-v41/bin/python -u v43/run_deformation_retrieval.py \
  --device cuda \
  --seeds 42 456 \
  --epochs 120 \
  --draws 40 \
  --accumulation 4 \
  --patience 20 \
  --top-k 3 \
  --memory-tokens 32 \
  --champion v42/checkpoints/v42_adapter_full_seed42_best.pt \
  --runs v43/runs/deformation_retrieval_seed42_456 \
  >& v43/runs/deformation_retrieval_seed42_456/console.log &
echo $! > v43/runs/deformation_retrieval_seed42_456/launcher.pid
