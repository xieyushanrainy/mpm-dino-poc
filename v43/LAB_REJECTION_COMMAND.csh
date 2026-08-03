#!/bin/tcsh
setenv PYTHONPATH "v2/src:v3/src:v4/src"
mkdir -p v43/runs/bad_neighbour_rejection_seed42_123_456
nohup .venv-v41/bin/python -u v43/run_bad_neighbour_rejection.py \
  --device cuda \
  --seeds 42 123 456 \
  --arms compact_baseline compatibility_gate compatibility_gate_wrong_training \
  --epochs 120 \
  --draws 40 \
  --accumulation 4 \
  --patience 20 \
  --top-k 3 \
  --memory-tokens 32 \
  --champion v42/checkpoints/v42_adapter_full_seed42_best.pt \
  --runs v43/runs/bad_neighbour_rejection_seed42_123_456 \
  >& v43/runs/bad_neighbour_rejection_seed42_123_456/console.log &
echo $! > v43/runs/bad_neighbour_rejection_seed42_123_456/launcher.pid
