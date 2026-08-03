mkdir -p v43/runs/causal_contact_source42_trainseed123
setenv PYTHONPATH "v2/src:v3/src:v4/src"
nohup .venv-v41/bin/python -u v43/run_causal_contact_replacement.py \
  --device cuda \
  --seeds 123 \
  --source-seed 42 \
  --variants static_control causal_continuous \
  --epochs 120 \
  --draws 40 \
  --patience 20 \
  --plateau-patience 5 \
  --contact-threshold 0.01 \
  --gate1e-root v42/runs/gate1e_seed42_456 \
  --oracle-root v42/runs/direct_decoder_probe_seed42_456 \
  --runs v43/runs/causal_contact_source42_trainseed123 \
  >& v43/runs/causal_contact_source42_trainseed123/console.log &
echo $! > v43/runs/causal_contact_source42_trainseed123/launcher.pid
