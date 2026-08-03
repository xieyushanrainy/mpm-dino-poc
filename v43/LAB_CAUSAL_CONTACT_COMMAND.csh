mkdir -p v43/runs/causal_contact_seed42_456
setenv PYTHONPATH "v2/src:v3/src:v4/src"
nohup .venv-v41/bin/python -u v43/run_causal_contact_replacement.py \
  --device cuda \
  --seeds 42 456 \
  --epochs 120 \
  --draws 40 \
  --patience 20 \
  --plateau-patience 5 \
  --contact-threshold 0.01 \
  --gate1e-root v42/runs/gate1e_seed42_456 \
  --oracle-root v42/runs/direct_decoder_probe_seed42_456 \
  --runs v43/runs/causal_contact_seed42_456 \
  >& v43/runs/causal_contact_seed42_456/console.log &
echo $! > v43/runs/causal_contact_seed42_456/launcher.pid
