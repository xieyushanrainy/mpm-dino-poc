#!/bin/tcsh
set run_root = v43/run/cpu_smoke_lab
mkdir -p $run_root
nohup python -u v43/smoke_retrieval.py \
  >& $run_root/nohup.log &
