#!/bin/tcsh
cd /cs/student/project_msc/2025/cgvi/yushaxie/mpm-dino-poc
mkdir -p v43/run/no_memory_contact_rotation
nohup .venv-v41/bin/python v43/run_no_memory_contact_rotation.py \
  --matrix v43/NO_MEMORY_CONTACT_ROTATION_MATRIX.json \
  --runs v43/run/no_memory_contact_rotation \
  --device cuda \
  --seeds 42 123 456 \
  --variants physical_only contact_torque_basis contact_shuffled \
  --epochs 120 --draws 40 --patience 20 \
  >& v43/run/no_memory_contact_rotation/console.log &
echo $! >! v43/run/no_memory_contact_rotation/launcher.pid
