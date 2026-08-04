#!/bin/tcsh
cd /cs/student/project_msc/2025/cgvi/yushaxie/mpm-dino-poc
mkdir -p v43/run/rotation_contact_curvature
nohup .venv-v41/bin/python v43/run_rotation_contact_curvature.py \
  --matrix v43/ROTATION_CONTACT_CURVATURE_MATRIX.json \
  --runs v43/run/rotation_contact_curvature \
  --device cuda \
  --seeds 42 123 456 \
  --variants geometry_base contact_timing contact_patch contact_curvature curvature_shuffled wrong_memory_contact \
  --epochs 120 --draws 40 --patience 20 \
  >& v43/run/rotation_contact_curvature/console.log &
echo $! >! v43/run/rotation_contact_curvature/launcher.pid
