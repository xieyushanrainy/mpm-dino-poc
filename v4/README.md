# V4 Particle-Dynamics POC

This implementation trains a particle-native two-frame autoregressive model on the balanced V4 package. The primary evidence run uses rigid and soft-body objects only. Fluid is loadable for quarantined diagnostics but is excluded from material-generalization claims until its Mantaflow tracer integration and collision semantics are regenerated.

## Environment

```bash
conda activate mpm-dino-poc
export PYTHONPATH="$PWD/v2/src:$PWD/v4/src"
```

## Prepare

```bash
python -m mpm_dino_v4.cli audit --dataset v4/dataset/packaged_balanced_90 --output v4/FLUID_AUDIT.json
python -m mpm_dino_v4.cli split --dataset v4/dataset/packaged_balanced_90 --output v4/splits_balanced_90.json
python -m mpm_dino_v4.cli prepare --dataset v4/dataset/packaged_balanced_90 --manifest v4/splits_balanced_90.json --output v4/cache
pytest -q v4/tests
```

The manifest fixes 20/5/5 UIDs per family before any windows are generated. Graphs use frame-0 geometry with mutual kNN defaults `candidate_k=12`, `max_neighbours=8`.

## Train matched controls

```bash
for mode in zero real scene_shuffled; do
  for seed in 42 123 456; do
    python -m mpm_dino_v4.cli train --cache v4/cache --manifest v4/splits_balanced_90.json --dino-mode "$mode" --seed "$seed" --output "v4/runs/$mode/seed$seed"
  done
done
```

The completed experiment compares real DINOv2, zero DINO, and deterministic scene-shuffled DINO. Point-shuffled DINO remains deferred.

Training samples uniformly across within-object target-acceleration quartiles. Validation and test retain their natural window distribution. Invalid DINO rows are zeroed before projection and accompanied by `dino_valid`.

## Evaluate

```bash
python -m mpm_dino_v4.cli evaluate --cache v4/cache --manifest v4/splits_balanced_90.json --output v4/runs/constant_velocity_test.json --device mps
python -m mpm_dino_v4.cli evaluate --cache v4/cache --manifest v4/splits_balanced_90.json --checkpoint v4/runs/real/seed42/best.pt --dino-mode real --seed 42 --output v4/runs/real/seed42/test_metrics.json
python -m mpm_dino_v4.cli aggregate v4/runs/{zero,real,scene_shuffled}/seed*/test_metrics.json --output v4/runs/aggregate.json
```

Evaluation writes raw object/window records plus window- and object-weighted one-step and autoregressive summaries for every horizon, including H1/H4/H8/H16. Aggregate and family-specific results are always retained together.

## Scientific gate

DINO is considered helpful only if real DINO beats both zero and scene-shuffled controls by more than seed variation. Fluid results must remain separate: the packaged fluid route exhibits an approximately 104 mm median first-frame COM drop, floor crossings in 27/30 objects, non-persistent tracer semantics, and eventual zero activity in many trajectories.

## Track B full-trajectory experiment

Track B predicts `X[2:61]` simultaneously from the first two frames. It uses a
gravity-aware ballistic reference, initial fixed-graph encoding, and four
factorized spatial/temporal blocks. The primary controls are pooled real DINOv2
and zero DINO:

```bash
for mode in zero real; do
  for seed in 42 123 456; do
    python -m mpm_dino_v4.cli train-full \
      --cache v4/cache \
      --manifest v4/splits_balanced_90.json \
      --dino-mode "$mode" \
      --seed "$seed" \
      --device mps \
      --epochs 300 \
      --batch-size 1 \
      --accumulation-steps 4 \
      --hidden-dim 128 \
      --blocks 4 \
      --heads 4 \
      --output "v4/runs/track_b/$mode/seed$seed"
  done
done
```

Evaluate a learned checkpoint or an analytic baseline:

```bash
python -m mpm_dino_v4.cli evaluate-full \
  --cache v4/cache \
  --manifest v4/splits_balanced_90.json \
  --checkpoint v4/runs/track_b/real/seed42/best.pt \
  --dino-mode real \
  --seed 42 \
  --device mps \
  --output v4/runs/track_b/real/seed42/test_metrics.json

python -m mpm_dino_v4.cli evaluate-full \
  --cache v4/cache \
  --manifest v4/splits_balanced_90.json \
  --baseline ballistic \
  --output v4/runs/track_b/baseline_ballistic_test.json
```

Every Track B run retains its resolved configuration, manifest hash, complete
history, best/last checkpoints, and evaluation output under
`v4/runs/track_b/`. Real-versus-zero results are provisional evidence only:
causal visual-information claims require a matched Track B shuffled control.
