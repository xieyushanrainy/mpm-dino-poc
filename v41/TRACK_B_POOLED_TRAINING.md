# Exact V4 Track B pooled-DINO replication on V4.1

Status: implementation and six-run matrix prepared; production training has not
been launched.

The optional shape-balanced loss follow-up is documented separately in
[`TRACK_B_SHAPE_LOSS_EXPERIMENT.md`](TRACK_B_SHAPE_LOSS_EXPERIMENT.md).

## Scientific question

Repeat the original V4 Track B pooled-DINO/FiLM architecture on the corrected
V4.1 dataset while using the same data, loss, training budget, guarded
selection, and evaluation policy as the V4.1 M1/M2/M6 screen.

This is a bridge experiment, not an architecture-identical control for
M1/M2/M6. Its matched control is its own zero-DINO condition.

## Fixed architecture

The model is the original `FullTrajectorySurrogate`, wrapped only to accept the
V4.1 loader's explicit `reference` tensor. The old architecture itself is
unchanged:

```text
DINO [B,N,384]
  -> LayerNorm -> Linear 384→64 -> SiLU -> Linear 64→16
  -> masked mean + masked max + valid fraction
  -> object condition [B,128]
  -> blockwise FiLM in four width-128 graph-temporal blocks
```

The physical path retains the original ballistic reference, two initial graph
layers, four graph-temporal blocks, four temporal-attention heads, and
COM/zero-mean-local correction heads. The zero condition retains every
projection, pooling, conditioning, and FiLM parameter and receives zero DINO
with the real `dino_valid` mask.

## V4.1 data and loss

- frozen manifest: `v41/manifests/v41_uid_splits.json`;
- training data: Panel Z plus Panel V through `V41TrajectoryDataset`;
- 40 UID/family-balanced draws per epoch;
- no family, solver, material, or VLM metadata enters the model;
- Panel Z and Panel V remain separate in reporting.

The shared loss is `compute_full_trajectory_loss`:

```text
1.00 residual
1.00 world position
0.50 COM
0.25 edge vector
0.10 edge length
0.25 key horizons (H4/H8/H16/H59)
```

This is the same implementation and weighting used by the V4.1 mechanisms.
Validation selection remains normalized mean RMSE over H16/H30/H40. Real DINO
must pass the matched-zero validation H1 guard before `best.pt` is eligible.

## Matrix

The dedicated entry point is:

```text
v41/run_track_b_pooled_matrix.py
```

It creates six runs:

```text
track_b_pooled × {zero, real} × {42, 123, 456}
```

Frozen defaults:

- CUDA FP32, AMP disabled;
- 150-epoch cap;
- early-stopping patience 30 after the H1 guard first passes;
- `ReduceLROnPlateau` patience 5;
- 40 balanced draws per epoch;
- AdamW learning rate `2e-4`;
- width 128, four blocks, four heads.

For every seed, zero trains first. The real run uses that seed's guarded zero
`best.pt` as its H1 reference. If zero has no guarded best checkpoint, the
matrix stops rather than substituting `last.pt`.

## Lab launch

From the repository root, after the existing V4.1 environment and tests pass:

```bash
mkdir -p v41/runs/track_b_pooled_cap150_p30_fp32

nohup bash -c '
  set -o pipefail
  PYTHONPATH=v2/src:v3/src:v4/src \
  .venv-v41/bin/python -u v41/run_track_b_pooled_matrix.py 2>&1 |
  tee -a v41/runs/track_b_pooled_cap150_p30_fp32/console.log
' > v41/runs/track_b_pooled_cap150_p30_fp32/nohup.log 2>&1 &

echo $! > v41/runs/track_b_pooled_cap150_p30_fp32/launcher.pid
```

Rerunning the identical command resumes incomplete runs. The entry point
rejects changes to the seeds, budget, device, AMP setting, or frozen manifest
inside an existing matrix directory.

## Evaluation

Only guarded `best.pt` checkpoints are scientifically eligible. Evaluate each
eligible checkpoint with `evaluate_v41`, which reports:

- H1/H8/H16/H30/H40/H59 and per-frame curves;
- world RMSE/MAE and COM error;
- centre-relative shape, edge-vector, and edge-length error;
- penetration rate/depth and active coverage;
- rigid Kabsch residual;
- separate Panel Z and Panel V summaries.

Apply the same frozen promotion rule used for M1/M2/M6. H59 remains diagnostic
only. Do not launch shuffled controls unless pooled real DINO first passes that
rule against its own matched zero condition.
