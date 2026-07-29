# V4.2

V4.2 is currently a design proposal for separating COM/rigid motion from local
deformation, correcting rotation-contaminated supervision, and testing
stage-aware deformation learning with controlled gradient routing.

Read in this order:

1. [`CONTEXT.md`](CONTEXT.md)
2. [`PLAN.md`](PLAN.md)

No V4.2 training is authorized or recorded in this directory yet.

Implementation scaffolding now lives in `v4/src/mpm_dino_v4/v42_*.py`.
It provides rotation-aware targets, protected global/local gradient routing,
stage metadata and separated losses. This does not record or authorize a
training run; experimental gates remain manual.

The learned local conditions are architecture-identical: geometry-only is the
zero-DINO control using the real validity mask, while real and point-shuffled
DINO use the same projection, four geometry-aware region tokens,
cross-attention adapter and canonical-displacement head.

Gate-0 contact geometry uses the mean floor gap of the lowest four active
surface points with an episode-adaptive tolerance, corroborated by a raw excess
vertical COM acceleration of `0.2g`. The same stage logic applies to all
families. This replaces the rejected 2% point-height quantile.

Run the training/validation-only Gate-0 audit from the repository root:

```bash
PYTHONPATH=v2/src:v4/src python v42/audit_targets_and_stages.py
```

The original Gate 1 found useful post-contact COM learning, but damaged the
near-exact ballistic H1 prediction and did not materially improve rotation over
an identity baseline. Gate 1B is the approved minimal repair:

- retain the existing ballistic-plus-residual COM parameterization;
- force the learned COM residual to exactly zero at H1;
- optimize rotation with smooth squared chordal distance while reporting
  geodesic radians;
- add a `0.25` key-horizon chordal rotation term at
  H1/H8/H16/H30/H40/H59;
- write `VALIDATION_BASELINES.json` from `best.pt`, stratified by family and
  Panel Z/V, comparing learned COM with ballistic COM and learned rotation
  with identity.

Run Gate 1B on the lab server:

```bash
PYTHONPATH=v2/src:v3/src:v4/src python -u v42/run_gate1.py \
  --device cuda \
  --seeds 42 456 \
  --epochs 120 \
  --draws 40 \
  --patience 20 \
  --plateau-patience 5 \
  --no-amp \
  --runs v42/runs/gate1b_seed42_456
```

This command does not launch Gate 2. Gate-1 checkpoints must be reviewed
before the protected local screen starts.
