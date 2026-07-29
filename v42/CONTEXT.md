# V4.2 context

## Status

V4.2 is a planning-stage continuation of the V4.1 experiments. No V4.2 model,
training run, split, or promotion rule has yet been approved. This directory
records the current scientific interpretation and the questions that must be
resolved before implementation.

The frozen V4.1 UID split and dataset remain the starting point. V4.2 must not
silently change the dataset, split, evaluator, or earlier results.

## Evidence inherited from V4.1

### Track-B COM result

Architecture-identical real-DINO training sometimes improved world/COM
prediction relative to zero and point-shuffled DINO. However, fixed-weight
interventions showed that:

- the COM output was bit-identical when DINO was zeroed or shuffled;
- complete trajectories changed only at nanometre scale;
- the learned local displacement was approximately zero;
- pointwise DINO alignment had negligible inference-time effect.

The defensible interpretation is therefore an **optimization-path effect**:
DINO-related local-loss gradients altered the shared physical trunk during
training and sometimes led the COM head to a better basin. The evidence does
not show that a useful DINO deformation representation caused the COM
improvement.

### Phase-2 local-shape result

The COM-normalized local-shape experiment compared physical-only, geometry,
real-DINO and point-shuffled conditions. It used a 75% soft / 25% rigid draw
mixture and excluded world-position and COM loss. All conditions converged to
an almost-zero local output and were effectively indistinguishable.

Two problems were subsequently identified:

1. COM subtraction removes translation but not global rotation. Consequently,
   centre-relative point and edge-vector losses can interpret rigid rotation
   as deformation.
2. Genuine impact/deformation frames are sparse and occur at different
   horizons for different episodes. Fixed horizons do not identify a common
   physical stage.

### Ground-truth deformation audit

The rotation-invariant audit found that deformation is not absent:

- soft-body H1 deformation is effectively zero;
- later soft deformation is thousands of times above the rigid numerical
  floor;
- validation/test H30 and H40 deformation is nevertheless often only about
  0.7--1.1% of object radius;
- deformation is strongly heterogeneous across UIDs and non-monotonic across
  horizons;
- Phase-2 local output was roughly 350--750 times smaller than the available
  validation H30/H40 geometric signal.

This supports both a data/supervision-dilution problem and a model-collapse
problem. It does not establish that DINO contains material or deformation
information.

## V4.2 central questions

V4.2 should answer three questions separately:

1. Can the available physical state predict COM reliably without DINO?
2. Can a rotation-correct local objective learn correctly timed deformation at
   all?
3. If local supervision is allowed to influence the physical trunk, does it
   reproducibly help or damage COM?

These questions must not be collapsed into a single world-RMSE comparison.

## Non-claims

V4.2 must not claim:

- DINO correspondence benefit without matched shuffled controls;
- inference-time DINO dependence from differences between separately trained
  checkpoints;
- material-property inference;
- obstacle generalization from the single wall episode;
- clean contact learning from proxy-contact trajectories;
- deformation benefit using rotation-contaminated centre-relative error alone;
- that an impact-stage label is available or required at inference.

## Authoritative inputs for the next design discussion

- `AGENTS.md`
- `CONTEXT.md`
- `V2_CONTEXT.md`
- `v4/CONTEXT.md`
- `v4/DESIGN.md`
- `v4/TRACKS_COMPARISON.md`
- `v4/V41_ARCHITECTURE_EXPERIMENT_MEMO.md`
- `v4/V41_LOCAL_SHAPE_PHASE2.md`
- `v4/V41_RESULTS.md`
- `v41/CONTEXT.md`
- `v41/manifests/v41_uid_splits.json`
- `v41/runs/local_shape_phase2_seed42_456/analysis_20260728/RESULTS.md`
- `v41/dataset/analysis_deformation_signal_20260728/RESULTS.md`
- `v42/PLAN.md`

