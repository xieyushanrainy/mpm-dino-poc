# V2 Plan: Reference Geometry and Local Deformation

## Objective

Determine whether persistent reference geometry and approximate local material neighbourhoods reduce recurrent twisting and compounding error on unseen PhysTwin interactions.

## Data changes

- Treat normalized frame-0 particle positions as effective reference positions `x0`; copy them unchanged through every window and rollout.
- Build a fixed initial surface-neighbour graph over real, non-padded particles. Start with 12 nearest-neighbour candidates, retain up to 8 mutual neighbours, and store a neighbour-validity mask. Never connect padded particles.
- Store rest relative vectors and lengths for every retained edge.
- Audit graph length distributions and visualize edges for cloth, rope and plush scenes before training; reject graph construction if it bridges obvious disconnected surfaces.

## Model inputs

Add particle-held features:

```text
x0                              3 channels
current reference displacement  x_t - x0, 3 channels
mean neighbour stretch          1 channel
neighbour stretch standard dev  1 channel
maximum absolute stretch        1 channel
```

For edge `(i,j)`, calculate current stretch without future leakage:

```text
s_ij(t) = ||x_i(t)-x_j(t)|| / (||x_i(0)-x_j(0)|| + eps) - 1
```

- Supply all nine new features to the particle head.
- Scatter `x_t-x0` and the three stretch statistics onto the grid, adding six U-Net input channels.
- Recompute deformation descriptors from predicted positions at every recurrent step.
- Do not feed target-frame deformation as input.

## Training losses

Retain V1 particle, occupancy, grid-velocity and consistency losses. Add masked losses over valid fixed edges:

```text
edge-vector loss:
  SmoothL1((x_i_pred-x_j_pred), (x_i_true-x_j_true))

edge-length loss:
  SmoothL1(||x_i_pred-x_j_pred||, ||x_i_true-x_j_true||)
```

Initial weights:

```text
edge vector: 0.25
edge length: 0.10
```

Log raw and weighted contributions. Rescale only if either new term is more than 10x larger or smaller than the particle term over a full epoch.

## Experiment sequence

1. Add cache-schema versioning, `x0`, neighbour indices/masks and graph visualization tests.
2. Implement V2-A (`x0` and reference displacement only) and train from scratch with the V1 one-step policy.
3. Implement V2-B deformation descriptors and edge losses; train from scratch using identical splits and budgets.
4. Select one-step checkpoints by validation particle distance and retain persistence/motion-stratified reports.
5. Fine-tune with randomized rollout lengths uniformly sampled from 1-4, then 1-8 only after validation improvement. Retain teacher-forced no-regression protection.
6. Run V2-C with DINO channels zeroed using the same seed, budget and checkpoint policy.
7. Compare V1/V2 at horizons 1, 4, 8, 16 and full sequence, reporting aggregate and motion-stratified errors.

## Acceptance criteria

V2-B is successful only if it:

- improves validation and held-out recurrent particle error over V1 at horizons 8 and 16;
- reduces unsupported twisting in all three held-out visualizations;
- preserves teacher-forced particle error within 10% of V1;
- does not increase out-of-grid fraction;
- shows improvement beyond V2-A, attributing value to neighbourhood deformation rather than extra coordinates alone.

DINO is considered useful only if V2-B outperforms the matched V2-C zero-DINO model.

## Known limitations

- Frame 0 may already contain stress or deformation and is only an effective reference.
- Surface kNN is not true material connectivity and can bridge folds or nearby disconnected surfaces.
- Edge losses regularize observed surface deformation, not volumetric conservation or constitutive physics.
- No force, torque, density, resting volume or measured material parameter is introduced.
- Architectural changes require new V2 checkpoints; V1 weights are baselines, not guaranteed initialization.

