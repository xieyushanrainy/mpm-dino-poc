# V2 Context: Deformation-Aware Particle-Grid Surrogate

## Read first

V2 continues from the completed V1 investigation. Before changing code, read:

1. `docs/v1/CONCLUSION.md`
2. `docs/v1/TRAINING_HISTORY.md`
3. `docs/v2/PLAN.md`
4. root `CONTEXT.md` for paper/dataset background

The canonical V1 comparison checkpoint is `v1/artifacts/rollout_s4/best.pt`.

## V1 state and failure

V1 recurrent particle state:

```text
current normalized position x_t
derived velocity v_t
persistent DINOv3 feature
validity and DINO-imputation flag
```

V1 grid input:

```text
object occupancy
object velocity
object validity
compressed DINO
controller occupancy
prescribed controller velocity
```

V1 predicts next grid occupancy/velocity and per-particle displacement. Predicted particles are recurrently scattered to rebuild the next authoritative grid.

The model predicts plausible teacher-forced motion but long recurrent rollouts twist and collapse. Step-2 and step-4 rollout training plateaued near persistence. Velocity damping and boundary-clamping diagnostics indicate they are late amplifiers, not the general initiating cause.

## V2 hypothesis

Current position and velocity do not identify elastic state. The model does not know the reference configuration or which particles were material neighbours. Current voxel proximity is not a substitute for material topology after deformation.

V2 tests whether adding an effective reference configuration and local relative deformation improves recurrent stability:

```text
persistent x_0
current displacement from reference x_t - x_0
fixed initial-neighbour graph
local stretch/deformation descriptors
next-state neighbour deformation supervision
```

This remains an approximation. PhysTwin supplies tracked surface points rather than rest volume, true mesh topology, deformation gradient or constitutive state. Frame 0 is treated as an effective reference configuration, not verified stress-free geometry.

## Fixed constraints from V1

- Continue using the same 16/3/3 scene split in `data/shared/splits/` for direct comparison.
- Keep frozen DINOv3 ViT-B/16 and the existing cached features initially.
- Keep 2,048 padded particles, 32-cubed grid, prescribed controller trajectory and object-local normalization. Read V1 caches from `data/v1/cache/` and write the incompatible V2 schema only under `data/v2/`.
- Do not claim force-conditioned or full MPM simulation.
- Do not use test scenes for checkpoint selection or hyperparameter tuning.
- Preserve the V1 checkpoint and diagnostic outputs as immutable baselines.

## Open scientific questions for V2

- Does reference geometry improve recurrence beyond merely increasing input dimension?
- Do approximate surface kNN edges represent material neighbourhoods well enough for cloth, rope and plush objects?
- Should local deformation input use stretch statistics, relative vectors or a regularized local deformation-gradient estimate?
- Can deformation features improve stationary and active-motion windows without degrading teacher-forced accuracy?
- Does DINO add value after explicit reference/deformation features are supplied?

## Required ablations

At minimum compare:

```text
V1: current state + DINO
V2-A: V1 + x_0 and x_t - x_0
V2-B: V2-A + fixed-neighbour deformation input/loss
V2-C: V2-B with zeroed DINO
```

This separates reference geometry, neighbourhood structure and DINO contribution.
