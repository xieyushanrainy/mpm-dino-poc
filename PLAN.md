# DINO-Conditioned Particle-Grid Dynamics POC

## Summary

Build an action-conditioned, MPM-inspired neural surrogate for the 22 PhysTwin/MatPhys scenarios. Persistent particles carry frozen DINOv3 appearance features; compressed features are scattered onto a 32-cubed grid so a 3D U-Net is explicitly DINO-conditioned. The predicted next grid guides per-particle displacement, after which the next input grid is reconstructed from moved particles.

This is a particle-grid neural dynamics surrogate, not a full MPM simulator: the observations do not contain constitutive parameters, deformation gradients, stresses, or true MPM grids.

## Data contract

- Read `final_data.pkl`, camera-0 RGB/depth/calibration, and controller trajectories from the PhysTwin dataset.
- Keep a deterministic maximum of 2,048 object tracks per scene and pad smaller scenes with a validity mask.
- Use frames 0 and 1 to initialize velocity and predict from frame 1 onward.
- Normalize into an object-local cube, retain metric scale and frame interval, and supply controller displacement for each predicted step.
- Keep particle identities during rollout and mask losses when either endpoint of a pseudo-track is invalid.
- Use frozen DINOv3 ViT-B/16 final normalized patch tokens. Reproject initial points into camera 0, depth-test them, bilinearly sample visible features, and use nearest-visible imputation plus a flag for occluded points.
- Cache 768D features once, project them trainably to 16D, retain them on particles, and scatter the compressed embeddings onto the grid at every step.

## Model and update contract

```text
particles at t: position, velocity, DINO, validity
        -> differentiable trilinear scatter
32^3 object occupancy, velocity, DINO and validity fields

controller action t->t+1
        -> separate scatter
32^3 controller occupancy and velocity fields

combined grid
        -> compact three-level 3D U-Net
32^3 -> 16^3 -> 8^3 -> 16^3 -> 32^3

outputs:
  next soft occupancy grid
  next object velocity grid
  full-resolution decoder features

sample predicted fields/features at each particle
+ particle position, velocity, DINO, imputation flag, scale, dt
        -> particle MLP
particle displacement
```

For recurrence, move particles, derive velocity as displacement divided by `dt`, rebuild the authoritative grid from moved particles, attach the next prescribed controller action, and repeat. The predicted grid guides particle motion but is not independently fed back.

## Training and evaluation

- Pretrain one-frame teacher-forced pairs with masked Huber losses.
- Initial weights: particle displacement 1.0, grid velocity 0.5, soft occupancy 0.25, moved-particle/grid consistency 0.25.
- Fine-tune with 2-, 4-, then 8-step unrolls and losses at every step.
- Select by validation eight-step particle error while retaining one-step error as a no-regression diagnostic.
- Primary deterministic scene-held-out split: 16 train, 3 validation, 3 test, stratified by action and object family.
- Also report exploratory rope-, cloth-, zebra-, and sloth-family-held-out evaluations.
- Report metric particle error at 1, 4, 8, 16, and full-rollout horizons, grid velocity error, occupancy IoU, validity coverage, and qualitative trajectories.
- Compare against persistence, constant velocity, and a matched trained model with DINO channels zeroed.
- First-stage success means beating persistence and constant velocity on held-out multistep particle error. DINO need not beat zero-DINO until the later material-information study.

## Defaults and deferred work

- Canonical view: camera 0; DINOv2 and multi-view fusion are deferred ablations.
- Initial grid: 32 cubed; 48 cubed is a later resolution ablation.
- DINOv3 weights and licence must be available; otherwise use DINOv2 ViT-B/14 behind the same cached-feature interface.
- Larger datasets, RGB/shuffled-DINO controls, and statistically meaningful unseen-object tests are later stages.
