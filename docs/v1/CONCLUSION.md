# V1 Conclusion: DINO-Conditioned Particle-Grid Dynamics

Date closed: 2026-07-04

## Executive conclusion

V1 demonstrates that frozen DINOv3 features can be attached to tracked 3D surface particles, projected into a learned 3D particle-grid surrogate, and used with prescribed controller motion to predict short-horizon deformation. Teacher-forced predictions are often plausible and remain substantially better than long recurrent rollouts.

V1 does **not** demonstrate stable long-horizon simulation or that DINO contributes material information. Recurrent predictions drift, develop unsupported twisting/shearing, and eventually collapse. Rollout fine-tuning at two and four steps produced only small validation improvements and did not beat persistence globally. Diagnostics show that velocity feedback and grid-boundary clamping amplify late errors but do not explain the initial instability.

The leading V2 hypothesis is that the recurrent state is physically incomplete: it contains current position, derived velocity and DINO, but no reference configuration, material connectivity, local deformation, deformation gradient, stress or other elastic memory.

## What V1 built

- Frozen DINOv3 ViT-B/16 features extracted from camera 0 and cached as 768D descriptors.
- Projection of visible RGB features onto multi-view RGB-D-derived 3D tracks, with nearest-visible feature imputation for camera-0-occluded points.
- Deterministic sampling/padding to 2,048 persistent surface particles per scene.
- A 32-cubed particle-grid-particle architecture:
  - object occupancy, velocity, validity and compressed DINO grid fields;
  - separate prescribed-controller occupancy and velocity fields;
  - a three-level 3D U-Net;
  - supervised next occupancy and grid velocity;
  - a particle head predicting continuous displacement.
- Action-conditioned recurrence using the next controller displacement, not force or torque.
- One-step and full-backpropagation rollout training, persistence baselines, teacher-forced guardrails, MPS/CUDA device support, visualization and velocity-feedback diagnostics.

## Canonical artifact

The retained V1 model is:

```text
v1/artifacts/rollout_s4/best.pt
```

It is the step-4 rollout checkpoint from epoch 3. Its weights are standalone even though its training lineage passed through earlier one-step and step-2 checkpoints.

Canonical validation metrics:

```text
four-step recurrent particle mean: 0.0071929494
four-step persistence mean:        0.0070527965
model / persistence:               1.019872
teacher-forced particle mean:      0.0041393263
teacher no-regression guard:       passed
```

The model is approximately 2% worse than persistence over aggregate four-step validation, while preserving useful teacher-forced behavior. Motion-stratified evaluation is better than the aggregate: V1 beats persistence on the highest-motion windows but loses on nearly stationary windows.

Canonical checkpoint evaluation over the three untouched test scenarios:

```text
model normalized particle mean:       0.0061416072
persistence normalized particle mean: 0.0049815430
constant-velocity mean:               0.015929565
model / persistence:                  1.2329
```

## Verified findings

### Data and conditioning

- The PhysTwin data supplies synchronized RGB-D observations, tracked surface trajectories and controller trajectories, not true MPM states.
- The 3D trajectories are pseudo-ground truth derived from CoTracker, depth and calibration; they are not complete volumetric reconstruction or measured continuum state.
- The dataset does not supply force, torque, resting volume, mass, density, material parameters, deformation gradient or stress.
- Controller motion is a model input during training and inference. Thirty controller points are voxelized into separate occupancy and prescribed-velocity channels for each update.
- DINO features persist with particle identity and are supplied to both the grid U-Net and particle head.

### Optimization

- The original unnormalized Huber particle term was numerically overwhelmed by grid losses. Normalized Smooth-L1 with beta 0.01 made particle and grid contributions comparable.
- Checkpoint selection by validation particle distance was more aligned with the task than selection by the aggregate grid objective.
- Reduce-on-plateau and relative-improvement early stopping operated correctly and stopped one-step/rollout continuation after genuine plateaus.
- Motion oversampling helped expose the model to active interactions but did not solve recurrent stability.

### Recurrence

- Teacher-forced predictions are much better than full recurrent predictions on unseen objects.
- Step-2 rollout fine-tuning gave a small guarded improvement; step-4 improved to near persistence but then plateaued.
- Longer continuation of the step-4 run did not beat its epoch-3 checkpoint.
- Full-sequence visualization extrapolates far beyond the trained four-step horizon and shows growing displacement, twisting and eventual collapse.

### Ruled-out or downgraded causes

- **Velocity feedback is not the primary short-horizon cause.** Sweeping
  `v_next = alpha * displacement/dt + (1-alpha) * v_previous` showed alpha 1.0 was best at horizons 4, 8 and 16. Strong damping only modestly delayed very late failure.
- **Grid clamping is not the general initiating cause.** Zebra particles leave the normalized domain late and clamping then worsens collapse, but unseen sloth and cloth accumulate large errors with zero out-of-grid particles.
- **More identical epochs are not enough.** Training error continued to improve while validation recurrent error plateaued or worsened.

## What remains unverified

- DINO has not been compared with a matched zero-DINO model. V1 proves that the architecture can consume DINO, not that DINO improves physics prediction or identifies material properties.
- No RGB, shuffled-DINO, DINOv2 or multi-view DINO ablations were completed.
- The unexpected twist has not been matched to any particular training trajectory. Cross-scene motion priors remain possible, but direct copying is not established.
- Material or force inference is not demonstrated.
- Generalization claims are limited by 22 scenarios, correlated object families and pseudo-label noise.
- Sparse manual `gt_track_3d.pkl` tracks were not integrated into final evaluation.

## Main V1 limitation

The recurrent state is not a Markov description of deformable mechanics. It includes current surface position and velocity but lacks the object's reference geometry and elastic memory. The grid also aggregates current spatial neighbours without knowing which particles were material neighbours initially. Consequently, different internal strain states can look similar to the network, and unsupported local shear/twist is weakly constrained.

V2 will test this diagnosis by adding persistent reference positions, fixed initial neighbourhoods, current local-deformation descriptors and neighbourhood deformation supervision.
