# V4.2 plan: separated COM and rotation-aware local deformation

## Purpose

V4.2 will separate stable global-motion learning from the unresolved local
deformation problem while retaining a controlled way to test whether local
supervision improves the physical trunk.

This document is a design plan, not authorization to train. Rotation handling,
target representation, gradient routing, sampling, and promotion rules must be
reviewed before implementation.

## Proposed decomposition

For point `p` at future time `t`, represent the prediction conceptually as:

```text
predicted point = predicted COM
                + predicted rigid orientation component
                + predicted local non-rigid deformation
```

The first two components describe global rigid motion. The final component
should be zero for a rigid object and non-zero only for genuine deformation.

```text
physical state ──> physical trunk ──> COM head
                         |
                         +──────────> rotation / rigid-motion representation
                         |
DINO or geometry ──> local adapter ─> local-deformation head
```

The exact rotation representation is deliberately unresolved.

## Design gate 1: address rotation before changing the sampler

This is the first V4.2 decision. COM-centred coordinates are insufficient
because they retain rotation. A rigid object can therefore receive a large
"shape" error despite having zero strain.

Candidate approaches:

### A. Kabsch-aligned supervision

At each supervised frame, align the target to a reference or predicted rigid
frame and train on the residual.

Advantages:

- simple and directly connected to the completed audit;
- produces an explicit non-rigid target;
- supplies a meaningful zero baseline for rigid objects.

Questions:

- align target to the reference, prediction to target, or both to a canonical
  frame?
- should gradients pass through the SVD/Kabsch operation?
- how should degenerate or highly symmetric point clouds be handled?
- does per-frame optimal alignment hide errors in predicted rotation timing?

### B. Explicit rotation head

Predict COM and global rotation, transform the reference shape with the
predicted rotation, then predict only the remaining deformation.

Advantages:

- makes global rigid motion and non-rigid motion explicit;
- allows rotation error and deformation error to be evaluated separately;
- preserves physical temporal timing.

Questions:

- quaternion, 6D rotation representation, or incremental angular velocity?
- how is rotation defined for deformable objects?
- is one global rotation meaningful during strong deformation?
- should the rotation head share the COM trunk?

### C. Rotation-invariant local targets

Train primarily on edge lengths, strain, distances, or other invariant local
geometry.

Advantages:

- avoids global alignment;
- does not penalize rigid rotation;
- closely targets deformation.

Questions:

- edge lengths alone do not uniquely reconstruct a 3D shape;
- chirality/direction and coherent displacement can be lost;
- a position-producing decoder still needs a coordinate frame.

### Provisional direction

Evaluate a hybrid of **explicit rigid-motion prediction plus invariant local
strain/edge supervision**, with Kabsch residual as an evaluation metric and
possibly an auxiliary target. Do not implement this choice until its geometry,
gradient behaviour, and failure cases have been reviewed.

## Separate objectives

### COM objective

The COM path should initially use physical inputs only:

```text
L_COM = COM trajectory error
      + optional COM velocity/acceleration consistency
      + predeclared key-horizon COM terms
```

It should use all valid frames and both families. Rotation and local DINO
features should not contaminate its target.

### Rigid-motion objective

If an explicit rotation head is adopted:

```text
L_rigid = rotation/orientation error
        + transformed-reference consistency
```

Rigid objects provide the cleanest supervision. For soft objects, the global
orientation definition must be fixed before training.

### Local-deformation objective

The local branch should use rotation-correct targets:

```text
L_local = non-rigid residual error
        + edge-strain error
        + edge-length consistency
        + rigid zero-deformation penalty
        + full-trajectory timing penalty
```

World-position and COM error should not be used as evidence that this branch
learned deformation.

## Controlled gradient routing

The earlier COM improvement may depend on local-loss gradients entering the
shared trunk. V4.2 should make this coupling explicit:

```text
trunk gradient = gradient(L_COM) + alpha * gradient(L_local)
```

Candidate settings:

- `alpha = 0`: local loss cannot alter the physical trunk;
- small `alpha`, such as `0.1`: weak controlled multi-task influence;
- `alpha = 1`: fully coupled reference condition.

The COM head itself should not receive local-loss gradients. The local head and
adapter should always receive their local gradients.

Two complementary training regimes should be distinguished:

1. **Protected staged training:** train the COM baseline, attach the local
   branch, freeze or weakly update the trunk, and verify that COM is preserved.
2. **Joint-from-initialization training:** required to test whether the earlier
   COM improvement is a reproducible optimization-path effect.

Protected staging can preserve an existing COM solution, but it cannot be
assumed to preserve the earlier real-DINO advantage.

## Impact-stage-aware supervision

Impact stage is training metadata, not an inference input.

Ground-truth training/validation trajectories may be partitioned descriptively
into:

```text
free flight -> contact onset -> compression
            -> maximum deformation -> recovery -> post-impact
```

Stage definitions should be derived from physical signals such as obstacle
distance/contact proxy, change in strain, strain magnitude, and its temporal
derivative. They must not be derived from test performance.

For the floor-only prototype, define the robust geometry signal as the mean
gap of the lowest four active surface points. This fixed-count lower tail
captures small rigid contact patches that the earlier 2% quantile missed while
remaining robust to a single penetrating/outlier point. Contact onset still
requires a raw excess COM acceleration of at least `0.2g`. The geometry
tolerance is episode-adaptive:
`max(10 mm, 0.25 * initial lowest-four gap)`. Because the centred second
difference marks the transition interval, label the following saved frame as
first visible contact. Apply this same rule to every family and regime; do not
introduce family-specific stage logic.

### Why stages help

- guarantee that scarce compression/recovery examples contribute gradients;
- retain pre-contact frames that penalize deformation occurring too early;
- retain recovery frames that penalize deformation persisting too long;
- distinguish timing failure from magnitude/shape failure;
- avoid assuming that H30 represents the same physical event for every UID.

### What stages do not do

- they do not tell the model the stage at inference;
- they do not make contact timing observable if required physical inputs are
  absent;
- they do not establish DINO benefit;
- they must not replace full-trajectory training.

### Proposed mixture

Use both:

1. full-trajectory batches for timing and global sequence consistency; and
2. event-emphasized windows or loss weighting for visibility of contact,
   compression, and recovery.

The mixture ratio is an open hyperparameter. A small prototype should compare
against the unchanged full-trajectory baseline.

## Inference contract

The intended model must infer future stages using only deployable inputs:

- observed positions and motion/velocity;
- horizon/time encoding;
- available obstacle/contact geometry;
- object geometry;
- DINO only in the conditions being tested.

Ground-truth future COM, deformation magnitude, Kabsch alignment, impact stage,
or future contact state must never be model inputs.

## Required diagnostics

Evaluation should keep Panel Z and Panel V separate and report:

### Global path

- world RMSE/MAE;
- COM error;
- H1, H8, H16, H30, H40 and H59;
- per-frame curves;
- penetration and active coverage.

### Rigid motion

- translation error;
- rotation/orientation error if defined;
- Kabsch residual;
- false local-deformation magnitude on rigid examples.

### Local deformation

- Kabsch-aligned point residual;
- edge-vector error only in a well-defined aligned frame;
- edge-length and normalized strain error;
- predicted versus ground-truth deformation magnitude;
- zero-prediction baseline;
- metrics by UID, deformation-magnitude bin, and impact stage;
- onset timing, peak timing and recovery timing if stage definitions are
  sufficiently reliable.

### DINO attribution

- real versus architecture-identical zero;
- point-shuffled DINO;
- fixed-weight real/zero/shuffled interventions;
- per-seed and paired per-UID effects;
- explicit separation between training-path and inference-time effects.

## Suggested experiment sequence

### Step 0: rotation and stage-definition audit

- decide the rigid/local coordinate decomposition;
- verify the chosen targets on synthetic rigid translation/rotation;
- visualize target residuals on representative rigid and soft episodes;
- define contact/deformation stages using training/validation only;
- measure counts and UID coverage per stage;
- freeze these definitions before model comparison.

### Step 1: physical-only COM/rigid baseline

- train the physical trunk, COM head, and any approved rotation head;
- establish stable multi-seed baselines;
- do not add DINO.
- use seeds 42 and 456, 40 UID-balanced draws per epoch, a 120-epoch cap,
  early-stopping patience 20 and plateau patience 5;
- select `best.pt` by validation mean full-trajectory global loss; test data
  remains untouched;
- review both completed checkpoints before authorizing Gate 2.

### Step 2: local learnability screen

- start from byte-identical physical baselines;
- compare a zero-output baseline with an architecture-identical token family:
  geometry-only/zero-DINO, aligned real DINO and point-shuffled DINO;
- use the same learned DINO projection, four geometry-aware region tokens,
  cross-attention adapter and canonical local head in all three learned
  conditions;
- define geometry-only as zero DINO values with the real `dino_valid` mask,
  so model capacity, validity information and initialization are matched;
- initially freeze the COM head and trunk (`alpha = 0`);
- use corrected local targets and full plus event-emphasized supervision;
- determine whether any local branch learns non-zero, correctly timed signal.

### Step 3: controlled multi-task coupling

Only if Step 2 establishes local learnability:

- compare `alpha = 0`, weak coupling and full coupling;
- test whether local gradients reproducibly improve COM without degrading
  local metrics, H1, penetration, or stability;
- distinguish staged preservation from joint-from-initialization effects.

### Step 4: expanded seeds and controls

Run the frozen multi-seed comparison and any required scene-shuffled control
only after the screening criteria are satisfied.

## Decisions to resolve in the next context

1. What is the mathematical definition of global rotation for soft objects?
2. Should V4.2 predict rotation explicitly, use Kabsch-aligned targets, or use a
   hybrid?
3. Where should gradients be stopped through alignment and between branches?
4. What input represents obstacle/contact geometry, and is it adequate for
   predicting impact timing?
5. How should impact stages be defined robustly from the available data?
6. Should event emphasis use sampling, loss weights, temporal windows, or a
   combination?
7. What zero-prediction and rigid-motion baselines are mandatory?
8. What screening and promotion criteria are scientifically meaningful?
9. Which experiments can fit the prototype time budget before a three-seed
   lab run?

## Current recommendation

Do not implement the sampler first. Resolve rotation and target decomposition,
then perform the training/validation stage audit. After those definitions are
fixed, implement the smallest local-learnability screen with a protected COM
baseline and explicit gradient routing.
