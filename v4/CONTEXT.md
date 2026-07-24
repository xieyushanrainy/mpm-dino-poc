# V4 Context: Force-Free Visual-Material Trajectory Modelling

Date: 2026-07-22

## Purpose

V4 restarts the POC on a new synthetic dataset with no applied control-force
field. The intended scientific advantage is a cleaner relationship between an
object's appearance/material assignment and its motion than in V1-V3, where
motion was dominated by prescribed controller trajectories or action metadata.

The v4 question is:

> Given a rendered object, its persistent 3D surface points and the initial
> state, do visual features improve prediction of the full gravity/contact
> trajectory beyond geometry, solver-route and explicit-material controls?

This wording is deliberately narrower than "infer material from motion" or
"replace MPM parameters with DINO." The dataset contains simulator outputs and
VLM-derived material labels, not measured material constants or real MPM state.

## Read first

1. Root [`CONTEXT.md`](../CONTEXT.md) for the paper and dataset lineage.
2. [`docs/v1/CONCLUSION.md`](../docs/v1/CONCLUSION.md) for the first recurrent
   particle-grid model and its failure modes.
3. [`docs/v2/CONTEXT.md`](../docs/v2/CONTEXT.md) and
   [`docs/v2/PLAN.md`](../docs/v2/PLAN.md) for reference geometry, fixed
   neighbourhoods and deformation losses.
4. [`v3/PLAN.md`](../v3/PLAN.md) and
   [`v3/artifacts/SEEN_FAMILY_UNSEEN_OBJECT_SEED_SWEEP_CONCLUSION.md`](../v3/artifacts/SEEN_FAMILY_UNSEEN_OBJECT_SEED_SWEEP_CONCLUSION.md)
   for particle-native architectures and the inconclusive DINO result.
5. Root [`CONTEXT.md`](../CONTEXT.md) also records PhysiFormer as an optional
   architecture reference for the wider POC, not as the required V4 basis.

## Previous POC context

### V1

V1 attached frozen DINOv3 features to 2,048 tracked PhysTwin surface points,
voxelized them on a 32-cubed grid and predicted recurrent displacement with a
3D U-Net and particle head. Teacher-forced predictions were useful, but the
four-step model was about 2% worse than persistence on aggregate validation and
about 23% worse on the untouched test set. Long rollouts twisted and collapsed.

The central diagnosis was incomplete recurrent state: current position and
velocity did not encode reference geometry, material neighbourhoods or elastic
memory. V1 also never established that DINO helped relative to a matched
zero-DINO model.

### V2

V2 added frame-0 reference positions, displacement from reference, a fixed
initial surface-neighbour graph, stretch descriptors and edge-vector/edge-length
losses. These are useful reusable ideas for persistent surface points, but they
remain approximations to continuum state: surface kNN is not true topology,
frame 0 is not guaranteed stress-free and no volumetric deformation gradient is
available.

### V3

V3 removed the 3D grid and continuous controller trajectory, screening direct
graph, object-latent graph and action-token graph architectures. The
`latent_graph` family was the most promising short-horizon architecture, but
DINO did not earn a mandatory role:

- on the normal no-bottleneck split, zero-DINO beat final-DINO by 8.36% on the
  mean H4/H8 validation error across seeds 45, 123 and 456;
- on a seen-family/unseen-object six-seed sweep, final-DINO was 3.27% worse in
  aggregate because of a large seed-123 failure, although it was 2.55% better
  when that seed was excluded;
- constrained-geometry experiments suggested weak object-level DINO identity,
  not reliable point-to-feature semantics;
- short rollout fine-tuning improved edge consistency but did not reliably
  improve H8/H16 particle error.

The transferable conclusion is to keep visual conditioning optional and always
compare it with zero, shuffled and geometry-only controls over multiple seeds.

## New v4 dataset

### Location and provenance

The user-facing description calls this `v4/data`, but the dataset currently
exists at:

```text
v4/dataset/packaged_dataset_1k/
```

Generation and packaging code is in:

```text
../../coderepos/DINO_VLM_sim_dataset_gen/
```

Generator commit inspected: `eab048cb5018a45baa2d3dc4aeed2a675bb5b2e5`.
The package index was created on 2026-07-22 and declares schema version 1.

Relevant entry points:

- `run_local_end_to_end.py`: resumable preparation, VLM, DINO and simulation
  pipeline;
- `blender_prepare_rigid.py`: point sampling, correspondence and provisional
  rigid preparation;
- `blender_simulate_selected.py`: selected Blender solver execution;
- `run_local_vlm_materials.py`: Qwen3-VL material assignment;
- `run_dino_features.py`: DINOv2 point-feature extraction;
- `package_test_dataset.py`: row-aligned standalone package construction;
- `validate_physics_dataset.py`: schema and correspondence checks.

### Verified package contract

The package contains 632 objects. Every sample has:

```text
trajectory_positions_m  float32 [61, 2048, 3]
dino_features           float16 [2048, 384]
dino_valid              bool    [2048]
point_material_ids      int32   [2048]
point_active            bool    [61, 2048]
```

The row contract is persistent: row `i` refers to the same sampled rest-surface
point across the trajectory, DINO vector, validity mask and material ID. All
samples are 2.0 seconds at 30 saved frames/s (`dt = 0.033333333`), in metres,
right-handed Blender XYZ with +Z up. Four rendered views are included.

Dataset-wide verified counts:

| Property | Value |
|---|---:|
| Objects | 632 |
| Rigid / fluid / soft-body routes | 601 / 30 / 1 |
| Frames per sample | 61 |
| Surface points per sample | 2,048 |
| DINO feature dimension | 384 |
| Median DINO-valid fraction | 65.4% |
| Applied force fields | 0 for all 632 |
| Gravity | `[0, 0, -9.81]` m/s2 for all 632 |

The package README says 632 samples even though the folder name contains `1k`.
Treat 632 as authoritative for this snapshot.

### Added soft-body assets and balanced 90-object package

The original 632-object package contained only one soft-body object, so it was
not used as the final three-family training set unchanged. A separate acquisition
run downloaded additional Objaverse assets originating from Sketchfab and built
30 homogeneous soft-body samples. Candidate annotation terms covered plush,
foam/sponge, rubber/silicone, cushions, plants and food. Annotation matching was
only a retrieval pre-filter; it was not accepted as the final material label.

Two candidate batches were screened:

| Batch | Selected candidates | Prepared | VLM screened | Successful soft simulations |
|---|---:|---:|---:|---:|
| `selection_soft_100` / `soft_dataset_100` | 100 | 100 | 90 | 21 |
| `selection_soft_60b` / `soft_dataset_60b` | 60 | 59 | 56 | 10 |

The screening and generation environment was the `dinopocdataset` Conda
environment. Four 2,048-pixel PBR views were rendered for each candidate.
`gpt-5.4-mini-2026-03-17` was used as the OpenAI VLM with minimum accepted
confidence `0.4`; unsupported, rigid and mixed-route candidates were rejected
rather than coerced into the soft-body route. Accepted objects were simulated
with Blender Soft Body under gravity only, for 2 seconds at 30 saved frames/s,
with 2,048 persistent surface points and DINOv2-Small features.

The final standalone soft package is:

```text
v4/dataset/packaged_soft_30/
```

It contains 30 VLM-approved, successfully simulated, validated and packaged
homogeneous soft-body objects. One additional successful simulation was not
needed for the final deterministic 30-object selection. After packaging and
hash/metadata validation, the 30 corresponding source GLBs were pruned to save
space; `PRUNED_SOURCE_GLBS.json` records the removed UIDs, paths and byte counts.
The compact package retains the four rendered views, point-aligned arrays and
metadata but not the source meshes. The reusable procedure is documented in
[`dataset/HOW_TO_ADD_SOFT_SAMPLES.md`](dataset/HOW_TO_ADD_SOFT_SAMPLES.md).

The main balanced dataset was then assembled at:

```text
v4/dataset/packaged_balanced_90/
```

It contains exactly 30 unique rigid, 30 fluid and 30 soft-body objects. Rigid
and fluid objects came from `packaged_dataset_1k`; soft objects came from
`packaged_soft_30`. The rigid subset was selected round-robin across primary
material categories, while all eligible fluid and soft objects were ordered
deterministically by category/confidence. `SELECTION_AUDIT.json` records every
UID, source package, route, primary material category and minimum VLM
confidence. This balanced package, rather than the imbalanced 632-object source,
is the default dataset for the current three-family architecture investigation.

The acquisition licence policy allowed any Objaverse-downloadable Sketchfab
licence. Asset-specific attribution and redistribution terms still need to be
recovered from Objaverse metadata before any external redistribution, especially
because the compact package does not itself include those licence records.

### What "no control force" does and does not mean

There is no configured external force field. This removes the controller/action
conditioning that complicated V1-V3 and makes full-trajectory modelling from an
initial state plausible.

However, motion is not caused by material alone. Every object starts with a
0.1 m drop above a floor and evolves under gravity, contact, restitution,
friction, object geometry, pose, solver route and numerical settings. Rigid
objects can reveal density/contact behavior only through the simulator's rigid
dynamics; they do not deform. Fluid and soft-body routes use different solvers,
so solver identity can become an easy proxy for material class.

The correct causal interpretation is therefore:

```text
trajectory = f(geometry, initial pose/state, gravity, floor/contact,
               solver route, simulator parameters, material parameters)
```

DINO/VLM features may correlate with the latter terms, but the dataset does not
isolate them automatically.

### Data-quality and scientific risks

1. **Source-package route imbalance.** 95.1% of the original 632-object package
   is rigid. A random split of that source can yield excellent aggregate metrics
   while learning little about deformable behavior. Use the balanced 90-object
   package for the main three-family study and still report per-route metrics.
2. **DINO version change.** The package uses `facebook/dinov2-small` (384-D),
   whereas V1-V3 used DINOv3 ViT-B/16. V4 results are not a direct continuation
   of earlier DINO ablations.
3. **Partial visual coverage.** Median `dino_valid` is about 65.4%; invalid
   surface points need an explicit imputation strategy and mask. Never interpret
   imputed vectors as observed material evidence.
4. **Time-varying activity.** `point_active` is `[T,N]`, not a static padding
   mask. Losses and attention must mask each frame independently, especially for
   fluid particles or solver failures/deactivation.
5. **VLM labels are hypotheses.** Material categories and parameters were
   inferred from renders using Qwen3-VL in the original source package and
   OpenAI `gpt-5.4-mini-2026-03-17` for the added soft-body batch. They are
   neither measured ground truth nor independent of the visual input tested.
6. **Possible target leakage.** Explicit material/solver fields used to generate
   the target can dominate DINO. Keep them as oracle baselines, not silently
   mixed into the main visual model.
7. **Split leakage.** Split by object UID and audit near-duplicate geometry or
   asset families. Do not split frames or points from one object across sets.
8. **Licensing.** The package intentionally omits licences and source meshes.
   Objaverse/Sketchfab attribution and redistribution rights must be recovered
   from the selection manifest before external release.

## PhysiFormer reference

### Paper and code

**Paper:** *PhysiFormer: Learning to Simulate Mechanics in World Space*  
**Authors:** Yiming Chen, Yushi Lan, Andrea Vedaldi  
**Year:** 2026  
**arXiv:** 2606.27364  
**Local PDF:** [`../../mpmpapers/physformer.pdf`](../../mpmpapers/physformer.pdf)  
**Local repository:** [`../../coderepos/PhysiFormer/`](../../coderepos/PhysiFormer/)  
**Commit inspected:** `f7cb285fe612266f50290f8411da4ec2650834df`

### Core formulation

PhysiFormer models the conditional full-trajectory distribution

```text
p(X_1:T | X_0, V_0, material)
```

directly in world-coordinate vertex space. It is a flow-matching diffusion
transformer with x-prediction/v-loss. During sampling, it starts from coordinate
noise and integrates an ODE from noise to a clean trajectory using 50 Heun
steps; the first frame is clamped to the initial position.

The paper's data has 49-frame triangular-mesh trajectories, rigid/elastic
material labels, per-vertex initial velocity and multiple interacting objects.
Training covers over 100k simulated scenes. Its main result relevant to this POC
is that one-shot full-trajectory generation avoids the exposure bias and error
accumulation that damaged autoregressive baselines.

### Architecture details

- Each noisy `(time, vertex)` 3D coordinate becomes a token.
- Initial position and initial velocity are separately embedded and
  broadcast-added to every temporal token for that vertex.
- Material conditioning is embedded by an MLP and added per vertex.
- The DiT backbone alternates attention axes in the repeated pattern
  `spatial -> temporal -> object -> temporal`.
- Spatial attention operates across all scene vertices within a frame;
  temporal attention follows each vertex across all frames; object attention
  gathers and pads vertices belonging to the same object.
- Coordinate-conditioned 3D RoPE supplies relative spatial position; 1D RoPE
  supplies time position.
- Sixteen learned register tokens are replicated into each factorized attention
  group and mean-reduced back to aggregate global context.
- AdaLN conditioning injects diffusion time/class context into each block.
- The paper's large configuration uses hidden dimension 1,024 and a 24-block
  `4 x 6` factorized stack. The released code also defines smaller configurable
  variants and supports per-frame masks, object IDs, scene tokens and per-object
  material vectors.

### Relevance to V4

PhysiFormer offers an alternative response to a dominant V1-V3 failure:
autoregressive rollout error. The v4 dataset has persistent point identities
and a fixed 61-frame horizon, so a full-trajectory coordinate-denoising
experiment is possible. This is one analysis option, not the default V4 design.

Useful concepts to reuse:

- full-trajectory, non-autoregressive prediction;
- separate spatial and temporal attention;
- coordinate RoPE rather than learned absolute vertex indices;
- first-frame position/velocity conditioning and clamping;
- frame-aware masks;
- register tokens/global object context;
- multiple stochastic samples and best/mean/distributional evaluation;
- trajectory MSE plus rigidity and momentum-consistency metrics where valid.

### Important incompatibilities

1. V4 has sampled surface point clouds, not triangle meshes. Mesh faces/topology
   are absent, so rendering topology and connected-component rigidity cannot be
   copied directly.
2. V4 has one object per sample. Object-level attention is unnecessary unless a
   scene later contains multiple bodies; a simpler `spatial -> temporal` stack
   is the appropriate starting point.
3. Dense spatial attention over 2,048 points for 61 frames is expensive:
   `O(T*N^2)` means roughly 256 million pair scores per spatial layer per
   sample, before heads/batches. The paper's training meshes are much smaller.
4. The released pretrained checkpoint is trained on different Genesis
   geometries, materials and normalization. Treat the code as an architectural
   reference, not a drop-in v4 checkpoint.
5. PhysiFormer conditions on explicit material type. Replacing that condition
   with DINO is a new hypothesis and needs explicit-material, solver-route,
   zero-DINO and shuffled-DINO controls.

## V4 candidate analysis tracks

V4 should not be committed to PhysiFormer in advance. Compare architectures
that answer different questions while sharing the same data contract, splits,
metrics and controls.

### Inputs and targets

```text
target:       X[0:61, 0:2048, xyz]
initial:      X0, V0 (finite difference or zero if verified)
visual:       DINOv2 feature + dino_valid/imputation flag per point
geometry:     rest-point coordinates and optional fixed kNN descriptors
context:      gravity, floor height, dt, solver route (control/oracle only)
mask:         point_active[t, i]
```

Normalize coordinates with a documented object-local transform but retain
physical scale as a conditioning scalar; scale affects gravity/contact motion.
Do not normalize each future frame independently.

### Track A: reuse V3 architectures

Adapt the V3 `latent_graph` and, where useful, `graph_direct` models to the new
dataset. Remove controller/action inputs, preserve reference geometry and fixed
neighbour features, and condition dynamics on gravity/time plus optional DINO
or explicit material information. This track provides continuity with the
previous POC and tests whether cleaner trajectories change the earlier weak,
seed-sensitive DINO result.

Both recurrent and analysis-only uses are valid. For recurrence, retain the V3
H1/H4/H8/H16 evaluation and edge-consistency diagnostics. For material
analysis, reuse the object-latent encoder to inspect whether visual or learned
latents separate solver routes/material categories without assuming that such
separation improves trajectory prediction.

### Track A1: V3-like autoregressive design under consideration

The current preferred recurrent formulation is a V3-like, particle-native
autoregressive model. It does **not** receive a family, material-class or
solver-route token. The model must predict the next state from visual features,
geometry and observed motion/form change. Family and solver labels may be used
for stratified sampling, diagnostics and per-family evaluation, but not as
network inputs.

#### Sliding-window contract

Training does not require a globally specified start frame or impact frame
`t0`. After splitting by object UID, construct sliding windows anywhere in each
61-frame trajectory:

```text
input:   X[t], X[t+1], reference geometry, DINO, masks, dt
target:  X[t+2]
t:       0 ... 58
```

Thus the two input frames are the two most recent observed states, not
necessarily frames 0 and 1. They provide a finite-difference velocity estimate

```text
V[t+1] = (X[t+1] - X[t]) / dt
```

and, for windows after deformation begins, observable changes in local shape.
First contact can be detected for sampling and evaluation, but it is neither a
required input nor a required origin for the model's time coordinate.

Split UIDs into train/validation/test **before** expanding trajectories into
windows. Adjacent windows and the 2,048 points from one object are correlated;
the effective independent sample count remains the number of objects, not the
number of generated windows.

#### Prediction parameterization

Prefer predicting displacement or the residual over a constant-velocity
baseline rather than absolute coordinates:

```text
X_cv[t+2]       = 2 * X[t+1] - X[t]
residual target = X[t+2] - X_cv[t+2]
prediction      = X_cv[t+2] + residual_model(...)
```

This makes the learned component focus on acceleration, collision response and
non-rigid deformation instead of relearning uniform translation. Preserve both
world-coordinate motion and object-centred motion in the state or loss:

```text
centre of mass:       C[t] = mean_i X[t,i]
local configuration:  Q[t,i] = X[t,i] - C[t]
```

World coordinates are necessary for gravity and floor contact. Centre-relative
coordinates, fixed reference kNN edges and edge-length/vector changes expose
local deformation without confusing it with free fall. The initial/reference
point cloud remains useful recurrent memory even when only two dynamic frames
are supplied.

#### Temporal sampling

Uniformly sampling all windows can over-represent easy free-fall and settled
states while under-representing short collision/deformation transients. The
training sampler may stratify or weight windows using target-side motion change,
for example the RMS second difference

```text
||X[t+2] - 2*X[t+1] + X[t]||
```

provided this value controls sampling only and is not passed to the model.
Report results under the natural, unweighted test distribution as well as
phase-stratified diagnostics. Sampling policy must not move windows from the
same UID across splits.

#### Teacher forcing and rollout

One-step teacher-forced accuracy is necessary but insufficient. Evaluate:

1. one-step prediction from two ground-truth frames;
2. autoregressive rollout seeded by two ground-truth frames, with predictions
   fed back as subsequent inputs;
3. error as a function of rollout horizon, including V3-compatible
   H1/H4/H8/H16 summaries;
4. per-family world-coordinate, centre-of-mass and centre-relative shape error;
5. edge consistency, Kabsch-aligned residual for rigid samples, floor
   penetration and active-point coverage.

Training should compare pure teacher forcing with scheduled/predicted-state
exposure or short rollout loss. Any rollout-training gain must be separated
from the DINO-conditioning claim.

#### Visual representation and controls

The current balanced package contains 30 rigid, 30 fluid and 30 soft-body
objects. Original mesh-aligned DINOv2-Small features are the preferred primary
representation because they are 384-D and have the most accurate point/image
correspondence. DINOv3 ViT-B/16 (768-D) remains an ablation: on the 90 objects
it modestly improved matched three-family linear classification but did not
improve global silhouette clustering.

The main autoregressive comparison should include at least geometry-only,
DINO-plus-geometry, shuffled-DINO and constant-velocity baselines. Do not
concatenate full 384-D DINOv2 and 768-D DINOv3 features by default. Project the
chosen frozen visual representation into a compact learned latent before graph
message passing.

#### Early-frame empirical caveat

For windows seeded at simulation frames 0 and 1, rigid and soft-body samples
contain effectively no local deformation. After removing translation and the
best rigid rotation, median frame-0-to-frame-1 non-rigid RMS displacement was
approximately `0.0000026%` of object radius for rigid and `0.00034%` for soft
body. These two frames mostly encode initial geometry and falling velocity, so
DINO supplies the only plausible material prior before interaction creates
observable form change.

The current fluid route is not temporally/physically comparable in this early
window. Its median centre-of-mass drop from frame 0 to 1 is about `112 mm`,
versus the analytical `5.45 mm` free-fall displacement over 1/30 s; 27/30 fluid
samples already have tracers below the declared floor at frame 1. Fluid points
are Mantaflow velocity-field tracers rather than persistent material particles.
Consequently, early fluid motion can expose a solver/pipeline signature. Do not
interpret strong frame-0/1 fluid discrimination as clean visual-material
inference until the fluid advection/time-scale and collision behavior are
corrected or regenerated.

This caveat does not invalidate arbitrary-time autoregressive windows, but it
does mean that a model can learn incompatible route dynamics from the current
targets. Report fluid separately and audit the fluid generator before making a
three-family physics-generalization claim.

### Track B: deterministic full-trajectory model

Predict the whole displacement trajectory with a compact factorized transformer
or graph-temporal model. This tests non-autoregressive prediction separately
from diffusion and separately from the specific PhysiFormer implementation.

### Track C: optional PhysiFormer-inspired experiment

After the shared data pipeline and deterministic baselines are validated, test
a compact flow-matching/full-trajectory model if warranted. Use scalable
spatial mixing such as fixed-kNN/local attention, anchors, or global register
tokens rather than assuming dense 2,048-point spatial attention. PhysiFormer
components are candidates for selective reuse, not architectural requirements.

Across all tracks, compare an object-pooled DINO latent first. Add per-point
DINO only if particle-shuffle controls show that point alignment matters.

### Required baselines and ablations

At minimum, compare under identical splits, seeds and budgets:

```text
B0  gravity/contact analytic or persistence/rigid-transform baseline
B1  geometry + initial state
B2  B1 + solver route (oracle diagnostic only; excluded from the main model)
B3  B1 + explicit VLM/simulator material parameters (oracle)
V1  B1 + pooled DINO
V2  B1 + point-aligned DINO
C1  zero-DINO
C2  scene-shuffled DINO
C3  point-shuffled DINO within each scene
```

The main DINO claim requires V1/V2 to beat B1 and the matched shuffled controls
by more than seed variance. B2 and B3 are oracle diagnostics that estimate the
information ceiling available from the simulator's route/material metadata;
they are not permitted inputs to the proposed autoregressive network.

### Splits and reporting

- Freeze UID-level train/validation/test manifests before tuning.
- Stratify by solver route and broad material category, but never move points or
  frames from the same UID across splits.
- Report rigid, fluid and soft-body results separately; do not use aggregate
  error alone.
- Use the balanced 90-object package for the main three-family experiment: 30
  rigid, 30 fluid and 30 soft-body objects. Preserve the original 632-object
  package as a separate imbalanced-source experiment rather than silently
  combining its additional rigid objects with the balanced evaluation.
- Report per-frame trajectory RMSE/MAE in metres, final-frame error, centre-of-
  mass error, pairwise-distance/shape error and active-point coverage.
- For rigid samples, add Kabsch-aligned rigidity residual and rigid-transform
  trajectory error.
- For stochastic models, report mean sample quality, best-of-K separately, and
  diversity/calibration; best-of-K must not be compared directly with a single
  deterministic prediction without disclosure.

## V4 acceptance criteria

V4 should be considered a successful material-grounding POC only if:

1. at least one model improves long-horizon error over the best simple baseline
   under matched evaluation; architecture-family comparisons must remain
   separate from the DINO/material-conditioning claim;
2. DINO improves held-out UID performance beyond geometry and matched shuffled
   controls across multiple seeds;
3. the improvement survives route-stratified reporting and is not explained
   solely by rigid/fluid solver classification;
4. full-trajectory predictions preserve rigid shape and do not exploit inactive
   points or invalid DINO rows;
5. all claims are framed as prediction of this synthetic simulator distribution,
   not recovery of real material constants.

If the DINO criteria fail, V4 can still support narrower architecture-specific
conclusions, such as improved V3 recurrence or improved full-trajectory
stability, while visual material conditioning remains unproven.
