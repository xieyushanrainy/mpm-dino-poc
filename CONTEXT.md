> Project status, 2026-07-04: V1 is concluded. See
> [`docs/v1/CONCLUSION.md`](docs/v1/CONCLUSION.md) and
> [`V2_CONTEXT.md`](V2_CONTEXT.md) before beginning V2 work.

## Papers and Datasets

### PhysCtrl

**Paper:** *PhysCtrl: Generative Physics for Controllable and Physics-Grounded Video Generation*  
**Authors:** Chen Wang, Chuhao Chen, Yiming Huang, Zhiyang Dou, Yuan Liu, Jiatao Gu, Lingjie Liu  
**Year:** 2025  
**URL:** https://arxiv.org/abs/2509.20358  
**DOI:** Unknown  
**Local PDF:** [PhysCtrl.pdf](../mpmpapers/PhysCtrl.pdf)

#### Research problem and core method

PhysCtrl learns a fast surrogate for physics simulation so that physically plausible object motion can guide controllable video generation. MPM and a rigid-body solver generate offline training trajectories. A conditional diffusion transformer then learns to denoise 24-frame trajectories of 2,048 corresponding 3D points, conditioned on the initial point cloud, force and application point, floor height, material class, Young's modulus $E$, and Poisson's ratio $\nu$.

Physical conditions are embedded by a jointly trained MLP. Spatial attention models interactions between points within each frame, while temporal attention follows each point across frames. Training combines trajectory reconstruction, velocity, MPM-inspired deformation-gradient, and floor-penetration losses. During forward inference, the frozen model generates trajectories from specified physical conditions. During optional inverse inference, the model remains frozen while an unknown condition such as $E$ is optimized to explain an observed 3D trajectory.

#### Relevance to this POC

- Provides scripts and a data contract for generating MPM ground-truth trajectories.
- Provides an explicit-parameter baseline conditioned on $E$, $\nu$, force, geometry, and material class.
- Suggests experiments that replace or augment $(E,\nu,[\mathrm{mat}])$ with DINO features projected onto object points.
- Provides trajectory losses, metrics, and inverse parameter estimation as possible baselines.
- Separates the useful simulation/trajectory components from the downstream video-generation system.

#### Assumptions and limitations

- Training trajectories are synthetic and inherit the MPM constitutive models and simulator settings.
- Density/mass is not a variable trajectory-model condition; the local generation configuration fixes density at 1,000. Whether this matches every released sample needs verification.
- Applied force follows a weight-normalized convention, so learned force effects depend on that convention.
- Learned and inferred parameters are simulator-effective values, not independently measured material constants.
- The paper reports little trajectory sensitivity to $\nu$, making it difficult to infer from motion.
- Inverse estimation requires a corresponding 3D point trajectory; RGB-only material inference is not demonstrated.
- Generalization beyond the simulated shapes, materials, parameter ranges, and force settings is not guaranteed.

#### Associated repository

**Local repository:** [physctrl](../coderepos/physctrl/)  
**Commit:** `803264f2901e9f14481ab366991976b517571178`  
**Tag:** Unknown; needs verification.

Relevant entry points:

- [`src/data_generation/generate_mpm_data.py`](../coderepos/physctrl/src/data_generation/generate_mpm_data.py): generates elastic, plasticine, and sand MPM trajectories in HDF5 format.
- [`src/data_generation/generate_rigid_data.py`](../coderepos/physctrl/src/data_generation/generate_rigid_data.py): generates rigid-body trajectories.
- [`src/data_generation/simulator/mpm/mpm_solver_warp.py`](../coderepos/physctrl/src/data_generation/simulator/mpm/mpm_solver_warp.py): Warp MPM solver.
- [`src/data_generation/configs/objaverse_mpm.json`](../coderepos/physctrl/src/data_generation/configs/objaverse_mpm.json): simulator defaults.
- [`src/dataset/traj_dataset.py`](../coderepos/physctrl/src/dataset/traj_dataset.py): trajectory and condition loader.
- [`src/model/dit.py`](../coderepos/physctrl/src/model/dit.py) and [`src/model/spacetime.py`](../coderepos/physctrl/src/model/spacetime.py): diffusion transformer and spatial-temporal blocks.
- [`src/train.py`](../coderepos/physctrl/src/train.py): trajectory-model training.
- [`src/eval.py`](../coderepos/physctrl/src/eval.py): trajectory inference and evaluation.
- [`src/utils/physparam.py`](../coderepos/physctrl/src/utils/physparam.py): inverse physical-parameter estimation.
- [`src/inference.py`](../coderepos/physctrl/src/inference.py): complete image-to-video pipeline.

**Potential reuse:** MPM data generation, HDF5 trajectory format, dataset loader, trajectory losses, parameter sweeps, and evaluation code.

**Reference only initially:** image-to-3D reconstruction, diffusion video pipeline, pretrained video model, and inverse parameter estimation. The transformer is an architectural baseline rather than a required dependency for the first POC.

---

### PhysCtrl Synthetic MPM Trajectory Dataset

**Dataset:** PhysCtrl synthetic trajectory dataset  
**Source:** PhysCtrl repository README; generated from Objaverse/ObjaverseXL objects, with the repository stating that the dataset is based on the open-source TRELLIS-500K collection.  
**Download status:** Not downloaded. The README says the full original dataset is not released because of its size, but provides a sample archive and regeneration scripts.  
**Local dataset path:** None.

#### Contents, structure, and formats

The paper describes approximately 550,000 simulated object animations across elastic, plasticine, sand, and rigid materials. The repository generates one HDF5 (`.h5`) file per rollout under a structure similar to:

```text
src/data_generation/data/objaverse/
├── raw/hf-objaverse-v1/glbs/
├── outputs_mpm/h5/
└── outputs_mpm/visualization/
```

The generator writes point positions `x`, velocities `v`, deformation gradients `F`, affine matrices `C`, particle volumes `vol`, floor height, drag point/mask/force, $E$, $\nu$, material type, and gravity flag. Each rollout uses 2,048 sampled points. The current generation configuration produces 48 frames, while the paper's learned model uses 24-frame trajectories; exact released-sample shapes and subsampling need verification.

#### Intended role in this POC

Regenerate a controlled subset rather than depend on the unavailable full dataset. Hold geometry, actions, and simulator settings fixed while varying selected material parameters, then compare:

1. explicit material-parameter conditioning;
2. DINO-only conditioning with selected parameters hidden;
3. DINO plus explicit parameters;
4. geometry-only or constant-feature controls.

The initial prediction target should be point trajectories or displacements. This tests whether DINO features can replace conditions in a learned trajectory model; it does not imply that arbitrary DINO vectors can directly replace constitutive parameters inside numerical MPM.

#### Required preprocessing

1. Download licence-compatible Objaverse/ObjaverseXL meshes and build metadata/UID lists.
2. Validate, normalize, and sample each object to 2,048 points.
3. Generate MPM rollouts over controlled material and force sweeps.
4. Create deterministic object-level train/validation/test splits.
5. Render consistent image views for DINO extraction; renderer and camera protocol need verification.
6. Project or interpolate 2D DINO features onto simulated surface points.
7. Define features for occluded surfaces and interior MPM particles.
8. Normalize coordinates, forces, frame intervals, and material parameters consistently.

#### Limitations, licensing, and data-quality concerns

- Exact reproduction depends on external object assets and local simulation because the full dataset is unavailable.
- Objaverse/Sketchfab assets can have asset-specific licences and redistribution restrictions.
- TRELLIS-500K and PhysCtrl dataset licensing terms need verification.
- No top-level licence was found in the local PhysCtrl repository; code-reuse permission needs verification.
- Mesh quality, watertightness, scale, thin geometry, and sampling may produce unstable MPM rollouts.
- DINO features may encode category, texture, lighting, and geometry rather than physical material properties.
- Synthetic trajectories depend on assumed density, constitutive models, contacts, and force conventions.
- If identical visual objects receive randomly varied material parameters, their unchanged DINO features cannot identify those variations. Appearance and simulated material labels must be correlated, or the POC must evaluate DINO as an auxiliary representation rather than a source of hidden parameter values.

#### Connection to paper and repository

This dataset trains PhysCtrl's trajectory diffusion model and can be recreated using [physctrl/src/data_generation](../coderepos/physctrl/src/data_generation/). For this POC, the generator and HDF5 contract are more relevant than the video-generation components.

---

### PhysTwin

**Paper:** *PhysTwin: Physics-Informed Reconstruction and Simulation of Deformable Objects from Videos*  
**Authors:** Hanxiao Jiang, Hao-Yu Hsu, Kaifeng Zhang, Hsin-Ni Yu, Shenlong Wang, Yunzhu Li  
**Year:** 2025  
**URL:** https://arxiv.org/abs/2503.17973  
**DOI:** Unknown  
**Local PDF:** [phystwin.pdf](../mpmpapers/phystwin.pdf)

#### Research problem and method

PhysTwin reconstructs an object-specific, simulation-ready digital twin from a short interaction recorded by three synchronized RGB-D cameras. It combines complete geometry generated with a learned shape prior, a spring-mass physics model, inferred spring topology and physical parameters, and 3D Gaussians for rendering.

Physical parameters are estimated separately for each video by replaying tracked hand actions and minimizing disagreement between simulated and observed 3D geometry and motion. Zero-order optimization handles discrete topology and global parameters; gradient-based optimization refines dense spring stiffness.

#### Relevance to this POC

PhysTwin provides the real RGB-D interaction dataset used by PhysWorld and MatPhys. Relevant assets include:

- RGB and depth observations;
- camera calibration;
- object and controller masks;
- dense CoTracker-derived 3D pseudo-trajectories;
- sparse manually annotated 3D evaluation tracks;
- reconstructed shape assets;
- control-point trajectories.

These provide potential real-world surface trajectories against which an MPM/DINO model could be trained or evaluated.

#### Assumptions and limitations

- Reconstruction requires three synchronized RGB-D views.
- Physics is fitted separately for each object and interaction.
- Spring parameters are effective simulator values, not measured material constants.
- The spring-mass model does not fully represent continuum, anisotropic, plastic, or fracture behaviour.
- Hand interaction is approximated using tracked control points and virtual springs.
- Dense object trajectories are CoTracker-and-depth-derived pseudo-labels, not ground truth.
- No measured force, torque, tactile, or material-property data was found.
- Generalization primarily concerns new interactions with known objects.

#### Associated repository

**Local repository:** [PhysTwin](../coderepos/PhysTwin/)  
**Commit:** `2b6630528141b9cba5a7677c8b88b2129b4a8390`  
**Tag:** No exact tag is checked out.

Relevant entry points:

- [`data_process/dense_track.py`](../coderepos/PhysTwin/data_process/dense_track.py): samples masked pixels and runs CoTracker3.
- [`data_process/data_process_track.py`](../coderepos/PhysTwin/data_process/data_process_track.py): maps 2D tracks to per-pixel 3D points and filters trajectories.
- [`data_process/data_process_pcd.py`](../coderepos/PhysTwin/data_process/data_process_pcd.py): constructs per-frame point-cloud images from RGB-D observations.
- [`data_process/data_process_sample.py`](../coderepos/PhysTwin/data_process/data_process_sample.py): produces final processed simulation data.
- [`evaluate_track.py`](../coderepos/PhysTwin/evaluate_track.py): evaluates predictions against sparse `gt_track_3d.pkl` trajectories.
- [`qqtt/engine/trainer_warp.py`](../coderepos/PhysTwin/qqtt/engine/trainer_warp.py): spring-mass fitting, simulation, and model-derived force visualisation.

**Potential reuse:**

- Dataset loading and RGB-D calibration.
- CoTracker-to-3D preprocessing.
- Object and controller trajectory extraction.
- Track masks and evaluation code.

**Reference only:**

- Spring-mass simulation and per-scene inverse optimization, unless used as a comparison baseline.
- Model-derived controller forces, which are not measured ground truth.

---

### PhysWorld

**Paper:** *PhysWorld: From Real Videos to World Models of Deformable Objects via Physics-Aware Demonstration Synthesis*  
**Authors:** Yu Yang, Zhilu Zhang, Xiang Zhang, Yihan Zeng, Hui Li, Wangmeng Zuo  
**Year:** 2025  
**URL:** https://arxiv.org/abs/2510.21447  
**DOI:** Unknown  
**Local PDF:** [physworld.pdf](../mpmpapers/physworld.pdf)

#### Research problem and method

PhysWorld addresses the tension between high-fidelity but slow physics simulation and fast but data-hungry learned world models.

Its pipeline is:

1. Construct an MPM digital twin from a short real interaction.
2. Select a constitutive-model family using a vision-language model.
3. Optimize global and then spatially varying MPM properties.
4. Generate 500 synthetic episodes per scenario using varied actions and part-aware material perturbations.
5. Train a property-conditioned GNN to predict particle displacement.
6. Fine-tune the GNN's physical-property inputs against the real video.

The runtime model is the GNN, not MPM.

#### Relevance to this POC

PhysWorld is the closest reference for producing MPM trajectories from the PhysTwin dataset. Relevant ideas include:

- converting reconstructed geometry into volumetric MPM particles;
- calibrating MPM motion against real 3D tracks;
- generating action-conditioned particle trajectories;
- using approximately 400 MPM substeps per video frame;
- conditioning trajectory prediction on spatial information;
- evaluating whether DINO features could replace or augment explicit material parameters.

PhysWorld does not use DINO. It uses PartField features only to correlate material-property perturbations within semantic parts.

#### Assumptions and limitations

- MPM calibration and GNN training remain object/scenario-specific.
- MPM parameters are effective estimates, not independently measured physical values.
- The GNN inherits modelling errors from its MPM teacher.
- Synthetic material perturbations change dynamics without changing visual appearance.
- Long autoregressive GNN rollouts may accumulate error.
- The pipeline is computationally expensive before runtime inference.
- The paper relies on 3D observations derived from the PhysTwin dataset rather than demonstrating RGB-only material identification.
- No uncertainty model is provided.

#### Associated repository

**Local repository:** [PhysWorld](../coderepos/PhysWorld/)  
**Commit:** `157a309e4f58634b0265cae7c1f4fc04b07394c0`  
**Tag:** No exact tag is checked out.  
**Repository status:** The local repository currently contains only a README and figures; implementation code and MPM entry points have not been released there.

**Potential reuse:**

- MPM setup and stepping.
- Geometry-to-MPM conversion.
- Real-trajectory calibration losses.
- Synthetic rollout generation.
- Action representation and temporal alignment with video frames.

**Reference only:**

- The object-specific GNN, unless the POC also includes learned trajectory prediction.
- PartField-based perturbation if DINO features are used instead.
- VLM constitutive-model selection if the POC fixes a single material model.

---

### MatPhys

**Paper:** *MatPhys: Learning Material-Aware Physics Parameters for Deformable Object Simulation from Videos*  
**Authors:** Yang Yang, Yiyan Wang, Zheming Liu, Naoya Iwamoto  
**Year:** 2026  
**URL:** https://arxiv.org/abs/2605.19386  
**DOI:** Unknown  
**Local PDF:** [matphys.pdf](../mpmpapers/matphys.pdf)

#### Research problem and method

MatPhys replaces PhysTwin's per-video parameter optimization with a shared feed-forward spring-parameter predictor. Its pipeline combines:

- TRELLIS2 reconstruction of 3D Gaussians from a keyframe;
- frozen DINO features lifted from 2D onto visible Gaussian centres;
- clustering into five semantic parts;
- part-level MLLM material priors;
- a learned ten-class material codebook;
- VideoMAE motion features from 32 frames;
- local spring geometry features;
- decoders predicting stiffness, damping, drag, friction, and collision elasticity.

The predicted parameters instantiate a spring-mass simulator. Training uses simulated-versus-observed tracking and Chamfer losses plus a weak material-prior regularizer.

#### Relevance to this POC

MatPhys provides the main reference for mapping 2D DINO features onto a 3D physical representation:

$$
u_i=\pi(x_i),\qquad f_i^{\mathrm{DINO}}=F(u_i)
$$

Each visible 3D Gaussian centre is projected into the keyframe and assigned the DINO feature at the corresponding image location. The features are then clustered into semantic 3D parts.

This supports a POC design in which DINO features are projected onto MPM particles or surface points and used as implicit material-conditioning features. MatPhys does not directly use DINO features to predict trajectories: DINO defines parts, while material-codebook features, VideoMAE motion, and geometry predict explicit spring parameters.

#### Assumptions and limitations

- DINO semantic parts may not correspond to true material boundaries.
- Static appearance cannot reveal internal filling, density, or stiffness.
- Invisible-region features use symmetry-based propagation, but the method is underspecified.
- The DINO layer, visibility test, interpolation method, symmetry detection, and missing-feature fallback need verification.
- MLLM material priors are uncertain and are not physical measurements.
- Inference is described as monocular, but training uses RGB-D-derived 3D supervision.
- The training set contains only 22 scenarios.
- The output is a spring-mass system, not MPM.
- Predicted parameters are effective simulator values rather than ground-truth material constants.

#### Associated repository

**Local repository:** Unknown.  
**Commit/tag:** Unknown.  
**Relevant entry points:** Unknown.

**Potential reuse:**

- Projection-based DINO 2D-to-3D feature mapping.
- Per-point semantic features and clustering.
- Combining visual features with observed probe motion.
- Learned material embeddings as an alternative to direct parameter regression.

A local implementation note is available at [`docs/reference/dino-2d-to-3d-mapping.md`](docs/reference/dino-2d-to-3d-mapping.md).

**Reference only:**

- MLLM material querying unless required by the POC.
- The learned codebook if the first POC directly conditions dynamics on DINO features.
- Spring-mass topology and parameter decoders.

---

### PhysTwin Dataset

**Dataset:** PhysTwin deformable-object interaction dataset  
**Source:** Introduced with the PhysTwin paper and reused by PhysWorld and MatPhys.  
**Local path:** [`../coderepos/PhysTwin/data/`](../coderepos/PhysTwin/data/)  
**Licence:** Unknown; needs verification.

#### Contents

The dataset contains 22 deformable-object interaction scenarios, including ropes, cloth, stuffed animals, and packages. Videos last approximately 1-10 seconds and contain lifting, stretching, pushing, and squeezing actions. Data were recorded with three synchronized Intel RealSense D455 RGB-D cameras.

Typical scenario structure:

```text
data/different_types/<scenario>/
├── color/{0,1,2}.mp4
├── depth/<camera>/<frame>.npy
├── mask/
│   ├── mask_info_<camera>.json
│   └── processed_masks.pkl
├── cotracker/
│   ├── <camera>.npz
│   └── <camera>.mp4
├── pcd/<frame>.npz
├── metadata.json
├── calibrate.pkl
├── track_process_data.pkl
├── final_data.pkl
├── gt_track_3d.pkl
├── split.json
└── shape/
```

Important formats:

- `cotracker/<camera>.npz`: `(T, N, 2)` 2D tracks and `(T, N)` visibility flags. Preprocessing initially samples up to 5,000 masked pixels per camera.
- `pcd/<frame>.npz`: per-camera, per-pixel `points`, `colors`, and `masks`, derived from RealSense depth and calibration.
- `final_data.pkl`: dense pseudo-3D object trajectories, controller trajectories, visibility and motion-validity masks, and surface/interior samples. Point index is intended to remain consistent across frames.
- `gt_track_3d.pkl`: nine sparse manually annotated 3D trajectories per video, used for evaluation rather than simulator fitting.
- `metadata.json`: camera intrinsics, resolution, frame rate, frame count, and device metadata.
- `calibrate.pkl`: camera transformation matrices.

#### Intended role in this POC

Potential uses include:

- supplying real RGB-D observations;
- extracting DINO features from RGB frames;
- projecting DINO features onto observed 3D surface points;
- providing controller trajectories as action inputs;
- supervising or evaluating MPM surface trajectories;
- using sparse manual tracks for held-out evaluation.

The dataset does not contain ground-truth MPM particles, MPM trajectories, material parameters, or measured interaction forces.

#### Required preprocessing

If using the provided processed files, the POC may begin from `final_data.pkl`, RGB frames, and calibration.

If reproducing from raw RGB-D:

1. Load synchronized RGB and depth frames.
2. Apply object and controller masks.
3. Run CoTracker3 on sampled masked RGB pixels.
4. Use the per-pixel depth point cloud to lift tracks into 3D.
5. Filter tracks using visibility, masks, and neighbourhood-motion consistency.
6. Transform observations into a common coordinate frame.
7. Associate DINO features with selected surface points.
8. Construct or initialize an MPM volumetric representation.

#### Limitations and data-quality concerns

- Dense trajectories are pseudo-labels generated by CoTracker3 and depth lookup.
- Tracks may fail under occlusion, fast motion, deformation, or missing depth.
- Invalid trajectories require visibility and motion-validity masks.
- Only nine sparse manually annotated trajectories are available for evaluation.
- No force/torque, tactile, density, stiffness, or other material ground truth is provided.
- The dataset is small for learning a transferable visual-to-physics model.
- The standard 7:3 split is temporal and does not by itself test unseen-object generalization.
- Material and action diversity are limited.
- Licence and redistribution constraints need verification before reuse.

#### Connection to papers and code

- PhysTwin created the dataset and its preprocessing pipeline.
- PhysWorld uses it to calibrate object-specific MPM digital twins and generate synthetic demonstrations.
- MatPhys uses it to train a shared visual-to-spring-parameter predictor.
- The local dataset is bundled with the PhysTwin repository.

---

## Previous POC: Discrete-Horizon MPM Grid Surrogate

### Purpose

- Learn direct augmented-grid transitions:
  $$
  G_m\rightarrow G_{m+k},\qquad k\in\{25,50,100,200\}.
  $$
- Test whether a normalized U-Net can replace multiple explicit MPM updates
  while retaining particle-position reconstruction.
- Current work focused on $k=25$.
- Direct true-state predictions are accurate, but recurrent predictions suffer
  substantial compounding error.
- Rollout-loss fine-tuning is the proposed next fix but is not implemented.

### Local files

Path base:

```text
/Users/yushanxie/workspace/cgvi/term3/mpm-DINO-poc/
```

Previous POC root:

```text
../mpm-grid-update-poc/discrete_horizon/
```

Key documentation:

- [`../mpm-grid-update-poc/discrete_horizon/DESIGN.md`](../mpm-grid-update-poc/discrete_horizon/DESIGN.md): current authoritative design and training policy.
- [`../mpm-grid-update-poc/discrete_horizon/writeup_2026-06-29.md`](../mpm-grid-update-poc/discrete_horizon/writeup_2026-06-29.md): detailed results, convergence history, compounding diagnosis, and rollout-loss proposal.
- [`../mpm-grid-update-poc/discrete_horizon/V3_IDEA_particle_material_hybrid.md`](../mpm-grid-update-poc/discrete_horizon/V3_IDEA_particle_material_hybrid.md): tentative mixed-material and constitutive-subcycling idea.

Key implementation:

- `generate_dataset.py`: fixed-horizon HDF5 generation.
- `dh_dataset.py`: splits, loading, and resolution-homogeneous batching.
- `dh_normalization.py`: resolution-aware transforms.
- `dh_model.py`: small U-Net.
- `dh_train_utils.py`: losses, metrics, and residual-noise augmentation.
- `train.py`: training, continuation, scheduling, and checkpoint selection.
- `evaluate.py`: held-out and persistence evaluation.
- `visualize_k25_sequences.py`: recurrent $G_0\rightarrow G_{250}$ visualization.
- `diagnose_late_frame_k25.py`: teacher-forced versus recurrent diagnosis.

Supporting code:

```text
../mpm-grid-update-poc/V1/
../standalone-code/mpm/mpm_learning_template.py
```

Dataset:

```text
../mpm-grid-update-poc/discrete_horizon/data/h5/
```

- Contains `k025`, `k050`, `k100`, and `k200`.
- Contains 4,800 pair files per horizon and occupies approximately 8 GB.

Current $k=25$ checkpoint:

```text
../mpm-grid-update-poc/discrete_horizon/runs/
  resume_k025_20260629_175441/best.pt
```

- Epoch 408.
- Validation particle mean: `4.2909e-4`.

Diagnostic outputs:

```text
../mpm-grid-update-poc/discrete_horizon/runs/
  resume_k025_20260629_175441/
    k25_sequence_visualization_epoch408/
    late_frame_diagnostic_epoch408/
```

### Architecture and workflow

```text
MPM particle rollout
  -> project start/end particle states to grids
  -> normalize grid channels
  -> U-Net predicts future grid and displacement
  -> interpolate displacement at particles
  -> grid and particle losses
```

Dataset variants:

```text
resolutions: 32, 48, 64
shapes:      varied blocks and circles
rollouts:    600
steps:       250 per rollout
```

Network interface:

```text
physical input: (B, 22, H, W)
network input:  (B, 24, H, W), including dx/dt conditions
output:         (B, 18, H, W)
```

Material conditioning in this POC:

```text
log10(E) and nu are spatially constant grid channels
one uniform material per rollout
material varies between rollouts
```

Typical commands, run from the previous POC root:

```bash
python train.py --horizon 25 --device mps

python train.py \
  --checkpoint runs/resume_k025_20260629_175441/best.pt \
  --epochs 100 --device mps

python evaluate.py \
  --checkpoint runs/resume_k025_20260629_175441/best.pt \
  --device mps
```

### Relevant implementation details

Resolution-aware transforms:

```text
mass, V0, count: log1p(value / dx^2)
momentum:        signed_log1p(momentum / dx^2)
velocity:        velocity * dt / dx
C:               C * dt
displacement:    displacement / dx
log10(E), nu:    range-normalized
```

Important model behavior:

- Predicts mass, momentum, velocity, $F$, $C$, $V_0$, particle count,
  displacement, and activity.
- Uses particle-position error as the primary checkpoint metric.
- Stores a separate objective-selected checkpoint.
- Supports MPS training and checkpoint continuation.

Important constraints:

- The simulator has one shared material object per particle set.
- HDF5 stores scalar $E,\nu$, not per-particle material parameters.
- Current checkpoints require explicit material channels.
- Grid-projected $F$ and $C$ are lossy summaries of particle state.
- Low direct-pair error does not imply recurrent stability.
- Rollout-loss training data and code remain unfinished.

### Experiments and findings

Most important result:

```text
early teacher-forced particle mean: 7.57e-4
late teacher-forced particle mean:  6.10e-4
final recurrent mean at G250:       2.12e-2
recurrent / late-teacher ratio:     34.8x
```

Conclusion:

- Late true states are generally predicted well.
- Predicted grids drift away from the simulator state distribution when reused.
- Compounding/off-manifold error is the dominant recurrent failure.
- The 64x64 cases show some additional late-stage weakness.

Successful elements:

- Resolution-aware normalization.
- Mixed-resolution and mixed-shape data.
- Particle-focused checkpoint selection.
- Lower-LR scheduling and early stopping.
- Per-variant evaluation.
- Teacher-forced versus recurrent diagnostics.

Insufficient elements:

- Constant high-LR continuation.
- Using aggregate composite loss as the main success metric.
- Residual-noise augmentation alone.
- Assuming direct accuracy guarantees recurrent accuracy.

### Relevance to mpm-DINO-poc

Potentially reusable:

- Fixed-horizon dataset organization.
- Grid targets and particle displacement reconstruction.
- Resolution-aware normalization for non-material channels.
- U-Net, training loop, and MPS resume support.
- Rollout-level splits and resolution batching.
- Evaluation, visualization, and compounding diagnostics.

Reference only:

- Explicit `log10(E)` and `nu` conditioning.
- Uniform-material generator.
- Existing 24-channel input stem and checkpoints.
- Current material normalization constants.

Necessary DINO adaptations:

- Replace or augment the two explicit material channels.
- Define DINO feature dimensionality, normalization, and grid alignment.
- Adapt the U-Net input stem.
- Decide how DINO features are propagated or recomputed recurrently.
- Retain explicit $E,\nu$ as evaluation labels if available.
- Repeat direct, per-variant, and recurrent diagnostics.

Current input contract:

```text
20 non-material physical channels
+ 2 explicit material channels
+ 2 dx/dt channels
= 24 channels
```

Replacing explicit material parameters gives:

```text
20 non-material physical channels
+ C_DINO feature channels
+ 2 dx/dt channels
= 22 + C_DINO channels
```

Existing checkpoints cannot consume arbitrary DINO features without changing
the input stem.

Recommended inspection order:

1. `DESIGN.md`
2. `writeup_2026-06-29.md`
3. `dh_normalization.py`
4. `dh_model.py`
5. `dh_train_utils.py`
6. `train.py`
7. `generate_dataset.py`
8. `diagnose_late_frame_k25.py`

### Decisions already made

These decisions apply only to the previous discrete-horizon experiment. They
should not constrain `mpm-DINO-poc`.

- Separate models were used for each horizon.
- The first implemented model was a normalized U-Net.
- Data used resolutions 32/48/64 and block/circle shapes.
- Particle error selected `best.pt`.
- Explicit $E,\nu$ were provided as grid-wide conditioning.
- Recurrent stability was evaluated separately from direct prediction.

Tentative previous-POC ideas, not decisions for the new project:

- Rollout-loss fine-tuning.
- Particle-held mixed materials.
- Stress/internal-force grid features.
- Coarse constitutive subcycling.

---

## Open Questions

### Sources and permissions

- Does the PhysTwin repository's MIT licence also cover the bundled dataset and derived trajectories?
- Is an official MatPhys implementation available?
- When will the PhysWorld implementation code be available?

### DINO representation

- Which DINO model, layer, token type, and input resolution should be used, and should its weights remain frozen?
- What feature dimension, normalization, and multi-resolution alignment should be used?
- How should image features be projected to surface points and assigned to occluded surfaces or interior MPM particles?
- How should DINO features be propagated or recomputed during recurrent prediction?

### Simulation and learning design

- Should the initial scope remain uniform-material, or include spatially varying materials?
- Which MPM constitutive model and fixed parameters should be used?
- Should rollout-loss fine-tuning be completed before or after introducing DINO features?
- How should the absence of measured force and material ground truth be handled?
- What experiment will distinguish material information in DINO features from correlated shape, texture, or motion cues?

### Real-data evaluation

- How should RGB-D, CoTracker, and camera-calibration noise be handled?
- Are the sparse manual PhysTwin tracks sufficient for evaluating predicted MPM trajectories?
