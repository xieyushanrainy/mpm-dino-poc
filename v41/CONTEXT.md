# V4.1 Dataset Context

> Status: dataset-generation POC, 2026-07-24.
>
> This file is the primary handoff for work using `v41/dataset/`. Read it before
> selecting V4.1 data, constructing splits, training a model, or interpreting
> physical results. The machine-readable authority is
> [`dataset/collection.json`](dataset/collection.json).

## Purpose

V4.1 extends the fixed, zero-velocity, floor-only V4 data toward a
physics-aware and scene-aware particle-dynamics dataset. The current snapshot
is deliberately small and modular. It supports:

- a balanced rigid-versus-soft zero-velocity free-fall comparison;
- rigid free fall with several initial linear velocities;
- one matched rigid floor-only versus wall-collision counterfactual;
- reuse of the same point-aligned DINOv2 features without new VLM inference;
- purpose-specific manifests so experiments do not need to filter opaque
  directories.

This snapshot is a POC, not a complete obstacle-generalization dataset. Fluid is
quarantined and excluded.

## Canonical paths

```text
v41/
├── CONTEXT.md
└── dataset/
    ├── README.md
    ├── collection.json
    ├── DATASET_COMPLETE
    ├── objects/<uid>/
    │   ├── static.npz
    │   └── source_metadata.json
    ├── episodes/<episode_id>/
    │   ├── trajectory.npz
    │   └── metadata.json
    ├── subsets/*.jsonl
    ├── source_glbs/
    │   ├── rigid/<uid>.glb
    │   └── soft_body/<uid>.glb
    └── visualizations/*.gif
```

Use paths relative to `v41/dataset/`. Do not hard-code the absolute source paths
recorded for provenance in `collection.json`.

## Current composition

There are 60 unique objects and 106 episodes:

| Family | Unique UIDs | Zero-velocity floor-only | Nonzero-velocity floor-only | Wall collision |
|---|---:|---:|---:|---:|
| Rigid | 30 | 30 | 45 | 1 |
| Soft body | 30 | 30 | 0 | 0 |
| Total | 60 | 60 | 45 | 1 |

The wall episode also has nonzero initial velocity, so rigid episodes comprise
30 zero-velocity controls and 46 nonzero-velocity episodes in total.

The 30 rigid and 30 soft UIDs are disjoint. The balanced zero-velocity subset is
therefore suitable for object-level rigid-versus-soft comparison, subject to
the soft-data limitations below.

## Purpose-specific manifests

Every JSONL row is a self-contained episode index containing the UID, family,
initial velocity, scene layout, collision flag, trajectory path, shared static
path, tags, and trajectory hash.

| Manifest | Episodes | Intended use |
|---|---:|---|
| `soft_free_fall_zero_velocity.jsonl` | 30 | Calibrated soft free fall |
| `rigid_free_fall_zero_velocity.jsonl` | 30 | Rigid free-fall control |
| `free_fall_zero_velocity_all_families.jsonl` | 60 | Balanced 30/30 comparison |
| `rigid_free_fall_initial_velocity.jsonl` | 45 | Rigid initial-velocity modelling |
| `rigid_free_fall_all.jsonl` | 75 | All floor-only rigid episodes |
| `rigid_collision_wall.jsonl` | 1 | Wall-collision POC only |
| `rigid_all.jsonl` | 76 | Every rigid episode |
| `all_episodes.jsonl` | 106 | Complete collection |

The files live under [`dataset/subsets/`](dataset/subsets/). Prefer these
manifests over rebuilding filters from filenames.

## Uniform tensor contract

Each episode trajectory contains:

```text
trajectory_positions_m   float32 [61, 2048, 3]
point_velocities_m_s     float32 [61, 2048, 3]
point_active             bool    [61, 2048]
times_s                  float32 [61]
point_ids                int32   [2048]
```

Each shared object-static file contains:

```text
reference_positions_m    float32 [2048, 3]
dino_features            float16 [2048, 384]
dino_valid               bool    [2048]
point_material_ids       int32   [2048]
```

The time horizon is 61 saved frames over 2 seconds at 30 Hz. Coordinates are
world-space metres in right-handed Blender XYZ with `+Z` up and the floor at
`z = 0`.

Point row identity is persistent. For a given UID, row `i` in
`reference_positions_m`, DINO arrays, and every trajectory frame refers to the
same sampled surface point. `point_ids` has been validated as `0..2047`.

Use `point_active[t, i]` in all losses and metrics even though the current rigid
and soft samples are generally fully active. Handle `dino_valid[i]` explicitly;
an invalid row is not observed visual evidence.

## Episode semantics and tags

Do not infer conditions only from the episode ID. Read the manifest fields:

```text
family:                       rigid | soft_body
initial_linear_velocity_m_s:  [vx, vy, vz]
initial_velocity_regime:      zero | nonzero
scene_layout_id:              floor_only | wall_x_0p65
has_obstacle_collision:       bool
tags:                         searchable semantic tags
```

All episodes use gravity `[0, 0, -9.81] m/s²`. Floor-only episodes still include
floor contact; “free fall” describes the release phase and does not mean a
contact-free two-second trajectory.

Rigid velocity variants use:

```text
[0.0,  0.0, 0.0] m/s
[0.5,  0.0, 0.0] m/s
[-0.5, 0.0, 0.0] m/s
[0.0,  0.5, 0.0] m/s
```

The single wall scene uses a static box centred at approximately
`[0.65, 0.0, 0.5] m` and a rigid object launched at `+0.5 m/s` along X. It has a
matched floor-only episode for the same UID and initial velocity.

## Provenance

No VLM or DINO inference was rerun for V4.1.

### Soft body

The 30 UIDs, reference points, DINOv2-small features, masks, material IDs, and
metadata come from the existing V4 `packaged_soft_30` lineage. Their trajectories
were regenerated in `DINO_VLM_sim_dataset_gen` after calibrating the Blender Soft
Body free-fall response.

### Rigid

- Fifteen UIDs have four newly generated V4.1 velocity variants.
- The other fifteen zero-velocity controls are copied from
  `v4/dataset/packaged_balanced_90`.
- Together they reproduce all 30 rigid UIDs in the V4 balanced package for the
  zero-velocity comparison.

### Collision

The wall subset contains one matched POC episode generated with Blender Bullet.
It is included to test data and scene-conditioning interfaces, not to support a
generalization claim.

### Source meshes

All 60 exact source GLBs are retained under `dataset/source_glbs/`, separated by
family. `collection.json` records each path and SHA-256 hash. These files retain
their asset-specific Objaverse/Sketchfab attribution and redistribution terms;
do not redistribute the archive without auditing those terms.

## Soft-body calibration

The original V4 soft trajectories fell far too slowly. The principal cause was
an incorrect mapping of contact friction into Blender Soft Body’s medium
friction, creating artificial drag and near-terminal velocity immediately after
release. Material damping was also multiplied too strongly.

The regenerated route uses:

```text
soft_body_gravity_scale          0.275
soft_body_medium_friction_scale  0.0
soft_body_damping_scale          1.0
soft_body_mass_scale             0.001
```

The gravity scale compensates for Blender Soft Body’s solver response; it is not
a claim that physical gravity changed.

Across all 30 regenerated objects:

- expected COM drop at 0.033/0.067/0.100 seconds:
  `0.00545 / 0.02180 / 0.04905 m`;
- observed mean drop:
  `0.00549 / 0.02176 / 0.04881 m`;
- maximum pre-contact drop error:
  approximately `0.00044 m`;
- maximum inferred initial-velocity error:
  approximately `0.0065 m/s`.

Thus early free-fall timing is calibrated well enough for the POC.

## Known limitations

### 1. Soft contact geometry is not fixed

The soft solver operates on a regular bounding-ellipsoid proxy. Original surface
points receive barycentrically interpolated proxy displacement. Collision is
therefore enforced on the ellipsoid, not directly on exported surface points.

Consequences:

- 27/30 regenerated soft objects contain at least one exported point below the
  floor;
- the worst observed minimum is approximately `z = -0.091 m`;
- soft collision/deformation trajectories must not yet be treated as clean
  contact ground truth.

The calibration fixed free-fall timing, not the proxy/contact representation.
For an initial experiment, evaluate pre-contact soft motion separately and
report penetration. Do not train a scene-contact model on these soft contacts
without addressing or explicitly quarantining them.

### 2. Collision diversity is insufficient

There is only one wall episode, one wall layout, one object UID, and one
collision timing. It can verify that a loader/model accepts scene geometry and
responds differently to a changed scene. It cannot measure unseen-layout,
unseen-contact-location, or object-scene generalization.

### 3. Initial velocity is rigid-only

Blender Soft Body did not reliably preserve a requested transform-based initial
velocity. Nonzero soft episodes remain blocked rather than being included with
incorrect labels.

### 4. Fluid is excluded

The prior Mantaflow tracer route had a severe time/velocity-scale issue and
different particle semantics. Fluid remains quarantined until its data
generation is repaired and validated.

### 5. Simulator-effective physics

Rigid uses Blender Bullet and soft uses Blender Soft Body. Parameters and
calibration values are simulator-effective controls, not measured real-world
material constants. Family/solver labels may be used for balancing, splits,
diagnostics, and per-family reporting, but must not be passed into the learned
primary network.

## Recommended experimental use

### Safe first comparisons

1. **Balanced zero-velocity free fall**
   - manifest: `free_fall_zero_velocity_all_families.jsonl`;
   - split by UID before creating temporal windows;
   - report rigid and soft separately;
   - distinguish pre-contact from post-contact metrics.

2. **Rigid initial-velocity response**
   - train/evaluate matched variants from
     `rigid_free_fall_initial_velocity.jsonl`;
   - include the explicit velocity vector or derive it from observed frames;
   - preserve object-level splits so velocity variants of one UID never cross
     train/validation/test boundaries.

3. **Scene-interface smoke test**
   - compare the one wall episode with its matched floor-only counterpart;
   - use it to validate SDF/normal/local-scene query plumbing;
   - do not report obstacle-generalization performance.

### Split rules

- Split by object UID, never by frame, point, or episode.
- Keep all velocity variants and scene variants for one UID in the same split.
- Freeze deterministic manifests before tuning.
- Report counts and metrics per family and per episode regime.
- Keep fluid absent rather than treating its absence as a negative class.

### DINO controls

The central visual question remains whether point-aligned DINO helps long-horizon
prediction without harming short-horizon dynamics. Use matched:

- real point-aligned DINO;
- zero-DINO;
- scene-shuffled DINO across UIDs;
- point-shuffled DINO within an object;
- geometry-only conditioning.

Evidence for DINO requires improvement over geometry and shuffled controls by
more than matched-seed variation. Do not use family, material class, solver
route, VLM parameters, or explicit material labels as primary-network inputs.

## Minimal loader pattern

```python
import json
from pathlib import Path
import numpy as np

root = Path("v41/dataset")
manifest = root / "subsets/free_fall_zero_velocity_all_families.jsonl"
rows = [json.loads(line) for line in manifest.read_text().splitlines() if line]

row = rows[0]
with np.load(root / row["object_static"], allow_pickle=False) as static:
    reference = static["reference_positions_m"]
    dino = static["dino_features"]
    dino_valid = static["dino_valid"]

with np.load(root / row["trajectory"], allow_pickle=False) as episode:
    positions = episode["trajectory_positions_m"]
    velocities = episode["point_velocities_m_s"]
    active = episode["point_active"]
    times = episode["times_s"]
```

The manifest row—not a family directory—is the experiment unit. This permits
overlapping subsets without duplicating trajectory arrays.

## Validation status

The collection validator checked:

- 60 unique object-static records;
- 106 unique episode records;
- all static and trajectory SHA-256 hashes;
- every tensor name and shape;
- row-aligned `point_ids`;
- all eight subset counts and episode references;
- 30 rigid and 30 soft reusable GLBs and their hashes.

`dataset/DATASET_COMPLETE` marks a completed build. Visual examples are under
[`dataset/visualizations/`](dataset/visualizations/).

