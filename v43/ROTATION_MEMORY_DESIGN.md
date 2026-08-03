# V4.3 isolated compact rotation-memory experiment

Date: 2026-08-03

## Frozen question and boundary

Test whether a compact memory selected by aligned DINO improves proper-Kabsch
rotation prediction for a single model trained on rigid and soft objects.  The
physical trunk, COM head, rotation head, region adapter and canonical
deformation path from `v42/checkpoints/v42_adapter_full_seed42_best.pt` are
read-only.  Identity, not the unpromoted V4.2 rotation head, is the prediction
base.  Fluid is excluded.  Family is used only by the sampler and reports.

The target is the proper row-vector Kabsch rotation from frame 1 to each of the
59 future frames, masked when the second/first singular-value ratio is below
`1e-3`.  The learned output is

```text
delta_theta[t] = sigmoid(g[t]) * theta_max * tanh(raw[t])
R_hat[t] = Exp(delta_theta[t])
```

with `theta_max = 20 degrees`.  This is a bounded per-frame residual and cannot
accumulate late-horizon angular drift.  The primary loss is mean SO(3)
geodesic error over valid frames, family-balanced by construction.  A small
second-difference smoothness penalty is declared but reported separately.

## Bank and retrieval contract

The immutable bank contains one entry per training UID from both rigid and
soft families (Panel Z is the canonical source episode).  Each entry stores
UID, family, split, panel, 59 rotation vectors, Kabsch validity, normalized
reference geometry, point and pooled DINO with validity, event phase, geometry
scale, representation and target provenance.  Its serialized content hash is
part of every run configuration.  Training queries exclude their UID;
validation queries use training sources only.  Test retrieval is rejected.

Unified cross-family retrieval is the only first-matrix policy.  Selected
source UIDs and families are recorded.  Family-matched retrieval is absent and
would be labelled an oracle-routing follow-up if separately authorized.

The reader forms compact source tokens from pooled/local DINO, aligned
geometry, phase, source rotation-vector prototypes and validity.  Detached
physical/global query features, geometry, query DINO/validity, contact-relative
features and oracle event time query the tokens.  Source rotations are latent
examples; no selected rotation is copied into the output.

## Frozen validation matrix

Seeds are 42, 123 and 456 with identical family-balanced UID sampling,
objectives and checkpoints.

| ID | Condition | Trained |
|---|---|---:|
| A | Identity | no |
| B | Frozen V4.2 rotation head | no |
| C | Zero-memory reader, exact identity | yes/control |
| D | Geometry retrieval, all DINO channels zero | yes |
| E | Aligned-DINO top-3 compact memory | yes |
| F | Deterministic scene-shuffled/wrong-object memory | yes |
| G | Point-shuffled DINO correspondence | yes if local DINO retained |
| H | Raw top-k copied rotation | report only |
| I | Oracle-best training memory | noncausal report only |

For E, fixed weights are evaluated with zero query DINO, zero memory DINO,
zero memory, wrong memory and shuffled correspondence.  Retrieval source
families and same/cross-family rates are saved for every applicable arm.

Validation reports degrees for family-balanced mean/median, rigid and soft,
Panels Z/V, H1/H8/H16/H30/H40/H59, pre/contact/post phases, every UID and seed,
identity improvement, H59 drift, inactive/static false rotation, gate activity
and fixed-weight changes.  Test remains sealed until this matrix is frozen.

## Promotion rule

The real-DINO arm is useful only if all seven rules in
`v43/ROTATION_HANDOVER_PROMPT.md` pass, including every-seed improvement for
both families and bit identity of protected COM and deformation parameters and
outputs.  No full matrix is launched by the implementation handoff.
