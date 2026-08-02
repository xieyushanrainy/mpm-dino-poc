# V4.2 experiment closeout

## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: validate/closeout
- Origin Date: 2026-08-02
- Verification Status: ANALYZED
- Evidence: downloaded V4.2 run artifacts and validation reports
- Test data used for V4.2 champion selection: no

## 1. Question and final architecture

V4.2 asked whether global motion, rotation, and non-rigid deformation could be
separated cleanly enough to learn deformation without allowing local or visual
losses to rewrite the physical trajectory model.

The retained decomposition is

\[
\hat x_{t,i}=\hat c_t+\hat R_t(q_i+\hat d_{t,i}),
\qquad q_i=x_{1,i}-c_1.
\]

The model has one shared DINO-free graph-temporal physical trunk and three
explicit heads:

1. **COM head** predicts global translation around a ballistic reference.
2. **Rotation head** predicts explicit accumulated rotation.
3. **Canonical deformation head** predicts zero-mean non-rigid displacement in
   the object frame after translation and Kabsch rotation are removed.

The retained deformation architecture uses four geometry-aware region tokens,
a point-query cross-attention adapter, and a canonical point decoder. V4.2
experiments froze the physical trunk, COM head, and rotation branch while
training the local path. This protected decomposition is the starting point for
V4.3.

## 2. What worked

### Protected three-way decomposition

The isolation contract worked technically. Local experiments kept protected
parameters bit-identical and left COM/rotation outputs unchanged. This makes it
possible to attribute later local improvements to the deformation path rather
than shared-trunk drift.

### Contact is the strongest deformation cue found in V4.2

Under the soft-event, target-amplitude-normalized objective, zero prediction has
loss approximately one. The controlled pointwise contact experiment produced:

| Condition | Validation objective | Predicted/target peak | Magnitude correlation | Timing |
|---|---:|---:|---:|---|
| Zero control | 0.9933 | 2.1% | 0.305 | onset absent; peak ~51 frames off |
| Curvature only | 0.9899 | 4.2% | 0.286 | still collapsed |
| Oracle contact | 0.8727 | 33.1% | 0.565 | onset 0; median peak 1--2 frames |
| Contact + curvature | 0.8625 | 28.4% | 0.650 | peak 0--3 frames; some onset failures |

Oracle contact therefore partially rescues even the older adapter. This is an
information-sufficiency result: contact was computed from future ground-truth
positions and is not yet deployable.

### Contact and event timing work together

The strongest V4.2 arm supplies pointwise oracle contact, fixed curvature
proxies, and oracle stage/event-relative time before the region adapter:

| Architecture/condition | Validation objective | Peak ratio | Strain RMSE |
|---|---:|---:|---:|
| Direct decoder, zero condition | 0.9935 | 2.7% | 0.03893 |
| Existing adapter, full oracle condition | **0.7781** | **46.6%** | 0.07819 |
| Direct point decoder, full oracle condition | 0.7921 | 42.9% | **0.04286** |

The adapter with the full condition is the best V4.2 reconstruction model. It
selects epochs 13 and 8 for seeds 42 and 456 rather than immediately collapsing.
Both seeds detect onset with median error zero and peak timing within two frames,
apart from one seed-dependent validation-object outlier.

The direct decoder proves that a simple pointwise MLP can use the oracle
information, but it does not beat the adapter on the selected reconstruction
objective. It does produce a much more strain-consistent field. The adapter is
therefore retained, with spatial regularization kept as follow-up work.

### Event-normalized soft-event supervision is a useful diagnostic contract

Restricting the optimized reconstruction metric to contact, compression, and
peak frames and dividing by each episode's target deformation energy makes the
static solution interpretable: zero is approximately one. The objective did not
solve deformation alone, but it made the contact/full-condition effects visible
and comparable across objects with different deformation amplitudes.

### DINO contains a sparse retrieval signal

Across all train-validation pairs, DINO distance is not monotonically related to
deformation similarity: global DINO Spearman is `0.032`, aligned point-DINO is
`0.128`, and global geometry is `0.142`. Nevertheless, aligned point-DINO
nearest-neighbour retrieval is better than global geometry for two of five
validation soft objects:

| Retrieval representation | Mean oracle-rescaled deformation error |
|---|---:|
| Global geometry neighbour | 0.950 |
| Global DINO neighbour | 0.821 |
| Aligned point-DINO neighbour | 0.769 |
| Oracle-best training field | 0.697 |

For one validation object, aligned DINO selects the oracle-best training field;
for another it reduces error from `0.997` to `0.176`. This supports treating
DINO as a sparse selector or retrieval key, not as a universal smooth material
coordinate.

For rotation, nearest DINO neighbours are better than nearest global-geometry
neighbours (`10.09°` versus `15.93°` for soft; `3.85°` versus `10.00°` for
rigid), but still worse than the identity baseline (`3.24°` soft, `1.57°`
rigid). DINO/geometry attention for rotation remains a V4.3 hypothesis, not a
V4.2 positive result.

## 3. What did not work

### Rotation remains unresolved

No Gate-1B--1F rotation experiment passed its promotion screen. Gate 1E is the
least damaging operational placeholder, not a solved rotation model. Contact-
driven Gate 1F produced a large seed-42 Panel-V improvement that did not
replicate and worsened rigid Panel Z. Identity remains a strong comparator.

### Geometry-only deformation collapses

Gate 2, family/rigid rebalancing (Gate 2B), and total-mass stage balancing
(Gate 2C) all fail the frozen screen. Predictions remain near zero, with absent
onset and roughly 51--52-frame peak errors. Fixed loss weights, rigid-object
imbalance, soft amplification up to 20x, and rare-stage mass balancing do not
solve the spatial field.

### Loss changes do not remove the single-frame spatial ceiling

Composite Huber, canonical-only Huber, and normalized MSE single-frame fits all
recover broad magnitude but leave about 29--33% residual RMS. Normalized MSE is
worse. Loss formulation is not the primary representational bottleneck.

### Temporal or material conditioning alone does not work

Late oracle temporal/material conditioning, event-normalized temporal-only
conditioning, and upstream temporal-only injection all remain near loss
`0.993`, recover only a few percent of target peak, and select epochs 1--3.
Temporal information becomes useful only when paired with localized contact in
the later full-condition experiment. Simulator material metadata alone has no
measurable effect under the tested interface.

### Generic stage templates and global geometry are insufficient

The per-stage affine field has approximately correct amplitude but validation
loss `2.435`, worse than zero. Geometry-neighbour transfer scores `0.950`, and
material-vector distance has Spearman `-0.069` with deformation similarity.
Global surface shape, stage identity, and the available material vector do not
identify object-specific deformation modes.

### Curvature is KIV, not established

The four inexpensive proxies—linearity, planarity, scattering, and normal
variation—do nothing without contact. Adding them to contact gives only a small
objective change and worsens strain. Because the later champion bundles
curvature with contact and event timing, its independent contribution remains
unproven. Do not claim curvature benefit without a new matched ablation.

### Naive direct DINO and naive field transfer are not supported

Earlier V4/V4.1 point concatenation, neighbourhood attention, staged DINO
adapters, global pooling, and split-region experiments did not demonstrate
robust inference-time DINO attribution. Some mean improvements were smaller
than seed variation or arose while the trained model was nearly insensitive to
DINO ablation.

The leakage-safe retrieval-transfer baseline is also negative. Raw DINO
neighbour fields have deformation loss `5.10--11.68`; training-only calibration
sets their amplitude approximately to zero. Raw retrieved rotations are worse
than identity, and calibration selects zero rotation. Therefore V4.3 must not
copy a vanilla nearest-neighbour trajectory directly.

## 4. V4.2 champion and limitations

The preserved research champion is:

```text
v42/checkpoints/v42_adapter_full_seed42_best.pt
SHA-256: 85dd4bb3024268ec1b2dedbf1a6ed09d1175628a8f1cdb995060fcf5430c4dfb
seed: 42
best epoch: 13
validation event-normalized MSE: 0.777190911769867
```

It retains the physics trunk plus COM/rotation/canonical-deformation heads and
uses the region adapter. It is a research checkpoint, not a deployable model:
its contact and event time come from future targets, its peak amplitude remains
only about half of ground truth, its strain error is high, and rotation is still
an operational placeholder.

## 5. Kept in view for V4.3

1. Retrieval-conditioned deformation rather than direct DINO regression.
2. Cross-attention from query points to retrieved DINO/geometry/contact-relative
   deformation memories.
3. Residual or gated use of retrieved fields so unreliable neighbours can be
   ignored instead of forcing transfer.
4. Contact-aware retrieval: contact-patch and non-contact points must not be
   treated as interchangeable.
5. Causal contact features derived from predicted motion and collider signed
   distance, closest point, normal, and relative velocity.
6. DINO/geometry attention for rotation as a separate controlled branch, always
   compared with identity and the protected V4.2 rotation head.
7. A normalized strain/edge regularizer to retain the champion's amplitude
   while improving local coherence.
8. Curvature only as an ablation, not a required feature.

## 6. Final V4.2 conclusion

V4.2 establishes that the protected physics-trunk/three-head decomposition is a
sound experimental platform and that localized contact plus event timing can
unlock substantial nonzero deformation in the region adapter. It does not solve
deformation, causal contact, rotation, or DINO attribution. The DINO evidence is
most consistent with a sparse retrieval/mixture role. V4.3 should preserve the
decomposition and test whether retrieved visual-mechanical memories improve the
conditioned deformation and, separately, rotation branches under strict
retrieval and shuffle controls.
