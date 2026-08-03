# Copy-ready V4.3 rotation-memory handover prompt

```text
Work in:
/Users/yushanxie/workspace/cgvi/term3/mpm-DINO-poc

We are starting a new, isolated rotation-retrieval experiment after closing the
V4.3 deformation-memory investigation. Read these files before changing code:

1. v43/V43_DEFORMATION_CLOSEOUT.md
2. v42/V42_EXPERIMENT_SUMMARY.md
3. v42/ROTATION_EXPERIMENT_RESULTS.md
4. v42/run/retrieval_transfer_baseline/RESULTS.md
5. v42/run/dino_learnability_audit/RESULTS.md
6. v43/FIELD_ATTENTION_DESIGN.md
7. v43/REJECTION_DESIGN.md

Objective
=========

Test whether DINO-selected compact mechanical memory can improve rotation
prediction when trained on both rigid and soft objects. This is a separate
controlled rotation branch. Do not reopen deformation architecture selection.

Why rigid belongs here
======================

Rigid and soft objects both have a meaningful object-level SO(3) rotation
target derived from Kabsch alignment. Rigid samples provide valuable rotation
supervision rather than a zero-deformation negative. Use both families in
training and validation, with family-balanced sampling and separate family,
panel and UID reporting. Do not include fluid in the first rotation experiment:
coherent object rotation and Kabsch validity are less reliable for fluid flow.

Do not add a rigid/soft classifier or use family as a model input. Family labels
may be used only for balanced sampling and stratified reporting. Build one
training-only bank containing rigid and soft UIDs, and allow retrieval across
the unified bank initially. Record retrieved-source families so cross-family
behavior can be audited. A later family-matched retrieval ablation is permitted
only if unified retrieval fails and must be labelled oracle routing.

Evidence and baselines
======================

- Identity rotation is strong: prior validation error was 3.244 degrees for
  soft and 1.572 degrees for rigid.
- Raw aligned-DINO top-1 copied rotation was much worse: 10.094 degrees for
  soft and 3.853 degrees for rigid.
- Training-only calibration selected zero rotation, equal to identity.
- DINO neighbours were better than geometry neighbours but remained worse than
  identity.
- No V4.2 learned rotation experiment passed its promotion screen.
- V4.3 deformation showed that compact latent memory is stronger than explicit
  field transfer. Retain compact memory; do not copy retrieved rotations.
- V4.3 wrong-scene training detected obvious shuffled negatives but did not
  identify harmful plausible neighbours. Do not assume that another shuffled
  negative alone solves rotation compatibility.

Protected architecture
======================

Freeze the DINO-free physical trunk, COM head and complete deformation path.
The first rotation experiment must not alter COM or canonical deformation.
Verify protected parameter and output identity before and after training.

Use identity as the primary rotation base because the protected V4.2 rotation
head is not a passed model. Predict a bounded, gated residual in the Lie algebra:

    delta_theta[t] = gate[t] * bounded_residual[t]
    R_hat[t] = Exp(delta_theta[t])

Also report the frozen V4.2 rotation head as a comparator. Do not force a
retrieved source rotation into the output. If accumulated angular dynamics are
tested, isolate them as a later arm; begin with bounded per-frame residuals to
avoid late-horizon drift.

Retrieval bank discipline
=========================

- Bank uses training UIDs only, rigid and soft.
- Training queries exclude the same UID.
- Validation/test rotations never choose neighbours, fit amplitude, calibrate
  gates, create prototypes or set hyperparameters.
- Test remains sealed until the validation design is frozen.
- Freeze and hash the bank/index for every matched arm.
- Store source UID, family, split, panel, time/event phase, valid Kabsch frames,
  DINO/point validity, geometry normalization, rotation representation and
  target provenance.
- Use the same retrieved source UIDs across architecture-matched arms.
- Record cross-family versus same-family retrieval rates.

Compact rotation-memory interface
=================================

Query tokens may contain detached physical/global features, normalized geometry,
query DINO and validity, collider/contact-relative features and oracle event time
for the initial sufficiency experiment.

Retrieved compact tokens may contain pooled/local source DINO, aligned geometry,
source event phase, source rotation-vector prototype or encoded trajectory,
Kabsch validity and provenance. Use attention and a bounded gate. Treat source
rotation as a latent mechanical example, not a copied prediction.

First controlled validation matrix
==================================

Use seeds 42, 123 and 456, identical balanced sampling, checkpoints, objectives
and parameter counts where practical:

A. identity rotation baseline;
B. frozen protected V4.2 rotation head;
C. zero-memory compact-reader control, exactly identity anchored;
D. geometry-retrieved compact rotation memory with DINO channels zero;
E. real aligned-DINO top-3 compact rotation memory;
F. deterministic scene-shuffled/wrong-object memory;
G. point-shuffled DINO correspondence control if pointwise DINO is retained;
H. raw top-k copied-rotation baseline, report only and never train from it;
I. oracle-best training-memory ceiling, noncausal report only.

For the trained real-DINO checkpoint, perform fixed-weight zero-query-DINO,
zero-memory-DINO, zero-memory, wrong-memory and correspondence ablations.
Separately trained arms alone are not sufficient attribution.

Targets, losses and metrics
===========================

Derive proper Kabsch rotations with the existing validity/degeneracy checks.
Use geodesic SO(3) loss as the primary objective, with a bounded residual and
optional smoothness penalty. Do not optimize world-position loss in the first
rotation experiment.

Report degrees, not only radians, for:

- family-balanced mean and median geodesic error;
- Panel Z and Panel V separately;
- H1, H8, H16, H30, H40 and H59;
- pre-contact, contact and post-contact phases;
- per-UID rows and seed variation;
- improvement relative to identity;
- late-horizon drift and false rotation on inactive/static intervals;
- gate activation and retrieved-source family;
- fixed-weight ablation changes.

Decision rules
==============

Do not claim useful DINO rotation memory unless:

1. real DINO memory beats identity in every seed for both rigid and soft;
2. it beats geometry, wrong-scene and point-shuffled controls beyond cross-seed
   variation;
3. fixed-weight DINO/memory ablation degrades the same checkpoint;
4. gains are UID-balanced and do not come only from rigid Panel V;
5. H59 and inactive/static rotation do not regress;
6. the gate suppresses harmful plausible neighbours, not only obvious shuffled
   negatives;
7. protected COM and deformation remain bit-identical.

Implementation requirements
===========================

- Create new rotation-memory modules, runner, tests, design and frozen matrix;
  do not overwrite V4.2 or V4.3 deformation evidence.
- Reuse existing Kabsch, SO(3), split, checkpoint and integrity helpers.
- Add tests for train-only bank construction, same-UID exclusion, deterministic
  retrieval, proper rotations, invalid Kabsch masking, identity/zero-memory
  equivalence, bounded residuals, shuffle controls, fixed-weight ablation,
  family-balanced sampling and protected COM/deformation identity.
- Run unit tests and a one-batch CPU optimizer smoke.
- Provide a copy-ready tcsh nohup lab command using .venv-v41/bin/python,
  console.log and launcher.pid.
- Save configs, bank/index hashes, histories, best/last checkpoints, UID/family/
  panel validation rows, ablations and RUN_COMPLETE markers.
- Do not use test data or launch the full matrix automatically.

First action
============

Audit the existing V4.2 rotation target/checkpoint contracts and write the
concrete compact rotation-memory design and frozen matrix. Then implement the
train-only rigid+soft bank, identity-anchored gated residual reader and matched
controls with tests. Explain any deviation before expanding scope.
```
