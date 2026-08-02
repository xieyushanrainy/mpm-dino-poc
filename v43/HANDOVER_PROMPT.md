# Copy-ready V4.3 handover prompt

```text
Work in:
/Users/yushanxie/workspace/cgvi/term3/mpm-DINO-poc

We are starting V4.3 after closing the V4.2 deformation investigation. Read
these files before changing code:

1. v42/V42_EXPERIMENT_SUMMARY.md
2. v42/checkpoints/MANIFEST.json
3. v42/run/direct_decoder_probe_seed42_456/RESULTS.md
4. v42/run/contact_curvature_seed42_456/RESULTS.md
5. v42/run/dino_learnability_audit/RESULTS.md
6. v42/run/retrieval_transfer_baseline/RESULTS.md
7. v42/ROTATION_EXPERIMENT_RESULTS.md

V4.3 objective
==============

Test whether DINO is useful as a retrieval key and attended mechanical memory,
rather than as a directly concatenated material feature or a copied nearest-
neighbour trajectory.

Preserve the V4.2 high-level architecture:

- one DINO-free graph-temporal physical trunk;
- explicit COM head;
- explicit rotation head;
- canonical zero-mean deformation head;
- world reconstruction x_hat[t,i] = c_hat[t] + R_hat[t](q_i+d_hat[t,i]);
- protected-path attribution checks so local/DINO experiments cannot silently
  alter COM, rotation, or the physical trunk unless a later experiment
  explicitly authorizes coupling.

The V4.2 research champion is:

v42/checkpoints/v42_adapter_full_seed42_best.pt
SHA-256: 85dd4bb3024268ec1b2dedbf1a6ed09d1175628a8f1cdb995060fcf5430c4dfb

It uses the existing geometry-region-token adapter with pointwise oracle
contact, curvature proxies, and oracle stage/event time. Its validation
event-normalized MSE is 0.7771909118 at seed 42 epoch 13. Treat it as an
oracle-conditioned reference/initialization, not a deployable checkpoint.

Evidence that constrains V4.3
=============================

- Oracle contact is the strongest V4.2 cue: objective ~0.873 versus ~0.993
  zero, peak amplitude ~33%, onset median 0, peak median 1--2 frames.
- Contact + event timing before the adapter is stronger: adapter_full averages
  ~0.778 and ~47% target peak across seeds.
- The direct point decoder learns with the full oracle condition but does not
  beat the adapter on the primary objective; keep the adapter architecture.
- Curvature alone is null and contact+curvature worsens strain. Curvature is
  KIV/ablation only.
- Temporal or material conditioning alone, late or upstream, is null.
- Geometry-only, loss reweighting, stage balancing, and loss changes collapse
  near zero.
- Global geometry similarity weakly predicts deformation similarity.
- Aligned point-DINO nearest-neighbour retrieval is promising for only 2/5
  validation soft objects: mean oracle-rescaled error 0.769 versus 0.950 for
  geometry retrieval and 0.697 oracle-best.
- This is sparse/local evidence, not a global DINO-material relationship:
  aligned-DINO pairwise Spearman is only 0.128.
- Vanilla field transfer fails leakage-safe calibration: raw top-k deformation
  loss is 5.10--11.68 and training calibration suppresses it to zero. Do not
  copy retrieved trajectories directly.
- DINO rotation neighbours beat geometry neighbours but remain worse than the
  identity baseline. Rotation attention is exploratory and must be isolated.

Primary V4.3 hypothesis
=======================

A query object can use its point-aligned DINO and geometry to retrieve a small
set of training-object mechanical memories. Query points then cross-attend to
retrieved source points/field tokens and predict a gated residual deformation.
The network must be able to ignore bad neighbours. Useful signal may be
category/prototype-like rather than smoothly regressed from DINO distance.

Required retrieval-bank discipline
==================================

- Build the bank from training UIDs only.
- For training queries, exclude the same UID (leave-one-UID-out retrieval).
- Never use validation/test target fields to retrieve, calibrate, fit amplitudes,
  choose neighbours, or create prototypes.
- Freeze and hash the bank/index used by every matched arm.
- Store source UID, split, stage/time, point-validity, DINO validity, geometry
  normalization, and field provenance for every memory.
- Keep test data sealed until a validation-only design is frozen.

Proposed memory and attention interface
=======================================

For each query point and time, query tokens should include:

- frozen physical point feature;
- normalized reference coordinate/local graph feature;
- query point DINO plus validity;
- contact-relative features: signed distance/gap, closest collider point,
  collider normal, relative normal/tangent velocity, contact probability or
  oracle contact in the first sufficiency experiment;
- stage/event-time condition (oracle initially for isolation).

Retrieved memory tokens may include:

- source point DINO and validity;
- aligned normalized source coordinate/local geometry;
- source contact-relative coordinate and contact-patch membership;
- source canonical deformation field or a learned deformation prototype by
  stage/time;
- optional source rotation representation for a later separate experiment.

Use cross-attention or a small mixture/prototype module. Predict a bounded gate
and residual correction; do not force the retrieved field into the output. Keep
the existing adapter/local head as the base path.

First validation-only experiment matrix
=======================================

Implement deformation retrieval before rotation retrieval. Use identical
physics/COM/rotation checkpoints, oracle contact/time, event-normalized soft
event objective, sampling, parameter counts where practical, and seeds 42/456.

A. no-retrieval memory / zero-memory control;
B. geometry-retrieved memory without DINO;
C. real aligned-DINO top-k retrieved memory;
D. scene-shuffled DINO retrieval or wrong-object memory;
E. point-shuffled query DINO/memory correspondence control;
F. optional oracle-best training-memory ceiling, reported only as a noncausal
   upper bound and never used for training or selection.

Also retain the raw top-k transfer baseline and zero-deformation baseline in
reports. Start with top-k fixed to a small predeclared value (for example 3);
do not search many k values on validation without accounting for selection.

Primary questions and decision rules
====================================

1. Does real retrieved memory beat zero memory in both seeds on the
   event-normalized validation objective?
2. Does it beat geometry retrieval and shuffled/wrong retrieval by more than
   cross-seed variation?
3. Does fixed-weight inference degrade when real query DINO or retrieved memory
   is replaced by its matched control? Separately trained checkpoints are not
   sufficient evidence of DINO use.
4. Does it improve deformation amplitude, spatial cosine/canonical NRMSE,
   strain/edge coherence, and onset/peak timing rather than only one scalar?
5. Are gains UID-balanced, or confined to the two validation objects already
   known to have useful neighbours?
6. Does the learned retrieval gate suppress bad neighbours and activate more
   strongly for genuinely transferable ones?

Do not claim effective DINO use unless real retrieval beats matched controls,
the effect replicates, and fixed-weight ablation shows behavioral dependence.

Contact plan
============

Use oracle contact/time in the first V4.3 retrieval experiment to isolate the
DINO-memory question. Once retrieval value is established, replace oracle
contact with a causal collider-relative representation derived from predicted
COM/rotation and geometry queries. The representation should support arbitrary
colliders, not only a floor. Keep contact patch and non-contact points distinct
in retrieval and attention.

Rotation plan
=============

Keep the current rotation head and identity baseline. Only after the deformation
retrieval attribution experiment is complete, test DINO+geometry retrieved
attention for rotation as a separate controlled branch. Compare identity,
protected V4.2 rotation, geometry retrieval, real DINO retrieval, and shuffled
retrieval. Do not allow a rotation experiment to change COM or deformation in
its first stage.

Implementation requirements
===========================

- Create new V4.3 modules/runners/tests; do not overwrite V4.2 evidence or
  champion files.
- Write a design document and frozen matrix config before the full run.
- Add unit tests for same-UID exclusion, split leakage, deterministic top-k,
  point alignment/masks, zero-memory equivalence, shuffle controls, attention
  gradients, fixed-weight ablation, and protected-parameter identity.
- Add a one-batch CPU smoke run, then provide a csh/tcsh nohup command for the
  lab server in the established v42 command style.
- Save complete configs, hashes, histories, best/last checkpoints, UID-level
  validation rows, and RUN_COMPLETE markers.
- Do not use test data or start an unreviewed large matrix automatically.

First action
============

Inspect the V4.2 code/checkpoint contracts and write the concrete V4.3 retrieval
design plus experiment matrix. Then implement the retrieval bank and matched
controls with tests. Explain any necessary deviation before expanding scope.
```
