# V4.3 retrieval-conditioned deformation design

> Priority update (2026-08-03): before launching the full retrieval matrix, run
> the matched causal-contact replacement experiment in
> [`CAUSAL_CONTACT_DESIGN.md`](CAUSAL_CONTACT_DESIGN.md). This does not change
> the frozen retrieval design; it measures how much of the V4.2 oracle base can
> be made deployable and determines which base condition V4.3 should inherit.

## Status and scope

This document freezes the first validation-only V4.3 design.  It tests whether
aligned DINO is useful as a key for an attended mechanical memory.  It does not
authorize test-set access, causal-contact claims, rotation retrieval, or a large
matrix run.  The V4.2 champion is an oracle-conditioned research reference.

## Protected model contract

The V4.2 graph-temporal physical trunk, COM head, and complete rotation branch
are loaded from the same seed-matched source and frozen.  The existing
full-condition region adapter and canonical head form the base deformation
path.  The retrieval module adds a zero-mean, gated canonical residual only:

```text
d_hat = zero_mean(d_base + sigmoid(g) * tanh(r) * residual_scale)
x_hat = c_hat + R_hat (q + d_hat)
```

The retrieval branch receives detached physical point features.  It cannot
alter COM, rotation, or the physical trunk.  Every optimizer step checks those
parameters against a bit-identical snapshot; smoke and final evaluation also
check COM and rotation outputs.

## Frozen bank and retrieval contract

The bank contains soft-body training UIDs only.  Each memory records source
UID, `train` split, event stage, normalized event time, point and DINO masks,
PCA-frame normalized coordinates, aligned DINO, oracle contact-patch features,
canonical target deformation, geometry scale/frame, and target-file
provenance.  The canonical manifest and tensor payload are hashed.  Every arm
must name the identical bank content hash.

Training retrieval excludes the query UID.  Validation retrieves only from the
training bank.  Ties are ordered by `(distance, source_uid)`.  No validation or
test target enters retrieval, alignment fitting, prototypes, calibration, or
neighbour choice.  Test remains sealed.

For each candidate, PCA proper-sign alignment is selected by geometry Chamfer.
Query points are matched to valid source points by nearest normalized aligned
coordinate.  Real-DINO ranking uses mean cosine distance on those matched valid
point pairs; geometry ranking uses Chamfer.  `k=3` is fixed before validation.
After alignment, each retrieved object is deterministically reduced to 32
evenly spaced valid memory tokens.  Full 2,048-by-2,048 cross-attention over 59
frames is not computationally practical; this fixed reduction is identical in
all arms and is not tuned on validation.

## Query, memory, and residual interface

Each query token contains detached physical state, normalized reference
coordinate, query DINO plus validity, oracle floor-contact features, and oracle
stage/event time.  Each memory token contains aligned source coordinate, source
DINO plus validity, source contact features/patch flag, canonical deformation,
stage, normalized event time, and source validity.  Query points cross-attend
to the flattened tokens from three retrieved objects.  A learned bounded gate
can suppress the residual.  Retrieved fields are inputs, never copied directly
to the output.

## Matched arms and attribution

All trained A--E arms have the same retrieval-module parameter count and base
checkpoint.  Controls change tensors/index selection, not architecture.

| Arm | Retrieval/memory condition |
|---|---|
| A `zero_memory` | all memory values and validity zero |
| B `geometry` | geometry-ranked sources; DINO channels zero |
| C `aligned_dino` | aligned point-DINO top-3 sources |
| D `scene_shuffled` | deterministic wrong-object permutation of C |
| E `point_shuffled` | C sources with deterministic within-source point permutation of DINO correspondence |
| F `oracle_best_ceiling` | noncausal validation report only; never trained or selected |

For the trained C checkpoint, fixed-weight inference additionally replaces (1)
query DINO, (2) memory DINO, (3) the full memory, and (4) point correspondence
with matched controls.  Separately trained arms alone are not attribution.

## Training and reporting

Seeds are 42 and 456.  Training uses soft objects and the V4.2 oracle
contact/curvature/event-time base condition, with the event-frame per-episode
amplitude-normalized canonical MSE as the selection objective.  Sampling,
epochs, optimizer, and protected sources are matched across A--E.  Reports save
complete arguments, checkpoint and bank hashes, histories, best/last weights,
UID-level validation rows, gate statistics, fixed-weight ablations, and a
`RUN_COMPLETE.json` marker.

Report event-normalized objective, canonical NRMSE, spatial vector cosine,
strain and edge coherence, predicted/target peak amplitude, onset/peak timing,
per-UID values, raw top-k transfer, and zero deformation.  A DINO-use claim
requires C to beat A, B, D, and E beyond cross-seed variation, in both seeds,
plus degradation under fixed-weight DINO/memory ablation and UID-balanced gains.

## Staged execution

1. Unit tests and one-batch CPU smoke test.
2. Review this design, matrix, bank manifest, and smoke artifacts.
3. Run A--E on validation only after approval.
4. Consider causal collider-relative contact only after retrieval value is
   established.  Rotation retrieval remains a separate later experiment.
