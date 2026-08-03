# Controlled compact-memory versus explicit-field attention experiment

Both arms use soft-body queries only, the identical training-only source UIDs,
aligned-DINO top-3 retrieval, oracle contact/event time, sampling, loss, seeds,
and frozen V4.2 champion.  Only the trainable memory reader changes.

`compact_memory` is the existing V4.3 reader: 32 aligned tokens per source with
a stage-mean field, followed by global query-to-memory attention.

`explicit_field_attention` maps one valid point from each of the same three
source objects to every query point.  For each query frame it selects the
source frame with the same oracle event stage and nearest normalized event
time.  Its source token contains aligned coordinate, DINO and validity, source
contact features, source canonical displacement, geometric correspondence
distance, and DINO cosine confidence. Attention is over the three matched
source points for each query point, followed by the same bounded residual gate.
The readers have 186,756 and 186,819 trainable parameters respectively, a
0.034% difference.

The first comparison is architectural, not a new family-routing experiment.
Rigid and fluid objects remain excluded. Test data remains sealed.  The field
arm additionally receives fixed-weight zero-field, shuffled-correspondence and
zero-memory ablations.  A later attribution matrix is permitted only if field
attention beats compact memory across the predeclared seeds and spatial metrics.
