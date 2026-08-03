# Compact-memory bad-neighbour rejection

This soft-only experiment retains the frozen V4.2 champion, aligned-DINO top-3
retrieval, compact 32-token source memories, objective, seeds and sampling.

The compatibility reader predicts one weight for each retrieved object from
the pooled query and source representations.  A conservative fixed null-memory
option is included, while the point-level residual gate decides whether all
memory should be rejected. Fixing the null prior prevents trivial collapse to
the frozen base.

The targeted arm receives a second training pass with deterministic
scene-shuffled memories.  Its wrong-memory prediction is penalized for moving
away from the frozen V4.2 base, and its non-null compatibility mass is
penalized.  Validation selection uses only the normal real-memory objective.
All negative construction is training-only and leave-one-UID-out.

Promotion requires improvement over compact memory in every seed, preservation
of good-UID gains, improvement of `6132c6...`, lower non-null mass for wrong
than real memory, and fixed-weight dependence on real memory.
