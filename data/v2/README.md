# V2 Generated Data

Write V2 cache-schema outputs here. Do not modify `../v1/cache/`; those files
are the frozen inputs used for the V1 baseline.

Schema version 1 adds normalized frame-0 `x0`, a non-padding `particle_mask`,
and fixed reciprocal neighbour tensors (`neighbour_indices`,
`neighbour_mask`, `rest_edge_vectors`, and `rest_edge_lengths`). Caches retain
the V1 tensors by value so V1 remains a directly comparable input baseline.
