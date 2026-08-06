# No-memory pointwise contact rotation adapter

Date: 2026-08-06

This experiment tests the closest rotation analogue of the successful V4.2
deformation repair. It uses no retrieval or memory bank. The existing
`DirectProbeConditionBuilder` computes one synchronized `[B,59,N,15]` oracle
condition that is passed unchanged to the frozen deformation model and the new
rotation adapter. The rotation path additionally receives normalized lever arm
`r=x-COM` and the floor-contact torque basis `r x n`.

The V4.2 physical trunk, COM and complete deformation path are frozen. Both
protected state and COM/deformation outputs must remain bit-identical. Identity
is the rotation base. A point-attention reader produces a bounded per-frame Lie
residual; its residual layer is zero-initialized and its gate starts near 0.01.

The objective balances frames with target rotation >=0.5 degrees against a
separate false-rotation penalty below 0.25 degrees, plus residual smoothness.
Natural-distribution validation remains the selection/reporting metric.

MVP matched variants, seeds 42/123/456:

1. `physical_only`: pointwise frozen physical features; all condition channels zero.
2. `contact_torque_basis`: recommended full interface: exact pointwise
   15-channel deformation condition plus normalized lever arm and `r x n`.
3. `contact_shuffled`: full interface with contact+curvature shuffled across
   points while event time stays synchronized.

The code retains intermediate diagnostic variants, but they are outside this
frozen MVP and must not be launched initially. If the full interface does not
beat both controls, stop. If it succeeds across all seeds and families, a later
small attribution follow-up may separate pointwise contact, lever arm and event
time.

Promotion requires the full interface to beat physical-only and shuffled
contact in every seed for both families; reduce static/H1 error; preserve H59;
and keep protected outputs bit-identical. Oracle contact makes this an
information-sufficiency experiment, not a deployable model. Test remains sealed.
