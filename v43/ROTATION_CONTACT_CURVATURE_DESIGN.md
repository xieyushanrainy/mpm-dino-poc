# Geometry rotation memory with contact and curvature

Date: 2026-08-04

## Question

Can contact timing and contact-patch geometry suppress the early/static false
rotation introduced by geometry-retrieved compact memory while retaining its
soft-body and H59 gains?  Does fixed local curvature add information beyond
contact timing and lever arm?

This is an oracle information-sufficiency follow-up.  Contact uses future
target positions and is not deployable.  It does not reopen DINO selection or
deformation architecture selection.  The V4.2 trunk, COM, rotation comparator
and full deformation path remain frozen and bit-identical.

## Matched interface

All arms use the same unified rigid+soft training-only bank, top-3 geometry
retrieval, seeds 42/123/456, sampler, loss, initialization policy and reader
parameter count.  Ten query channels are reserved in every arm:

1. mean signed gap, proximity and normal velocity (3);
2. proximity-weighted normalized contact centroid/lever arm (3);
3. proximity-weighted linearity, planarity, scattering and normal variation
   (4).

Unused channels are exactly zero.  Family is not a model input.  The floor
normal is fixed; the contact centroid supplies the lever arm relative to COM.

## Frozen matrix

| Arm | Rotation-query condition | Retrieval |
|---|---|---|
| A `geometry_base` | all ten channels zero | geometry top-3 |
| B `contact_timing` | contact channels only | same geometry top-3 |
| C `contact_patch` | contact + lever arm | same geometry top-3 |
| D `contact_curvature` | contact + lever arm + curvature | same geometry top-3 |
| E `curvature_shuffled` | D with deterministic point-shuffled curvature | same geometry top-3 |
| F `wrong_memory_contact` | D | deterministic wrong memory |

Primary promotion requires B/C/D to improve over the matched A arm in all
seeds for both families, reduce H1 and inactive/static false rotation, preserve
H59, and beat E/F beyond seed variation.  Curvature is useful only if D beats C
and E.  Report results with and without the dominant validation UID
`db01c9486b1d4590ab7d31836b1df4d9`.  Test remains sealed.
