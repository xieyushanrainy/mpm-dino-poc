# Seen-Family Unseen-Object DINO Seed Sweep Conclusion

## Question

Does final-DINO help the normal no-bottleneck V3 `latent_graph` model when
validation holds out unseen objects from families represented in training?

This is a stronger DINO-conditioning check than the original POC split because
the validation scenes are not the same object instances seen during training,
while their broad families remain represented.

## Split

Training manifest:

```text
data/shared/splits/seen_family_unseen_object_train.txt
```

Validation manifest:

```text
data/shared/splits/seen_family_unseen_object_val.txt
```

Held-out validation scenes:

- `double_stretch_zebra`
- `single_lift_cloth_4`
- `single_push_rope_1`
- `single_push_sloth`

## Configuration

- Architecture: `latent_graph`
- Geometry: full/default geometry, no bottleneck
- DINO modes: `final`, `zero`
- Device: `mps`
- Schedule: max 60 epochs, early-stop on plateau
- Primary metric: recurrent validation mean of H4 and H8 particle error

## Results

| Seed | final-DINO H4/H8 | zero-DINO H4/H8 | Final delta vs zero |
|---:|---:|---:|---:|
| 45 | 0.00993983 | 0.01068792 | -7.00% |
| 123 | 0.01415952 | 0.01063260 | +33.17% |
| 456 | 0.01041388 | 0.01086223 | -4.13% |
| 789 | 0.01082381 | 0.01094651 | -1.12% |
| 999 | 0.01044639 | 0.01047143 | -0.24% |
| 2024 | 0.01163536 | 0.01168609 | -0.43% |

Lower is better. Negative delta means final-DINO beats zero-DINO.

Aggregate across all six seeds:

| Seeds | final-DINO | zero-DINO | Final delta vs zero |
|---|---:|---:|---:|
| All seeds | 0.01123646 | 0.01088113 | +3.27% |
| Excluding seed 123 | 0.01065185 | 0.01093083 | -2.55% |

## Seed 123 Outlier

Seed `123` is a clear failure case for final-DINO. The issue appears already in
one-step validation, not only in recurrent rollout:

| Mode | Epochs | Best one-step val particle |
|---|---:|---:|
| final-DINO | 10 | 0.00492966 |
| zero-DINO | 41 | 0.00394243 |

Final-DINO stopped much earlier and landed in a worse optimization basin, while
zero-DINO continued improving. This suggests DINO introduces a seed-sensitive
nuisance channel or optimization instability in the current setup.

## Interpretation

The added seeds changed the conclusion from "DINO fails outright" to a more
nuanced result:

- Excluding seed `123`, final-DINO wins consistently on H4/H8 across
  `45`, `456`, `789`, `999`, and `2024`.
- The wins are small for the later seeds: `789`, `999`, and `2024` are close to
  ties.
- Including seed `123`, zero-DINO still wins the aggregate because the
  final-DINO failure on that seed is large.

Final-DINO therefore shows a weak positive signal on the seen-family/unseen-object
split, but the effect is not robust enough to make DINO mandatory in the main
no-bottleneck `latent_graph` architecture.

## Recommendation

Treat DINO as optional and currently weak:

1. Do not require DINO for the main V3 `latent_graph` model yet.
2. Keep the seen-family/unseen-object split as the best current diagnostic for
   visual/object conditioning.
3. If pursuing DINO further, focus on stabilizing training or regularizing the
   DINO branch before adding scene-shuffle controls.
4. Only run scene-shuffle once final-DINO is robustly better than zero-DINO
   without excluding a major failed seed.

The fair current conclusion is:

```text
Final-DINO may help slightly on unseen objects from seen families, but the
current implementation is seed-sensitive. Zero-DINO remains the safer default
for the main no-bottleneck V3 architecture.
```
