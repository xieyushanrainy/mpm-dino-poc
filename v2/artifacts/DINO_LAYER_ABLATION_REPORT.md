# V2 DINO Layer and Zero-DINO Ablation

Date: 2026-07-08

## Question

Does the choice of frozen DINOv3 transformer layer affect V2 dynamics, and
does DINO improve over a parameter-matched model receiving zero visual
features?

## Experimental design

The comparison used four variants:

- **Block 11:** final DINOv3 ViT-B/16 transformer block used by the original
  V1/V2 pipeline.
- **Block 9:** a slightly earlier high-level block.
- **Block 6:** a middle transformer block.
- **Zero-DINO:** the same architecture and learned DINO projection module, but
  with every cached DINO input set to zero.

All variants used the same:

- 16/3/3 scene split and V2 cache geometry;
- particle identities, effective reference positions, and neighbour graphs;
- DINO projection/imputation coordinates;
- model architecture and parameter count;
- initialization seed, optimizer, losses, and plateau policy;
- MPS device and `mpm-dino-poc` environment.

Block 11 was re-extracted as a preprocessing control. Its features were exactly
equal to the existing last-layer cache for all 22 scenes, confirming that the
layer comparison did not introduce a projection or track-alignment change.

Each new variant was trained from scratch until the same validation plateau
criterion used for the main V2 model. The particle-selected one-step checkpoint
was then evaluated recurrently for four steps without rollout fine-tuning.

## Results

| DINO input | Best epoch | One-step validation | Model / persistence | Four-step recurrent | Four-step teacher |
|---|---:|---:|---:|---:|---:|
| **Block 11 (final)** | 16 | **0.00383268** | **0.9919** | **0.00651959** | **0.00386395** |
| Zero-DINO | 21 | 0.00384066 | 0.9939 | 0.00671408 | 0.00397377 |
| Block 6 | 19 | 0.00397097 | 1.0277 | 0.00704909 | 0.00408202 |
| Block 9 | 21 | 0.00397561 | 1.0289 | 0.00725028 | 0.00412763 |

Lower is better.

Relative to zero-DINO, final-block DINO improves:

- one-step validation by **0.21%**;
- four-step recurrence by **2.90%**.

Relative to the intermediate layers, final-block DINO improves four-step
recurrence by **7.5%** versus block 6 and **10.1%** versus block 9.

Zero-DINO still beats four-step persistence, with a model/persistence ratio of
approximately `0.952`. Therefore most of V2's improvement comes from reference
geometry and local deformation state rather than DINO alone.

## Conclusion

The final DINO layer is the best of the tested representations. Moving to an
earlier layer does not help and materially worsens recurrent validation.

DINO provides a positive but modest improvement over zero-DINO. The advantage
is almost negligible for one-step prediction and becomes more visible at four
steps. This pattern is consistent with DINO supplying a small persistent
identity/appearance cue that helps recurrence, while explicit reference and
deformation features carry most of the predictive value.

Under the V2 plan's literal criterion, DINO is useful because the matched
final-layer model outperforms zero-DINO. Scientifically, however, a 2.9%
four-step margin from one seed is not strong enough for a definitive claim.

## Limitations and next decision

- Results use one training seed and three validation scenes.
- No confidence interval or scene-stratified significance test has been run.
- The untouched test scenes have not been used in this ablation.
- Zero-DINO retains the DINO projection parameters for parameter matching; its
  input is constant rather than removing the branch architecturally.
- Only blocks 6, 9, and 11 were compared.

Before spending more time on layer selection, repeat **only block 11 and
zero-DINO** with two additional seeds and compare per-scene four-step errors.
If the final-layer advantage persists, retain block 11. If it disappears,
treat DINO as optional and focus model development on recurrent deformation
state and optimization rather than visual-layer tuning.

## Multi-seed validation

The final-layer and zero-DINO comparison was subsequently repeated with seeds
123 and 456 using the same plateau policy.

| Seed | Final one-step | Zero one-step | Final four-step | Zero four-step |
|---:|---:|---:|---:|---:|
| 42 | **0.00383268** | 0.00384066 | **0.00651959** | 0.00671408 |
| 123 | **0.00383368** | 0.00385494 | **0.00627159** | 0.00645327 |
| 456 | 0.00390425 | **0.00380491** | 0.00659505 | **0.00654371** |
| Mean | 0.00385687 | **0.00383350** | **0.00646208** | 0.00657035 |
| Sample SD | 0.00004103 | 0.00002577 | 0.00016923 | 0.00013243 |

Final-layer DINO wins both metrics for two of three seeds, but seed 456 reverses
the ordering. On the three-seed mean:

- zero-DINO is 0.61% better at one step;
- final-layer DINO is 1.65% better at four steps.

The four-step paired advantage for DINO is `0.0001083`, smaller than the
between-seed standard deviation of either variant. With only three seeds and
three validation scenes, this is weak and inconsistent evidence rather than a
robust DINO benefit.

### Updated conclusion

The claim that final-layer DINO is better than zero-DINO is **not validated
robustly across seeds**. DINO may offer a small recurrent benefit, but its size
is comparable to initialization variance and it does not improve mean one-step
accuracy. Treat DINO as optional for the current architecture and attribute
the main V2 gains to reference geometry and local deformation features.

## Artifacts

```text
v2/runs/v2b_mps/one_step/best.pt                 # block 11
v2/runs/dino_ablation/block06/one_step/best.pt
v2/runs/dino_ablation/block09/one_step/best.pt
v2/runs/dino_ablation/zero/one_step/best.pt
v2/runs/dino_ablation/*/eval_h4/best.pt
```
