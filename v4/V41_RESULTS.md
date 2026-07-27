# V4.1 correspondence-preserving architecture screen

Date: 2026-07-26

This document supplements, and does not replace, the prior V4 evidence in
[`RESULTS.md`](RESULTS.md) and [`TRACKS_COMPARISON.md`](TRACKS_COMPARISON.md).
It separates retrieved-run integrity, validation-only checkpoint selection, and
test conclusions. Raw artifacts are under
`v41/runs/lab_cap150_p30_fp32/analysis_20260726/`.

## Conclusion

No V4.1 visual mechanism passes the frozen promotion rule.

- M1 real DINO improves the three-seed Panel Z mean at H30 and H40 and wins
  2/3 and 3/3 paired seeds, respectively. It nevertheless fails promotion
  because test H1 is 17.8% worse than its matched zero control.
- M6 real DINO is worse in the Panel Z three-seed mean at H30 and H40 and its
  test H1 is 97.5% worse than zero.
- M2 cannot receive a three-seed scientific verdict. Real seeds 42 and 456
  never passed the validation H1 guard and therefore have no guarded `best.pt`.
  Their diagnostic last checkpoints were not substituted.

H59 is diagnostic only. Neither the M1 H59 deterioration nor any isolated M2
seed result can promote a mechanism. Because no mechanism promotes, no
scene-shuffled or point-shuffled training is authorized by the reviewed rule.
The experiment therefore supplies no shuffled-control evidence for a
point-correspondence claim.

## Implementation and retrieved-run integrity

The retrieved matrix has the expected 21 stages: 12 M1/M2 runs, three M6
stage-1 trunks, and six M6 stage-2 runs. All histories parse, contain monotonic
epochs, and are finite. Configurations consistently record CUDA FP32 without
AMP, seeds 42/123/456, width 128, four blocks, four heads, 40 UID-balanced
draws, a 150-epoch cap, early-stopping patience 30, and plateau patience 5.
The frozen manifest and collection hashes match. Every accepted guarded
`best.pt` matches its completion record and a guard-eligible history epoch.
M6 real and zero stage 2 have identical `starting_trunk_sha256` values within
each seed.

Five runs reached the 150-epoch cap:

- M1 real seed 456;
- M1 zero seeds 123 and 456;
- M2 real seeds 42 and 456.

The last two are complete diagnostic runs with
`complete_no_h1_eligible_checkpoint`. There are no `RUN_FAILED.json` or
`RUNNING.json` markers in the retrieved matrix.

One retrieval/provenance anomaly remains: M1 zero seed 123 `last.pt` has SHA-256
`883acb1a...`, while its completion record names `c6760357...`. Its selected
`best.pt` independently matches the recorded hash `7f9307f3...`; the mismatched
last checkpoint was not used. The complete pre-interpretation report and file
inventory are in
[`INTEGRITY_REPORT.md`](../v41/runs/lab_cap150_p30_fp32/analysis_20260726/INTEGRITY_REPORT.md)
and `integrity_report.json`.

The implementation retains the reviewed physical backbone, losses, fixed UID
split, real validity mask in zero controls, and separate Panel Z/V evaluation.
The full test evaluator reports H1, H8, H16, H30, H40, H59, every intervening
frame, world RMSE/MAE, COM error, centre-relative shape error, edge-vector and
edge-length error, penetration rate/depth, active coverage, and rigid Kabsch
residual.

## Validation-only selection

Guarded checkpoints were selected by validation normalized mean RMSE over
H16/H30/H40, subject to the matched-zero validation H1 guard. All 16 evaluated
checkpoints passed their saved validation guard. Test results were not used to
choose checkpoints.

M1 and M6 each have six eligible guarded checkpoints. M2 has all three zero
checkpoints but only real seed 123. M6 real/zero stage 2 starts from the
byte-identical per-seed stage-1 trunk, as predeclared.

## Untouched test results

Values below are three-seed object-weighted means in millimetres. “Wins” counts
paired real-DINO seeds with lower RMSE. Panel Z and Panel V are never combined.
M2 is omitted from three-seed tables because its real condition has only one
eligible seed.

### Panel Z: balanced rigid/soft, zero velocity

| Mechanism | Condition | H1 | H8 | H16 | H30 | H40 | H59 |
|---|---|---:|---:|---:|---:|---:|---:|
| M1 | real | 8.92 | 62.00 | 54.14 | 39.58 | 31.13 | 66.63 |
| M1 | zero | 7.57 | 63.80 | 56.37 | 41.50 | 33.08 | 51.17 |
| M1 | real wins | 0/3 | 3/3 | 3/3 | 2/3 | 3/3 | 0/3 |
| M6 | real | 8.93 | 60.24 | 52.28 | 43.94 | 35.30 | 55.11 |
| M6 | zero | 4.52 | 61.66 | 56.30 | 42.64 | 34.01 | 43.93 |
| M6 | real wins | 0/3 | 2/3 | 3/3 | 2/3 | 2/3 | 1/3 |

M1's real-minus-zero mean RMSE deltas are -1.91 mm at H30 and -1.95 mm
at H40, but the UID-bootstrap 95% intervals are respectively
[-6.85, 2.48] mm and [-6.88, 2.27] mm. With only ten test UIDs, both intervals
include zero. M6's corresponding deltas are +1.30 mm and +1.28 mm, with
intervals [-5.57, 8.37] mm and [-4.67, 7.17] mm.

At H30/H40, M1's lower world error is mostly lower COM error; its
centre-relative shape and edge errors are essentially unchanged. The gain also
coincides with more penetration: at H40, real/zero penetration rates are
3.70%/1.26% and mean depths are 0.270/0.087 mm. M6 likewise increases H40
penetration from 4.21% to 8.80% and depth from 0.351 to 1.398 mm. Thus neither
mechanism supports a cleaner-contact interpretation.

### Panel V: rigid-only variable velocity

The test split contains three episodes from only one UID. These values are
episode means within that UID, followed by seed means; a UID bootstrap is
degenerate and cannot quantify object-level generalization.

| Mechanism | Condition | H1 | H8 | H16 | H30 | H40 | H59 |
|---|---|---:|---:|---:|---:|---:|---:|
| M1 | real | 14.67 | 14.41 | 14.33 | 13.48 | 23.85 | 63.60 |
| M1 | zero | 15.97 | 12.91 | 12.22 | 12.41 | 19.63 | 56.58 |
| M6 | real | 13.19 | 12.04 | 9.81 | 18.11 | 19.55 | 66.03 |
| M6 | zero | 12.11 | 8.81 | 6.72 | 15.60 | 19.58 | 53.98 |

Panel V is dominated by COM error; rigid shape and Kabsch residuals remain
sub-millimetre. Active coverage is 1.0 throughout both panels. These results
describe this held-out UID and its velocity episodes, not broad
variable-velocity object generalization.

### Physical metrics at promotion horizons

The table reports real/zero three-seed means. Distances are millimetres; rates
and coverage are fractions. Kabsch is rigid-only.

| Panel | Mechanism | H | RMSE | MAE | COM | shape | edge-vector | edge-length | penetration rate | depth | coverage | Kabsch |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Z | M1 | 30 | 39.58/41.50 | 37.33/39.22 | 32.68/34.99 | 19.53/19.52 | 1.943/1.942 | 0.038/0.036 | .0406/.0174 | .333/.096 | 1/1 | .235/.202 |
| Z | M1 | 40 | 31.13/33.08 | 29.11/30.96 | 19.52/22.46 | 21.50/21.50 | 1.746/1.746 | 0.055/0.053 | .0370/.0126 | .270/.087 | 1/1 | .248/.218 |
| Z | M6 | 30 | 43.94/42.64 | 41.75/40.22 | 36.98/36.08 | 19.51/19.51 | 1.941/1.941 | 0.035/0.035 | .0855/.0323 | 1.313/.234 | 1/1 | .183/.185 |
| Z | M6 | 40 | 35.30/34.01 | 33.13/31.57 | 24.17/23.35 | 21.50/21.50 | 1.746/1.746 | 0.052/0.052 | .0880/.0421 | 1.398/.351 | 1/1 | .184/.188 |
| V | M1 | 30 | 13.48/12.41 | 13.48/12.41 | 13.47/12.40 | .428/.369 | .032/.027 | .018/.015 | .0036/.0036 | .017/.022 | 1/1 | .371/.317 |
| V | M1 | 40 | 23.85/19.63 | 23.85/19.63 | 23.85/19.62 | .453/.401 | .034/.030 | .019/.016 | .0078/.0023 | .053/.016 | 1/1 | .390/.342 |
| V | M6 | 30 | 18.11/15.60 | 18.11/15.60 | 18.10/15.60 | .347/.343 | .026/.025 | .013/.014 | .0114/.0079 | .110/.052 | 1/1 | .289/.292 |
| V | M6 | 40 | 19.55/19.58 | 19.55/19.58 | 19.55/19.58 | .354/.352 | .026/.026 | .014/.014 | .0040/.0053 | .023/.019 | 1/1 | .291/.297 |

All horizons, per-frame curves, per-seed values, per-UID paired deltas,
bootstrap intervals, and paired sign-flip uncertainty are saved in
`aggregate_summary.json`, `paired_per_object_deltas.json`, and
`raw_evaluations/*.json`. Statistical uncertainty is screening-level only:
Panel Z has ten test UIDs and Panel V has one.

## Frozen promotion decision

The rule requires, at H30 or H40: a better three-seed real mean, at least two
paired-seed wins, and test H1 no more than 10% worse than zero.

| Mechanism | H30 mean/wins | H40 mean/wins | test H1 real/zero | Decision |
|---|---|---|---:|---|
| M1 | better, 2/3 | better, 3/3 | 1.178 | not promoted: H1 guard fails |
| M2 | incomplete | incomplete | unavailable | not evaluable: two guarded real checkpoints absent |
| M6 | worse, 2/3 | worse, 2/3 | 1.975 | not promoted: mean and H1 criteria fail |

No shuffled-control runs are proposed because the prerequisite promotion did
not occur.

## Relation to prior Track B evidence

The earlier pooled-DINO Track B comparison used the same broad non-causal
backbone family but is not an architecture-identical control for M1/M2/M6. Its
Panel-Z predecessor test used ten rigid/soft zero-velocity objects and reported
real pooled DINO worse than zero at H1/H8/H16 (29.76/62.90/57.79 mm versus
23.81/57.40/51.35 mm), with an aggregate H59 improvement driven principally by
rigid motion.

V4.1 M1 changes that pattern at medium horizons: correspondence-preserving real
DINO lowers the H30/H40 mean relative to its own matched zero path. That is a
weak architecture-specific signal, not promotion: it fails the test H1
constraint, its ten-UID intervals include zero, and it worsens H59 and
penetration. M6 does not reproduce the medium-horizon gain. M2 is incomplete.
The combined evidence therefore does not establish useful DINO conditioning or
point-correspondence benefit.

## Claim boundaries

These results do not demonstrate obstacle generalization from the single wall
episode, clean soft-body contact learning from proxy-contact trajectories,
material-property inference, or correspondence benefit without shuffled
controls. Panels Z and V remain separate, and H59 is never used for promotion.

## Exact pooled Track B bridge experiment

The original V4 Track B pooled-DINO/FiLM architecture was subsequently repeated
on V4.1 with the shared V4.1 loss, UID-balanced sampler, FP32 budget, validation
selector and H1 guard. The retrieved six-run matrix is artifact-complete, but
real seed 123 never passed the validation H1 guard and has no guarded `best.pt`.
The frozen three-seed promotion rule is therefore not scientifically
evaluable.

On the two eligible matched seeds, real pooled DINO is worse than zero on Panel
Z by 10.9% at H30 and 22.8% at H40, losing both seeds at both horizons. Panel V
is also substantially worse from H8 through H40. Real improves the two-seed
mean at H59, but H59 is diagnostic only. The bridge experiment is not promoted
and does not justify shuffled controls.

The full integrity and test report is
[`../v41/runs/track_b_pooled_cap150_p30_fp32/analysis_20260727/RESULTS.md`](../v41/runs/track_b_pooled_cap150_p30_fp32/analysis_20260727/RESULTS.md).
