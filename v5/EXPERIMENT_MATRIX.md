# V5 controlled experiment matrix

Date: 2026-08-05

## Purpose

This matrix validates the staged V5 claims with the fewest necessary branches.
It is a plan only: no run is authorized by this document. Seeds are `42`, `123`
and `456`; selection uses validation only and test remains sealed.

## Common contract

- Random initialization for all V5 neural components; no V4 weights.
- Dataset and split fixed to the V4.1 manifest hash in `DESIGN.md`.
- Identical preprocessing, sampling budget and stopping policy within a stage.
- COM and rotation jointly update one shared physical trunk. It is frozen
  downstream except in an explicitly named deformation fine-tuning arm.
- Soft samples train deformation; rigid and soft samples train global motion.
- Primary deformation metric: event-amplitude-normalized canonical MSE.
- Every result records seed, UID breakdown, configuration, checkpoint hashes
  and protected-path equality checks.

## Matrix

| Gate | Arm | Trained component | Required comparison or decision | Promotion rule |
|---|---|---|---|---|
| 0 | Integrity audit | None | Split hash, sealed-test check, causal feature provenance, canonical-target invariants | All checks pass |
| 1A | Ballistic COM | None | Analytic baseline | Report baseline |
| 1B/2B | Shared learned global motion | Shared trunk, COM residual and rotation head | Report COM and rotation metrics separately; compare rotation with identity | Select one joint checkpoint per seed, then make the global rotation fallback decision |
| 2A | Identity rotation | None | Required rotation baseline | Report three-seed mean |
| 2B | Learned rotation | Rotation head | Compare with 2A using SO(3) geodesic error | Adopt globally only if three-seed mean beats identity; otherwise freeze identity |
| 3 | Pointwise interaction | Interaction encoder plus auxiliary heads | Contact and event-time auxiliary metrics; no analytic contact inputs | Select own best checkpoint, then freeze |
| 4A | Zero deformation | None | Objective reference, expected near 1 | Report baseline |
| 4B | Causal base | Deformation decoder on frozen 1/2/3 | Compare its three-seed mean with the prior two-seed reference `0.91098` | `< 0.91098` qualifies; `<= 0.7025` succeeds and stops; otherwise stop |
| 4C | Shared-trunk fine-tune (optional) | Deformation decoder plus scaled gradients into shared trunk | Compare with 4B and report COM/rotation drift | Separate named arm; simplicity only breaks comparable-performance ties |
| 5A | Zero memory | None | Frozen 4B output | Must reproduce 4B |
| 5B | Compact memory | Bank tokenizer/reader and scalar gate | Top-3 DINO object retrieval, 32 tokens/source, causal phase interpolation | Final success if three-seed mean `<= 0.7025` |
| 5C | Memory attribution | No retraining | Zero memory and zero/replaced query DINO object-selection ablations | Diagnostic; disclose direction and UID concentration |

## Gate details

### Gate 0: integrity before learning

Verify the authoritative split SHA-256, 40/10/10 UID counts, 61-frame and
2,048-point shapes, Kabsch proper-rotation targets, zero-mean canonical targets
and event-amplitude normalization. Trace every inference tensor to frames 0/1,
static geometry, fixed environment information or an upstream causal
prediction. Fail closed on any target-derived query feature.

### Gate 1: COM

Train one shared global model per seed from random initialization. COM and
rotation losses jointly update its physical trunk. Persist ballistic and
residual COM outputs separately so the physics contribution remains auditable.

### Gate 2: rotation

Train the rotation head jointly with the shared trunk and COM head. Compare the
three-seed mean with identity, then make one global choice. Do not choose
identity for some seeds and learned rotation for others.

### Gate 3: interaction representation

Do not rerun a timing-only arm: the matched V4.3 evidence already rejects it.
The V5 interaction stage learns the pointwise latent using contact and
event-relative-time auxiliary supervision. Select it by a predeclared aggregate
of those auxiliary validation metrics, freeze it and remove training labels
from all inference calls.

### Gate 4: standalone causal base

The default arm trains only the deformation decoder on soft samples. The
optional 4C arm uses a predeclared scaled gradient into the shared trunk while
keeping COM/rotation heads and interaction bit-identical, and reports induced
global-motion drift. Evaluate all three seeds before applying the mean threshold.
Memory is forbidden unless the selected causal base strictly beats
`0.91098` and remains above `0.7025`.

### Gate 5: optional memory

Rebuild only the permitted 20-UID, three-phase training bank. Learn compact
tokens and the reader from scratch. Freeze every earlier stage. Training
queries exclude their own UID. The scalar gate begins near zero and the fixed
residual bound is written to the manifest before runs begin.

Arm 5C does not create a new trained matrix. It applies fixed-weight ablations
to 5B to show whether the final checkpoint depends on memory and DINO-based
object selection. Per-UID harm and gain are reported even though balance is not
a promotion requirement.

## Decision outcomes

1. If 4B mean is `>= 0.91098`, V5 architecture fails; do not add memory.
2. If 4B mean is `< 0.91098` and `<= 0.7025`, V5 succeeds without memory.
3. If 4B mean lies in `(0.7025, 0.91098)`, execute Gate 5.
4. If 5B mean is `<= 0.7025`, V5 succeeds with compact memory.
5. Otherwise report a qualified causal-base improvement but not V5 success.

No test-set evaluation is included in this planning matrix. A separate explicit
authorization is required after the architecture and all thresholds are frozen.
