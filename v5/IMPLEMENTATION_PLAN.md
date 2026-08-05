# V5 implementation plan

Date: 2026-08-05

## Scope

Implementation was authorized and the initial package, staged trainers, memory
branch, evaluator, CLI and tests are complete. Full experiment training remains
a separate action. This document retains the work packages and acceptance
checks as the implementation contract.

Implemented package:

```text
v5/src/mpm_dino_v5
```

Verification suite:

```text
v5/tests
```

## 1. Package and configuration skeleton

Create a V5 package alongside the V4 implementation rather than modifying V4
behavior. Define versioned configuration schemas for data, architecture,
training stages, bank construction and evaluation. Every run manifest records
the split hash, seed, initialization policy, input provenance, stage source
hashes and test-sealing status.

Acceptance checks:

- importing V5 does not alter V4 modules or artifacts;
- configs reject V4 checkpoint paths as initialization sources;
- configs make the stage freeze set explicit;
- a dry-run manifest contains all reproducibility fields.

## 2. Reuse audited data and target utilities

Wrap the V4.1 loader and verified geometry/target utilities behind V5-owned
interfaces. Add provenance tags distinguishing inference inputs, training-only
labels and evaluation-only targets. Keep family out of model tensors.

Acceptance checks:

- split and dataset hashes match the design;
- target shapes and point IDs are stable;
- canonical targets are zero-mean and use proper Kabsch rotations;
- an inference batch contains no future query targets or event labels.

## 3. Implement factorized global motion

Implement the ballistic COM function, randomly initialized residual COM head,
proper-SO(3) rotation head and identity fallback. Expose rigid future positions
as a detached interface for the interaction stage.

Acceptance checks:

- zero COM residual reproduces the analytic ballistic path;
- predicted rotations are proper rotations;
- identity and learned rotation share the same evaluation path;
- freezing prevents parameter and output changes downstream.

## 4. Implement the learned interaction representation

Implement a small pointwise encoder over centered reference geometry and the
frozen predicted rigid trajectory. Add training-only contact and event-time
heads. Ensure the deformation-facing API returns only the latent and causal
metadata; ground-truth auxiliary labels never enter its forward inputs.

Acceptance checks:

- gradients cannot reach frozen COM/rotation;
- DINO, family and material labels are absent from encoder inputs;
- perturbing or deleting future labels does not change inference output;
- auxiliary heads and metrics handle pre-event and missing-event frames with a
  predeclared mask.

## 5. Implement the standalone deformation decoder

Implement a soft-only decoder from reference geometry and frozen interaction
latents. Enforce zero-mean canonical deformation in the forward path. Reuse the
audited event-amplitude-normalized objective and metric implementation.

Acceptance checks:

- predicted deformation has zero pointwise mean per frame;
- no deformation gradient reaches earlier stages;
- zero prediction scores approximately one under the primary objective;
- the three-seed aggregation applies the strict and inclusive thresholds
  exactly as specified.

## 6. Implement optional memory behind a gate

Do this work only if the causal base qualifies but misses the final target.
Build a V5 bank generator restricted to the 20 permitted soft training UIDs and
the three existing phases. Implement leave-one-UID-out retrieval, top-3
object-level DINO selection, 32 learned compact tokens per source entry, linear
phase interpolation, compact cross-attention and the near-zero scalar gate.

Acceptance checks:

- bank manifest rejects non-training or out-of-scope UIDs/phases;
- learned tokens and reader weights are newly initialized;
- training retrieval never returns the query UID;
- zero memory reproduces the frozen causal base;
- residuals are bounded and zero-mean;
- DINO affects only source-object selection.

## 7. Evaluation and reporting

Build one evaluator shared by baselines and trained arms. Produce per-seed,
three-seed mean and per-UID tables plus the diagnostic physical and efficiency
metrics required by `DESIGN.md`. Add protected-path hashes and causal-input
audits to every result bundle.

Acceptance checks:

- the evaluator never loads sealed test entries during validation work;
- identity and learned rotation are compared before the global fallback choice;
- memory ablations use fixed weights and do not retrain;
- result summaries state formal success separately from diagnostic quality.

## 8. Execution order and stop conditions

1. Review and freeze all planning defaults in a versioned matrix config.
2. Run Gate 0 integrity tests.
3. Train and freeze COM for all three seeds.
4. Train rotation for all seeds; make the global identity fallback decision.
5. Train and freeze the interaction encoder for all seeds.
6. Train and evaluate the standalone causal deformation base.
7. Stop on failure or base success according to the thresholds.
8. Only when eligible, build/train/evaluate compact memory.
9. Freeze the selected architecture before requesting any sealed-test run.

## 9. Pre-execution review items

The following values require one written choice before implementation begins:

- model widths, depths and latent/token dimensions;
- auxiliary contact/time target definitions and aggregate selection metric;
- residual bound and gate initial logit;
- optimizer, learning rate, sampling budget, epoch cap and patience;
- tolerance for the simplicity tie-breaker;
- exact checkpoint and artifact directory naming.

These are bounded engineering choices. They must not alter the architectural
contracts or promotion thresholds established by the interview.
