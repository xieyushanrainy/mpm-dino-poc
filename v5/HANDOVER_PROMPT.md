# Copy-ready V5 planning and grilling prompt

```text
Work in:
/Users/yushanxie/workspace/cgvi/term3/mpm-DINO-poc

Use Plan mode for this task. Use the `grill-me` skill after reading the context.
Do not implement code, freeze an architecture, write a controlled experiment
matrix or start training until the interview has clarified the design and I
explicitly approve moving forward.

Purpose
=======

I want to design a cleaner V5 trained from scratch on the same dataset. It may
combine the strongest causal-contact and oracle compact-memory ideas, but that
combination is not yet a decision. Preserve evidence-backed mechanisms where
appropriate, challenge unnecessary inherited complexity, and keep all major V5
choices open during the initial grilling session.

Read first
==========

Start with these context documents:

1. v5/HANDOVER.md
2. v41/dataset/README.md
3. v41/manifests/v41_uid_splits.json
4. v42/V42_EXPERIMENT_SUMMARY.md
5. v43/V43_DEFORMATION_CLOSEOUT.md
6. v43/run/causal_contact_seed42_456/RESULTS.md
7. v43/run/rotation_memory/RESULTS.md

Then inspect these authoritative architecture/config/code files only as needed:

8. v42/checkpoints/MANIFEST.json
9. v43/CAUSAL_CONTACT_DESIGN.md
10. v43/DESIGN.md
11. v43/run/causal_contact_seed42_456/MATRIX_CONFIG.json
12. v43/run/causal_contact_seed42_456/causal_continuous/seed42/config.json
13. v43/run/bad_neighbour_rejection_seed42_123_456/compact_baseline/seed123/config.json
14. v43/run/bad_neighbour_rejection_seed42_123_456/BANK_MANIFEST.json
15. v4/src/mpm_dino_v4/v41_data.py
16. v4/src/mpm_dino_v4/v42_model.py
17. v4/src/mpm_dino_v4/v43_causal_contact.py
18. v4/src/mpm_dino_v4/v43_retrieval.py
19. v4/src/mpm_dino_v4/v43_train.py

Current oracle architecture
===========================

The strongest oracle-conditioned deformation system is composite:

- frozen V4.2 full-condition base with explicit COM, rotation and zero-mean
  canonical deformation;
- future-ground-truth contact, stage and event time condition the base;
- aligned-DINO top-3 retrieval from a train-only soft-object bank;
- 32 compact tokens per source;
- compact cross-attention predicts a bounded gated residual;
- d_hat = zero_mean(d_base + gate * residual).

Base checkpoint:
v42/checkpoints/v42_adapter_full_seed42_best.pt
objective: 0.7771909118

Best reader:
v43/run/bad_neighbour_rejection_seed42_123_456/compact_baseline/seed123/best.pt
objective: 0.6629586637

Compact-baseline three-seed mean: 0.7025. It is a non-deployable oracle
research result. DINO is supported mainly as a sparse object/prototype selector;
exact pointwise mechanics correspondence and reliable bad-neighbour rejection
were not established.

Current causal architecture
===========================

The best causal candidate uses a frozen Gate-1E physical/COM/rotation model to
construct a zero-deformation rigid future trajectory. Fixed analytic operations
derive pointwise signed floor gap, smooth proximity, vertical velocity and time
relative to predicted contact. Together with four static geometry proxies and
seven zero stage slots, this forms a 15-channel condition. A learned condition
projection, geometry-region adapter and zero-mean canonical deformation head
are trained while the global path remains protected.

Best checkpoint:
v43/run/causal_contact_seed42_456/causal_continuous/seed42/best.pt
objective: 0.8373412609

Seed 456 scored 0.98462; the two-seed mean was 0.91098. Continuous pointwise
contact showed useful signal, but the effect did not replicate and was
UID-concentrated. Timing alone was null. Gate-1E rotation remains an operational
placeholder, not a promoted solution.

Evidence-backed mechanisms to understand
========================================

- COM/rotation/canonical-deformation factorization;
- ballistic COM plus learned residual;
- zero-mean canonical deformation and Kabsch-aligned targets;
- pointwise contact localization rather than timing alone;
- event-amplitude-normalized deformation objective;
- competent base plus bounded memory correction;
- compact latent memory rather than copied fields;
- DINO as object/prototype retrieval signal;
- train-only, leave-one-UID-out memory discipline;
- protected global paths for attribution.

Critical incompatibility
========================

The oracle memory implementation uses ground-truth query stage/event
information when selecting/building stage-specific memory. The causal model has
no memory reader. They cannot simply be connected and called a causal model.
How a causal query accesses source phase is an open V5 design question.

Dataset
=======

Dataset: v41/dataset
Split: v41/manifests/v41_uid_splits.json
Manifest SHA-256:
2f4add1aa6f6f68b965d031cf12bf60f981a10f628a09bdc7f75a574cd42949b

There are 40 train, 10 validation and 10 sealed test UIDs. Each episode has 61
frames and 2,048 points; observe frames 0/1 and predict 59 future frames. Rigid
and soft are available; fluid was excluded. Family may support sampling and
reporting but is not a model input.

Grilling instructions
=====================

After reading, first give me a concise evidence summary and a list of decisions
that remain open. Then invoke `grill-me` and interview me relentlessly but one
consequential question at a time. Challenge assumptions and expose trade-offs,
especially around:

- the primary V5 goal and meaning of match/beat;
- what training from scratch means;
- which proven mechanisms must remain;
- whether V5 needs a base-plus-memory residual;
- whether a memory bank is acceptable at inference;
- causal phase access to training memories;
- DINO's role and shape correspondence;
- COM/rotation scope and identity rotation;
- how rigid samples should be used;
- complexity, latency and controlled evaluation.

Maintain a live decision log with three categories: decided, tentative and
open. Do not turn tentative answers into frozen requirements. Do not draft
v5/DESIGN.md or an experiment matrix until I explicitly say the grilling phase
is complete and authorize planning output.
```
