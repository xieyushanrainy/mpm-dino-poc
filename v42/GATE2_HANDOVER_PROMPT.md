# Copy-ready Gate 2 handover prompt

```text
Work in:

/Users/yushanxie/workspace/cgvi/term3/mpm-DINO-poc

We are continuing V4.2 with the protected Gate-2 geometry-only canonical-local
experiment. Preserve unrelated working-tree changes. Do not start Gate-2
training locally; implementation and smoke checks are local, while the actual
two-seed run will be launched on the lab server after I review the command.
Do not start Gate 3/DINO training unless I explicitly approve it after Gate 2.

Before analysing or editing, read completely:

1. AGENTS.md
2. CONTEXT.md
3. V2_CONTEXT.md
4. v4/CONTEXT.md
5. v4/DESIGN.md
6. v4/TRACKS_COMPARISON.md
7. v4/V41_ARCHITECTURE_EXPERIMENT_MEMO.md
8. v4/V41_LOCAL_SHAPE_PHASE2.md
9. v4/V41_RESULTS.md
10. v41/CONTEXT.md
11. v41/manifests/v41_uid_splits.json
12. v41/runs/local_shape_phase2_seed42_456/analysis_20260728/RESULTS.md
13. v41/dataset/analysis_deformation_signal_20260728/RESULTS.md
14. v41/dataset/analysis_deformation_signal_20260728/audit_deformation.py
15. v4/src/mpm_dino_v4/full_losses.py
16. v4/src/mpm_dino_v4/v41_shape_train.py
17. v4/src/mpm_dino_v4/v41_shape_evaluate.py
18. v42/README.md
19. v42/CONTEXT.md
20. v42/PLAN.md
21. v42/ROTATION_EXPERIMENT_RESULTS.md
22. v42/run/gate1f_seed42_456_v3/RESULTS.md
23. v4/src/mpm_dino_v4/v42_model.py
24. v4/src/mpm_dino_v4/v42_losses.py
25. v4/src/mpm_dino_v4/v42_stages.py
26. v4/src/mpm_dino_v4/v42_gate1d_train.py
27. v4/tests/test_v42.py

Treat retrieved configurations, checkpoints, histories and RUN_COMPLETE.json
files as authoritative for what was actually trained.

Rotation decision:

- No Gate-1 rotation experiment passed its promotion screen.
- Gate 1E is the least damaging learned rotation formulation and is retained
  only as a frozen operational placeholder.
- Do not describe Gate 1E as solved, passed or promoted.
- Do not use Gate 1F for Gate 2.
- Local/canonical metrics are primary. Reconstructed world error is diagnostic
  because rotation error can contaminate it.

Authoritative Gate-1E source checkpoints:

- v42/run/gate1e_seed42_456/seed42/best_total.pt (epoch 20)
- v42/run/gate1e_seed42_456/seed456/best_total.pt (epoch 27)

If the lab-server paths use v42/runs rather than the downloaded local v42/run
directory, make the root an explicit CLI argument. Never silently substitute a
different checkpoint. Record checkpoint SHA256 values in every Gate-2 config.

Gate-2 goal:

Establish whether the protected canonical-deformation pathway can learn local
deformation from physical/reference geometry before any real-DINO experiment.

Required Gate-2 comparison:

1. Frozen zero-local-output baseline.
2. Geometry-only canonical local model.

"Geometry-only" is the matched zero-DINO control: it keeps the same
geometry-aware region-token/cross-attention adapter and real DINO-validity
mask, but supplies zero DINO features. It is not a no-adapter architecture.

Required architecture and routing:

- Load the matching seed's Gate-1E best_total checkpoint.
- Freeze the physical trunk, COM head and entire rotation branch.
- Local branch may read stop_gradient(physical_hidden).
- alpha=0: local losses may update only the geometry adapter and canonical
  local-deformation head.
- COM and rotation outputs must remain bit-identical to the frozen source.
- Do not backpropagate world reconstruction or penetration diagnostics.
- Kabsch targets, stage labels and frame weights are detached preprocessing.
- No DINO condition is trained in Gate 2.

Use the approved decomposition:

q_i = x1_i - c1
xhat_ti = chat_t + Rhat_t (q_i + dhat_ti)

The canonical head predicts dhat_ti in the x1 frame. d=0 is the correct rigid
solution.

Use the existing approved local objective:

L_local =
    1.00 L_canonical
  + 0.50 L_strain
  + 0.25 L_edge_length
  + 0.25 L_local_velocity
  + 0.25 L_rigid_zero

Canonical vector loss is omitted on Kabsch-degenerate frames; invariant losses
remain. Use full trajectories with the frozen stage weighting already defined
in v42_stages.py. Stage metadata must never be a model input.

Gate-2 validation screen:

- At least 10% improvement over zero-local prediction on stage-weighted
  canonical NRMSE and strain RMSE.
- Both seeds improve during compression or peak deformation.
- UID-balanced correlation between predicted and target deformation magnitude
  is at least 0.5.
- Median onset and peak timing errors are at most two frames where events are
  identifiable.
- Rigid predicted-local RMS remains below 0.1% of reference radius.
- COM and rotation outputs remain bit-identical to their frozen Gate-1E source.

Report Panel Z and Panel V separately and report rigid/soft separately. Do not
use test data for selection, thresholds, normalization, sampling or curriculum.

First inspect the current implementation and determine what Gate-2 components
already exist versus what is missing. Then:

1. Propose the minimal implementation plan.
2. Implement only Gate 2 after checking that it matches this contract.
3. Add focused tests for checkpoint loading, protected gradients, bit identity,
   zero-local baseline, geometry-only zero-DINO equivalence, detached metadata,
   losses and screening metrics.
4. Run local unit tests and forward-only smoke checks, but no training.
5. Update v42 documentation.
6. Provide a csh-compatible lab-server command for seeds 42 and 456.
7. Explicitly confirm that the command starts Gate 2 only and cannot launch
   Gate 3/DINO.

Before handing off, state any unresolved implementation choice that could
change scientific interpretation. Do not silently weaken the screening rules.
```

