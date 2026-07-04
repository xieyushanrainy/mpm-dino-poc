# V2 Handoff

The next chat should begin with:

```text
Read V2_CONTEXT.md, then docs/v2/CONTEXT.md and docs/v2/PLAN.md.
Treat paths as relative to mpm-DINO-poc.
```

Primary documents:

- `docs/v1/CONCLUSION.md`: final V1 findings and limitations.
- `docs/v1/TRAINING_HISTORY.md`: consolidated run lineage and metrics.
- `docs/v1/ARTIFACTS.md`: canonical model and diagnostics.
- `docs/v2/CONTEXT.md`: assumptions and scientific handoff.
- `docs/v2/PLAN.md`: decision-complete V2 experiment plan.

Canonical V1 baseline:

```text
v1/artifacts/rollout_s4/best.pt
```

V2 objective: test whether persistent frame-0 reference positions, fixed initial surface-neighbour structure and local relative-deformation supervision improve recurrent stability, while separately measuring whether DINO contributes beyond those explicit geometric features.
