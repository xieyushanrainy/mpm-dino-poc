# V5 planning context

The initial `grill-me` interview is complete and the V5 planning direction was
approved on 2026-08-05. No V5 code or training has been started.

Start with:

1. [`HANDOVER.md`](HANDOVER.md) — dataset, current oracle/causal architectures,
   evidence-backed mechanisms and open questions.
2. [`HANDOVER_PROMPT.md`](HANDOVER_PROMPT.md) — copy-ready Plan-mode and
   `grill-me` prompt for a new Codex conversation.
3. [`DESIGN.md`](DESIGN.md) — approved V5 goals, architecture and safeguards.
4. [`EXPERIMENT_MATRIX.md`](EXPERIMENT_MATRIX.md) — staged controlled matrix
   and promotion rules.
5. [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md) — implementation work
   packages and acceptance checks.

Implementation was subsequently authorized. Full experiment training and
sealed-test evaluation remain separate explicit actions.

## Implementation

The V5 implementation now lives in `v5/src/mpm_dino_v5/`. It provides:

- validated configuration and causal-input contracts;
- ballistic-plus-residual COM and learned/identity rotation paths;
- learned pointwise interaction and zero-mean canonical deformation modules;
- stage freezing, protected hashes and promotion decisions;
- restricted top-3 compact memory with three-phase interpolation;
- event-normalized losses, validation aggregation and staged trainers.

Run the integrity audit without loading sealed-test trajectories:

```bash
PYTHONPATH=v2/src:v4/src:v5/src python -m mpm_dino_v5.cli audit \
  v41/manifests/v41_uid_splits.json
```

Create a validated configuration:

```bash
PYTHONPATH=v2/src:v4/src:v5/src python -m mpm_dino_v5.cli init-config \
  v5/config.local.json
```

Train one stage explicitly with `python -m mpm_dino_v5.cli train --help`.
Downstream stages require the promoted V5 checkpoints from earlier stages.
No trainer accepts a V4 initialization checkpoint.

The optional memory branch is disabled by default. After the causal base is
formally eligible, rebuild its restricted bank with `build-bank`, set
`memory.enabled` to `true` in a copied config, and train the `memory` stage.

## Shared physical v3 correction

The independent COM/rotation implementation is superseded. `V5SharedPhysicalModel`
restores the V4.2-class pointwise graph/temporal trunk and trains it jointly from
the COM and rotation losses, while retaining ballistic-plus-residual COM and an
identity rotation baseline. Its contract is `v5_shared_physical_v3`; no V4 or
older V5 weights are loaded. Start the three-seed global run with
`v5/run_shared_global_v3.csh`.

Interaction freezes the promoted shared-global checkpoint. Deformation freezes
it by default (`--trunk-gradient-scale 0`), but a separately named experimental
arm may pass a value in `(0, 1]` to fine-tune the shared trunk through a scaled
gradient. COM and rotation head weights remain protected in either case.
