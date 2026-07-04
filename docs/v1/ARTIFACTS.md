# V1 Artifact Index

## Canonical model

- `v1/artifacts/rollout_s4/best.pt`: retained V1 checkpoint, step-4 epoch 3.
- `v1/artifacts/rollout_s4/history.jsonl`: canonical step-4 training history.

Checkpoint SHA-256:

```text
7ebc021bec0002c8b8f1ad958b633362e11722bdfa0547b9285d865cf1b40be5
```

## Canonical diagnostics

- `v1/artifacts/rollout_s4/velocity_feedback_val.csv`: rolling validation windows.
- `v1/artifacts/rollout_s4/velocity_feedback_other_unseen_short.csv`: unseen sloth and cloth short horizons.
- `v1/artifacts/rollout_s4/velocity_feedback_unseen_zebra.csv`: long zebra trajectory.
- `v1/artifacts/rollout_s4/velocity_feedback_push_sloth_long.csv`: long sloth trajectory.
- `v1/artifacts/rollout_s4/velocity_feedback_lift_cloth_long.csv`: long cloth trajectory.
- `v1/artifacts/rollout_s4/unseen_push_sloth.mp4`: unseen sloth visualization.
- `v1/artifacts/rollout_s4/unseen_lift_cloth4.mp4`: unseen cloth visualization.
- `v1/artifacts/rollout_s4/unseen_zebra_alpha025.mp4`: damped zebra diagnostic.

## Non-canonical predecessors

The following directories establish training lineage but are superseded:

```text
archive/v1_runs/one_step
archive/v1_runs/multiscene_resume
archive/v1_runs/multiscene_particle
archive/v1_runs/multiscene_particle_2
archive/v1_runs/rollout_s2
archive/v1_runs/rollout_s4_2_no_good
```

They have not been deleted automatically. Their meaningful results are consolidated in `TRAINING_HISTORY.md`, allowing manual archival or deletion later.
