# V2-B MPS Training Summary

Date: 2026-07-04

## Configuration

- Environment: `mpm-dino-poc`
- Device: Apple MPS
- Split: unchanged V1 16/3/3 scene split
- Model: V2-B reference positions, reference displacement, fixed mutual-kNN
  neighbourhoods, stretch descriptors, and edge vector/length losses
- Selection: validation particle distance with a 0.5% meaningful-improvement
  threshold
- Scheduling: ReduceLROnPlateau with guarded early stopping

## One-step result

The one-step stage ran for 24 epochs and stopped after eight epochs without a
0.5% relative improvement. Learning rate reductions occurred at epochs 10,
20, and 24. The retained checkpoint is epoch 16:

```text
validation particle mean: 0.0038326809
validation persistence:   0.0038641035
model / persistence:      0.991868
```

This is 6.2% lower than the documented V1 one-step validation particle error
of `0.0040866636`.

## Step-2 rollout result

Initial evaluation of the retained one-step weights:

```text
recurrent particle mean: 0.0046884059
recurrent persistence:   0.0049138445
model / persistence:     0.954122
teacher-forced mean:     0.00382382
teacher guard limit:     0.00420620
```

V1-style rollout fine-tuning was tested at `2.5e-5` and `1e-5`. Both caused
immediate guarded regression. A final conservative run used `2.5e-6`, teacher
weight `1.0`, plateau reduction to `1e-6`, and early-stopped after six epochs.
No trained epoch passed the teacher guard or beat the epoch-0 recurrent
incumbent. The best trained epoch had recurrent mean `0.0054544936` and failed
the teacher guard.

The retained `best.pt` is therefore the guarded epoch-0 checkpoint. It is 8.1%
lower than V1's retained step-2 recurrent error of `0.0051012548`.

## Step-4 rollout result

Initial evaluation from the guarded step-2 checkpoint:

```text
recurrent particle mean: 0.0065195928
recurrent persistence:   0.0070527965
model / persistence:     0.924398
teacher-forced mean:     0.00386395
teacher guard limit:     0.00425034
```

Fine-tuning began at `1e-5`, reduced to `5e-6`, and early-stopped after six
epochs without a guarded improvement. Every trained epoch failed the teacher
guard and regressed recurrence. The lowest trained recurrent error was
`0.0082210180` at epoch 3.

The retained `best.pt` is again the guarded epoch-0 checkpoint. Its recurrent
error is 9.4% lower than V1's canonical step-4 validation error of
`0.0071929494`, and it beats four-step persistence by 7.6%.

## Numerical issue found and fixed

The first rollout attempt exposed an infinite derivative from `sqrt(0)` in the
neighbour-stretch standard deviation at degree-1 particles. The descriptor now
uses `sqrt(variance + eps^2)`. An explicit two-step finite-gradient regression
test covers this recurrent-only failure mode.

## Retained lineage

```text
v2/runs/v2b_mps/one_step/best.pt
v2/runs/v2b_mps/rollout_s2_guarded/best.pt
v2/runs/v2b_mps/rollout_s4/best.pt
```

The three retained files contain identical model tensors because neither
rollout stage produced a guarded improvement. Their metadata and evaluation
horizons differ. Failed and diagnostic branches are retained separately under
`v2/runs/v2b_mps/` and are not canonical.
