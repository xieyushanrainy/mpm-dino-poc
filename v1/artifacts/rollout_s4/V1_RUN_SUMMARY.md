# Canonical V1 Run

This directory is the retained V1 result. See:

- `../../../docs/v1/CONCLUSION.md`
- `../../../docs/v1/TRAINING_HISTORY.md`
- `../../../docs/v1/ARTIFACTS.md`

Canonical checkpoint:

```text
best.pt
SHA-256: 7ebc021bec0002c8b8f1ad958b633362e11722bdfa0547b9285d865cf1b40be5
```

Configuration:

```text
rollout steps:             4
best epoch:                3
grid resolution:           32
U-Net base width:          24
batch size:                1
learning rate:             1e-5
particle Smooth-L1 beta:   0.01
rollout discount:          0.9
teacher loss weight:       0.5
motion oversampling:       2.0
teacher guardrail ratio:   1.1
```

Validation:

```text
recurrent particle mean:   0.0071929494
recurrent persistence:     0.0070527965
model/persistence ratio:   1.019872
teacher-forced mean:       0.0041393263
teacher guardrail:         passed
```

Later step-4 continuation did not improve this checkpoint.
