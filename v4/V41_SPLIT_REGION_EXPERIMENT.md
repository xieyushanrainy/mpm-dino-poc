# V4.1 Strict-Local-DINO Region-Token Experiment

Date prepared: 2026-07-27

## Scientific question

Does real DINO improve local deformation prediction when it is prevented from
influencing the global centre-of-mass path?

The primary comparison is real point-aligned DINO versus an
architecture-identical zero-DINO control. Existing Track B results are
contextual evidence only because the architecture and loss are not identical.

## Frozen architecture

The mechanism name is `split_region`.

- A DINO-free graph-temporal physical trunk processes motion, geometry,
  gravity, timing, and reference-graph features.
- The COM head reads only the physical hidden state. There is no computational
  path from DINO, DINO validity, region tokens, or the local adapter to COM.
- A reference-geometry-aware point encoder pools DINO into four learned region
  tokens with masked attention.
- A zero-initialized cross-attention adapter injects those tokens only into a
  copy of the physical state used by the local deformation head.
- The local displacement is explicitly mean-centred across valid points.
- Real and zero runs use the same model; zero mode changes only the DINO tensor.

This tests an object/region-level DINO signal. It does not establish useful
point correspondence without a later point-shuffled control.

## Frozen loss

The loss profile is `legacy_shape_aux_v1`:

```text
total = original Track B loss + 0.2 * shape-only auxiliary
```

The original Track B objective is unchanged, including its H4/H8/H16/H59
key-horizon term. The auxiliary does not duplicate world-position or COM loss:

```text
shape-only auxiliary =
    1.0 * radius-normalized centre-relative shape
  + 0.5 * normalized edge strain
  + 0.25 * shape error at H16/H30/H40/H59
```

H59 is retained as a training stability anchor but remains diagnostic-only for
promotion.

## Frozen training matrix

- DINO modes: `zero`, `real`
- Seeds: `42`, `123`, `456`
- Device: CUDA FP32, AMP disabled
- Maximum epochs: 100
- Early-stopping patience: 15
- ReduceLROnPlateau patience: 5
- UID-balanced draws per epoch: 40
- Width: 128
- Graph-temporal blocks: 4
- Attention heads: 4

For every seed, the complete real and zero models must have the same
`starting_model_sha256`. The runner stops if they differ. The real run uses the
matched zero best checkpoint only to define the unchanged H1 validation guard;
it does not load trained zero weights.

## Evaluation and promotion

Evaluate guarded `best.pt` checkpoints at H1, H8, H16, H30, H40, and H59,
keeping Panels Z and V separate.

Apply the existing V4.1 promotion rule:

1. Real beats matched zero in the three-seed mean at Panel Z H30 or H40.
2. Real wins at least two of the three paired seeds at that horizon.
3. Test H1 is no more than 10% worse than matched zero.

H59 cannot promote the mechanism. Do not run scene- or point-shuffled controls
unless the mechanism first promotes.

## Lab command

Run from the repository root in the prepared environment:

```bash
PYTHONUNBUFFERED=1 PYTHONPATH=v4/src:v3/src:v2/src python v41/run_track_b_split_region_matrix.py --device cuda --epochs 100 --patience 15 --draws 40 --no-amp --runs v41/runs/track_b_split_region_cap100_p15_fp32
```

The command is resumable. It refuses to mix changed matrix settings or a
changed manifest into the same output directory.
