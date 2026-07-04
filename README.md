# DINO-conditioned particle-grid dynamics POC

The implementation follows [PLAN.md](PLAN.md). It is an MPM-inspired learned
surrogate: particles are the recurrent state, a DINO-conditioned 3D U-Net
predicts next-grid guidance, and a particle head predicts continuous motion.

## Implemented

- normalized trilinear particle-to-grid scatter and grid-to-particle gather;
- frozen-feature projection onto both particles and the grid;
- `32^3 -> 16^3 -> 8^3 -> 16^3 -> 32^3` 3D U-Net;
- supervised next-occupancy and next-velocity grid heads;
- grid-guided particle displacement head;
- PhysTwin `final_data.pkl` pair loader with deterministic point sampling and
  validity-masked targets;
- camera-0 DINOv3 extraction, reprojection, depth visibility, and
  nearest-visible imputation;
- one-step masked Huber training losses and checkpointing.

Rollout-loss fine-tuning, held-out split manifests, full evaluation, and 3D
visualization are the next implementation phase. One-step training should be
validated before recurrent training is enabled.

## Environment

Create the isolated Apple Silicon environment (TensorFlow is intentionally not
installed):

```bash
conda env create -f environment.yml
conda activate mpm-dino-poc
```

To rebuild it after dependency changes:

```bash
conda env remove -n mpm-dino-poc
conda env create -f environment.yml
```

Run commands from this directory and expose the package locally:

```bash
export PYTHONPATH="$PWD"
```

DINOv3 extraction uses Hugging Face Transformers and may require accepting the
model-weight licence or authenticating before download.

The environment pins NumPy 1.26 because older compiled scientific packages and
TensorFlow builds commonly fail when imported under NumPy 2. NumPy, SciPy,
PyTorch, and torchvision are all installed as pip wheels; do not replace NumPy
or SciPy with Conda OpenBLAS builds on macOS, because their LLVM OpenMP runtime
conflicts with PyTorch's bundled runtime. The extraction script also explicitly
disables Transformers' optional TensorFlow backend.
`torchvision==0.23.0` is paired with `torch==2.8.0`, as required by the fast
DINOv3 image processor.

The code selects devices in the order CUDA, MPS, CPU. The same environment file
is portable to Linux/CUDA, where pip resolves the Linux PyTorch wheel; for a
cluster with a mandated CUDA toolkit, install the matching PyTorch 2.8 wheel
from the cluster's PyTorch index and keep torchvision at 0.23.

The U-Net downsamples with stride-2 `Conv3d`, not `MaxPool3d` or `AvgPool3d`.
PyTorch 2.8 does not implement 3D pooling on MPS, whereas strided 3D convolution
and transposed 3D convolution run natively on both MPS and CUDA.

## Prepare a scene

Extract persistent camera-0 DINO features:

```bash
python extract_dino_features.py \
  ../coderepos/PhysTwin/data/different_types/single_lift_zebra \
  --output data/features/single_lift_zebra.npz
```

Build the compact training cache:

```bash
python prepare_scene.py \
  --final-data ../coderepos/PhysTwin/data/different_types/single_lift_zebra/final_data.pkl \
  --dino-features data/features/single_lift_zebra.npz \
  --output data/cache/single_lift_zebra.pt
```

The feature file must remain aligned with the original `final_data.pkl` track
order. `prepare_scene.py` then applies the same deterministic point subset to
tracks and features.

## One-step training

Prepare all 22 scenes (existing features/caches are reused):

```bash
python prepare_dataset.py
```

```bash
python train.py data/cache/*.pt --device mps --epochs 20
```

For the predefined scene-level split, start a multi-scene run:

```bash
python train.py @data/splits/poc_train.txt \
  --val-caches @data/splits/poc_val.txt \
  --device mps --epochs 20 --output runs/multiscene
```

Continue from the existing zebra checkpoint for 40 additional epochs. Zebra is
in the training split, so this continuation does not leak into the supplied
validation or test lists:

```bash
python train.py @data/splits/poc_train.txt \
  --val-caches @data/splits/poc_val.txt \
  --checkpoint runs/one_step/last.pt \
  --device mps --batch-size 2 --epochs 5 --lr 1e-4 \
  --output runs/multiscene_resume
```

Continuation restores model, optimizer, and (when present) scheduler state.
New runs save `last.pt`, validation-selected `best.pt`, and `history.jsonl`.
Five multi-scene epochs contain far more optimizer updates than five epochs of
the original single-scene run; inspect validation history before requesting a
large epoch count. Evaluate the held-out split with:

```bash
python evaluate.py runs/multiscene_resume/best.pt @data/splits/poc_test.txt
```

### Particle-focused continuation and rollout gate

The current trainer uses normalized Smooth-L1 (`particle_beta=0.01`), selects
`best.pt` by validation particle mean distance, schedules LR on that same
metric, and reports the ratio to validation persistence. When resuming an older
composite-loss checkpoint it loads weights only and resets Adam/scheduler,
because the particle-gradient scale changed.

```bash
python train.py @data/splits/poc_train.txt \
  --val-caches @data/splits/poc_val.txt \
  --checkpoint runs/multiscene_resume/best.pt \
  --device mps --batch-size 2 --epochs 5 --lr 1e-4 \
  --particle-beta 0.01 \
  --lr-patience 3 --lr-factor 0.5 --min-lr 1e-6 \
  --early-stop-patience 8 --min-relative-improvement 0.005 \
  --rollout-gate-ratio 0.9 \
  --output runs/multiscene_particle
```

`rollout_ready=True` requires validation particle error to be at most 90% of
persistence. Do not use the test split to schedule, stop, or open this gate.

For a low-memory smoke run:

```bash
python train.py data/cache/single_lift_zebra.pt \
  --device mps --resolution 16 --base 8 --epochs 1
```

The default scientific configuration remains resolution 32 and base width 24.

Evaluate a checkpoint against persistence and constant-velocity baselines:

```bash
python evaluate.py runs/one_step/last.pt data/cache/*.pt --device mps
```

Render ground truth, recurrent rollout, and teacher-forced predictions:

```bash
python visualize_rollout.py \
  runs/one_step/last.pt \
  data/cache/single_lift_zebra.pt
```

This writes `rollout_visualization.mp4` and frame-level metric CSV beside the
checkpoint. Use `--max-frames 8` for a quick smoke render.

## Tests

```bash
pytest -q
```

Tests cover transfer conservation/shape behavior, DINO-to-U-Net gradients,
output contracts, and finite one-step losses.
