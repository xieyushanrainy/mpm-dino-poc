# Frozen V1 Implementation

V1 is concluded and should not be extended for V2. Its findings and lineage are
documented in:

- `../docs/v1/CONCLUSION.md`
- `../docs/v1/TRAINING_HISTORY.md`
- `../docs/v1/ARTIFACTS.md`

Canonical checkpoint:

```text
v1/artifacts/rollout_s4/best.pt
```

Run all commands from the repository root:

```bash
conda activate mpm-dino-poc
export PYTHONPATH="$PWD/v1/src"
pytest -q v1/tests
```

Representative commands:

```bash
python v1/scripts/extract_dino_features.py \
  ../coderepos/PhysTwin/data/different_types/single_lift_zebra \
  --output data/shared/dino/single_lift_zebra.npz

python v1/scripts/prepare_scene.py \
  --final-data ../coderepos/PhysTwin/data/different_types/single_lift_zebra/final_data.pkl \
  --dino-features data/shared/dino/single_lift_zebra.npz \
  --output data/v1/cache/single_lift_zebra.pt

python v1/scripts/evaluate.py \
  v1/artifacts/rollout_s4/best.pt \
  @data/shared/splits/poc_test.txt

python v1/scripts/visualize_rollout.py \
  v1/artifacts/rollout_s4/best.pt \
  data/v1/cache/double_stretch_zebra.pt \
  --output v1/artifacts/rollout_s4/zebra_recheck.mp4
```

The environment is defined by root `environment.yml`. Frozen V1 caches are in
`data/v1/cache`, shared DINO features and splits are in `data/shared`, and
historical non-canonical runs are under `archive/v1_runs/`.
