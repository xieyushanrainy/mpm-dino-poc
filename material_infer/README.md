# Material inference from frozen DINO features

This folder implements two object-level probes on the balanced V4 package:

1. three-way `rigid` / `fluid` / `soft_body` classification;
2. `log10(E)` or Poisson-ratio regression on soft-body objects only.

Each UID is exactly one independent example. DINO features are masked, mean/max
pooled, standardized and PCA-projected. The scaler and PCA are fit on training
UIDs only. The official object-level V4 split remains unchanged.

The targets are VLM-assigned simulator priors, not measured physical constants.
The experiment tests whether those packaged labels are recoverable from DINO.

## Environment

Use the repository environment:

```bash
conda activate mpm-dino-poc
cd mpm-DINO-poc/material_infer
```

The default dataset is
`../v4/dataset/packaged_balanced_90_dinov2_dinov3`; override it with
`--dataset` if needed.

## Audit

```bash
python run_experiment.py audit --output outputs/audit.json
```

## Family classification

Linear and MLP probes:

```bash
python run_experiment.py run --task family --model linear --output outputs/family_linear_real
python run_experiment.py run --task family --model mlp --output outputs/family_mlp_real
```

Pass `--device mps` on Apple Silicon to train the probe on MPS.

Matched controls:

```bash
python run_experiment.py run --task family --model mlp --control scene_shuffled --output outputs/family_mlp_shuffled
python run_experiment.py run --task family --model mlp --control label_permuted --output outputs/family_mlp_permuted
python run_experiment.py run --task family --model linear --feature-source geometry --output outputs/family_geometry
python run_experiment.py run --task family --model linear --feature-source valid_fraction --pca-components 1 --output outputs/family_valid_fraction
```

## Soft-body regression

Use fewer PCA components for the 20-object training set:

```bash
python run_experiment.py run --task soft --target log10_E --model linear --pca-components 8 --output outputs/soft_E_linear_real
python run_experiment.py run --task soft --target nu --model linear --pca-components 8 --output outputs/soft_nu_linear_real
python run_experiment.py run --task soft --target log10_E --model mlp --pca-components 8 --hidden-dim 8 --output outputs/soft_E_mlp_real
python run_experiment.py run --task soft --target log10_E --model mlp --pca-components 8 --hidden-dim 8 --control scene_shuffled --output outputs/soft_E_mlp_shuffled
```

Every run writes a `summary.json`, per-seed metrics and predictions, loss
history, and a checkpoint containing both model and preprocessing state.

## Soft-object cross-validation

Six-fold cross-validation covers all 30 soft objects. Each fold uses 20 objects
for training, 5 for validation and 5 for testing; preprocessing is refit inside
each fold.

```bash
python run_experiment.py cross-validate --target log10_E --model mlp --device mps --control real --output runs/cv_E_real
python run_experiment.py cross-validate --target log10_E --model mlp --device mps --control scene_shuffled --output runs/cv_E_shuffled
```

## Tests

```bash
pytest -q tests
```
