# How to generate another soft-body batch

Run commands from:

```bash
cd /Users/yushanxie/workspace/cgvi/term3/coderepos/DINO_VLM_sim_dataset_gen
source ~/.zshrc
conda activate dinopocdataset
```

Use new directory names for every batch so an interrupted run can be resumed.

## 1. Select and download candidates

Aim for substantially more candidates than the required final count. The run that produced
30 accepted samples needed about 150 candidates because rigid, mixed-material, low-confidence,
and failed assets were rejected.

```bash
python sample_objaverse_soft_selection.py \
  --selection-dir selection_soft_NEW \
  --data-dir data \
  --target-count 100 \
  --candidate-pool-size 1000 \
  --max-uid-scan 50000 \
  --minimum-score 5 \
  --seed 1234 \
  --exclude-path /Users/yushanxie/workspace/cgvi/term3/mpm-DINO-poc/v4/dataset/packaged_dataset_1k \
  --exclude-path /Users/yushanxie/workspace/cgvi/term3/mpm-DINO-poc/v4/dataset/packaged_soft_30
```

If an Objaverse batch download stalls, stop it and resume missing UIDs individually:

```bash
python download_missing_selection.py \
  --selection-dir selection_soft_NEW \
  --data-dir data \
  --timeout 90
```

## 2. Render four views

```bash
python rendering/render_selected_pbr.py \
  --selection-dir selection_soft_NEW \
  --output-dir soft_render_NEW \
  --data-dir data \
  --blender /Applications/Blender.app/Contents/MacOS/Blender \
  --resolution 2048 \
  --samples 0 \
  --device CPU \
  --min-free-gpu-mb 0
```

The renderer is resumable and skips complete objects.

## 3. Run DINO, OpenAI material screening, and soft-body simulation

`--limit` is the number of accepted simulations required. `--candidate-limit` is the larger
number to screen. The action configuration below applies gravity only and no control force.

```bash
python run_local_end_to_end.py \
  --selection-dir selection_soft_NEW \
  --render-root soft_render_NEW \
  --output-dir soft_dataset_NEW \
  --data-dir data \
  --blender /Applications/Blender.app/Contents/MacOS/Blender \
  --limit 30 \
  --candidate-limit 100 \
  --points-per-body 2048 \
  --duration 2 \
  --simulation-fps 30 \
  --save-fps 30 \
  --vlm-backend openai \
  --vlm-model gpt-5.4-mini-2026-03-17 \
  --openai-workers 4 \
  --openai-detail low \
  --openai-reasoning-effort low \
  --openai-max-output-tokens 5000 \
  --device auto \
  --real-simulation-routes soft_body \
  --min-vlm-confidence 0.4 \
  --visualize-trajectories
```

Rerun the same command to resume. Never lower the confidence or homogeneous-route checks just
to reach the target; instead add more candidates. Complex multi-part assets may require a
larger `--openai-max-output-tokens` value.

## 4. Package and then prune GLBs

Do a first packaging run without `--prune-source-glbs` if you want to inspect the package.
Add that flag only after you are ready to delete the packaged source GLBs. Multiple resumable
runs can be combined by repeating `--dataset-dir` and `--selection-dir`.

```bash
python package_v4_exchange_dataset.py \
  --dataset-dir soft_dataset_NEW \
  --selection-dir selection_soft_NEW \
  --output-dir /Users/yushanxie/workspace/cgvi/term3/mpm-DINO-poc/v4/dataset/packaged_soft_NEW \
  --glb-root data/hf-objaverse-v1/glbs \
  --limit 30 \
  --required-route soft_body \
  --points-per-object 2048 \
  --dino-dimension 384 \
  --min-vlm-confidence 0.4 \
  --overwrite \
  --prune-source-glbs
```

Success means the package has `PACKAGE_COMPLETE`, `dataset.json`, 30 object directories, and
`PRUNED_SOURCE_GLBS.json`. Each `sample.npz` should contain exactly:

- `trajectory_positions_m` with shape `(61, 2048, 3)`
- `dino_features`
- `dino_valid`
- `point_material_ids`
- `point_active`

Check that every metadata file reports `solver_route: soft_body` and an empty `force_fields`
list. Treat VLM material values as visual priors, not measured physical ground truth.
