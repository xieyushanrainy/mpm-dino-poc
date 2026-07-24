from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .evaluate import summarize
from .full_data import FullTrajectoryDataset, ballistic_trajectory
from .full_train import FULL_MODEL_KEYS, load_full_model, move
from .metrics import metric_values


def _trajectory_rows(batch, prediction):
    rows = []
    frames = prediction.shape[1]
    for index in range(frames):
        target = batch["target"][:, index]
        mask = batch["target_mask"][:, index]
        values = metric_values(
            prediction[:, index],
            target,
            mask,
            batch["x0"],
            batch["neighbour_indices"],
            batch["neighbour_mask"],
            batch["floor_z"],
            list(batch["family"]),
        )
        for object_index, uid in enumerate(batch["uid"]):
            row = {
                "uid": uid,
                "family": batch["family"][object_index],
                "start_t": 0,
                "horizon": index + 1,
                "active_points": int(mask[object_index].sum().detach().cpu()),
            }
            for key, value in values.items():
                row[key] = float(value[object_index].detach().cpu())
            row["cv_rmse_m"] = float("nan")
            row["cv_relative_improvement"] = float("nan")
            rows.append(row)
    return rows


def evaluate_full(
    cache,
    manifest,
    split,
    dino_mode,
    seed,
    output,
    checkpoint_path=None,
    baseline=None,
    device="mps",
    families=("rigid", "soft_body"),
):
    if (checkpoint_path is None) == (baseline is None):
        raise ValueError("provide exactly one of checkpoint_path or baseline")
    dataset = FullTrajectoryDataset(cache, manifest, split, families, dino_mode, seed)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    model = None
    if checkpoint_path:
        model, config = load_full_model(checkpoint_path, device)
        if config["dino_mode"] != dino_mode or int(config["seed"]) != seed:
            raise ValueError("evaluation mode and seed must match checkpoint")
        model.eval()
    rows = []
    target_device = torch.device(device)
    for raw in loader:
        batch = move(raw, target_device)
        if model is not None:
            with torch.no_grad():
                prediction = model(
                    **{key: batch[key] for key in FULL_MODEL_KEYS}
                ).position
        elif baseline == "ballistic":
            prediction = ballistic_trajectory(
                batch["x0"], batch["x1"], batch["gravity"], batch["dt"], 59
            )
        elif baseline == "constant_velocity":
            h = torch.arange(
                1, 60, device=target_device, dtype=batch["x0"].dtype
            )[None, :, None, None]
            prediction = batch["x1"][:, None] + h * (
                batch["x1"] - batch["x0"]
            )[:, None]
        else:
            raise ValueError(f"unsupported full-trajectory baseline: {baseline}")
        rows.extend(_trajectory_rows(batch, prediction))
    payload = {
        "model_family": "v4_full_trajectory",
        "checkpoint": str(checkpoint_path) if checkpoint_path else None,
        "baseline": baseline,
        "dino_mode": dino_mode,
        "seed": seed,
        "split": split,
        "families": list(families),
        "summary": summarize(rows),
        "object_rows": rows,
    }
    Path(output).write_text(json.dumps(payload, indent=2, allow_nan=True) + "\n")
    return payload


def compare_tracks(track_b_paths, track_a_paths, baseline_paths, output):
    track_b = [json.loads(Path(path).read_text()) for path in track_b_paths]
    track_a_payloads = [json.loads(Path(path).read_text()) for path in track_a_paths]
    baselines = [json.loads(Path(path).read_text()) for path in baseline_paths]
    track_a = []
    for payload in track_a_payloads:
        rows = [row for row in payload["rollout_rows"] if int(row["start_t"]) == 0]
        track_a.append(
            {
                "dino_mode": payload["dino_mode"],
                "seed": payload["seed"],
                "summary": summarize(rows),
            }
        )

    result = {"track_b": {}, "track_a_frame0": {}, "baselines": {}}
    for destination, payloads, summary_key in (
        (result["track_b"], track_b, "summary"),
        (result["track_a_frame0"], track_a, "summary"),
    ):
        for mode in sorted({payload["dino_mode"] for payload in payloads}):
            selected = [payload for payload in payloads if payload["dino_mode"] == mode]
            destination[mode] = {}
            for family in ("aggregate", "rigid", "soft_body"):
                for horizon in (1, 4, 8, 16, 59):
                    key = f"object_weighted/{family}/H{horizon}"
                    values = [
                        payload[summary_key][key]["rmse_m"]
                        for payload in selected
                        if key in payload[summary_key]
                    ]
                    if values:
                        destination[mode][f"{family}/H{horizon}"] = {
                            "mean_rmse_m": float(np.mean(values)),
                            "std_rmse_m": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                            "runs": len(values),
                        }
    for payload in baselines:
        name = payload["baseline"]
        result["baselines"][name] = {}
        for family in ("aggregate", "rigid", "soft_body"):
            for horizon in (1, 4, 8, 16, 59):
                key = f"object_weighted/{family}/H{horizon}"
                if key in payload["summary"]:
                    result["baselines"][name][f"{family}/H{horizon}"] = payload["summary"][key]
    Path(output).write_text(json.dumps(result, indent=2, allow_nan=True) + "\n")
    return result
