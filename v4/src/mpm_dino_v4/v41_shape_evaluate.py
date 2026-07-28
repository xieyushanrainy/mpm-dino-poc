from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from mpm_dino_v2.deformation import edge_validity, gather_neighbours
from .model import masked_mean
from .v41_data import MODEL_INPUT_KEYS, V41TrajectoryDataset
from .v41_model import build_v41_model
from .v41_train import move


def evaluate_local_shape(
    root, manifest, checkpoint, output, split="test", device="cpu",
):
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = state["config"]
    model = build_v41_model(
        config["mechanism"],
        hidden_dim=config["hidden_dim"],
        blocks=config["blocks"],
        heads=config["heads"],
        dropout=config["dropout"],
    ).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    dataset = V41TrajectoryDataset(
        root, manifest, split, config["dino_mode"], config["seed"],
        families=("soft_body", "rigid"),
    )
    rows = []
    with torch.no_grad():
        for raw in DataLoader(dataset, batch_size=1, shuffle=False):
            batch = move(raw, torch.device(device))
            prediction = model(**{
                key: batch[key] for key in MODEL_INPUT_KEYS
            })
            target_com = masked_mean(
                batch["target"], batch["target_mask"], dim=2,
            )
            target_shape = batch["target"] - target_com[:, :, None]
            ballistic_com = masked_mean(
                prediction.ballistic, batch["target_mask"], dim=2,
            )
            predicted_shape = (
                prediction.ballistic - ballistic_com[:, :, None]
                + prediction.residual_local
            )
            predicted_shape = (
                predicted_shape
                - masked_mean(
                    predicted_shape, batch["target_mask"], dim=2,
                )[:, :, None]
            )
            reference_com = masked_mean(
                batch["reference"], batch["input_mask"],
            )
            radius = torch.linalg.vector_norm(
                batch["reference"] - reference_com[:, None], dim=-1,
            ).masked_fill(
                ~batch["input_mask"], 0,
            ).amax(1).clamp_min(1e-6)
            for horizon in (1, 8, 16, 30, 40, 59):
                index = horizon - 1
                mask = batch["target_mask"][:, index]
                squared = (
                    predicted_shape[:, index] - target_shape[:, index]
                ).square().sum(-1)
                shape_rmse = torch.sqrt(
                    (squared * mask).sum(1) / mask.sum(1).clamp_min(1)
                )
                indices = batch["neighbour_indices"]
                valid_edges = edge_validity(
                    mask, indices, batch["neighbour_mask"],
                )
                pred_vectors = (
                    gather_neighbours(
                        predicted_shape[:, index], indices,
                    ) - predicted_shape[:, index, :, None]
                )
                target_vectors = (
                    gather_neighbours(target_shape[:, index], indices)
                    - target_shape[:, index, :, None]
                )
                vector_squared = (
                    pred_vectors - target_vectors
                ).square().sum(-1)
                edge_vector_rmse = torch.sqrt(
                    (vector_squared * valid_edges).sum(1).sum(1)
                    / valid_edges.sum(1).sum(1).clamp_min(1)
                )
                rest = batch["rest_edge_lengths"].clamp_min(1e-8)
                pred_strain = (
                    torch.linalg.vector_norm(pred_vectors, dim=-1) - rest
                ) / rest
                target_strain = (
                    torch.linalg.vector_norm(target_vectors, dim=-1) - rest
                ) / rest
                strain_squared = (pred_strain - target_strain).square()
                strain_rmse = torch.sqrt(
                    (strain_squared * valid_edges).sum(1).sum(1)
                    / valid_edges.sum(1).sum(1).clamp_min(1)
                )
                local_rms = torch.sqrt(
                    (
                        prediction.residual_local[:, index].square().sum(-1)
                        * mask
                    ).sum(1) / mask.sum(1).clamp_min(1)
                )
                rows.append({
                    "uid": batch["uid"][0],
                    "episode_id": batch["episode_id"][0],
                    "family": batch["family"][0],
                    "panel": batch["panel"][0],
                    "horizon": horizon,
                    "shape_rmse_m": float(shape_rmse.cpu()),
                    "shape_nrmse": float((shape_rmse / radius).cpu()),
                    "edge_vector_rmse_m": float(edge_vector_rmse.cpu()),
                    "strain_rmse": float(strain_rmse.cpu()),
                    "predicted_local_rms_m": float(local_rms.cpu()),
                })
    summary = {}
    for family in ("soft_body", "rigid"):
        for horizon in (1, 8, 16, 30, 40, 59):
            selected = [
                row for row in rows
                if row["family"] == family and row["horizon"] == horizon
            ]
            by_uid = defaultdict(list)
            for row in selected:
                by_uid[row["uid"]].append(row)
            metrics = {}
            for key in (
                "shape_rmse_m", "shape_nrmse", "edge_vector_rmse_m",
                "strain_rmse", "predicted_local_rms_m",
            ):
                metrics[key] = float(np.mean([
                    np.mean([row[key] for row in values])
                    for values in by_uid.values()
                ]))
            summary[f"{family}/H{horizon}"] = {
                "uids": len(by_uid),
                "episodes": len(selected),
                **metrics,
            }
    payload = {
        "checkpoint": str(Path(checkpoint).resolve()),
        "config": config,
        "split": split,
        "summary": summary,
        "object_rows": rows,
    }
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(
        json.dumps(payload, indent=2, allow_nan=True) + "\n"
    )
    return payload
