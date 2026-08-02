from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, default_collate

from .v41_data import V41TrajectoryDataset
from .v41_train import move
from .v42_gate2 import _batch_targets_and_stages
from .v42_oracle import EVENT_STAGES, material_vector
from .v42_stages import ImpactStage


def _pca_frame(points):
    covariance = points.T @ points / max(len(points), 1)
    values, vectors = torch.linalg.eigh(covariance)
    order = torch.argsort(values, descending=True)
    values, vectors = values[order], vectors[:, order]
    if torch.linalg.det(vectors) < 0:
        vectors[:, -1] *= -1
    return values, vectors


def _descriptor(points, eigenvalues):
    radius = torch.linalg.vector_norm(points, dim=-1)
    quantiles = torch.linspace(0.05, 0.95, 19, device=points.device)
    radial = torch.quantile(radius, quantiles)
    projected = points.abs()
    axis_quantiles = torch.cat([
        torch.quantile(projected[:, axis], quantiles[::3])
        for axis in range(3)
    ])
    return torch.cat((eigenvalues, radial, axis_quantiles))


def _stage_field(targets, stages, target_mask, stage, point_mask):
    selected = stages.labels[0, 1:].eq(int(stage))
    valid = target_mask[0] & targets.valid_rotation[0, :, None] & selected[:, None]
    count = valid.sum(0)
    field = (
        targets.displacement[0] * valid[..., None]
    ).sum(0) / count.clamp_min(1)[..., None]
    available = point_mask & count.gt(0)
    return field, available, int(selected.sum())


def _record(dataset, index, device, dataset_root):
    batch = move(default_collate([dataset[index]]), device)
    targets, stages = _batch_targets_and_stages(batch)
    mask = batch["input_mask"][0]
    normalized = targets.reference_shape[0] / targets.radius[0]
    points = normalized[mask]
    eigenvalues, frame = _pca_frame(points)
    canonical_points = points @ frame
    fields = {}
    for stage in EVENT_STAGES:
        field, available, frames = _stage_field(
            targets, stages, batch["target_mask"], stage, mask,
        )
        fields[int(stage)] = {
            "field": (field[mask] / targets.radius[0]) @ frame,
            "available": available[mask],
            "frames": frames,
        }
    uid = batch["uid"][0]
    metadata_path = Path(dataset_root) / "objects" / uid / "source_metadata.json"
    metadata = json.loads(metadata_path.read_text())
    dino = batch["dino"][0, mask]
    dino_valid = batch["dino_valid"][0, mask]
    normalized_dino = F.normalize(dino, dim=-1)
    pooled_dino = normalized_dino[dino_valid].mean(0)
    pooled_dino = F.normalize(pooled_dino, dim=0)
    return {
        "uid": uid, "episode_id": batch["episode_id"][0],
        "points": canonical_points, "eigenvalues": eigenvalues,
        "descriptor": _descriptor(points, eigenvalues),
        "frame": frame, "fields": fields,
        "dino": normalized_dino, "dino_valid": dino_valid,
        "pooled_dino": pooled_dino,
        "material": material_vector(metadata).to(device),
        "material_raw": metadata.get("simulation", {}).get(
            "body_parameters", [{}]
        )[0],
    }


def _subsample(points, maximum=256):
    if len(points) <= maximum:
        return points
    indices = torch.linspace(
        0, len(points) - 1, maximum, device=points.device,
    ).round().long()
    return points[indices]


PROPER_SIGNS = (
    (1.0, 1.0, 1.0), (1.0, -1.0, -1.0),
    (-1.0, 1.0, -1.0), (-1.0, -1.0, 1.0),
)


def _geometry_alignment(source, target):
    target_small = _subsample(target["points"])
    best = None
    for values in PROPER_SIGNS:
        sign = source["points"].new_tensor(values)
        source_small = _subsample(source["points"] * sign)
        distances = torch.cdist(source_small, target_small)
        chamfer = distances.min(1).values.square().mean() + (
            distances.min(0).values.square().mean()
        )
        candidate = (float(chamfer), sign)
        if best is None or candidate[0] < best[0]:
            best = candidate
    return best


def _field_metrics(source, target, sign, stage):
    source_stage, target_stage = source["fields"][stage], target["fields"][stage]
    source_valid = source_stage["available"]
    target_valid = target_stage["available"]
    if not source_valid.any() or not target_valid.any():
        return None
    source_points = source["points"][source_valid] * sign
    target_points = target["points"][target_valid]
    nearest = torch.cdist(target_points, source_points).argmin(1)
    predicted = source_stage["field"][source_valid][nearest] * sign
    truth = target_stage["field"][target_valid]
    target_energy = truth.square().sum().clamp_min(1e-12)
    raw_mse = (predicted - truth).square().sum() / target_energy
    scale = (predicted * truth).sum() / predicted.square().sum().clamp_min(1e-12)
    scaled = predicted * scale
    scaled_mse = (scaled - truth).square().sum() / target_energy
    cosine = F.cosine_similarity(predicted, truth, dim=-1)
    return {
        "raw_normalized_mse": float(raw_mse),
        "best_scalar": float(scale),
        "rescaled_normalized_mse": float(scaled_mse),
        "point_vector_cosine": float(cosine.mean()),
        "positive_direction_fraction": float(cosine.gt(0).float().mean()),
        "raw_predicted_to_target_rms": float(torch.sqrt(
            predicted.square().mean() / truth.square().mean().clamp_min(1e-12)
        )),
        "target_points": target_points.detach().cpu(),
        "truth": truth.detach().cpu(),
        "predicted": predicted.detach().cpu(),
    }


def _rank(values):
    order = np.argsort(values)
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.arange(len(values), dtype=float)
    return ranks


def _spearman(x, y):
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return None
    return float(np.corrcoef(_rank(np.asarray(x)), _rank(np.asarray(y)))[0, 1])


def geometry_learnability_audit(
    dataset_root, manifest, output, device="cpu", nearest_k=3,
):
    device = torch.device(device)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    train_dataset = V41TrajectoryDataset(
        dataset_root, manifest, "train", "zero", 42,
        families=("soft_body",),
    )
    validation_dataset = V41TrajectoryDataset(
        dataset_root, manifest, "validation", "zero", 42,
        families=("soft_body",),
    )
    train = [
        _record(train_dataset, index, device, dataset_root)
        for index in range(len(train_dataset))
    ]
    validation = [
        _record(validation_dataset, index, device, dataset_root)
        for index in range(len(validation_dataset))
    ]
    pairs, validation_rows = [], []
    plot_payload = []
    for target in validation:
        candidates = []
        for source in train:
            chamfer, sign = _geometry_alignment(source, target)
            descriptor_distance = float(torch.linalg.vector_norm(
                source["descriptor"] - target["descriptor"]
            ))
            material_distance = float(torch.linalg.vector_norm(
                source["material"] - target["material"]
            ))
            stage_metrics = {
                stage.name.lower(): _field_metrics(
                    source, target, sign, int(stage),
                ) for stage in EVENT_STAGES
            }
            usable = [value for value in stage_metrics.values() if value]
            mean_rescaled = float(np.mean([
                value["rescaled_normalized_mse"] for value in usable
            ])) if usable else None
            pair = {
                "validation_uid": target["uid"], "train_uid": source["uid"],
                "geometry_chamfer": chamfer,
                "descriptor_distance": descriptor_distance,
                "material_distance": material_distance,
                "mean_rescaled_normalized_mse": mean_rescaled,
                "stage_metrics": {
                    key: (
                        {k: v for k, v in value.items() if not torch.is_tensor(v)}
                        if value else None
                    ) for key, value in stage_metrics.items()
                },
                "train_material": source["material_raw"],
                "validation_material": target["material_raw"],
            }
            pairs.append(pair)
            candidates.append((pair, stage_metrics))
        candidates.sort(key=lambda item: item[0]["geometry_chamfer"])
        for rank, (pair, _) in enumerate(candidates, 1):
            pair["geometry_rank_for_validation"] = rank
        best_field = min(
            candidates,
            key=lambda item: (
                float("inf") if item[0]["mean_rescaled_normalized_mse"] is None
                else item[0]["mean_rescaled_normalized_mse"]
            ),
        )
        nearest = candidates[0]
        validation_rows.append({
            "validation_uid": target["uid"],
            "nearest_geometry": nearest[0],
            "top_geometry_neighbours": [item[0] for item in candidates[:nearest_k]],
            "best_field_match": best_field[0],
            "best_field_geometry_rank": best_field[0]["geometry_rank_for_validation"],
        })
        peak = ImpactStage.PEAK_DEFORMATION.name.lower()
        plot_payload.append({
            "validation_uid": target["uid"],
            "train_uid": nearest[0]["train_uid"],
            "geometry_chamfer": nearest[0]["geometry_chamfer"],
            "metrics": nearest[1].get(peak),
        })
    geometry, material, errors = [], [], []
    for pair in pairs:
        if pair["mean_rescaled_normalized_mse"] is not None:
            geometry.append(pair["geometry_chamfer"])
            material.append(pair["material_distance"])
            errors.append(pair["mean_rescaled_normalized_mse"])
    summary = {
        "train_soft_uids": len(train), "validation_soft_uids": len(validation),
        "pair_count": len(pairs),
        "geometry_distance_vs_field_error_spearman": _spearman(geometry, errors),
        "material_distance_vs_field_error_spearman": _spearman(material, errors),
        "nearest_geometry_mean_rescaled_mse": float(np.mean([
            row["nearest_geometry"]["mean_rescaled_normalized_mse"]
            for row in validation_rows
        ])),
        "oracle_best_train_mean_rescaled_mse": float(np.mean([
            row["best_field_match"]["mean_rescaled_normalized_mse"]
            for row in validation_rows
        ])),
        "oracle_best_mean_geometry_rank": float(np.mean([
            row["best_field_geometry_rank"] for row in validation_rows
        ])),
    }
    report = {
        "audit": "geometry_learnability_nearest_training_fields_v1",
        "fit_split": "train", "evaluation_split": "validation",
        "test_used": False,
        "geometry_alignment": (
            "radius normalization, PCA axes, best proper sign, symmetric chamfer"
        ),
        "field_transfer": (
            "nearest source point in aligned normalized coordinates; stage-mean "
            "canonical displacement; best scalar reported separately"
        ),
        "summary": summary, "validation_rows": validation_rows,
        "all_pairs": pairs,
    }
    json_path = output / "GEOMETRY_LEARNABILITY_AUDIT.json"
    json_path.write_text(json.dumps(report, indent=2) + "\n")
    _plot_audit(validation_rows, pairs, plot_payload, output)
    return report


def _plot_audit(validation_rows, pairs, plot_payload, output):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    uids = [row["validation_uid"] for row in validation_rows]
    for uid in uids:
        selected = [pair for pair in pairs if pair["validation_uid"] == uid]
        axes[0].scatter(
            [pair["geometry_chamfer"] for pair in selected],
            [pair["mean_rescaled_normalized_mse"] for pair in selected],
            s=18, alpha=0.65, label=uid[:6],
        )
    axes[0].set_xlabel("Aligned geometry Chamfer (lower = more similar)")
    axes[0].set_ylabel("Best-rescaled deformation error (lower = better)")
    axes[0].set_title("Does geometry similarity predict deformation similarity?")
    axes[0].legend(fontsize=7, ncol=2)
    nearest_errors = [
        row["nearest_geometry"]["mean_rescaled_normalized_mse"]
        for row in validation_rows
    ]
    best_errors = [
        row["best_field_match"]["mean_rescaled_normalized_mse"]
        for row in validation_rows
    ]
    x = np.arange(len(uids))
    axes[1].bar(x - 0.18, nearest_errors, 0.36, label="nearest geometry")
    axes[1].bar(x + 0.18, best_errors, 0.36, label="best train field (oracle)")
    axes[1].set_xticks(x, [uid[:6] for uid in uids], rotation=30)
    axes[1].set_ylabel("Best-rescaled deformation error")
    axes[1].set_title("Geometry neighbour versus attainable train match")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "geometry_learnability_summary.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(len(plot_payload), 3, figsize=(11, 3 * len(plot_payload)))
    if len(plot_payload) == 1:
        axes = axes[None]
    for row, payload in enumerate(plot_payload):
        metrics = payload["metrics"]
        if metrics is None:
            continue
        points = metrics["target_points"].numpy()
        truth = metrics["truth"].numpy()
        predicted = metrics["predicted"].numpy() * metrics["best_scalar"]
        take = np.linspace(0, len(points) - 1, min(180, len(points))).astype(int)
        for column, (field, title) in enumerate((
            (truth, "validation target"),
            (predicted, "nearest-geometry transferred field"),
            (predicted - truth, "residual"),
        )):
            ax = axes[row, column]
            ax.scatter(points[take, 0], points[take, 2], s=3, alpha=0.25)
            ax.quiver(
                points[take, 0], points[take, 2],
                field[take, 0], field[take, 2],
                angles="xy", scale_units="xy", scale=1, width=0.003,
            )
            ax.set_aspect("equal")
            ax.set_title(title if row == 0 else "")
            ax.set_ylabel(
                f"{payload['validation_uid'][:6]} <- {payload['train_uid'][:6]}"
                if column == 0 else ""
            )
    fig.tight_layout()
    fig.savefig(output / "peak_field_nearest_geometry.png", dpi=180)
    plt.close(fig)
