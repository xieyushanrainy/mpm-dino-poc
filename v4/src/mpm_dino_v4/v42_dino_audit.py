from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .v41_data import V41TrajectoryDataset
from .v42_geometry_audit import (
    _field_metrics, _geometry_alignment, _record, _spearman,
)
from .v42_oracle import EVENT_STAGES


def _dino_pair(source, target, sign):
    global_cosine = F.cosine_similarity(
        source["pooled_dino"][None], target["pooled_dino"][None], dim=-1,
    )[0]
    source_valid = source["dino_valid"]
    target_valid = target["dino_valid"]
    source_points = source["points"][source_valid] * sign
    target_points = target["points"][target_valid]
    nearest = torch.cdist(target_points, source_points).argmin(1)
    aligned_cosine = F.cosine_similarity(
        source["dino"][source_valid][nearest],
        target["dino"][target_valid], dim=-1,
    )
    return {
        "global_pooled_cosine": float(global_cosine),
        "global_pooled_distance": float(1 - global_cosine),
        "aligned_point_cosine_mean": float(aligned_cosine.mean()),
        "aligned_point_cosine_median": float(aligned_cosine.median()),
        "aligned_point_distance": float(1 - aligned_cosine.mean()),
        "aligned_valid_target_points": int(target_valid.sum()),
    }


def dino_learnability_audit(
    dataset_root, manifest, output, device="cpu", nearest_k=3,
):
    device = torch.device(device)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    train_dataset = V41TrajectoryDataset(
        dataset_root, manifest, "train", "real", 42,
        families=("soft_body",),
    )
    validation_dataset = V41TrajectoryDataset(
        dataset_root, manifest, "validation", "real", 42,
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
    all_pairs, rows = [], []
    for target in validation:
        candidates = []
        for source in train:
            chamfer, sign = _geometry_alignment(source, target)
            dino = _dino_pair(source, target, sign)
            stage_metrics = {
                stage.name.lower(): _field_metrics(
                    source, target, sign, int(stage),
                ) for stage in EVENT_STAGES
            }
            usable = [value for value in stage_metrics.values() if value]
            error = float(np.mean([
                value["rescaled_normalized_mse"] for value in usable
            ])) if usable else None
            pair = {
                "validation_uid": target["uid"], "train_uid": source["uid"],
                "geometry_chamfer": chamfer, **dino,
                "mean_rescaled_normalized_mse": error,
                "stage_metrics": {
                    key: (
                        {k: v for k, v in value.items() if not torch.is_tensor(v)}
                        if value else None
                    ) for key, value in stage_metrics.items()
                },
                "train_material": source["material_raw"],
                "validation_material": target["material_raw"],
            }
            all_pairs.append(pair)
            candidates.append(pair)
        by_global = sorted(candidates, key=lambda p: p["global_pooled_distance"])
        by_aligned = sorted(candidates, key=lambda p: p["aligned_point_distance"])
        by_geometry = sorted(candidates, key=lambda p: p["geometry_chamfer"])
        by_field = sorted(candidates, key=lambda p: p["mean_rescaled_normalized_mse"])
        for ranking, key in (
            (by_global, "global_dino_rank"),
            (by_aligned, "aligned_dino_rank"),
            (by_geometry, "geometry_rank"),
        ):
            for rank, pair in enumerate(ranking, 1):
                pair[key] = rank
        rows.append({
            "validation_uid": target["uid"],
            "nearest_global_dino": by_global[0],
            "nearest_aligned_dino": by_aligned[0],
            "nearest_geometry": by_geometry[0],
            "oracle_best_field": by_field[0],
            "top_global_dino": by_global[:nearest_k],
            "top_aligned_dino": by_aligned[:nearest_k],
        })
    usable = [p for p in all_pairs if p["mean_rescaled_normalized_mse"] is not None]
    errors = [p["mean_rescaled_normalized_mse"] for p in usable]
    summary = {
        "train_soft_uids": len(train), "validation_soft_uids": len(validation),
        "pair_count": len(usable),
        "global_dino_distance_vs_field_error_spearman": _spearman(
            [p["global_pooled_distance"] for p in usable], errors,
        ),
        "aligned_dino_distance_vs_field_error_spearman": _spearman(
            [p["aligned_point_distance"] for p in usable], errors,
        ),
        "geometry_distance_vs_field_error_spearman": _spearman(
            [p["geometry_chamfer"] for p in usable], errors,
        ),
        "nearest_global_dino_mean_rescaled_mse": float(np.mean([
            row["nearest_global_dino"]["mean_rescaled_normalized_mse"]
            for row in rows
        ])),
        "nearest_aligned_dino_mean_rescaled_mse": float(np.mean([
            row["nearest_aligned_dino"]["mean_rescaled_normalized_mse"]
            for row in rows
        ])),
        "nearest_geometry_mean_rescaled_mse": float(np.mean([
            row["nearest_geometry"]["mean_rescaled_normalized_mse"]
            for row in rows
        ])),
        "oracle_best_field_mean_rescaled_mse": float(np.mean([
            row["oracle_best_field"]["mean_rescaled_normalized_mse"]
            for row in rows
        ])),
        "oracle_best_mean_global_dino_rank": float(np.mean([
            row["oracle_best_field"]["global_dino_rank"] for row in rows
        ])),
        "oracle_best_mean_aligned_dino_rank": float(np.mean([
            row["oracle_best_field"]["aligned_dino_rank"] for row in rows
        ])),
    }
    report = {
        "audit": "dino_similarity_vs_deformation_similarity_v1",
        "fit_split": "train", "evaluation_split": "validation",
        "test_used": False,
        "dino_model": "facebook/dinov2-small",
        "global_similarity": (
            "cosine of mean L2-normalized valid point features"
        ),
        "aligned_similarity": (
            "mean point-feature cosine after PCA/Chamfer geometry alignment "
            "and nearest-surface-point matching"
        ),
        "deformation_metric": (
            "stage-mean canonical field transfer; mean best-scalar-rescaled "
            "normalized MSE over contact/compression/peak"
        ),
        "summary": summary, "validation_rows": rows, "all_pairs": all_pairs,
    }
    path = output / "DINO_LEARNABILITY_AUDIT.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    _plot_dino_audit(rows, usable, output)
    return report


def _plot_dino_audit(rows, pairs, output):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for uid in [row["validation_uid"] for row in rows]:
        selected = [p for p in pairs if p["validation_uid"] == uid]
        axes[0].scatter(
            [p["global_pooled_distance"] for p in selected],
            [p["mean_rescaled_normalized_mse"] for p in selected],
            s=20, alpha=0.65, label=uid[:6],
        )
    axes[0].set_xlabel("Global pooled DINO cosine distance")
    axes[0].set_ylabel("Best-rescaled deformation error")
    axes[0].set_title("Does DINO similarity predict deformation similarity?")
    axes[0].legend(fontsize=7, ncol=2)
    labels = [row["validation_uid"][:6] for row in rows]
    x = np.arange(len(rows))
    width = 0.25
    axes[1].bar(x - width, [
        row["nearest_global_dino"]["mean_rescaled_normalized_mse"] for row in rows
    ], width, label="global DINO")
    axes[1].bar(x, [
        row["nearest_aligned_dino"]["mean_rescaled_normalized_mse"] for row in rows
    ], width, label="aligned-point DINO")
    axes[1].bar(x + width, [
        row["nearest_geometry"]["mean_rescaled_normalized_mse"] for row in rows
    ], width, label="geometry")
    axes[1].set_xticks(x, labels, rotation=30)
    axes[1].set_ylabel("Transferred deformation error")
    axes[1].set_title("Nearest neighbour selected by each representation")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "dino_learnability_summary.png", dpi=180)
    plt.close(fig)
