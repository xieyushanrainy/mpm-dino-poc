#!/usr/bin/env python3
"""Train-to-validation geometry versus ground-truth rotation audit."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "v4" / "src"))

from mpm_dino_v4.v42_geometry_audit import _geometry_alignment, _pca_frame  # noqa: E402
from mpm_dino_v4.v42_rotation_audit import (  # noqa: E402
    geodesic_error, proper_kabsch,
)


def ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    result = np.empty(len(values), dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        result[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return result


def spearman(left, right):
    left, right = np.asarray(left), np.asarray(right)
    if len(left) < 3 or left.std() == 0 or right.std() == 0:
        return None
    return float(np.corrcoef(ranks(left), ranks(right))[0, 1])


def load_record(dataset_root: Path, row: dict) -> dict:
    with np.load(dataset_root / row["trajectory"], allow_pickle=False) as data:
        positions = data["trajectory_positions_m"].astype(np.float64)
        active = data["point_active"].astype(bool)
    reference = positions[1]
    reference_valid = active[0] & active[1]
    centered = reference[reference_valid] - reference[reference_valid].mean(0)
    radius = max(float(np.linalg.norm(centered, axis=1).max()), 1e-12)
    points = torch.from_numpy(centered / radius).double()
    _, frame_t = _pca_frame(points)
    frame = frame_t.numpy()
    canonical_points = points @ frame_t
    with np.load(dataset_root / row["object_static"], allow_pickle=False) as data:
        dino = torch.from_numpy(data["dino_features"].astype(np.float32))[reference_valid]
        dino_valid = torch.from_numpy(data["dino_valid"].astype(bool))[reference_valid]
    dino = F.normalize(dino, dim=-1)
    rotations, rotation_valid = [], []
    minimum_ratio = 1.0
    for frame_index in range(2, len(positions)):
        valid = reference_valid & active[frame_index]
        rotation, _, ratio = proper_kabsch(reference, positions[frame_index], valid)
        rotations.append(frame.T @ rotation @ frame)
        rotation_valid.append(ratio >= 1e-3)
        minimum_ratio = min(minimum_ratio, ratio)
    return {
        "uid": row["uid"], "family": row["family"],
        "points": canonical_points, "rotations": np.asarray(rotations),
        "rotation_valid": np.asarray(rotation_valid),
        "dino": dino, "dino_valid": dino_valid,
        "minimum_kabsch_ratio": minimum_ratio,
    }


def rotation_distance(source: dict, target: dict, sign: torch.Tensor) -> tuple[float, int]:
    transform = np.diag(sign.cpu().numpy())
    source_rotation = transform @ source["rotations"] @ transform
    valid = source["rotation_valid"] & target["rotation_valid"]
    distances = [
        geodesic_error(source_rotation[index], target["rotations"][index])
        for index in np.flatnonzero(valid)
    ]
    return float(np.mean(distances)), len(distances)


def aligned_dino_distance(source: dict, target: dict, sign: torch.Tensor) -> float:
    source_valid, target_valid = source["dino_valid"], target["dino_valid"]
    source_points = source["points"][source_valid] * sign
    target_points = target["points"][target_valid]
    nearest = torch.cdist(target_points, source_points).argmin(1)
    cosine = F.cosine_similarity(
        source["dino"][source_valid][nearest], target["dino"][target_valid], dim=-1,
    )
    return float(1.0 - cosine.mean())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("v41/dataset"))
    parser.add_argument(
        "--manifest", type=Path,
        default=Path("v41/manifests/v41_uid_splits.json"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("v42/run/geometry_rotation_similarity_audit"),
    )
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())

    records = {}
    for split in ("train", "validation"):
        rows = [manifest["episodes"][episode_id]
                for episode_id in manifest["splits"][split]["panel_z"]]
        records[split] = [load_record(args.dataset, row) for row in rows]

    all_pairs, family_summaries = [], {}
    for family in ("soft_body", "rigid"):
        family_train = [row for row in records["train"] if row["family"] == family]
        family_validation = [row for row in records["validation"] if row["family"] == family]
        train = [row for row in family_train if row["rotation_valid"].any()]
        validation = [row for row in family_validation if row["rotation_valid"].any()]
        pairs = []
        for target in validation:
            candidates = []
            for source in train:
                chamfer, sign = _geometry_alignment(source, target)
                rotation_rad, frames_compared = rotation_distance(source, target, sign)
                dino_distance = aligned_dino_distance(source, target, sign)
                pair = {
                    "family": family,
                    "validation_uid": target["uid"],
                    "train_uid": source["uid"],
                    "geometry_chamfer": chamfer,
                    "aligned_dino_distance": dino_distance,
                    "rotation_mean_geodesic_rad": rotation_rad,
                    "rotation_mean_geodesic_deg": float(np.degrees(rotation_rad)),
                    "frames_compared": frames_compared,
                    "source_minimum_kabsch_ratio": source["minimum_kabsch_ratio"],
                    "target_minimum_kabsch_ratio": target["minimum_kabsch_ratio"],
                }
                candidates.append(pair)
                pairs.append(pair)
                all_pairs.append(pair)
            by_geometry = sorted(candidates, key=lambda row: row["geometry_chamfer"])
            by_dino = sorted(candidates, key=lambda row: row["aligned_dino_distance"])
            by_rotation = sorted(candidates, key=lambda row: row["rotation_mean_geodesic_rad"])
            for rank, pair in enumerate(by_geometry, 1):
                pair["geometry_rank_for_validation"] = rank
            for rank, pair in enumerate(by_dino, 1):
                pair["dino_rank_for_validation"] = rank
            for pair in candidates:
                pair["dino_geometry_rank_fusion"] = 0.5 * (
                    pair["geometry_rank_for_validation"]
                    + pair["dino_rank_for_validation"]
                )
            by_fusion = sorted(candidates, key=lambda row: (
                row["dino_geometry_rank_fusion"],
                row["dino_rank_for_validation"],
                row["geometry_rank_for_validation"],
            ))
            best_rotation = by_rotation[0]
            best_rotation["is_oracle_rotation_neighbour"] = True
            by_geometry[0]["is_nearest_geometry_neighbour"] = True
            by_dino[0]["is_nearest_dino_neighbour"] = True
            by_fusion[0]["is_nearest_fusion_neighbour"] = True
        geometry = [row["geometry_chamfer"] for row in pairs]
        dino = [row["aligned_dino_distance"] for row in pairs]
        fusion = [row["dino_geometry_rank_fusion"] for row in pairs]
        rotation = [row["rotation_mean_geodesic_rad"] for row in pairs]
        nearest = [row for row in pairs if row.get("is_nearest_geometry_neighbour")]
        nearest_dino = [row for row in pairs if row.get("is_nearest_dino_neighbour")]
        nearest_fusion = [row for row in pairs if row.get("is_nearest_fusion_neighbour")]
        oracle = [row for row in pairs if row.get("is_oracle_rotation_neighbour")]
        family_summaries[family] = {
            "train_uids": len(train), "validation_uids": len(validation),
            "excluded_train_uids_no_well_conditioned_rotation": [
                row["uid"] for row in family_train if not row["rotation_valid"].any()
            ],
            "excluded_validation_uids_no_well_conditioned_rotation": [
                row["uid"] for row in family_validation if not row["rotation_valid"].any()
            ],
            "pair_count": len(pairs),
            "geometry_distance_vs_rotation_distance_spearman": spearman(geometry, rotation),
            "aligned_dino_distance_vs_rotation_distance_spearman": spearman(dino, rotation),
            "dino_geometry_rank_fusion_vs_rotation_distance_spearman": spearman(fusion, rotation),
            "nearest_geometry_mean_rotation_error_deg": float(np.mean([
                row["rotation_mean_geodesic_deg"] for row in nearest
            ])),
            "nearest_aligned_dino_mean_rotation_error_deg": float(np.mean([
                row["rotation_mean_geodesic_deg"] for row in nearest_dino
            ])),
            "nearest_dino_geometry_fusion_mean_rotation_error_deg": float(np.mean([
                row["rotation_mean_geodesic_deg"] for row in nearest_fusion
            ])),
            "oracle_best_mean_rotation_error_deg": float(np.mean([
                row["rotation_mean_geodesic_deg"] for row in oracle
            ])),
            "oracle_best_mean_geometry_rank": float(np.mean([
                row["geometry_rank_for_validation"] for row in oracle
            ])),
        }

    report = {
        "audit": "geometry_similarity_vs_rotation_similarity_v1",
        "fit_split": "train", "evaluation_split": "validation",
        "panel": "Z_zero_initial_velocity", "test_used": False,
        "geometry": "radius-normalized PCA alignment, best proper sign, symmetric squared Chamfer",
        "dino": "aligned-point DINOv2 cosine distance after the same PCA/sign alignment and nearest-surface-point matching",
        "dino_geometry_fusion": "equal-weight mean of within-validation-object DINO and geometry ranks",
        "rotation": "mean full-trajectory SO(3) geodesic distance after expressing rotations in the matched PCA/sign frame",
        "frames_compared": 59,
        "summary": family_summaries,
        "pairs": all_pairs,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "GEOMETRY_ROTATION_SIMILARITY.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(family_summaries, indent=2))


if __name__ == "__main__":
    main()
