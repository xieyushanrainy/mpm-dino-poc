#!/usr/bin/env python3
"""Create a reproducible stratified PCA projection of packaged DINO features."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--points-per-object", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    root = Path(args.dataset_dir).resolve()
    dataset = json.loads((root / "dataset.json").read_text(encoding="utf-8"))
    rng = np.random.default_rng(args.seed)
    features, families, uids = [], [], []
    object_means, object_families, object_uids = [], [], []
    for record in dataset["objects"]:
        uid, family = record["uid"], record["solver_route"]
        with np.load(root / record["sample"], allow_pickle=False) as arrays:
            values = arrays["dino_features"].astype(np.float32)
            valid = arrays["dino_valid"].astype(bool)
        values = values[valid]
        if len(values) < args.points_per_object:
            raise ValueError(f"{uid}: only {len(values)} valid DINO rows")
        indices = rng.choice(len(values), args.points_per_object, replace=False)
        sampled = values[indices]
        features.append(sampled)
        families.extend([family] * len(sampled)); uids.extend([uid] * len(sampled))
        object_means.append(values.mean(axis=0)); object_families.append(family); object_uids.append(uid)
    matrix = np.concatenate(features, axis=0)
    means = np.stack(object_means)
    pca = PCA(n_components=2, svd_solver="randomized", random_state=args.seed)
    point_xy = pca.fit_transform(matrix)
    mean_xy = pca.transform(means)
    np.savez_compressed(
        root / "dino_feature_pca_projection.npz",
        point_xy=point_xy.astype(np.float32), family=np.asarray(families), uid=np.asarray(uids),
        object_mean_xy=mean_xy.astype(np.float32), object_family=np.asarray(object_families),
        object_uid=np.asarray(object_uids), explained_variance_ratio=pca.explained_variance_ratio_,
        pca_components=pca.components_.astype(np.float32), pca_mean=pca.mean_.astype(np.float32),
    )
    family_names = ("rigid", "fluid", "soft_body")
    summary = {
        "method": "PCA fitted to a deterministic stratified sample of point-level DINO embeddings",
        "seed": args.seed, "points_per_object": args.points_per_object,
        "point_count": len(matrix), "object_count": len(means),
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "families": {},
    }
    for family in family_names:
        mask = np.asarray(families) == family
        subset = point_xy[mask]
        summary["families"][family] = {
            "point_count": int(mask.sum()), "centroid": subset.mean(axis=0).tolist(),
            "std": subset.std(axis=0).tolist(),
        }
    (root / "dino_feature_pca_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    # A compact subset is sufficient for the interactive scatter while the NPZ preserves all rows.
    visual_indices = np.concatenate([
        rng.choice(np.flatnonzero(np.asarray(families) == family), 600, replace=False)
        for family in family_names
    ])
    visual = {
        "variance": [round(float(value * 100), 2) for value in pca.explained_variance_ratio_],
        "points": [
            [round(float(point_xy[i, 0]), 4), round(float(point_xy[i, 1]), 4), families[i]]
            for i in visual_indices
        ],
        "means": [
            [round(float(x), 4), round(float(y), 4), family, uid]
            for (x, y), family, uid in zip(mean_xy, object_families, object_uids)
        ],
    }
    (root / "dino_feature_pca_visual.json").write_text(
        json.dumps(visual, separators=(",", ":")), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
