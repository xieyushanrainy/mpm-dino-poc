#!/usr/bin/env python3
"""High-dimensional family-separability analysis for the balanced DINO dataset."""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.spatial.distance import pdist
from scipy.stats import kruskal, mannwhitneyu
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, silhouette_samples, silhouette_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, normalize


FAMILIES = ("rigid", "fluid", "soft_body")


def bh_adjust(pvalues: list[float]) -> list[float]:
    order = np.argsort(pvalues)
    adjusted = np.empty(len(pvalues), dtype=float)
    running = 1.0
    for rank_index in range(len(order) - 1, -1, -1):
        original_index = order[rank_index]
        rank = rank_index + 1
        running = min(running, pvalues[original_index] * len(pvalues) / rank)
        adjusted[original_index] = running
    return adjusted.tolist()


def permutation_p(observed: float, values: np.ndarray, *, greater: bool = True) -> float:
    if greater:
        return float((1 + np.sum(values >= observed)) / (len(values) + 1))
    return float((1 + np.sum(values <= observed)) / (len(values) + 1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--permutations", type=int, default=2000)
    parser.add_argument("--feature-key", default="dino_features")
    parser.add_argument("--valid-key", default="dino_valid")
    parser.add_argument("--output", help="Defaults to dino_high_dimensional_analysis.json in dataset-dir")
    args = parser.parse_args()
    root = Path(args.dataset_dir).resolve()
    output = Path(args.output).resolve() if args.output else root / "dino_high_dimensional_analysis.json"
    dataset = json.loads((root / "dataset.json").read_text(encoding="utf-8"))
    rng = np.random.default_rng(args.seed)
    means, labels, uids = [], [], []
    point_spread = []
    for record in dataset["objects"]:
        with np.load(root / record["sample"], allow_pickle=False) as arrays:
            values = arrays[args.feature_key].astype(np.float64)
            values = values[arrays[args.valid_key].astype(bool)]
        mean = values.mean(axis=0)
        means.append(mean); labels.append(record["solver_route"]); uids.append(record["uid"])
        normalized_points = normalize(values)
        normalized_mean = normalize(mean[None, :])[0]
        point_spread.append(float(np.mean(1.0 - normalized_points @ normalized_mean)))
    x = np.stack(means); y = np.asarray(labels); x_norm = normalize(x)

    # Leave-one-out distance to the family centroid: one compactness value per independent object.
    loo_distance = np.empty(len(x_norm), dtype=float)
    for family in FAMILIES:
        indices = np.flatnonzero(y == family)
        for index in indices:
            other = indices[indices != index]
            centroid = normalize(x_norm[other].mean(axis=0, keepdims=True))[0]
            loo_distance[index] = 1.0 - float(x_norm[index] @ centroid)

    within = {}
    for family in FAMILIES:
        indices = np.flatnonzero(y == family)
        pairwise = pdist(x_norm[indices], metric="cosine")
        values = loo_distance[indices]
        within[family] = {
            "n_objects": len(indices),
            "mean_pairwise_cosine_distance": float(pairwise.mean()),
            "median_pairwise_cosine_distance": float(np.median(pairwise)),
            "mean_leave_one_out_centroid_distance": float(values.mean()),
            "std_leave_one_out_centroid_distance": float(values.std(ddof=1)),
            "mean_point_to_object_mean_cosine_distance": float(np.mean(np.asarray(point_spread)[indices])),
        }

    groups = [loo_distance[y == family] for family in FAMILIES]
    kw = kruskal(*groups)
    kw_eta2 = max(0.0, float((kw.statistic - len(FAMILIES) + 1) / (len(y) - len(FAMILIES))))
    pairwise_tests = []
    raw_p = []
    for left, right in combinations(FAMILIES, 2):
        test = mannwhitneyu(loo_distance[y == left], loo_distance[y == right], alternative="two-sided")
        n1, n2 = np.sum(y == left), np.sum(y == right)
        rank_biserial = 2.0 * float(test.statistic) / (n1 * n2) - 1.0
        pairwise_tests.append({"left": left, "right": right, "u": float(test.statistic), "rank_biserial": rank_biserial})
        raw_p.append(float(test.pvalue))
    adjusted = bh_adjust(raw_p)
    for row, p, q in zip(pairwise_tests, raw_p, adjusted):
        row.update({"p_value": p, "bh_adjusted_p": q})

    silhouette = float(silhouette_score(x_norm, y, metric="cosine"))
    silhouette_by_family = {
        family: float(silhouette_samples(x_norm, y, metric="cosine")[y == family].mean())
        for family in FAMILIES
    }
    perm_silhouette = np.empty(args.permutations)
    for i in range(args.permutations):
        perm_silhouette[i] = silhouette_score(x_norm, rng.permutation(y), metric="cosine")

    # Euclidean distance-based R²: fraction of total centroid variance explained by family labels.
    grand = x_norm.mean(axis=0)
    total_ss = float(np.sum((x_norm - grand) ** 2))
    def family_r2(permuted_y: np.ndarray) -> float:
        within_ss = 0.0
        for family in FAMILIES:
            subset = x_norm[permuted_y == family]
            within_ss += float(np.sum((subset - subset.mean(axis=0)) ** 2))
        return 1.0 - within_ss / total_ss
    r2 = family_r2(y)
    perm_r2 = np.asarray([family_r2(rng.permutation(y)) for _ in range(args.permutations)])

    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=20, random_state=args.seed)
    # Store fold-level metrics; repeated out-of-fold predictions are not collapsed into a single score.
    fold_balanced, fold_f1 = [], []
    aggregate_confusion = np.zeros((len(FAMILIES), len(FAMILIES)), dtype=int)
    classifier = make_pipeline(
        StandardScaler(), LogisticRegression(
            C=1.0, max_iter=2000, class_weight="balanced", random_state=args.seed,
            solver="liblinear",
        )
    )
    splits = list(cv.split(x, y))
    for train, test in splits:
        classifier.fit(x[train], y[train]); predicted = classifier.predict(x[test])
        fold_balanced.append(balanced_accuracy_score(y[test], predicted))
        fold_f1.append(f1_score(y[test], predicted, average="macro"))
        aggregate_confusion += confusion_matrix(y[test], predicted, labels=FAMILIES)
    observed_cv = float(np.mean(fold_balanced))
    # Re-stratify each shuffled label vector so every fold retains all three classes.
    cv_permutations = min(200, args.permutations)
    perm_cv = np.empty(cv_permutations)
    for iteration in range(cv_permutations):
        yp = rng.permutation(y); scores = []
        null_cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=args.seed + iteration)
        for train, test in null_cv.split(x, yp):
            classifier.fit(x[train], yp[train]); scores.append(balanced_accuracy_score(yp[test], classifier.predict(x[test])))
        perm_cv[iteration] = np.mean(scores)
    pairwise_classification = []
    for left, right in combinations(FAMILIES, 2):
        mask = np.isin(y, (left, right)); xp, yp = x[mask], y[mask]
        pair_cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=20, random_state=args.seed)
        scores = []
        for train, test in pair_cv.split(xp, yp):
            classifier.fit(xp[train], yp[train])
            scores.append(balanced_accuracy_score(yp[test], classifier.predict(xp[test])))
        pairwise_classification.append({
            "left": left, "right": right, "mean_balanced_accuracy": float(np.mean(scores)),
            "std_fold_balanced_accuracy": float(np.std(scores, ddof=1)),
        })

    result = {
        "analysis_unit": f"object (mean of valid point-level {x.shape[1]}D DINO embeddings)",
        "feature_key": args.feature_key, "valid_key": args.valid_key,
        "n_objects": len(y), "n_per_family": {family: int(np.sum(y == family)) for family in FAMILIES},
        "seed": args.seed, "permutations": args.permutations,
        "within_family_compactness": within,
        "compactness_test": {
            "test": "Kruskal-Wallis on object-level leave-one-out cosine distance to family centroid",
            "h": float(kw.statistic), "p_value": float(kw.pvalue), "eta_squared_h": kw_eta2,
            "pairwise_mann_whitney_bh": pairwise_tests,
        },
        "silhouette": {
            "metric": "cosine", "overall": silhouette, "by_family": silhouette_by_family,
            "permutation_p_greater": permutation_p(silhouette, perm_silhouette),
            "permutation_null_mean": float(perm_silhouette.mean()),
            "permutation_null_std": float(perm_silhouette.std(ddof=1)),
        },
        "distance_based_separability": {
            "metric": "Euclidean on L2-normalized object means", "family_r_squared": r2,
            "permutation_p_greater": permutation_p(r2, perm_r2),
            "permutation_null_mean": float(perm_r2.mean()),
        },
        "linear_classification": {
            "model": "standardized multinomial logistic regression",
            "cv": "20x repeated stratified 5-fold CV",
            "mean_balanced_accuracy": observed_cv,
            "std_fold_balanced_accuracy": float(np.std(fold_balanced, ddof=1)),
            "mean_macro_f1": float(np.mean(fold_f1)),
            "std_fold_macro_f1": float(np.std(fold_f1, ddof=1)),
            "chance_balanced_accuracy": 1 / 3,
            "permutation_p_greater": permutation_p(observed_cv, perm_cv),
            "permutation_count": cv_permutations,
            "permutation_null_mean": float(perm_cv.mean()),
            "row_normalized_confusion_matrix": (
                aggregate_confusion / aggregate_confusion.sum(axis=1, keepdims=True)
            ).tolist(),
            "confusion_labels": list(FAMILIES),
            "pairwise": pairwise_classification,
        },
        "limitations": [
            "Family is confounded with source collection and simulation pipeline; separation is not purely material-driven.",
            "DINO is an appearance/semantic representation, not a calibrated physical-material encoder.",
            "Object means discard spatial and multimodal structure within each object.",
            "Only 30 objects per family are available; uncertainty remains material.",
        ],
    }
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
