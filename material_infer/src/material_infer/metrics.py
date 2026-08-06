from __future__ import annotations

import numpy as np


def classification_metrics(y: np.ndarray, pred: np.ndarray, classes: list[str]) -> dict:
    confusion = np.zeros((len(classes), len(classes)), dtype=np.int64)
    for target, prediction in zip(y, pred):
        confusion[int(target), int(prediction)] += 1
    recalls, f1s = [], []
    per_class = {}
    for i, name in enumerate(classes):
        tp = confusion[i, i]
        precision = tp / max(1, confusion[:, i].sum())
        recall = tp / max(1, confusion[i].sum())
        f1 = 2 * precision * recall / max(1e-12, precision + recall)
        recalls.append(recall); f1s.append(f1)
        per_class[name] = {"precision": float(precision), "recall": float(recall), "f1": float(f1), "support": int(confusion[i].sum())}
    return {
        "accuracy": float((y == pred).mean()), "balanced_accuracy": float(np.mean(recalls)),
        "macro_f1": float(np.mean(f1s)), "confusion": confusion.tolist(), "per_class": per_class,
    }


def _ranks(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=np.float64)
    start = 0
    while start < len(x):
        end = start + 1
        while end < len(x) and x[order[end]] == x[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2
        start = end
    return ranks


def regression_metrics(y: np.ndarray, pred: np.ndarray, target: str) -> dict:
    error = np.abs(y - pred)
    ry, rp = _ranks(y), _ranks(pred)
    denom = np.sqrt(((ry - ry.mean()) ** 2).sum() * ((rp - rp.mean()) ** 2).sum())
    spearman = float(((ry - ry.mean()) * (rp - rp.mean())).sum() / denom) if denom > 0 else 0.0
    result = {"mae": float(error.mean()), "median_ae": float(np.median(error)), "spearman": spearman}
    if target == "log10_E":
        result["within_0.5_log10"] = float((error <= 0.5).mean())
    return result


def bootstrap_interval(y: np.ndarray, pred: np.ndarray, fn, key: str, seed: int = 20260805, draws: int = 2000) -> list[float]:
    rng, values = np.random.default_rng(seed), []
    for _ in range(draws):
        idx = rng.integers(0, len(y), len(y))
        values.append(float(fn(y[idx], pred[idx])[key]))
    return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]
