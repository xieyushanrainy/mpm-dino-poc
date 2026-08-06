from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import numpy as np

from .data import load_records
from .experiment import run_experiment
from .metrics import regression_metrics


def _balanced_target_folds(records, target: str, folds: int) -> list[list[str]]:
    """Deterministic serpentine allocation across the sorted target range."""
    key = "log10_e" if target == "log10_E" else "nu"
    rows = sorted(
        ((uid, float(getattr(record, key))) for uid, record in records.items() if record.family == "soft_body"),
        key=lambda item: (item[1], item[0]),
    )
    if len(rows) % folds:
        raise ValueError(f"{len(rows)} soft objects cannot be divided evenly into {folds} folds")
    result = [[] for _ in range(folds)]
    for block_start in range(0, len(rows), folds):
        block = rows[block_start:block_start + folds]
        order = range(folds) if (block_start // folds) % 2 == 0 else reversed(range(folds))
        for fold, row in zip(order, block):
            result[fold].append(row[0])
    return result


def run_cross_validation(args) -> dict:
    records = load_records(args.dataset)
    folds = _balanced_target_folds(records, args.target, args.folds)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    fold_summaries = []
    all_soft = set().union(*map(set, folds))
    if len(all_soft) != 30:
        raise ValueError(f"expected 30 unique soft objects, found {len(all_soft)}")

    for test_index in range(args.folds):
        validation_index = (test_index + 1) % args.folds
        test_uids = folds[test_index]
        validation_uids = folds[validation_index]
        train_uids = sorted(all_soft - set(test_uids) - set(validation_uids))
        fold_dir = output / f"fold{test_index}"
        fold_dir.mkdir(exist_ok=True)
        manifest = {
            "schema_version": 1,
            "description": "soft-object cross-validation fold",
            "splits": {"train": train_uids, "validation": validation_uids, "test": test_uids},
        }
        manifest_path = fold_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        run_args = Namespace(**{
            **vars(args), "command": "run", "task": "soft", "manifest": manifest_path,
            "output": fold_dir / "run",
        })
        delattr(run_args, "folds")
        fold_summaries.append(run_experiment(run_args))

    seed_results = []
    for seed_pos, seed in enumerate(args.seeds):
        predictions = []
        for summary in fold_summaries:
            predictions.extend(summary["runs"][seed_pos]["predictions"])
        if len(predictions) != 30 or len({row["uid"] for row in predictions}) != 30:
            raise ValueError("cross-validation predictions do not cover exactly 30 unique soft UIDs")
        y = np.asarray([row["target"] for row in predictions], dtype=np.float64)
        pred = np.asarray([row["prediction"] for row in predictions], dtype=np.float64)
        seed_results.append({"seed": seed, "metrics": regression_metrics(y, pred, args.target), "predictions": predictions})

    metric_names = list(seed_results[0]["metrics"])
    aggregate = {
        name: {
            "mean": float(np.mean([run["metrics"][name] for run in seed_results])),
            "std": float(np.std([run["metrics"][name] for run in seed_results])),
            "values": [run["metrics"][name] for run in seed_results],
        }
        for name in metric_names
    }
    result = {
        "config": {key: (str(value) if isinstance(value, Path) else value) for key, value in vars(args).items()},
        "folds": folds, "runs": seed_results, "aggregate": aggregate,
    }
    (output / "cv_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return result
