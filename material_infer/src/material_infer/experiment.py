from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .data import FAMILIES, ObjectRecord, load_manifest, load_records, make_feature
from .metrics import bootstrap_interval, classification_metrics, regression_metrics
from .model import FeatureTransform, Probe


def _seed_everything(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def _uids(manifest: dict, records: dict[str, ObjectRecord], split: str, task: str) -> list[str]:
    uids = list(manifest["splits"][split])
    return [uid for uid in uids if task == "family" or records[uid].family == "soft_body"]


def _scene_shuffle(x: np.ndarray, uids: list[str], seed: int) -> np.ndarray:
    order = sorted(range(len(uids)), key=lambda i: hashlib.sha256(f"donor:{seed}:{uids[i]}".encode()).hexdigest())
    donor = np.empty(len(order), dtype=np.int64)
    for pos, index in enumerate(order):
        donor[index] = order[(pos + 1) % len(order)]
    return x[donor]


def _features(records, uids, source, dino_key) -> np.ndarray:
    return np.stack([make_feature(records[uid], source, dino_key) for uid in uids])


def _targets(records, uids, task, target):
    if task == "family":
        return np.asarray([FAMILIES.index(records[uid].family) for uid in uids], dtype=np.int64)
    key = "log10_e" if target == "log10_E" else "nu"
    return np.asarray([getattr(records[uid], key) for uid in uids], dtype=np.float32)


def _train(model, x_train, y_train, x_val, y_val, task, lr, weight_decay, epochs, patience, device):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    xt, xv = torch.from_numpy(x_train).to(device), torch.from_numpy(x_val).to(device)
    if task == "family":
        yt, yv, criterion = torch.from_numpy(y_train).long().to(device), torch.from_numpy(y_val).long().to(device), nn.CrossEntropyLoss()
    else:
        yt = torch.from_numpy(y_train).float()[:, None].to(device)
        yv = torch.from_numpy(y_val).float()[:, None].to(device)
        criterion = nn.MSELoss()
    best, best_loss, stale = None, float("inf"), 0
    history = []
    for epoch in range(epochs):
        model.train(); optimizer.zero_grad(); loss = criterion(model(xt), yt); loss.backward(); optimizer.step()
        model.eval()
        with torch.no_grad(): val_loss = float(criterion(model(xv), yv))
        history.append({"epoch": epoch + 1, "train_loss": float(loss.detach()), "validation_loss": val_loss})
        if val_loss < best_loss - 1e-8:
            best_loss, stale = val_loss, 0
            best = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= patience: break
    model.load_state_dict(best)
    return history, best_loss


def run_experiment(args) -> dict:
    config = {key: (str(value) if isinstance(value, Path) else value) for key, value in vars(args).items()}
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("--device mps requested, but torch.backends.mps.is_available() is false")
    device = torch.device(args.device)
    records, manifest = load_records(args.dataset), load_manifest(args.manifest)
    split_uids = {split: _uids(manifest, records, split, args.task) for split in ("train", "validation", "test")}
    raw_x = {split: _features(records, uids, args.feature_source, args.dino_key) for split, uids in split_uids.items()}
    raw_y = {split: _targets(records, uids, args.task, args.target) for split, uids in split_uids.items()}
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    run_results = []
    for seed in args.seeds:
        _seed_everything(seed)
        x = {key: value.copy() for key, value in raw_x.items()}
        y = {key: value.copy() for key, value in raw_y.items()}
        if args.control == "scene_shuffled":
            for split in x:
                x[split] = _scene_shuffle(x[split], split_uids[split], seed)
        if args.control == "label_permuted":
            y["train"] = np.random.default_rng(seed).permutation(y["train"])

        transform = FeatureTransform.fit(x["train"], args.pca_components)
        x = {split: transform.transform(values) for split, values in x.items()}
        target_mean, target_scale = 0.0, 1.0
        if args.task == "soft":
            target_mean, target_scale = float(y["train"].mean()), float(y["train"].std())
            target_scale = target_scale if target_scale > 1e-8 else 1.0
            y_fit = {split: ((value - target_mean) / target_scale).astype(np.float32) for split, value in y.items()}
        else:
            y_fit = y
        model = Probe(x["train"].shape[1], len(FAMILIES) if args.task == "family" else 1, args.model, args.hidden_dim, args.dropout).to(device)
        history, best_loss = _train(model, x["train"], y_fit["train"], x["validation"], y_fit["validation"], args.task, args.lr, args.weight_decay, args.epochs, args.patience, device)
        model.eval()
        with torch.no_grad(): raw_pred = model(torch.from_numpy(x["test"]).to(device)).cpu().numpy()
        if args.task == "family":
            pred = raw_pred.argmax(1)
            metrics = classification_metrics(y["test"], pred, list(FAMILIES))
            metrics["macro_f1_ci95"] = bootstrap_interval(y["test"], pred, lambda a, b: classification_metrics(a, b, list(FAMILIES)), "macro_f1")
            display_pred = [FAMILIES[i] for i in pred]
            display_target = [FAMILIES[i] for i in y["test"]]
        else:
            pred = raw_pred[:, 0] * target_scale + target_mean
            metrics = regression_metrics(y["test"], pred, args.target)
            metrics["mae_ci95"] = bootstrap_interval(y["test"], pred, lambda a, b: regression_metrics(a, b, args.target), "mae")
            display_pred, display_target = pred.tolist(), y["test"].tolist()
        seed_dir = output / f"seed{seed}"; seed_dir.mkdir(exist_ok=True)
        cpu_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
        torch.save({"model": cpu_state, "preprocess": transform.state_dict(), "target_mean": target_mean, "target_scale": target_scale, "args": config}, seed_dir / "best.pt")
        (seed_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n")
        result = {"seed": seed, "best_validation_loss": best_loss, "metrics": metrics,
                  "predictions": [{"uid": uid, "target": target, "prediction": prediction} for uid, target, prediction in zip(split_uids["test"], display_target, display_pred)]}
        (seed_dir / "test_metrics.json").write_text(json.dumps(result, indent=2) + "\n")
        run_results.append(result)
    primary = "macro_f1" if args.task == "family" else "mae"
    values = [run["metrics"][primary] for run in run_results]
    summary = {"config": config, "split_uids": split_uids, "runs": run_results,
               "aggregate": {"metric": primary, "mean": float(np.mean(values)), "std": float(np.std(values)), "values": values}}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary
