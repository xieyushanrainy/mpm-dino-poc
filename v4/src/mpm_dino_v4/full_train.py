from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .full_data import FullTrajectoryDataset
from .full_losses import compute_full_trajectory_loss
from .full_model import FullTrajectorySurrogate


FULL_MODEL_KEYS = (
    "x0",
    "x1",
    "input_mask",
    "dino",
    "dino_valid",
    "dt",
    "gravity",
    "floor_z",
    "neighbour_indices",
    "neighbour_mask",
    "rest_edge_vectors",
    "rest_edge_lengths",
)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def move(batch: dict, device: torch.device) -> dict:
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def horizon_rmse(output, batch, indices=(7, 15)) -> float:
    values = []
    for index in indices:
        mask = batch["target_mask"][:, index]
        squared = (output.position[:, index] - batch["target"][:, index]).square().sum(-1)
        per_object = torch.sqrt(
            (squared * mask).sum(1) / mask.sum(1).clamp_min(1)
        )
        values.append(per_object.mean())
    return float(torch.stack(values).mean().detach().cpu())


def run_epoch(
    model,
    loader,
    device,
    optimizer=None,
    accumulation_steps: int = 4,
    max_batches: int | None = None,
):
    training = optimizer is not None
    model.train(training)
    names = (
        "total",
        "residual",
        "position",
        "com",
        "edge_vector",
        "edge_length",
        "key_horizons",
    )
    totals = {name: 0.0 for name in names}
    selections = []
    batches = 0
    if training:
        optimizer.zero_grad(set_to_none=True)
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for raw in loader:
            batch = move(raw, device)
            output = model(**{key: batch[key] for key in FULL_MODEL_KEYS})
            loss = compute_full_trajectory_loss(output, batch)
            if training:
                (loss.total / accumulation_steps).backward()
                if (batches + 1) % accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
            for name in names:
                totals[name] += float(getattr(loss, name).detach().cpu())
            selections.append(horizon_rmse(output, batch))
            batches += 1
            if max_batches is not None and batches >= max_batches:
                break
    if training and batches % accumulation_steps:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    metrics = {name: value / max(batches, 1) for name, value in totals.items()}
    metrics["selection_h8_h16_rmse_m"] = float(np.mean(selections)) if selections else float("inf")
    return metrics


def train_full_model(
    cache,
    manifest,
    output,
    dino_mode,
    seed,
    device="mps",
    epochs=300,
    batch_size=1,
    accumulation_steps=4,
    hidden_dim=128,
    blocks=4,
    heads=4,
    dropout=0.1,
    lr=2e-4,
    patience=30,
    max_batches=None,
    families=("rigid", "soft_body"),
):
    seed_everything(seed)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    train_data = FullTrajectoryDataset(cache, manifest, "train", families, dino_mode, seed)
    val_data = FullTrajectoryDataset(cache, manifest, "validation", families, dino_mode, seed)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=0)
    target_device = torch.device(device)
    model = FullTrajectorySurrogate(
        hidden_dim=hidden_dim, blocks=blocks, heads=heads, dropout=dropout
    ).to(target_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=10, threshold=0.005,
        threshold_mode="rel", min_lr=1e-6
    )
    config = {
        "model_family": "v4_full_trajectory",
        "cache": str(Path(cache).resolve()),
        "manifest_sha256": hashlib.sha256(
            json.dumps(manifest, sort_keys=True).encode()
        ).hexdigest(),
        "dino_mode": dino_mode,
        "seed": seed,
        "device": device,
        "epochs": epochs,
        "batch_size": batch_size,
        "accumulation_steps": accumulation_steps,
        "hidden_dim": hidden_dim,
        "blocks": blocks,
        "heads": heads,
        "dropout": dropout,
        "lr": lr,
        "patience": patience,
        "max_batches": max_batches,
        "families": list(families),
        "frames": 59,
    }
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    best = float("inf")
    reference_best = float("inf")
    stale = 0
    with (output / "history.jsonl").open("w") as history:
        for epoch in range(1, epochs + 1):
            train = run_epoch(
                model, train_loader, target_device, optimizer,
                accumulation_steps, max_batches
            )
            validation = run_epoch(
                model, val_loader, target_device, None,
                accumulation_steps, max_batches
            )
            selection = validation["selection_h8_h16_rmse_m"]
            scheduler.step(selection)
            meaningful = selection < reference_best * 0.995
            if meaningful:
                reference_best, stale = selection, 0
            else:
                stale += 1
            record = {
                "epoch": epoch,
                "train": train,
                "validation": validation,
                "selection": selection,
                "lr": optimizer.param_groups[0]["lr"],
            }
            history.write(json.dumps(record) + "\n")
            history.flush()
            state = {
                "model": model.state_dict(),
                "config": config,
                "epoch": epoch,
                "validation": validation,
                "selection": selection,
            }
            torch.save(state, output / "last.pt")
            if selection < best:
                best = selection
                torch.save(state, output / "best.pt")
            print(
                f"epoch={epoch:03d} train={train['total']:.6g} "
                f"val_h8h16={selection:.6g} lr={record['lr']:.3g} stale={stale}",
                flush=True,
            )
            if stale >= patience:
                break
    return output / "best.pt"


def load_full_model(checkpoint_path, device="cpu"):
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = state["config"]
    model = FullTrajectorySurrogate(
        hidden_dim=config["hidden_dim"],
        blocks=config["blocks"],
        heads=config["heads"],
        dropout=config["dropout"],
        frames=config.get("frames", 59),
    )
    model.load_state_dict(state["model"])
    model.to(device)
    return model, config
