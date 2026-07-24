from __future__ import annotations

import json
import random
import hashlib
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import QuartileBatchSampler, WindowDataset
from .losses import compute_loss
from .model import V4ParticleSurrogate


MODEL_KEYS = ("x_prev", "x_curr", "mask_prev", "mask_curr", "reference", "dino", "dino_valid", "dt",
              "gravity", "floor_z", "neighbour_indices", "neighbour_mask", "rest_edge_vectors", "rest_edge_lengths")


def seed_everything(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def move(batch, device):
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def run_epoch(model, loader, device, optimizer=None, max_batches=None):
    model.train(optimizer is not None); totals = {key: 0.0 for key in ("total", "residual", "position", "com", "edge_vector", "edge_length")}; batches = 0
    context = torch.enable_grad() if optimizer else torch.no_grad()
    with context:
        for raw in loader:
            batch = move(raw, device); output = model(**{key: batch[key] for key in MODEL_KEYS}); loss = compute_loss(output, batch)
            if optimizer:
                optimizer.zero_grad(set_to_none=True); loss.total.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
            for key in totals: totals[key] += float(getattr(loss, key).detach().cpu())
            batches += 1
            if max_batches is not None and batches >= max_batches: break
    return {key: value / max(batches, 1) for key, value in totals.items()}


def train_model(cache, manifest, output, dino_mode, seed, device="mps", epochs=60, batch_size=2,
                hidden_dim=128, layers=3, dino_embed_dim=16, lr=2e-4, patience=8, families=("rigid", "soft_body"), max_batches=None):
    seed_everything(seed); output = Path(output); output.mkdir(parents=True, exist_ok=True)
    train_data = WindowDataset(cache, manifest, "train", families, dino_mode, seed)
    val_data = WindowDataset(cache, manifest, "validation", families, dino_mode, seed)
    train_loader = DataLoader(train_data, batch_sampler=QuartileBatchSampler(train_data, batch_size, seed), num_workers=0)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=0)
    target = torch.device(device); model = V4ParticleSurrogate(384, dino_embed_dim, hidden_dim, layers).to(target)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=3, min_lr=1e-6)
    config = {"cache": str(Path(cache).resolve()), "dino_mode": dino_mode, "seed": seed, "device": device,
              "epochs": epochs, "batch_size": batch_size, "hidden_dim": hidden_dim, "layers": layers,
              "dino_embed_dim": dino_embed_dim, "lr": lr, "patience": patience, "families": list(families), "max_batches": max_batches}
    config["manifest_sha256"] = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    best, stale = float("inf"), 0
    with (output / "history.jsonl").open("w") as history:
        for epoch in range(1, epochs + 1):
            train = run_epoch(model, train_loader, target, optimizer, max_batches); val = run_epoch(model, val_loader, target, None, max_batches)
            scheduler.step(val["total"]); record = {"epoch": epoch, "train": train, "validation": val, "lr": optimizer.param_groups[0]["lr"]}
            history.write(json.dumps(record) + "\n"); history.flush()
            state = {"model": model.state_dict(), "config": config, "epoch": epoch, "validation": val}
            torch.save(state, output / "last.pt")
            if val["total"] < best * 0.995:
                best, stale = val["total"], 0; torch.save(state, output / "best.pt")
            else: stale += 1
            print(f"epoch={epoch:03d} train={train['total']:.6g} val={val['total']:.6g} stale={stale}", flush=True)
            if stale >= patience: break
    return output / "best.pt"


def load_model(checkpoint: str | Path, device="cpu"):
    state = torch.load(checkpoint, map_location="cpu", weights_only=False); config = state["config"]
    model = V4ParticleSurrogate(384, config["dino_embed_dim"], config["hidden_dim"], config["layers"])
    model.load_state_dict(state["model"]); return model.to(device), config
