from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .v41_data import MODEL_INPUT_KEYS, UIDBalancedSampler, V41TrajectoryDataset
from .v41_train import (
    atomic_torch_save, move, restore_rng_state, rng_state, seed_all,
    state_sha256,
)
from .v42_geometry import canonical_targets, rotation_geodesic
from .v42_losses import compute_v42_global_losses
from .v42_model import V42RotationAwareSurrogate


GLOBAL_PREFIXES = (
    "initial_node.", "initial_graph.", "time_projection.", "token.",
    "blocks.", "v42_com_head.", "rotation_head.",
)


def gate1_parameters(model):
    return [
        parameter for name, parameter in model.named_parameters()
        if name.startswith(GLOBAL_PREFIXES)
    ]


def gate1_scores(output, batch, targets):
    result = {}
    for horizon in (1, 8, 16, 30, 40, 59):
        if horizon > output.com.shape[1]:
            continue
        index = horizon - 1
        valid = batch["target_mask"][:, index].any(1)
        com_error = torch.linalg.vector_norm(
            output.com[:, index] - targets.com[:, index], dim=-1,
        )
        rotation_error = rotation_geodesic(
            output.rotation[:, index], targets.rotation[:, index],
        )
        rotation_valid = valid & targets.valid_rotation[:, index]
        result[f"h{horizon}_com_nrmse"] = (
            (com_error / targets.radius) * valid
        ).sum() / valid.sum().clamp_min(1)
        result[f"h{horizon}_rotation_rad"] = (
            rotation_error * rotation_valid
        ).sum() / rotation_valid.sum().clamp_min(1)
    return result


def run_gate1_epoch(
    model, loader, device, optimizer=None, accumulation=4,
    amp=False, scaler=None, max_batches=None,
):
    training = optimizer is not None
    model.train(training)
    totals, counts, batches = {}, {}, 0
    parameters = gate1_parameters(model)
    if training:
        optimizer.zero_grad(set_to_none=True)
    with torch.enable_grad() if training else torch.no_grad():
        for raw in loader:
            batch = move(raw, device)
            with torch.autocast(
                device_type=device.type, dtype=torch.float16, enabled=amp,
            ):
                output = model(**{
                    key: batch[key] for key in MODEL_INPUT_KEYS
                })
                targets = canonical_targets(
                    batch["x1"], batch["target"], batch["input_mask"],
                    batch["target_mask"],
                )
                loss = compute_v42_global_losses(output, batch, targets)
            if training:
                value = loss.total / accumulation
                if scaler is not None:
                    scaler.scale(value).backward()
                else:
                    value.backward()
                if (batches + 1) % accumulation == 0:
                    if scaler is not None:
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(parameters, 1.0)
                    if scaler is not None:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
            values = {
                "loss": loss.total,
                **{
                    f"loss_{name}": getattr(loss, name)
                    for name in loss.__dataclass_fields__ if name != "total"
                },
                **gate1_scores(output, batch, targets),
            }
            for key, value in values.items():
                totals[key] = totals.get(key, 0.0) + float(
                    value.detach().cpu()
                )
                counts[key] = counts.get(key, 0) + 1
            batches += 1
            if max_batches and batches >= max_batches:
                break
    if training and batches % accumulation:
        if scaler is not None:
            scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    return {
        key: value / max(counts[key], 1) for key, value in totals.items()
    }


def train_v42_gate1(
    root, manifest, output, seed, device="cuda", epochs=120,
    draws_per_epoch=40, hidden_dim=128, blocks=4, heads=4, dropout=0.1,
    lr=2e-4, accumulation=4, patience=20, plateau_patience=5,
    amp=False, resume=True, max_batches=None,
):
    seed_all(seed)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    train_ds = V41TrajectoryDataset(
        root, manifest, "train", "zero", seed,
        families=("soft_body", "rigid"),
    )
    validation_ds = V41TrajectoryDataset(
        root, manifest, "validation", "zero", seed,
        families=("soft_body", "rigid"),
    )
    sampler = UIDBalancedSampler(train_ds, draws_per_epoch, seed)
    train_loader = DataLoader(
        train_ds, batch_size=1, sampler=sampler, num_workers=0,
    )
    validation_loader = DataLoader(
        validation_ds, batch_size=1, shuffle=False, num_workers=0,
    )
    model = V42RotationAwareSurrogate(
        local_mode="zero", hidden_dim=hidden_dim, blocks=blocks,
        heads=heads, dropout=dropout, frames=59,
    ).to(device)
    parameters = gate1_parameters(model)
    optimizer = torch.optim.AdamW(
        parameters, lr=lr, weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=plateau_patience,
        threshold=0.005, threshold_mode="rel", min_lr=1e-6,
    )
    use_amp = bool(amp)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    config = {
        "experiment": "v42_gate1_physical_com_rotation",
        "seed": seed, "device": device, "epochs": epochs,
        "draws_per_epoch": draws_per_epoch, "hidden_dim": hidden_dim,
        "blocks": blocks, "heads": heads, "dropout": dropout, "lr": lr,
        "accumulation": accumulation, "patience": patience,
        "plateau_patience": plateau_patience, "amp": use_amp,
        "manifest_content_sha256": manifest["manifest_content_sha256"],
        "starting_model_sha256": state_sha256(model.state_dict()),
        "trainable_prefixes": list(GLOBAL_PREFIXES),
        "dino_mode": "zero",
        "families": ["soft_body", "rigid"],
        "loss": {
            "implementation": "compute_v42_global_losses",
            "com_weights": {
                "position": 1.0, "velocity": 0.25,
                "acceleration": 0.10, "key_horizons": 0.25,
            },
            "rotation_weights": {
                "rigid": 1.0, "soft_kabsch_gauge": 0.25,
                "rigid_fit": 0.25,
            },
            "key_horizons": [1, 8, 16, 30, 40, 59],
            "kabsch_degeneracy_ratio": 1e-3,
        },
        "selection": {
            "split": "validation",
            "metric": "mean full-trajectory global loss",
            "test_used": False,
        },
    }
    (output / "config.json").write_text(
        json.dumps(config, indent=2) + "\n"
    )
    best, stale, start_epoch = float("inf"), 0, 1
    last_path = output / "last.pt"
    if resume and last_path.exists() and not (
        output / "RUN_COMPLETE.json"
    ).exists():
        state = torch.load(
            last_path, map_location="cpu", weights_only=False,
        )
        immutable = (
            "seed", "draws_per_epoch", "hidden_dim", "blocks", "heads",
            "dropout", "lr", "accumulation", "patience",
            "plateau_patience", "amp", "manifest_content_sha256",
        )
        if any(
            state["config"].get(key) != config.get(key) for key in immutable
        ):
            raise ValueError(
                "refusing to resume with changed Gate-1 configuration"
            )
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        if use_amp and state.get("scaler"):
            scaler.load_state_dict(state["scaler"])
        best, stale = float(state["best"]), int(state["stale"])
        start_epoch = int(state["epoch"]) + 1
        sampler.epoch = int(state["sampler_epoch"])
        restore_rng_state(state["rng_state"])
    mode = "a" if start_epoch > 1 else "w"
    epoch = start_epoch - 1
    with (output / "history.jsonl").open(mode) as history:
        for epoch in range(start_epoch, epochs + 1):
            started = time.perf_counter()
            train = run_gate1_epoch(
                model, train_loader, torch.device(device), optimizer,
                accumulation, use_amp, scaler, max_batches,
            )
            validation = run_gate1_epoch(
                model, validation_loader, torch.device(device), None,
                accumulation, use_amp, None, max_batches,
            )
            selection = validation["loss"]
            scheduler.step(selection)
            improved = selection < best
            stale = 0 if improved else stale + 1
            best = min(best, selection)
            record = {
                "epoch": epoch, "train": train,
                "validation": validation, "selection": selection,
                "lr": optimizer.param_groups[0]["lr"],
                "seconds": time.perf_counter() - started,
            }
            history.write(json.dumps(record) + "\n")
            history.flush()
            state = {
                "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict() if use_amp else None,
                "config": config, "epoch": epoch, "train": train,
                "validation": validation, "selection": selection,
                "best": best, "stale": stale, "sampler_epoch": sampler.epoch,
                "rng_state": rng_state(),
            }
            atomic_torch_save(state, output / "last.pt")
            if improved:
                atomic_torch_save(state, output / "best.pt")
            print(
                f"epoch={epoch:03d} global={selection:.6g} "
                f"stale={stale}", flush=True,
            )
            if stale >= patience:
                break
    best_path = output / "best.pt"
    completion = {
        "status": "complete", "last_epoch": epoch,
        "best_selection": best,
        "best_checkpoint_sha256": hashlib.sha256(
            best_path.read_bytes()
        ).hexdigest(),
        "last_checkpoint_sha256": hashlib.sha256(
            (output / "last.pt").read_bytes()
        ).hexdigest(),
        "history_sha256": hashlib.sha256(
            (output / "history.jsonl").read_bytes()
        ).hexdigest(),
    }
    (output / "RUN_COMPLETE.json").write_text(
        json.dumps(completion, indent=2) + "\n"
    )
    return best_path

