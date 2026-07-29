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
        ballistic_error = torch.linalg.vector_norm(
            output.ballistic_com[:, index] - targets.com[:, index], dim=-1,
        )
        result[f"h{horizon}_ballistic_com_nrmse"] = (
            (ballistic_error / targets.radius) * valid
        ).sum() / valid.sum().clamp_min(1)
        identity = torch.eye(
            3, device=output.rotation.device, dtype=output.rotation.dtype,
        ).expand_as(output.rotation[:, index])
        identity_error = rotation_geodesic(
            identity, targets.rotation[:, index],
        )
        result[f"h{horizon}_identity_rotation_rad"] = (
            identity_error * rotation_valid
        ).sum() / rotation_valid.sum().clamp_min(1)
    return result


@torch.no_grad()
def write_gate1_validation_baselines(
    model, loader, device, output_path, amp=False,
):
    """Write mandatory family/panel model-versus-physics comparisons."""
    model.eval()
    horizons = (1, 8, 16, 30, 40, 59)
    accumulators = {}
    for raw in loader:
        batch = move(raw, device)
        with torch.autocast(
            device_type=device.type, dtype=torch.float16, enabled=amp,
        ):
            prediction = model(**{
                key: batch[key] for key in MODEL_INPUT_KEYS
            })
            targets = canonical_targets(
                batch["x1"], batch["target"], batch["input_mask"],
                batch["target_mask"],
            )
        for sample in range(prediction.com.shape[0]):
            family = batch["family"][sample]
            panel = batch["panel"][sample]
            group = accumulators.setdefault(
                (family, panel), {"uids": set(), "horizons": {}},
            )
            group["uids"].add(batch["uid"][sample])
            for horizon in horizons:
                if horizon > prediction.com.shape[1]:
                    continue
                index = horizon - 1
                valid = bool(batch["target_mask"][sample, index].any())
                if not valid:
                    continue
                values = group["horizons"].setdefault(horizon, {
                    "episodes": 0,
                    "model_com_m_sum": 0.0,
                    "ballistic_com_m_sum": 0.0,
                    "model_com_nrmse_sum": 0.0,
                    "ballistic_com_nrmse_sum": 0.0,
                    "rotation_valid": 0,
                    "model_rotation_rad_sum": 0.0,
                    "identity_rotation_rad_sum": 0.0,
                })
                model_com = torch.linalg.vector_norm(
                    prediction.com[sample, index] - targets.com[sample, index]
                )
                ballistic_com = torch.linalg.vector_norm(
                    prediction.ballistic_com[sample, index]
                    - targets.com[sample, index]
                )
                radius = targets.radius[sample]
                values["episodes"] += 1
                values["model_com_m_sum"] += float(model_com.cpu())
                values["ballistic_com_m_sum"] += float(ballistic_com.cpu())
                values["model_com_nrmse_sum"] += float(
                    (model_com / radius).cpu()
                )
                values["ballistic_com_nrmse_sum"] += float(
                    (ballistic_com / radius).cpu()
                )
                if bool(targets.valid_rotation[sample, index]):
                    model_rotation = rotation_geodesic(
                        prediction.rotation[sample, index],
                        targets.rotation[sample, index],
                    )
                    identity_rotation = rotation_geodesic(
                        torch.eye(
                            3, device=device, dtype=prediction.rotation.dtype,
                        ),
                        targets.rotation[sample, index],
                    )
                    values["rotation_valid"] += 1
                    values["model_rotation_rad_sum"] += float(
                        model_rotation.cpu()
                    )
                    values["identity_rotation_rad_sum"] += float(
                        identity_rotation.cpu()
                    )
    report = {
        "schema": "v42_gate1b_validation_baselines_v1",
        "split": "validation",
        "test_used": False,
        "comparisons": {
            "com": "learned ballistic residual versus ballistic COM",
            "rotation": "predicted rotation versus identity rotation",
        },
        "strata": {},
    }
    for (family, panel), group in sorted(accumulators.items()):
        result = {
            "uid_count": len(group["uids"]),
            "uids": sorted(group["uids"]),
            "horizons": {},
        }
        for horizon, values in sorted(group["horizons"].items()):
            episodes = values.pop("episodes")
            rotation_valid = values.pop("rotation_valid")
            result["horizons"][f"h{horizon}"] = {
                "episode_count": episodes,
                "rotation_valid_count": rotation_valid,
                "model_com_m": values["model_com_m_sum"] / episodes,
                "ballistic_com_m": (
                    values["ballistic_com_m_sum"] / episodes
                ),
                "model_com_nrmse": (
                    values["model_com_nrmse_sum"] / episodes
                ),
                "ballistic_com_nrmse": (
                    values["ballistic_com_nrmse_sum"] / episodes
                ),
                "model_rotation_rad": (
                    values["model_rotation_rad_sum"] / rotation_valid
                    if rotation_valid else None
                ),
                "identity_rotation_rad": (
                    values["identity_rotation_rad_sum"] / rotation_valid
                    if rotation_valid else None
                ),
            }
        report["strata"][f"{family}/panel_{panel}"] = result
    Path(output_path).write_text(json.dumps(report, indent=2) + "\n")
    return report


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
        "experiment": "v42_gate1b_ballistic_anchor_chordal_rotation",
        "model_contract_version": "gate1b_v1",
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
                "full_trajectory_chordal": 1.0,
                "key_horizon_chordal": 0.25,
                "rigid_fit": 0.25,
            },
            "com_parameterization": (
                "ballistic COM plus learned residual; residual exactly zero H1"
            ),
            "rotation_training_metric": (
                "0.5 * squared Frobenius/chordal distance"
            ),
            "rotation_reporting_metric": "clamped geodesic angle radians",
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
            "experiment", "model_contract_version", "seed",
            "draws_per_epoch", "hidden_dim", "blocks", "heads",
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
    best_state = torch.load(
        best_path, map_location=device, weights_only=False,
    )
    model.load_state_dict(best_state["model"])
    baseline_path = output / "VALIDATION_BASELINES.json"
    write_gate1_validation_baselines(
        model, validation_loader, torch.device(device),
        baseline_path, use_amp,
    )
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
        "validation_baselines_sha256": hashlib.sha256(
            baseline_path.read_bytes()
        ).hexdigest(),
    }
    (output / "RUN_COMPLETE.json").write_text(
        json.dumps(completion, indent=2) + "\n"
    )
    return best_path
