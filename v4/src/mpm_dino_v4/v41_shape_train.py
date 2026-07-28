from __future__ import annotations

import hashlib
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Sampler

from .full_losses import compute_local_shape_trajectory_loss
from .model import masked_mean
from .v41_data import MODEL_INPUT_KEYS, V41TrajectoryDataset
from .v41_model import build_v41_model
from .v41_train import (
    atomic_torch_save, move, restore_rng_state, rng_state, seed_all,
    state_sha256,
)


def local_shape_scores(output, batch):
    target, mask = batch["target"], batch["target_mask"]
    reference, input_mask = batch["reference"], batch["input_mask"]
    reference_com = masked_mean(reference, input_mask)
    radius = torch.linalg.vector_norm(
        reference - reference_com[:, None], dim=-1
    ).masked_fill(~input_mask, 0).amax(1).clamp_min(1e-6)
    target_com = masked_mean(target, mask, dim=2)
    target_shape = target - target_com[:, :, None]
    ballistic_com = masked_mean(output.ballistic, mask, dim=2)
    predicted_shape = (
        output.ballistic - ballistic_com[:, :, None]
        + output.residual_local
    )
    predicted_shape = (
        predicted_shape
        - masked_mean(predicted_shape, mask, dim=2)[:, :, None]
    )
    result = {}
    for horizon in (1, 8, 16, 30, 40, 59):
        index = horizon - 1
        point_mask = mask[:, index]
        squared = (
            predicted_shape[:, index] - target_shape[:, index]
        ).square().sum(-1)
        rmse = torch.sqrt(
            (squared * point_mask).sum(1)
            / point_mask.sum(1).clamp_min(1)
        )
        result[f"h{horizon}_shape_rmse_m"] = rmse.mean()
        result[f"h{horizon}_shape_nrmse"] = (rmse / radius).mean()
    result["selection_shape_nrmse"] = torch.stack([
        result[f"h{horizon}_shape_nrmse"]
        for horizon in (16, 30, 40)
    ]).mean()
    return result


def run_local_shape_epoch(
    model, loader, device, optimizer=None, accumulation=4,
    scaler=None, amp=False, max_batches=None,
):
    training = optimizer is not None
    model.train(training)
    totals, counts, batches = {}, {}, 0
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
                loss = compute_local_shape_trajectory_loss(output, batch)
            if training:
                value = loss.total / accumulation
                if scaler is not None:
                    scaler.scale(value).backward()
                else:
                    value.backward()
                if (batches + 1) % accumulation == 0:
                    if scaler is not None:
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    if scaler is not None:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
            scores = local_shape_scores(output, batch)
            values = {
                "loss": loss.total,
                **{
                    f"loss_{name}": getattr(loss, name)
                    for name in loss.__dataclass_fields__
                    if name != "total"
                },
                **scores,
                **{
                    f"{batch['family'][0]}_{key}": value
                    for key, value in scores.items()
                },
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
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    return {
        key: value / max(counts[key], 1)
        for key, value in totals.items()
    }


def physical_trunk_sha256(model):
    excluded = (
        "dino_projection.", "region_encoder.", "visual.",
        "local_head.", "com_head.",
    )
    return state_sha256({
        key: value for key, value in model.state_dict().items()
        if not key.startswith(excluded)
    })


class Phase2FamilySampler(Sampler[int]):
    """Deterministic 3:1 soft/rigid draws, balanced by UID within family."""

    def __init__(self, dataset, draws, seed, soft_fraction=0.75):
        self.dataset = dataset
        self.draws = draws
        self.seed = seed
        self.soft_fraction = soft_fraction
        self.epoch = 0
        self.by_family = defaultdict(list)
        for uid in dataset.uids:
            family = dataset.rows[dataset.by_uid[uid][0]]["family"]
            self.by_family[family].append(uid)
        if not {"soft_body", "rigid"} <= self.by_family.keys():
            raise ValueError("Phase 2 requires both soft_body and rigid UIDs")

    def __len__(self):
        return self.draws

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        self.epoch += 1
        soft_draws = round(self.draws * self.soft_fraction)
        schedule = ["soft_body"] * soft_draws + [
            "rigid"
        ] * (self.draws - soft_draws)
        order = torch.randperm(len(schedule), generator=generator).tolist()
        for position in order:
            family = schedule[position]
            uids = self.by_family[family]
            uid = uids[int(torch.randint(
                len(uids), (), generator=generator,
            ))]
            episodes = self.dataset.by_uid[uid]
            yield episodes[int(torch.randint(
                len(episodes), (), generator=generator,
            ))]


def train_v41_local_shape(
    root, manifest, output, condition, seed, device="cuda",
    epochs=80, draws_per_epoch=40, hidden_dim=128, blocks=4, heads=4,
    dropout=0.1, lr=2e-4, accumulation=4, patience=15,
    plateau_patience=5, amp=False, resume=True, max_batches=None,
    geometry_reference=None,
):
    contracts = {
        "physical_only": ("none", "zero"),
        "geometry_tokens": ("split_region", "zero"),
        "real_dino": ("split_region", "real"),
        "point_shuffled": ("split_region", "point_shuffled"),
    }
    if condition not in contracts:
        raise ValueError(condition)
    mechanism, dino_mode = contracts[condition]
    seed_all(seed)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    train_ds = V41TrajectoryDataset(
        root, manifest, "train", dino_mode, seed,
        families=("soft_body", "rigid"),
    )
    validation_ds = V41TrajectoryDataset(
        root, manifest, "validation", dino_mode, seed,
        families=("soft_body", "rigid"),
    )
    sampler = Phase2FamilySampler(
        train_ds, draws_per_epoch, seed, soft_fraction=0.75,
    )
    train_loader = DataLoader(
        train_ds, batch_size=1, sampler=sampler, num_workers=0,
    )
    validation_loader = DataLoader(
        validation_ds, batch_size=1, shuffle=False, num_workers=0,
    )
    model = build_v41_model(
        mechanism, hidden_dim=hidden_dim, blocks=blocks, heads=heads,
        dropout=dropout,
    ).to(device)
    starting_model = state_sha256(model.state_dict())
    starting_physical_trunk = physical_trunk_sha256(model)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=plateau_patience,
        threshold=0.005, threshold_mode="rel", min_lr=1e-6,
    )
    use_amp = bool(amp)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    geometry_h1 = None
    if geometry_reference is not None:
        reference_state = torch.load(
            geometry_reference, map_location="cpu", weights_only=False,
        )
        geometry_h1 = float(
            reference_state["validation"]["soft_body_h1_shape_nrmse"]
        )
    config = {
        "experiment": "v41_com_normalized_local_shape_phase2",
        "condition": condition,
        "mechanism": mechanism,
        "dino_mode": dino_mode,
        "seed": seed,
        "device": device,
        "epochs": epochs,
        "draws_per_epoch": draws_per_epoch,
        "hidden_dim": hidden_dim,
        "blocks": blocks,
        "heads": heads,
        "dropout": dropout,
        "lr": lr,
        "accumulation": accumulation,
        "patience": patience,
        "plateau_patience": plateau_patience,
        "amp": use_amp,
        "families": ["soft_body", "rigid"],
        "sampling": {
            "soft_fraction": 0.75,
            "rigid_fraction": 0.25,
            "draws_at_default_40": {"soft_body": 30, "rigid": 10},
        },
        "manifest_content_sha256": manifest["manifest_content_sha256"],
        "starting_model_sha256": starting_model,
        "starting_physical_trunk_sha256": starting_physical_trunk,
        "geometry_reference": (
            str(geometry_reference) if geometry_reference else None
        ),
        "loss": {
            "implementation": "compute_local_shape_trajectory_loss",
            "future_com_model_input": False,
            "world_position_loss": False,
            "com_loss": False,
            "normalization": "fixed_reference_radius",
            "weights": {
                "center_relative_shape": 1.0,
                "edge_vector": 0.25,
                "normalized_edge_strain": 0.5,
                "key_horizons": 0.25,
                "rigid_zero_local_residual": 0.25,
            },
            "key_horizons": [16, 30, 40, 59],
            "frame_weighting": (
                "0.25 + normalized ground-truth centre-relative "
                "deformation; normalized to mean one per episode"
            ),
        },
        "selection": {
            "split": "validation",
            "metric": "mean shape NRMSE at H16/H30/H40",
            "family": "soft_body",
            "h1_guard": (
                "no more than 10% worse than geometry_tokens"
                if geometry_reference else None
            ),
        },
    }
    (output / "config.json").write_text(
        json.dumps(config, indent=2) + "\n"
    )

    best = float("inf")
    stale = 0
    start_epoch = 1
    seen_eligible = geometry_h1 is None
    last_path = output / "last.pt"
    if resume and last_path.exists() and not (
        output / "RUN_COMPLETE.json"
    ).exists():
        state = torch.load(
            last_path, map_location="cpu", weights_only=False,
        )
        prior = state["config"]
        immutable = (
            "condition", "mechanism", "dino_mode", "seed", "epochs",
            "draws_per_epoch", "hidden_dim", "blocks", "heads", "dropout",
            "lr", "accumulation", "patience", "plateau_patience", "amp",
            "manifest_content_sha256",
        )
        if any(prior.get(key) != config.get(key) for key in immutable):
            raise ValueError(
                "refusing to resume with changed Phase-2 configuration"
            )
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        if use_amp and state.get("scaler"):
            scaler.load_state_dict(state["scaler"])
        best = float(state.get("best", state["selection"]))
        stale = int(state.get("stale", 0))
        start_epoch = int(state["epoch"]) + 1
        seen_eligible = bool(
            state.get("seen_eligible", seen_eligible)
        )
        sampler.epoch = int(
            state.get("sampler_epoch", state["epoch"])
        )
        if "rng_state" in state:
            restore_rng_state(state["rng_state"])

    mode = "a" if start_epoch > 1 else "w"
    epoch = start_epoch - 1
    with (output / "history.jsonl").open(mode) as history:
        for epoch in range(start_epoch, epochs + 1):
            started = time.perf_counter()
            train = run_local_shape_epoch(
                model, train_loader, torch.device(device), optimizer,
                accumulation, scaler, use_amp, max_batches,
            )
            validation = run_local_shape_epoch(
                model, validation_loader, torch.device(device), None,
                accumulation, None, use_amp, max_batches,
            )
            if "soft_body_selection_shape_nrmse" not in validation:
                raise ValueError(
                    "validation produced no soft_body examples; "
                    "Phase-2 selection requires the full soft validation set"
                )
            selection = validation["soft_body_selection_shape_nrmse"]
            scheduler.step(selection)
            eligible = (
                geometry_h1 is None
                or validation["soft_body_h1_shape_nrmse"]
                <= 1.10 * geometry_h1
            )
            seen_eligible = seen_eligible or eligible
            record = {
                "epoch": epoch,
                "train": train,
                "validation": validation,
                "selection": selection,
                "h1_guard_eligible": eligible,
                "lr": [group["lr"] for group in optimizer.param_groups],
                "seconds": time.perf_counter() - started,
            }
            history.write(json.dumps(record) + "\n")
            history.flush()
            improved = eligible and selection < best
            if improved:
                best = selection
                stale = 0
            elif seen_eligible:
                stale += 1
            state = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict() if use_amp else None,
                "config": config,
                "epoch": epoch,
                "validation": validation,
                "selection": selection,
                "best": best,
                "stale": stale,
                "seen_eligible": seen_eligible,
                "sampler_epoch": sampler.epoch,
                "rng_state": rng_state(),
            }
            atomic_torch_save(state, output / "last.pt")
            if improved:
                atomic_torch_save(state, output / "best.pt")
            print(
                f"epoch={epoch:03d} shape={selection:.6g} "
                f"h1={validation['soft_body_h1_shape_nrmse']:.6g} "
                f"eligible={eligible} stale={stale}",
                flush=True,
            )
            if seen_eligible and stale >= patience:
                break

    best_path = output / "best.pt"
    completion = {
        "status": (
            "complete" if best_path.exists()
            else "complete_no_h1_eligible_checkpoint"
        ),
        "last_epoch": epoch,
        "best_selection_shape_nrmse": (
            best if best_path.exists() else None
        ),
        "h1_guard_ever_eligible": best_path.exists(),
        "best_checkpoint_sha256": (
            hashlib.sha256(best_path.read_bytes()).hexdigest()
            if best_path.exists() else None
        ),
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
    return best_path if best_path.exists() else output / "last.pt"
