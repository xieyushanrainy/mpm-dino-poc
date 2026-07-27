from __future__ import annotations

import hashlib
import json
import random
import time
from pathlib import Path
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from .full_losses import (
    compute_full_trajectory_loss, compute_shape_balanced_trajectory_loss,
)
from .v41_data import MODEL_INPUT_KEYS, UIDBalancedSampler, V41TrajectoryDataset
from .v41_model import build_v41_model


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def move(batch, device):
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


def validation_scores(output, batch):
    radius = torch.linalg.vector_norm(batch["reference"] - batch["reference"].mean(1, keepdim=True), dim=-1).amax(1).clamp_min(1e-6)
    result = {}
    for horizon in (1, 16, 30, 40, 59):
        i = horizon - 1
        mask = batch["target_mask"][:, i]
        sq = (output.position[:, i] - batch["target"][:, i]).square().sum(-1)
        rmse = torch.sqrt((sq * mask).sum(1) / mask.sum(1).clamp_min(1))
        result[f"h{horizon}_rmse_m"] = rmse.mean()
        result[f"h{horizon}_nrmse"] = (rmse / radius).mean()
    result["selection_nrmse"] = torch.stack([result[f"h{h}_nrmse"] for h in (16,30,40)]).mean()
    return result


def run_epoch(model, loader, device, optimizer=None, accumulation=4, max_batches=None,
              scaler=None, amp=False, loss_profile="legacy"):
    training = optimizer is not None
    model.train(training)
    totals, batches = {}, 0
    if training: optimizer.zero_grad(set_to_none=True)
    with torch.enable_grad() if training else torch.no_grad():
        for raw in loader:
            batch = move(raw, device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                output = model(**{k: batch[k] for k in MODEL_INPUT_KEYS})
                if loss_profile == "legacy":
                    loss = compute_full_trajectory_loss(output, batch)
                elif loss_profile == "shape_balanced_v1":
                    loss = compute_shape_balanced_trajectory_loss(output, batch)
                else:
                    raise ValueError(f"unsupported V4.1 loss profile: {loss_profile}")
            if training:
                if scaler is not None:
                    scaler.scale(loss.total / accumulation).backward()
                else:
                    (loss.total / accumulation).backward()
                if (batches + 1) % accumulation == 0:
                    if scaler is not None:
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    if scaler is not None:
                        scaler.step(optimizer); scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
            values = {
                "loss": loss.total,
                **{
                    f"loss_{name}": getattr(loss, name)
                    for name in loss.__dataclass_fields__
                    if name != "total"
                },
                **validation_scores(output, batch),
            }
            for key, value in values.items(): totals[key] = totals.get(key, 0.) + float(value.detach().cpu())
            batches += 1
            if max_batches and batches >= max_batches: break
    if training and batches % accumulation:
        if scaler is not None:
            scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if scaler is not None:
            scaler.step(optimizer); scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    return {k: v/max(batches,1) for k,v in totals.items()}


def state_sha256(state):
    digest = hashlib.sha256()
    for key, value in sorted(state.items()):
        digest.update(key.encode()); digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def atomic_torch_save(value, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def rng_state() -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_state(value: dict) -> None:
    random.setstate(value["python"])
    np.random.set_state(value["numpy"])
    torch.set_rng_state(value["torch"])
    if value.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(value["cuda"])


def train_v41(root, manifest, output, mechanism, dino_mode, seed, device="mps",
              epochs=160, draws_per_epoch=40, hidden_dim=128, blocks=4, heads=4,
              dropout=0.1, lr=2e-4, trunk_lr=2e-5, accumulation=4, patience=30,
              max_batches=None, stage1_checkpoint=None, zero_reference=None,
              amp=None, resume=True, plateau_patience=5,
              loss_profile="legacy"):
    if loss_profile not in {"legacy", "shape_balanced_v1"}:
        raise ValueError(f"unsupported V4.1 loss profile: {loss_profile}")
    seed_all(seed)
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    train_ds = V41TrajectoryDataset(root, manifest, "train", dino_mode, seed)
    val_ds = V41TrajectoryDataset(root, manifest, "validation", dino_mode, seed)
    train_sampler = UIDBalancedSampler(train_ds, draws_per_epoch, seed)
    train_loader = DataLoader(train_ds, batch_size=1, sampler=train_sampler, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)
    model = build_v41_model(
        mechanism, hidden_dim=hidden_dim, blocks=blocks, heads=heads,
        dropout=dropout,
    ).to(device)
    starting_trunk = None
    if mechanism == "m6" and stage1_checkpoint:
        state = torch.load(stage1_checkpoint, map_location="cpu", weights_only=False)
        missing, unexpected = model.load_state_dict(state["model"], strict=False)
        if any(not key.startswith(("visual.", "dino_projection.")) for key in missing) or unexpected:
            raise ValueError(f"incompatible M6 trunk: {missing=} {unexpected=}")
        starting_trunk = state_sha256(model.trunk_state_dict())
        groups = [
            {"params": list(model.dino_projection.parameters()) + list(model.visual.parameters()), "lr": lr},
            {"params": [p for n,p in model.named_parameters() if not n.startswith(("dino_projection.","visual."))], "lr": trunk_lr},
        ]
        optimizer = torch.optim.AdamW(groups, weight_decay=1e-4)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=.5, patience=plateau_patience, threshold=.005,
        threshold_mode="rel", min_lr=1e-6,
    )
    use_amp = (str(device).startswith("cuda") if amp is None else bool(amp))
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    zero_h1 = None
    if zero_reference:
        zero_h1 = float(torch.load(zero_reference, map_location="cpu", weights_only=False)["validation"]["h1_rmse_m"])
    config = {
        "mechanism": mechanism, "dino_mode": dino_mode, "seed": seed, "device": device,
        "epochs": epochs, "draws_per_epoch": draws_per_epoch, "hidden_dim": hidden_dim,
        "blocks": blocks, "heads": heads, "dropout": dropout, "lr": lr,
        "trunk_lr": trunk_lr, "accumulation": accumulation, "patience": patience,
        "manifest_content_sha256": manifest["manifest_content_sha256"],
        "stage1_checkpoint": str(stage1_checkpoint) if stage1_checkpoint else None,
        "starting_trunk_sha256": starting_trunk,
        "zero_reference": str(zero_reference) if zero_reference else None,
        "amp": use_amp, "resume": resume,
        "plateau_patience": plateau_patience,
        "loss_profile": loss_profile,
        "loss": (
            {
                "implementation": "compute_full_trajectory_loss",
                "normalization": "world_metres",
                "smooth_l1_beta": 0.001,
                "weights": {
                    "residual": 1.0, "position": 1.0, "com": 0.5,
                    "edge_vector": 0.25, "edge_length": 0.1,
                    "key_horizons": 0.25,
                },
                "key_horizons": [4, 8, 16, 59],
            }
            if loss_profile == "legacy"
            else {
                "implementation": "compute_shape_balanced_trajectory_loss",
                "normalization": "per_object_fixed_reference_radius",
                "smooth_l1_beta": 0.01,
                "weights": {
                    "world": 1.0, "com": 0.5, "shape": 1.0,
                    "strain": 0.5, "key_horizons": 0.25,
                },
                "key_horizons": [16, 30, 40],
                "strain": "(edge_length-rest_length)/rest_length",
            }
        ),
        "architecture_contract": (
            "exact_v4_track_b_pooled_dino_film"
            if mechanism == "track_b_pooled"
            else "v41_correspondence_preserving"
        ),
    }
    (output/"config.json").write_text(json.dumps(config, indent=2)+"\n")
    best, stale, start_epoch, seen_eligible = float("inf"), 0, 1, zero_h1 is None
    last_path = output/"last.pt"
    if resume and last_path.exists() and not (output/"RUN_COMPLETE.json").exists():
        state = torch.load(last_path, map_location="cpu", weights_only=False)
        prior = state["config"]
        immutable = ("mechanism","dino_mode","seed","hidden_dim","blocks","heads",
                     "dropout","lr","trunk_lr","accumulation","manifest_content_sha256",
                     "draws_per_epoch","plateau_patience","loss_profile")
        if any(
            prior.get(k, "legacy" if k == "loss_profile" else None)
            != config.get(k)
            for k in immutable
        ):
            raise ValueError("refusing to resume with a changed scientific configuration")
        model.load_state_dict(state["model"])
        if "optimizer" in state: optimizer.load_state_dict(state["optimizer"])
        if "scheduler" in state: scheduler.load_state_dict(state["scheduler"])
        if "scaler" in state and use_amp: scaler.load_state_dict(state["scaler"])
        best=float(state.get("best",state["selection"]))
        stale=int(state.get("stale",0)); start_epoch=int(state["epoch"])+1
        seen_eligible=bool(state.get("seen_eligible", seen_eligible))
        train_sampler.epoch = int(state.get("sampler_epoch", state["epoch"]))
        if "rng_state" in state:
            restore_rng_state(state["rng_state"])
    mode = "a" if start_epoch > 1 else "w"
    epoch = start_epoch - 1
    with (output/"history.jsonl").open(mode) as history:
        for epoch in range(start_epoch, epochs+1):
            started=time.perf_counter()
            train=run_epoch(
                model, train_loader, torch.device(device), optimizer,
                accumulation, max_batches, scaler, use_amp, loss_profile,
            )
            validation=run_epoch(
                model, val_loader, torch.device(device), None,
                accumulation, max_batches, None, use_amp, loss_profile,
            )
            selection=validation["selection_nrmse"]; scheduler.step(selection)
            eligible = zero_h1 is None or validation["h1_rmse_m"] <= 1.10*zero_h1
            seen_eligible = seen_eligible or eligible
            record={"epoch":epoch,"train":train,"validation":validation,"selection":selection,
                    "h1_guard_eligible":eligible,"lr":[g["lr"] for g in optimizer.param_groups],
                    "seconds":time.perf_counter()-started}
            history.write(json.dumps(record)+"\n"); history.flush()
            improved = eligible and selection < best
            if improved:
                best=selection; stale=0
            elif seen_eligible:
                stale += 1
            state={"model":model.state_dict(),"optimizer":optimizer.state_dict(),
                   "scheduler":scheduler.state_dict(),"scaler":scaler.state_dict() if use_amp else None,
                   "config":config,"epoch":epoch,"validation":validation,
                   "selection":selection,"best":best,"stale":stale,
                   "seen_eligible":seen_eligible,
                   "sampler_epoch": train_sampler.epoch, "rng_state": rng_state()}
            atomic_torch_save(state,output/"last.pt")
            if improved:
                atomic_torch_save(state,output/"best.pt")
            print(f"epoch={epoch:03d} val={selection:.6g} h1={validation['h1_rmse_m']:.6g} eligible={eligible} stale={stale}",flush=True)
            if seen_eligible and stale >= patience: break
    guarded_checkpoint = output/"best.pt"
    if not guarded_checkpoint.exists():
        # Preserve the complete diagnostic run without mislabelling an
        # H1-ineligible checkpoint as scientifically selectable.
        diagnostic = output/"ineligible_last.pt"
        atomic_torch_save(torch.load(output/"last.pt", map_location="cpu", weights_only=False), diagnostic)
    completion = {
        "status": "complete" if guarded_checkpoint.exists() else "complete_no_h1_eligible_checkpoint",
        "last_epoch": epoch, "best_selection_nrmse": best if guarded_checkpoint.exists() else None,
        "h1_guard_ever_eligible": guarded_checkpoint.exists(),
        "best_checkpoint_sha256": (
            hashlib.sha256(guarded_checkpoint.read_bytes()).hexdigest()
            if guarded_checkpoint.exists() else None
        ),
        "last_checkpoint_sha256": hashlib.sha256((output/"last.pt").read_bytes()).hexdigest(),
        "history_sha256": hashlib.sha256((output/"history.jsonl").read_bytes()).hexdigest(),
    }
    (output/"RUN_COMPLETE.json").write_text(json.dumps(completion,indent=2)+"\n")
    return guarded_checkpoint if guarded_checkpoint.exists() else output/"last.pt"


def benchmark_v41(root, manifest, mechanism, device="mps", hidden_dim=128, blocks=4, heads=4):
    seed_all(42)
    ds=V41TrajectoryDataset(root,manifest,"train","real",42)
    batch=move(next(iter(DataLoader(ds,batch_size=1))),torch.device(device))
    model=build_v41_model(
        mechanism, hidden_dim=hidden_dim, blocks=blocks, heads=heads,
        dropout=.1,
    ).to(device).train()
    if device=="mps": torch.mps.empty_cache()
    started=time.perf_counter()
    output=model(**{k:batch[k] for k in MODEL_INPUT_KEYS})
    loss=compute_full_trajectory_loss(output,batch).total; loss.backward()
    if device=="mps": torch.mps.synchronize()
    elapsed=time.perf_counter()-started
    peak=int(torch.mps.driver_allocated_memory()) if device=="mps" else 0
    return {"mechanism":mechanism,"parameters":sum(p.numel() for p in model.parameters()),
            "trainable_parameters":sum(p.numel() for p in model.parameters() if p.requires_grad),
            "iteration_seconds":elapsed,"peak_driver_allocated_bytes":peak,
            "loss":float(loss.detach().cpu())}
