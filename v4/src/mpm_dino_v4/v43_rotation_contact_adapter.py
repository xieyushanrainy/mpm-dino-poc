from __future__ import annotations

import json
import math
import time
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from .v41_data import MODEL_INPUT_KEYS, UIDBalancedSampler, V41TrajectoryDataset
from .v41_train import atomic_torch_save, move, seed_all
from .v42_contact_curvature import (
    CONTACT_DIM, DIRECT_CONDITION_DIM, POINT_CONDITION_DIM,
    DirectProbeConditionBuilder,
)
from .v42_gate2 import _batch_targets_and_stages, file_sha256
from .v43_rotation_memory import (
    assert_protected_identity, geodesic_radians, protected_snapshot, so3_exp,
)
from .v43_rotation_train import HORIZONS, summarize
from .v43_train import load_champion


VARIANTS = (
    "physical_only", "pooled_contact", "pointwise_contact",
    "contact_lever", "contact_torque_basis", "contact_shuffled",
    "zero_event_time",
)
TOKEN_DIM = 128 + DIRECT_CONDITION_DIM + 3 + 3


def synchronized_rotation_tokens(batch: dict, physical_hidden: Tensor, condition: Tensor,
                                 variant: str, seed: int = 0) -> tuple[Tensor, Tensor]:
    """Build [B,T,N,F] tokens from the exact deformation condition tensor."""
    if variant not in VARIANTS:
        raise ValueError(variant)
    b, t, n = physical_hidden.shape[:3]
    point_condition = condition.clone()
    centre = (batch["x1"] * batch["input_mask"][..., None]).sum(1)
    centre = centre / batch["input_mask"].sum(1).clamp_min(1)[..., None]
    lever = batch["x1"] - centre[:, None]
    radius = torch.linalg.vector_norm(lever, dim=-1).masked_fill(
        ~batch["input_mask"], 0).amax(1).clamp_min(1e-6)
    lever = lever / radius[:, None, None]
    floor_normal = lever.new_tensor([0., 0., 1.]).expand_as(lever)
    torque_basis = torch.linalg.cross(lever, floor_normal, dim=-1)

    if variant == "physical_only":
        point_condition.zero_(); lever.zero_(); torque_basis.zero_()
    elif variant == "pooled_contact":
        valid = batch["target_mask"].to(point_condition.dtype)
        pooled = (point_condition * valid[..., None]).sum(2)
        pooled = pooled / valid.sum(2).clamp_min(1)[..., None]
        point_condition = pooled[:, :, None].expand_as(point_condition).clone()
        lever.zero_(); torque_basis.zero_()
    elif variant == "pointwise_contact":
        lever.zero_(); torque_basis.zero_()
    elif variant == "contact_lever":
        torque_basis.zero_()
    elif variant == "contact_shuffled":
        rows = []
        for index in range(b):
            generator = torch.Generator(device="cpu").manual_seed(seed + index * 1009)
            permutation = torch.randperm(n, generator=generator).to(point_condition.device)
            shuffled = point_condition[index].clone()
            # Shuffle contact+curvature together while preserving synchronized event time.
            shuffled[..., :POINT_CONDITION_DIM] = shuffled[:, permutation, :POINT_CONDITION_DIM]
            rows.append(shuffled)
        point_condition = torch.stack(rows)
    elif variant == "zero_event_time":
        point_condition[..., POINT_CONDITION_DIM:] = 0

    lever_frames = lever[:, None].expand(-1, t, -1, -1)
    torque_frames = torque_basis[:, None].expand(-1, t, -1, -1)
    tokens = torch.cat((physical_hidden.detach(), point_condition,
                        lever_frames, torque_frames), -1)
    valid = batch["target_mask"] & batch["input_mask"][:, None]
    return tokens, valid


class PointwiseContactRotationAdapter(nn.Module):
    """Contact-point attention with an identity-biased bounded Lie residual."""
    def __init__(self, hidden_dim=128, heads=4, max_degrees=20.):
        super().__init__()
        self.token = nn.Sequential(nn.Linear(TOKEN_DIM, hidden_dim), nn.GELU(),
                                   nn.Linear(hidden_dim, hidden_dim))
        self.query = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.attention = nn.MultiheadAttention(hidden_dim, heads, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
        self.residual = nn.Linear(hidden_dim, 3)
        self.gate = nn.Linear(hidden_dim, 1)
        nn.init.zeros_(self.residual.weight); nn.init.zeros_(self.residual.bias)
        nn.init.zeros_(self.gate.weight); nn.init.constant_(self.gate.bias, -4.6)
        self.max_radians = math.radians(max_degrees)

    def forward(self, tokens: Tensor, valid: Tensor):
        b, t, n, _ = tokens.shape
        encoded = self.token(tokens).reshape(b * t, n, -1)
        mask = valid.reshape(b * t, n)
        active = mask.any(-1)
        safe = mask.clone(); safe[:, 0] |= ~active
        query = self.query.expand(b * t, -1, -1)
        pooled, _ = self.attention(query, encoded, encoded, key_padding_mask=~safe)
        hidden = self.norm(pooled[:, 0]).reshape(b, t, -1)
        gate = torch.sigmoid(self.gate(hidden)) * active.reshape(b, t, 1)
        delta = gate * self.max_radians * torch.tanh(self.residual(hidden))
        return so3_exp(delta), delta, gate


def rotation_activity_loss(rotation: Tensor, delta: Tensor, target: Tensor, valid: Tensor,
                           static_weight=.25, smoothness_weight=.01):
    identity = torch.eye(3, dtype=target.dtype, device=target.device).expand_as(target)
    target_angle = geodesic_radians(identity, target)
    error = geodesic_radians(rotation, target)
    active = valid & target_angle.ge(math.radians(.5))
    static = valid & target_angle.lt(math.radians(.25))
    active_loss = error[active].mean() if active.any() else error[valid].mean()
    false_rotation = geodesic_radians(rotation, identity)
    static_loss = false_rotation[static].mean() if static.any() else error.new_zeros(())
    second = delta[:, 2:] - 2 * delta[:, 1:-1] + delta[:, :-2]
    smooth_valid = valid[:, 2:] & valid[:, 1:-1] & valid[:, :-2]
    smooth = second.square().sum(-1)[smooth_valid].mean() if smooth_valid.any() else error.new_zeros(())
    return active_loss + static_weight * static_loss + smoothness_weight * smooth, active_loss, static_loss


def _rows(rotation, gate, base_rotation, targets, batch, valid):
    target = targets.rotation
    errors = geodesic_radians(rotation, target)
    identity = torch.eye(3, device=rotation.device).expand_as(target)
    identity_errors = geodesic_radians(identity, target)
    base_errors = geodesic_radians(base_rotation, target)
    rows = []
    for index in range(rotation.shape[0]):
        mask = valid[index]
        values = errors[index, mask].detach()
        row = {"uid": batch["uid"][index], "family": batch["family"][index],
               "panel": batch["panel"][index], "valid_kabsch_frames": int(mask.sum()),
               "mean_error_deg": math.degrees(float(values.mean())) if values.numel() else None,
               "median_error_deg": math.degrees(float(values.median())) if values.numel() else None,
               "identity_mean_error_deg": math.degrees(float(identity_errors[index, mask].mean())) if mask.any() else None,
               "v42_mean_error_deg": math.degrees(float(base_errors[index, mask].mean())) if mask.any() else None,
               "gate_mean": float(gate[index].detach().mean()), "cross_family_rate": 0.}
        row["improvement_vs_identity_deg"] = (row["identity_mean_error_deg"] - row["mean_error_deg"]
                                                if row["mean_error_deg"] is not None else None)
        static = mask & identity_errors[index].le(math.radians(.25))
        row["inactive_static_false_rotation_deg"] = math.degrees(float(
            geodesic_radians(rotation[index, static], identity[index, static]).detach().mean()
        )) if static.any() else None
        for horizon in HORIZONS:
            j = horizon - 1
            row[f"H{horizon}_deg"] = math.degrees(float(errors[index, j].detach())) if mask[j] else None
        rows.append(row)
    return rows


def run_epoch(model, base, loader, builder, device, variant, seed, optimizer=None,
              accumulation=4, max_batches=None):
    training = optimizer is not None
    model.train(training); base.eval()
    if training: optimizer.zero_grad(set_to_none=True)
    total, rows, batches = 0., [], 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for raw in loader:
            batch = move(raw, device)
            targets, stages = _batch_targets_and_stages(batch)
            condition = builder(batch, stages)
            with torch.no_grad():
                base_output = base(**{**{k: batch[k] for k in MODEL_INPUT_KEYS},
                                      "oracle_condition": condition})
            tokens, point_valid = synchronized_rotation_tokens(
                batch, base_output.physical_hidden, condition, variant, seed,
            )
            rotation, delta, gate = model(tokens, point_valid)
            finite = torch.isfinite(targets.rotation).all((-1, -2))
            valid = targets.valid_rotation & finite
            target = torch.where(finite[..., None, None], targets.rotation,
                                 torch.eye(3, device=device).expand_as(targets.rotation))
            loss, active, static = rotation_activity_loss(rotation, delta, target, valid)
            if training:
                (loss / accumulation).backward()
            total += float(loss.detach()); batches += 1
            rows.extend(_rows(rotation, gate, base_output.rotation, targets, batch, valid))
            if training and batches % accumulation == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step(); optimizer.zero_grad(set_to_none=True)
            if max_batches and batches >= max_batches: break
    if training and batches % accumulation:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step(); optimizer.zero_grad(set_to_none=True)
    return total / max(1, batches), rows


def train_contact_adapter(root, manifest, champion, output, seed, variant, *, device="cuda",
                          epochs=120, draws=40, lr=2e-4, accumulation=4, patience=20,
                          max_degrees=20., max_batches=None):
    seed_all(seed); device = torch.device(device); output = Path(output); output.mkdir(parents=True, exist_ok=True)
    train_ds = V41TrajectoryDataset(root, manifest, "train", "real", seed, families=("rigid", "soft_body"))
    val_ds = V41TrajectoryDataset(root, manifest, "validation", "real", seed, families=("rigid", "soft_body"))
    train_loader = DataLoader(train_ds, batch_size=1, sampler=UIDBalancedSampler(train_ds, draws, seed))
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)
    base, source = load_champion(champion, device); protected = protected_snapshot(base)
    model = PointwiseContactRotationAdapter(max_degrees=max_degrees).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    builder = DirectProbeConditionBuilder(True)
    # Cache a protected output, not only parameters.
    fixed_raw = next(iter(val_loader)); fixed = move(fixed_raw, device)
    fixed_targets, fixed_stages = _batch_targets_and_stages(fixed)
    fixed_condition = builder(fixed, fixed_stages)
    with torch.no_grad():
        fixed_before = base(**{**{k: fixed[k] for k in MODEL_INPUT_KEYS},
                               "oracle_condition": fixed_condition})
        com_before = fixed_before.com.detach().cpu().clone()
        deformation_before = fixed_before.canonical_displacement.detach().cpu().clone()
    config = {"experiment": "v43_no_memory_pointwise_contact_rotation_v1", "variant": variant,
              "seed": seed, "champion": str(champion), "champion_sha256": file_sha256(champion),
              "champion_epoch": source["epoch"], "memory_bank_used": False,
              "condition_contract": "shared_deformation_15d_plus_lever_and_cross_normal",
              "oracle_contact": True, "test_data_used": False, "epochs": epochs,
              "draws": draws, "lr": lr, "accumulation": accumulation,
              "max_residual_degrees": max_degrees,
              "trainable_parameters": sum(p.numel() for p in model.parameters())}
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    best, stale, best_state = float("inf"), 0, None
    with (output / "history.jsonl").open("w") as history:
        for epoch in range(1, epochs + 1):
            started = time.time()
            train_loss, _ = run_epoch(model, base, train_loader, builder, device, variant, seed,
                                      optimizer, accumulation, max_batches)
            _, rows = run_epoch(model, base, val_loader, builder, device, variant, seed,
                                max_batches=max_batches)
            summary = summarize(rows); score = summary["family_balanced_mean_deg"]
            if score is None or not math.isfinite(score): raise RuntimeError("non-finite validation score")
            improved = score < best; best, stale = min(best, score), 0 if improved else stale + 1
            record = {"epoch": epoch, "train_objective": train_loss, "validation": summary,
                      "seconds": time.time() - started}
            history.write(json.dumps(record) + "\n"); history.flush()
            state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                     "config": config, **record, "best_family_balanced_deg": best}
            atomic_torch_save(state, output / "last.pt")
            if improved: atomic_torch_save(state, output / "best.pt"); best_state = state
            print(f"contact-rotation variant={variant} seed={seed} epoch={epoch:03d} val_deg={score:.5f}", flush=True)
            if stale >= patience: break
    model.load_state_dict(best_state["model"])
    assert_protected_identity(base, protected)
    with torch.no_grad():
        fixed_after = base(**{**{k: fixed[k] for k in MODEL_INPUT_KEYS},
                              "oracle_condition": fixed_condition})
    output_identical = (torch.equal(com_before, fixed_after.com.detach().cpu()) and
                        torch.equal(deformation_before, fixed_after.canonical_displacement.detach().cpu()))
    if not output_identical: raise RuntimeError("protected COM/deformation output changed")
    report = {"variant": variant, "seed": seed, "best_epoch": best_state["epoch"],
              "validation": best_state["validation"], "memory_bank_used": False,
              "protected_state_bit_identical": True, "protected_output_bit_identical": True,
              "test_data_used": False}
    (output / "VALIDATION_RESULTS.json").write_text(json.dumps(report, indent=2) + "\n")
    complete = {"status": "complete", "best_epoch": best_state["epoch"],
                "best_checkpoint_sha256": file_sha256(output / "best.pt"),
                "last_checkpoint_sha256": file_sha256(output / "last.pt"),
                "memory_bank_used": False, "protected_com_deformation_bit_identical": True,
                "protected_output_bit_identical": True, "test_data_used": False}
    (output / "RUN_COMPLETE.json").write_text(json.dumps(complete, indent=2) + "\n")
