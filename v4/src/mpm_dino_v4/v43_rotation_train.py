from __future__ import annotations

import json
import math
import time
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from .model import masked_mean
from .v41_data import MODEL_INPUT_KEYS, UIDBalancedSampler, V41TrajectoryDataset
from .v41_train import atomic_torch_save, move, seed_all
from .v42_contact_curvature import DirectProbeConditionBuilder
from .v42_gate2 import _batch_targets_and_stages, file_sha256
from .v43_rotation_memory import (
    ARMS, CompactRotationReader, RotationBank, assert_protected_identity,
    geodesic_radians, protected_snapshot, retrieve_rotation, so3_exp,
)
from .v43_train import load_champion


MEMORY_DIM = 384 + 3 + 1 + 1 + 1  # pooled DINO, rotvec, valid, phase, geometry distance
QUERY_DIM = 128 + 384 + 1 + 3     # detached physical, pooled DINO, phase, geometry summary
HORIZONS = (1, 8, 16, 30, 40, 59)


def _pooled_dino(batch: dict) -> Tensor:
    valid = batch["input_mask"] & batch["dino_valid"]
    pooled = masked_mean(batch["dino"], valid)
    return torch.nn.functional.normalize(pooled, dim=-1)


def _query_geometry(batch: dict) -> tuple[Tensor, Tensor, Tensor]:
    centre = masked_mean(batch["x1"], batch["input_mask"])
    shape = batch["x1"] - centre[:, None]
    radius = torch.linalg.vector_norm(shape, dim=-1).masked_fill(
        ~batch["input_mask"], 0).amax(1).clamp_min(1e-6)
    coordinates = shape / radius[:, None, None]
    valid = batch["input_mask"]
    summary = torch.stack((
        radius,
        torch.linalg.vector_norm(coordinates, dim=-1).masked_fill(~valid, 0).sum(1)
        / valid.sum(1).clamp_min(1),
        valid.float().mean(1),
    ), -1)
    return coordinates, valid, summary


def _target_rotations(batch: dict) -> tuple[Tensor, Tensor]:
    targets, _ = _batch_targets_and_stages(batch)
    return targets.rotation, targets.valid_rotation


def _phase_labels(stages) -> tuple[Tensor, list[str]]:
    labels = stages.labels[:, 1:]
    # Stage numbering is retained in rows; broad phases are stable diagnostics.
    names = []
    for value in labels[0].tolist():
        names.append("pre_contact" if value <= 0 else ("contact" if value <= 2 else "post_contact"))
    phase = torch.linspace(0, 1, labels.shape[1], device=labels.device)[None]
    return phase, names


class RotationMemoryModel(nn.Module):
    def __init__(self, base: nn.Module, bank: RotationBank, arm: str, seed: int,
                 top_k: int, max_degrees: float):
        super().__init__()
        self.base, self.bank, self.arm, self.seed, self.top_k = base, bank, arm, seed, top_k
        self.reader = CompactRotationReader(QUERY_DIM, MEMORY_DIM, 128, 4, max_degrees)

    def _selected(self, batch, coordinates, valid, mode):
        return retrieve_rotation(
            self.bank, batch["uid"][0], batch["split_name"], coordinates[0].detach().cpu(),
            batch["dino"][0].detach().cpu(), valid[0].detach().cpu(),
            batch["dino_valid"][0].detach().cpu(), mode=mode, k=self.top_k, seed=self.seed,
        )

    def forward_batch(self, batch: dict, condition: Tensor, split: str, ablation: str | None = None):
        inputs = {key: batch[key] for key in MODEL_INPUT_KEYS}
        inputs["oracle_condition"] = condition
        with torch.no_grad():
            base_output = self.base(**inputs)
        coordinates, valid, geometry_summary = _query_geometry(batch)
        pooled = _pooled_dino(batch)
        targets, stages = _batch_targets_and_stages(batch)
        phase, phase_names = _phase_labels(stages)
        mode = self.arm
        if ablation == "wrong_memory": mode = "scene_shuffled"
        batch["split_name"] = split
        selected = self._selected(batch, coordinates, valid, mode)
        family_rows, uid_rows, memories, masks = [], [], [], []
        for entry in selected:
            dino = entry.pooled_dino.to(coordinates)
            if self.arm in {"zero_memory", "geometry"} or ablation == "zero_memory_dino":
                dino = torch.zeros_like(dino)
            distance = torch.cdist(coordinates[0, valid[0]].float(),
                                   entry.coordinates[entry.point_valid].to(coordinates).float())
            geometry_distance = (distance.min(1).values.square().mean()
                                 + distance.min(0).values.square().mean())
            token = torch.cat((
                dino[None].expand(59, -1), entry.rotation_vectors.to(coordinates),
                entry.kabsch_valid.to(coordinates)[:, None].float(),
                entry.event_phase.to(coordinates)[:, None],
                geometry_distance.reshape(1, 1).expand(59, 1),
            ), -1)
            memories.append(token); masks.append(entry.kabsch_valid.to(valid.device))
            family_rows.append(entry.family); uid_rows.append(entry.uid)
        memory = torch.stack(memories, 1)[None]
        memory_valid = torch.stack(masks, 1)[None]
        if self.arm == "zero_memory" or ablation == "zero_memory":
            memory.zero_(); memory_valid.zero_()
        query_dino = pooled
        if self.arm in {"zero_memory", "geometry"} or ablation == "zero_query_dino":
            query_dino = torch.zeros_like(query_dino)
        point_mask = batch["input_mask"][:, None, :, None]
        physical = (base_output.physical_hidden.detach() * point_mask).sum(2)
        physical = physical / point_mask.sum(2).clamp_min(1)
        query = torch.cat((physical, query_dino[:, None].expand(-1, 59, -1),
                           phase[..., None], geometry_summary[:, None].expand(-1, 59, -1)), -1)
        if not memory_valid.any():
            delta = query.new_zeros(query.shape[0], query.shape[1], 3)
            gate = query.new_zeros(query.shape[0], query.shape[1], 1)
            rotation = torch.eye(3, dtype=query.dtype, device=query.device).expand(
                query.shape[0], query.shape[1], 3, 3)
        else:
            rotation, delta, gate = self.reader(query, memory, memory_valid)
        return rotation, delta, gate, base_output.rotation, targets, stages, phase_names, uid_rows, family_rows


def _smoothness(delta: Tensor, valid: Tensor) -> Tensor:
    if delta.shape[1] < 3: return delta.new_zeros(())
    second = delta[:, 2:] - 2 * delta[:, 1:-1] + delta[:, :-2]
    mask = valid[:, 2:] & valid[:, 1:-1] & valid[:, :-2]
    return (second.square().sum(-1) * mask).sum() / mask.sum().clamp_min(1)


def run_epoch(model, loader, device, builder, split, optimizer=None, smoothness_weight=.01,
              accumulation=4, max_batches=None, ablation=None):
    training = optimizer is not None
    model.base.eval(); model.reader.train(training)
    if training: optimizer.zero_grad(set_to_none=True)
    total, rows, batches = 0., [], 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for raw in loader:
            batch = move(raw, device)
            _, stages = _batch_targets_and_stages(batch)
            condition = builder(batch, stages)
            rotation, delta, gate, v42_rotation, targets, stages, phase_names, source_uids, source_families = \
                model.forward_batch(batch, condition, split, ablation)
            target_rotation, valid = targets.rotation, targets.valid_rotation
            finite = torch.isfinite(target_rotation).all((-1, -2))
            valid = valid & finite
            target_rotation = torch.where(
                finite[..., None, None], target_rotation,
                torch.eye(3, dtype=target_rotation.dtype, device=device).expand_as(target_rotation),
            )
            errors = geodesic_radians(rotation, target_rotation)
            primary = errors[valid].mean() if valid.any() else errors.new_zeros(())
            smoothness = _smoothness(delta, valid)
            loss = primary + smoothness_weight * smoothness
            if training and loss.requires_grad:
                (loss / accumulation).backward()
            total += float(primary.detach()); batches += 1
            identity_errors = geodesic_radians(torch.eye(3, device=device).expand_as(target_rotation), target_rotation)
            v42_errors = geodesic_radians(v42_rotation, target_rotation)
            valid_values = errors[0, valid[0]]
            has_valid = bool(valid_values.numel())
            row = {"uid": batch["uid"][0], "family": batch["family"][0], "panel": batch["panel"][0],
                   "valid_kabsch_frames": int(valid[0].sum()),
                   "mean_error_deg": math.degrees(float(valid_values.detach().mean())) if has_valid else None,
                   "median_error_deg": math.degrees(float(valid_values.detach().median())) if has_valid else None,
                   "identity_mean_error_deg": math.degrees(float(identity_errors[0, valid[0]].mean())) if has_valid else None,
                   "v42_mean_error_deg": math.degrees(float(v42_errors[0, valid[0]].mean())) if has_valid else None,
                   "gate_mean": float(gate.detach().mean()), "source_uids": source_uids,
                   "source_families": source_families,
                   "cross_family_rate": sum(f != batch["family"][0] for f in source_families) / len(source_families)}
            row["improvement_vs_identity_deg"] = (
                row["identity_mean_error_deg"] - row["mean_error_deg"]
                if row["identity_mean_error_deg"] is not None and row["mean_error_deg"] is not None
                else None
            )
            static = valid[0] & identity_errors[0].le(math.radians(.25))
            row["inactive_static_false_rotation_deg"] = (
                math.degrees(float(geodesic_radians(
                    rotation[0, static], torch.eye(3, device=device).expand(int(static.sum()), 3, 3)
                ).mean())) if static.any() else None
            )
            for horizon in HORIZONS:
                index = horizon - 1
                row[f"H{horizon}_deg"] = math.degrees(float(errors[0, index])) if valid[0, index] else None
            for phase_name in ("pre_contact", "contact", "post_contact"):
                mask = valid[0] & torch.tensor([p == phase_name for p in phase_names], device=device)
                row[phase_name + "_deg"] = math.degrees(float(errors[0, mask].mean())) if mask.any() else None
            rows.append(row)
            if training and batches % accumulation == 0:
                torch.nn.utils.clip_grad_norm_(model.reader.parameters(), 1.0)
                optimizer.step(); optimizer.zero_grad(set_to_none=True)
            if max_batches and batches >= max_batches: break
    if training and batches % accumulation and any(p.grad is not None for p in model.reader.parameters()):
        torch.nn.utils.clip_grad_norm_(model.reader.parameters(), 1.0)
        optimizer.step(); optimizer.zero_grad(set_to_none=True)
    return total / max(batches, 1), rows


def summarize(rows: list[dict]) -> dict:
    def mean(key, selected):
        values = [r[key] for r in selected if r.get(key) is not None]
        return sum(values) / len(values) if values else None
    result = {"uid_rows": rows}
    for family in ("rigid", "soft_body"):
        selected = [r for r in rows if r["family"] == family]
        result[family] = {"mean_error_deg": mean("mean_error_deg", selected),
                          "identity_mean_error_deg": mean("identity_mean_error_deg", selected),
                          "median_of_uid_medians_deg": mean("median_error_deg", selected)}
    family_means = [result[f]["mean_error_deg"] for f in ("rigid", "soft_body") if result[f]["mean_error_deg"] is not None]
    result["family_balanced_mean_deg"] = sum(family_means) / len(family_means)
    result["cross_family_retrieval_rate"] = mean("cross_family_rate", rows)
    for panel in ("Z", "V"):
        result["panel_" + panel + "_mean_deg"] = mean("mean_error_deg", [r for r in rows if r["panel"] == panel])
    return result


def train_rotation_memory(root, manifest, champion, bank, output, seed, arm, *, device="cuda",
                          epochs=120, draws=40, lr=2e-4, accumulation=4, patience=20,
                          top_k=3, max_degrees=20., smoothness_weight=.01, max_batches=None):
    seed_all(seed); device = torch.device(device); output = Path(output); output.mkdir(parents=True, exist_ok=True)
    train_ds = V41TrajectoryDataset(root, manifest, "train", "real", seed, families=("rigid", "soft_body"))
    val_ds = V41TrajectoryDataset(root, manifest, "validation", "real", seed, families=("rigid", "soft_body"))
    train_loader = DataLoader(train_ds, batch_size=1, sampler=UIDBalancedSampler(train_ds, draws, seed))
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)
    base, source = load_champion(champion, device)
    protected = protected_snapshot(base)
    model = RotationMemoryModel(base, bank, arm, seed, top_k, max_degrees).to(device)
    optimizer = torch.optim.AdamW(model.reader.parameters(), lr=lr, weight_decay=1e-4)
    builder = DirectProbeConditionBuilder(True)
    config = {"experiment": "v43_rotation_memory_v1", "seed": seed, "arm": arm,
              "bank_sha256": bank.content_sha256, "champion": str(champion),
              "champion_sha256": file_sha256(champion), "champion_epoch": source["epoch"],
              "families": ["rigid", "soft_body"], "family_as_model_input": False,
              "top_k": top_k, "max_residual_degrees": max_degrees, "epochs": epochs,
              "draws": draws, "lr": lr, "accumulation": accumulation,
              "smoothness_weight": smoothness_weight, "test_data_used": False}
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    best, stale, best_state = float("inf"), 0, None
    with (output / "history.jsonl").open("w") as history:
        for epoch in range(1, epochs + 1):
            started = time.time()
            train_loss, _ = run_epoch(model, train_loader, device, builder, "train", optimizer,
                                      smoothness_weight, accumulation, max_batches)
            val_loss, rows = run_epoch(model, val_loader, device, builder, "validation",
                                       max_batches=max_batches)
            summary = summarize(rows); score = summary["family_balanced_mean_deg"]
            if score is None or not math.isfinite(score):
                raise RuntimeError("non-finite family-balanced validation score")
            improved = score < best; best, stale = min(best, score), 0 if improved else stale + 1
            record = {"epoch": epoch, "train_geodesic_rad": train_loss,
                      "validation": summary, "seconds": time.time() - started}
            history.write(json.dumps(record) + "\n"); history.flush()
            state = {"reader": model.reader.state_dict(), "optimizer": optimizer.state_dict(),
                     "config": config, **record, "best_family_balanced_deg": best}
            atomic_torch_save(state, output / "last.pt")
            if improved: atomic_torch_save(state, output / "best.pt"); best_state = state
            print(f"rotation arm={arm} seed={seed} epoch={epoch:03d} val_deg={score:.5f}", flush=True)
            if stale >= patience: break
    model.reader.load_state_dict(best_state["reader"])
    ablations = {}
    if arm == "aligned_dino":
        for name in ("zero_query_dino", "zero_memory_dino", "zero_memory", "wrong_memory"):
            _, rows = run_epoch(model, val_loader, device, builder, "validation", max_batches=max_batches,
                                ablation=name)
            ablations[name] = summarize(rows)
        # With pooled compact DINO, correspondence shuffle is mathematically invariant and is declared explicitly.
        ablations["point_shuffled_correspondence"] = {"not_applicable": True,
            "reason": "v1 compact reader retains pooled, not pointwise, DINO"}
    assert_protected_identity(base, protected)
    report = {"arm": arm, "seed": seed, "best_epoch": best_state["epoch"],
              "validation": best_state["validation"], "fixed_weight_ablations": ablations,
              "protected_state_bit_identical": True, "bank_sha256": bank.content_sha256,
              "test_data_used": False}
    (output / "VALIDATION_RESULTS.json").write_text(json.dumps(report, indent=2) + "\n")
    complete = {"status": "complete", "best_epoch": best_state["epoch"],
                "best_checkpoint_sha256": file_sha256(output / "best.pt"),
                "last_checkpoint_sha256": file_sha256(output / "last.pt"),
                "protected_com_deformation_bit_identical": True,
                "test_data_used": False}
    (output / "RUN_COMPLETE.json").write_text(json.dumps(complete, indent=2) + "\n")
    return report
