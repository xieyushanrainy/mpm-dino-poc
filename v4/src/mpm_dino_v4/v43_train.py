from __future__ import annotations

import json
import time
from dataclasses import replace
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from .model import masked_mean
from .v41_data import MODEL_INPUT_KEYS, UIDBalancedSampler, V41TrajectoryDataset
from .v41_train import atomic_torch_save, move, seed_all
from .v42_contact_curvature import DirectProbeConditionBuilder, oracle_floor_contact_features
from .v42_gate2 import _batch_targets_and_stages, file_sha256
from .v42_losses import compute_v42_local_losses
from .v42_model import V42RotationAwareSurrogate
from .v42_oracle import EVENT_STAGES, event_normalized_canonical_mse
from .v42_oracle import temporal_features
from .v43_field_attention import FieldAttentionModel, FieldBank, FieldEntry
from .v43_retrieval import (
    AttendedMechanicalMemory, MemoryEntry, RetrievalBank,
    materialize_aligned, retrieve,
)


ARMS = ("zero_memory", "geometry", "aligned_dino", "scene_shuffled", "point_shuffled")
MEMORY_DIM = 3 + 384 + 1 + 3 + 3
QUERY_DIM = 128 + 3 + 384 + 1 + 15


def load_champion(path, device):
    state = torch.load(path, map_location="cpu", weights_only=False)
    config = state["config"]
    if config.get("model_contract_version") != "direct_point_decoder_probe_v1":
        raise ValueError("V4.3 requires the reviewed V4.2 adapter-full contract")
    source = state["model"]
    hidden = source["v42_com_head.1.weight"].shape[0]
    blocks = 1 + max(int(k.split(".")[1]) for k in source if k.startswith("blocks."))
    model = V42RotationAwareSurrogate(
        local_mode="geometry", hidden_dim=hidden, blocks=blocks, heads=4,
        dropout=.1, frames=59, local_trunk_alpha=0.,
        rotation_parameterization="axis_angle", rotation_attention=True,
        rotation_dynamics=True, oracle_condition_dim=15,
        oracle_injection="adapter",
    ).to(device)
    model.load_state_dict(source, strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, state


def build_bank(root, manifest):
    dataset = V41TrajectoryDataset(root, manifest, "train", "real", 42,
                                   families=("soft_body",))
    entries, seen = [], set()
    for index in range(len(dataset)):
        item = dataset[index]
        if item["uid"] in seen or item["panel"] != "Z":
            continue
        seen.add(item["uid"])
        batch = {key: (value[None] if torch.is_tensor(value) else [value])
                 for key, value in item.items()}
        targets, stages = _batch_targets_and_stages(batch)
        point_mask = batch["input_mask"][0]
        coordinates = targets.reference_shape[0] / targets.radius[0]
        contact = oracle_floor_contact_features(batch)[0]
        for stage in EVENT_STAGES:
            selected = stages.labels[0, 1:].eq(int(stage))
            valid = batch["target_mask"][0] & selected[:, None]
            count = valid.sum(0)
            deformation = (targets.displacement[0] * valid[..., None]).sum(0)
            deformation = deformation / count.clamp_min(1)[..., None]
            contact_mean = (contact * valid[..., None]).sum(0)
            contact_mean = contact_mean / count.clamp_min(1)[..., None]
            entries.append(MemoryEntry(
                uid=item["uid"], split="train", stage=int(stage),
                event_time=.5, coordinates=coordinates.cpu(),
                dino=batch["dino"][0].cpu(),
                deformation=(deformation / targets.radius[0]).cpu(),
                point_valid=(point_mask & count.gt(0)).cpu(),
                dino_valid=batch["dino_valid"][0].cpu(),
                contact=contact_mean.cpu(), geometry_scale=float(targets.radius[0]),
                field_provenance=item["episode_id"] + ":canonical_target",
            ))
    return RetrievalBank(entries)


def build_field_bank(root, manifest):
    dataset = V41TrajectoryDataset(root, manifest, "train", "real", 42,
                                   families=("soft_body",))
    entries, seen = [], set()
    for index in range(len(dataset)):
        item = dataset[index]
        if item["uid"] in seen or item["panel"] != "Z":
            continue
        seen.add(item["uid"])
        batch = {key: (value[None] if torch.is_tensor(value) else [value])
                 for key, value in item.items()}
        targets, stages = _batch_targets_and_stages(batch)
        entries.append(FieldEntry(
            uid=item["uid"], split="train",
            coordinates=(targets.reference_shape[0] / targets.radius[0]).cpu(),
            dino=batch["dino"][0].cpu(),
            point_valid=batch["input_mask"][0].cpu(),
            dino_valid=batch["dino_valid"][0].cpu(),
            displacement=(targets.displacement[0] / targets.radius[0]).cpu(),
            contact=oracle_floor_contact_features(batch)[0].cpu(),
            stages=stages.labels[0, 1:].cpu(),
            event_time=temporal_features(stages, 59)[0, :, -1].cpu(),
            provenance=item["episode_id"] + ":canonical_trajectory",
        ))
    return FieldBank(entries)


class V43RetrievalModel(nn.Module):
    def __init__(self, base, bank, arm, top_k=3, memory_tokens=32, seed=42):
        super().__init__()
        self.base, self.bank, self.arm = base, bank, arm
        self.top_k, self.memory_tokens, self.seed = top_k, memory_tokens, seed
        self.memory = AttendedMechanicalMemory(
            QUERY_DIM, MEMORY_DIM, hidden_dim=128, heads=4,
        )

    def _memory(self, batch, stages, coordinates, query_dino, query_dino_valid,
                *, memory_dino_ablate=False, memory_ablate=False,
                correspondence_ablate=False):
        frames, n = batch["target"].shape[1:3]
        values = coordinates.new_zeros(frames, self.top_k, self.memory_tokens, MEMORY_DIM)
        masks = torch.zeros(frames, self.top_k, self.memory_tokens,
                            dtype=torch.bool, device=coordinates.device)
        cache = {}
        for frame in range(frames):
            stage = int(stages.labels[0, frame + 1])
            if stage not in [int(s) for s in EVENT_STAGES]:
                continue
            if stage not in cache:
                selected = retrieve(
                    self.bank, batch["uid"][0], "train" if batch.get("split") == "train" else "validation",
                    coordinates[0].detach().cpu(), query_dino[0].detach().cpu(),
                    batch["input_mask"][0].detach().cpu(), query_dino_valid[0].detach().cpu(),
                    stage=stage, mode=self.arm, k=self.top_k,
                    shuffle_seed=self.seed,
                )
                materialize_mode = "point_shuffled" if correspondence_ablate else self.arm
                tokens, valid = materialize_aligned(
                    coordinates[0].detach().cpu(), batch["input_mask"][0].detach().cpu(),
                    selected, materialize_mode, self.seed, self.memory_tokens,
                )
                if memory_dino_ablate:
                    tokens[..., 3:388] = 0
                if memory_ablate:
                    tokens.zero_(); valid.zero_()
                cache[stage] = (tokens.to(coordinates), valid.to(coordinates.device))
            values[frame], masks[frame] = cache[stage]
        return values[None], masks[None]

    def forward_batch(self, batch, stages, condition, split, ablation=None):
        ablation = ablation or {}
        inputs = {key: batch[key] for key in MODEL_INPUT_KEYS}
        inputs["oracle_condition"] = condition
        base_output = self.base(**inputs)
        reference_com = masked_mean(batch["x1"], batch["input_mask"])
        shape = batch["x1"] - reference_com[:, None]
        radius = torch.linalg.vector_norm(shape, dim=-1).masked_fill(
            ~batch["input_mask"], 0).amax(1).clamp_min(1e-6)
        coordinates = shape / radius[:, None, None]
        use_dino = self.arm in {"aligned_dino", "scene_shuffled", "point_shuffled"}
        query_dino = batch["dino"] if use_dino else torch.zeros_like(batch["dino"])
        query_dino_valid = batch["dino_valid"] if use_dino else torch.zeros_like(batch["dino_valid"])
        retrieval_dino, retrieval_dino_valid = query_dino, query_dino_valid
        if ablation.get("query_dino"):
            query_dino = torch.zeros_like(query_dino)
            query_dino_valid = torch.zeros_like(query_dino_valid)
        point_condition = condition
        query = torch.cat((
            base_output.physical_hidden.detach(),
            coordinates[:, None].expand(-1, base_output.physical_hidden.shape[1], -1, -1),
            query_dino[:, None].expand(-1, base_output.physical_hidden.shape[1], -1, -1),
            query_dino_valid[:, None, :, None].expand(-1, base_output.physical_hidden.shape[1], -1, -1).to(query_dino.dtype),
            point_condition,
        ), dim=-1)
        batch["split"] = split
        memory, memory_valid = self._memory(
            batch, stages, coordinates, retrieval_dino, retrieval_dino_valid,
            memory_dino_ablate=ablation.get("memory_dino", False),
            memory_ablate=ablation.get("memory", False),
            correspondence_ablate=ablation.get("correspondence", False),
        )
        displacement, gate = self.memory(
            base_output.canonical_displacement, query, memory, memory_valid,
            batch["input_mask"], return_gate=True,
        )
        canonical_shape = shape[:, None] + displacement
        rotated = torch.einsum("btni,btij->btnj", canonical_shape, base_output.rotation)
        output = replace(
            base_output, canonical_displacement=displacement,
            canonical_shape=canonical_shape,
            position=base_output.com[:, :, None] + rotated,
        )
        return output, gate, memory_valid


def run_epoch(model, loader, device, condition_builder, split, optimizer=None,
              accumulation=4, max_batches=None, ablation=None):
    training = optimizer is not None
    model.base.eval(); model.memory.train(training)
    if training: optimizer.zero_grad(set_to_none=True)
    total, rows, batches = 0., [], 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for raw in loader:
            batch = move(raw, device)
            targets, stages = _batch_targets_and_stages(batch)
            condition = condition_builder(batch, stages)
            output, gate, memory_valid = model.forward_batch(
                batch, stages, condition, split, ablation,
            )
            loss = event_normalized_canonical_mse(output, batch, targets, stages, None, None)
            labels = stages.labels[:, 1:]
            selected = torch.zeros_like(labels, dtype=torch.bool)
            for event_stage in EVENT_STAGES:
                selected |= labels.eq(int(event_stage))
            valid = (
                batch["target_mask"] & targets.valid_rotation[:, :, None]
                & selected[:, :, None]
            )
            error = output.canonical_displacement - targets.displacement
            canonical_nrmse = torch.sqrt(
                (error.square().sum(-1) * valid).sum()
                / (3 * valid.sum()).clamp_min(1)
            ) / targets.radius.mean().clamp_min(1e-8)
            dot = (output.canonical_displacement * targets.displacement).sum(-1)
            denominator = (
                torch.linalg.vector_norm(output.canonical_displacement, dim=-1)
                * torch.linalg.vector_norm(targets.displacement, dim=-1)
            ).clamp_min(1e-8)
            spatial_cosine = ((dot / denominator) * valid).sum() / valid.sum().clamp_min(1)
            pred_curve = torch.sqrt(
                (output.canonical_displacement.square().sum(-1) * batch["target_mask"]).sum(2)
                / batch["target_mask"].sum(2).clamp_min(1)
            )
            target_curve = torch.sqrt(
                (targets.displacement.square().sum(-1) * batch["target_mask"]).sum(2)
                / batch["target_mask"].sum(2).clamp_min(1)
            )
            peak_ratio = pred_curve.max() / target_curve.max().clamp_min(1e-8)
            local_losses = compute_v42_local_losses(
                output, batch, targets=targets,
                frame_weights=stages.weights[:, 1:],
                soft_deformation_amplification_cap=1.0,
                soft_deformation_quantile=.95,
                soft_deformation_floor_fraction=.005,
                family_balanced=True, rigid_family_weight=.25,
                rigid_zero_weight=0.0,
            )
            # The zero-memory arm is an exact frozen-base control and therefore
            # intentionally has no differentiable retrieval path.
            if training and loss.requires_grad:
                (loss / accumulation).backward()
            total += float(loss.detach()); batches += 1
            rows.append({"uid": batch["uid"][0], "objective": float(loss.detach()),
                         "canonical_nrmse": float(canonical_nrmse.detach()),
                         "spatial_cosine": float(spatial_cosine.detach()),
                         "predicted_to_target_peak_ratio": float(peak_ratio.detach()),
                         "strain_loss": float(local_losses.strain.detach()),
                         "edge_length_loss": float(local_losses.edge_length.detach()),
                         "gate_mean": float(gate.detach().mean()),
                         "valid_memory_tokens": int(memory_valid.sum())})
            if training and loss.requires_grad and batches % accumulation == 0:
                torch.nn.utils.clip_grad_norm_(model.memory.parameters(), 1.0)
                optimizer.step(); optimizer.zero_grad(set_to_none=True)
            if max_batches and batches >= max_batches: break
    if training and batches % accumulation and any(
        parameter.grad is not None for parameter in model.memory.parameters()
    ):
        torch.nn.utils.clip_grad_norm_(model.memory.parameters(), 1.0)
        optimizer.step(); optimizer.zero_grad(set_to_none=True)
    return total / max(batches, 1), rows


def train_v43(root, manifest, champion, bank, output, seed, arm, *, device="cuda",
              epochs=120, draws=40, lr=2e-4, accumulation=4, patience=20,
              max_batches=None, top_k=3, memory_tokens=32,
              architecture="compact_memory"):
    seed_all(seed); device = torch.device(device); output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    train_ds = V41TrajectoryDataset(root, manifest, "train", "real", seed,
                                    families=("soft_body",))
    val_ds = V41TrajectoryDataset(root, manifest, "validation", "real", seed,
                                  families=("soft_body",))
    train_loader = DataLoader(train_ds, batch_size=1,
                              sampler=UIDBalancedSampler(train_ds, draws, seed))
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)
    base, source = load_champion(champion, device)
    protected = {name: value.detach().cpu().clone()
                 for name, value in base.state_dict().items()}
    if architecture == "compact_memory":
        model = V43RetrievalModel(base, bank, arm, top_k, memory_tokens, seed)
    elif architecture == "explicit_field_attention":
        model = FieldAttentionModel(base, bank, seed, top_k)
    else:
        raise ValueError(f"unknown architecture: {architecture}")
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.memory.parameters(), lr=lr, weight_decay=1e-4)
    builder = DirectProbeConditionBuilder(True)
    config = {"experiment": "v43_compact_vs_field_attention_v1", "arm": arm,
              "architecture": architecture,
              "seed": seed, "epochs": epochs, "draws": draws, "lr": lr,
              "accumulation": accumulation, "patience": patience,
              "top_k": top_k, "memory_tokens": memory_tokens,
              "bank_sha256": bank.content_sha256,
              "trainable_parameters": sum(
                  parameter.numel() for parameter in model.memory.parameters()
              ),
              "champion": str(champion), "champion_sha256": file_sha256(champion),
              "champion_epoch": source["epoch"], "test_data_used": False}
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    best, stale, best_state = float("inf"), 0, None
    with (output / "history.jsonl").open("w") as history:
        for epoch in range(1, epochs + 1):
            started = time.time()
            train_loss, _ = run_epoch(model, train_loader, device, builder, "train",
                                      optimizer, accumulation, max_batches)
            val_loss, rows = run_epoch(model, val_loader, device, builder,
                                       "validation", max_batches=max_batches)
            improved = val_loss < best
            best, stale = min(best, val_loss), 0 if improved else stale + 1
            record = {"epoch": epoch, "train_objective": train_loss,
                      "validation_objective": val_loss, "validation_rows": rows,
                      "seconds": time.time() - started}
            history.write(json.dumps(record) + "\n"); history.flush()
            state = {"model": model.memory.state_dict(), "optimizer": optimizer.state_dict(),
                     "config": config, **record, "best": best, "stale": stale}
            atomic_torch_save(state, output / "last.pt")
            if improved:
                atomic_torch_save(state, output / "best.pt"); best_state = state
            print(f"arm={arm} seed={seed} epoch={epoch:03d} val={val_loss:.6f}", flush=True)
            if stale >= patience: break
    model.memory.load_state_dict(best_state["model"])
    fixed_weight_ablations = {}
    if arm == "aligned_dino":
        ablations = ({
            "zero_source_deformation": {"source_deformation": True},
            "zero_memory": {"memory": True},
            "point_shuffled_correspondence": {"correspondence": True},
        } if architecture == "explicit_field_attention" else {
            "zero_query_dino": {"query_dino": True},
            "zero_memory_dino": {"memory_dino": True},
            "zero_memory": {"memory": True},
            "point_shuffled_correspondence": {"correspondence": True},
        })
        for name, ablation in ablations.items():
            objective, rows = run_epoch(
                model, val_loader, device, builder, "validation",
                max_batches=max_batches, ablation=ablation,
            )
            fixed_weight_ablations[name] = {
                "validation_objective": objective, "validation_rows": rows,
            }
    protected_identical = all(
        torch.equal(base.state_dict()[name].detach().cpu(), value)
        for name, value in protected.items()
    )
    if not protected_identical:
        raise RuntimeError("protected V4.2 base changed")
    report = {"arm": arm, "seed": seed, "best_epoch": best_state["epoch"],
              "best_validation_objective": best_state["validation_objective"],
              "validation_rows": best_state["validation_rows"],
              "fixed_weight_ablations": fixed_weight_ablations,
              "bank_sha256": bank.content_sha256, "test_data_used": False}
    (output / "VALIDATION_RESULTS.json").write_text(json.dumps(report, indent=2) + "\n")
    complete = {"status": "complete", "best_epoch": best_state["epoch"],
                "best_validation_objective": best_state["validation_objective"],
                "best_checkpoint_sha256": file_sha256(output / "best.pt"),
                "last_checkpoint_sha256": file_sha256(output / "last.pt"),
                "protected_base_frozen": protected_identical,
                "fixed_weight_ablations_saved": bool(fixed_weight_ablations),
                "test_data_used": False}
    (output / "RUN_COMPLETE.json").write_text(json.dumps(complete, indent=2) + "\n")
