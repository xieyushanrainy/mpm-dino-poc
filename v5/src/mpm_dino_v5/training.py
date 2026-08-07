from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from mpm_dino_v4.v41_data import UIDBalancedSampler
from mpm_dino_v4.v42_losses import compute_v42_global_losses

from .config import V5Config
from .data import dataset, interaction_labels, model_inputs, targets_and_stages
from .losses import (
    com_trajectory_loss,
    event_normalized_deformation_mse,
    interaction_auxiliary_loss,
    rotation_geodesic_mean,
)
from .memory import V5MemoryBank, V5MemoryModule
from .model import (
    V5COMModel,
    V5DeformationDecoder,
    V5InteractionEncoder,
    V5RotationModel,
    V5SharedPhysicalModel,
    centered_reference,
    reconstruct_positions,
)
from .staging import assert_unchanged, freeze, snapshot


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def move(batch: dict, device: torch.device) -> dict:
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def atomic_save(value: dict, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


@dataclass(frozen=True)
class TrainOptions:
    epochs: int = 120
    draws_per_epoch: int = 40
    learning_rate: float = 2e-4
    weight_decay: float = 1e-4
    patience: int = 20
    accumulation: int = 4
    max_batches: int | None = None


def loaders(root, manifest, seed, families, options: TrainOptions):
    train = dataset(root, manifest, "train", families=families, seed=seed)
    validation = dataset(root, manifest, "validation", families=families, seed=seed)
    sampler = UIDBalancedSampler(train, options.draws_per_epoch, seed)
    return (
        DataLoader(train, batch_size=1, sampler=sampler, num_workers=0),
        DataLoader(validation, batch_size=1, shuffle=False, num_workers=0),
    )


def _optimizer(module: nn.Module, options: TrainOptions):
    parameters = [parameter for parameter in module.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError("stage has no trainable parameters")
    return torch.optim.AdamW(parameters, lr=options.learning_rate, weight_decay=options.weight_decay)


def _run_batches(module, loader, device, step, optimizer=None, accumulation=4, max_batches=None):
    training = optimizer is not None
    module.train(training)
    if training:
        optimizer.zero_grad(set_to_none=True)
    total = 0.0
    count = 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for raw in loader:
            batch = move(raw, device)
            loss = step(batch)
            if training:
                (loss / accumulation).backward()
                if (count + 1) % accumulation == 0:
                    torch.nn.utils.clip_grad_norm_(module.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
            total += float(loss.detach().cpu())
            count += 1
            if max_batches is not None and count >= max_batches:
                break
    if training and count % accumulation:
        torch.nn.utils.clip_grad_norm_(module.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    return total / max(count, 1)


def _fit(stage: str, module: nn.Module, train_loader, validation_loader, device, step, output: Path, seed: int, config: V5Config, options: TrainOptions, protected=(), metadata=None):
    output.mkdir(parents=True, exist_ok=True)
    optimizer = _optimizer(module, options)
    protected_hashes = snapshot(protected)
    best = float("inf")
    bad_epochs = 0
    history = []
    for epoch in range(1, options.epochs + 1):
        train_score = _run_batches(module, train_loader, device, step, optimizer, options.accumulation, options.max_batches)
        validation_score = _run_batches(module, validation_loader, device, step, None, options.accumulation, options.max_batches)
        assert_unchanged(protected, protected_hashes)
        history.append({"epoch": epoch, "train": train_score, "validation": validation_score})
        if validation_score < best:
            best = validation_score
            bad_epochs = 0
            atomic_save({
                "contract": "v5_random_initialization_stage_v1",
                "architecture_contract": getattr(module, "contract_version", None),
                "stage": stage,
                "seed": seed,
                "model": module.state_dict(),
                "validation": validation_score,
                "config": config.to_dict(),
                "options": options.__dict__,
                "protected": protected_hashes,
                "metadata": dict(metadata or {}),
            }, output / "best.pt")
        else:
            bad_epochs += 1
        if bad_epochs >= options.patience:
            break
    (output / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    return best


def build_modules(config: V5Config, device="cpu"):
    config.validate()
    values = config.model
    return (
        V5COMModel(
            values.hidden_dim, values.frames, values.dropout,
            values.blocks, values.heads,
        ).to(device),
        V5RotationModel(values.hidden_dim, values.frames, values.dropout).to(device),
        V5InteractionEncoder(values.interaction_dim, values.blocks, values.dropout).to(device),
        V5DeformationDecoder(values.interaction_dim, values.hidden_dim, values.blocks, values.dropout).to(device),
    )


def build_shared_modules(config: V5Config, device="cpu"):
    config.validate()
    values = config.model
    return (
        V5SharedPhysicalModel(
            values.hidden_dim, values.frames, values.dropout,
            values.blocks, values.heads,
        ).to(device),
        V5InteractionEncoder(
            values.interaction_dim, values.blocks, values.dropout,
            physical_dim=values.hidden_dim,
        ).to(device),
        V5DeformationDecoder(
            values.interaction_dim, values.hidden_dim, values.blocks,
            values.dropout, physical_dim=values.hidden_dim,
        ).to(device),
    )


class SharedDeformationStage(nn.Module):
    contract_version = "v5_shared_deformation_stage_v3"

    def __init__(self, physical, interaction, decoder, trunk_gradient_scale):
        super().__init__()
        self.physical = physical
        self.interaction = interaction
        self.decoder = decoder
        self.trunk_gradient_scale = float(trunk_gradient_scale)

    def train(self, mode: bool = True):
        super().train(mode)
        self.interaction.eval()
        if not mode or self.trunk_gradient_scale == 0:
            self.physical.eval()
        return self


def load_v5_stage(module: nn.Module, path: str | Path, expected_stage: str, device="cpu") -> dict:
    state = torch.load(path, map_location=device, weights_only=False)
    if state.get("contract") != "v5_random_initialization_stage_v1" or state.get("stage") != expected_stage:
        raise ValueError(f"expected a V5 {expected_stage} checkpoint")
    required_contract = getattr(module, "contract_version", None)
    if required_contract is not None and state.get("architecture_contract") != required_contract:
        raise ValueError(
            f"incompatible {expected_stage} architecture: expected {required_contract}; "
            "the simplified V5 checkpoint must not be reused"
        )
    module.load_state_dict(state["model"], strict=True)
    return state


def train_com(root, manifest, output, seed, config=V5Config(), options=TrainOptions(), device="cpu"):
    seed_all(seed)
    device = torch.device(device)
    train_loader, validation_loader = loaders(root, manifest, seed, ("rigid", "soft_body"), options)
    com, _, _, _ = build_modules(config, device)

    def step(batch):
        targets, _ = targets_and_stages(batch)
        predicted, _ = com(**model_inputs(batch))
        return com_trajectory_loss(
            predicted, targets.com, targets.radius,
            batch["target_mask"].any(2),
        ).total

    return _fit("com", com, train_loader, validation_loader, device, step, Path(output), seed, config, options)


def train_shared_global(root, manifest, output, seed, config=V5Config(), options=TrainOptions(), device="cpu"):
    """Jointly update one trunk from both COM and rotation objectives."""
    seed_all(seed)
    device = torch.device(device)
    train_loader, validation_loader = loaders(
        root, manifest, seed, ("rigid", "soft_body"), options,
    )
    physical, _, _ = build_shared_modules(config, device)

    def step(batch):
        targets, _ = targets_and_stages(batch)
        output_value = physical(**model_inputs(batch))
        return compute_v42_global_losses(output_value, batch, targets).total

    return _fit(
        "shared_global", physical, train_loader, validation_loader, device,
        step, Path(output), seed, config, options,
    )


def train_shared_interaction(root, manifest, output, seed, global_checkpoint, use_identity_rotation=False, config=V5Config(), options=TrainOptions(), device="cpu"):
    seed_all(seed)
    device = torch.device(device)
    train_loader, validation_loader = loaders(root, manifest, seed, ("soft_body",), options)
    physical, interaction, _ = build_shared_modules(config, device)
    load_v5_stage(physical, global_checkpoint, "shared_global", device)
    freeze(physical)

    def step(batch):
        targets, stages = targets_and_stages(batch)
        with torch.no_grad():
            global_output = physical(**model_inputs(batch))
            reference_shape, _, _ = centered_reference(
                batch["x1"], batch["input_mask"],
            )
            rigid_position = (
                global_output.com[:, :, None] + reference_shape[:, None]
                if use_identity_rotation else global_output.position
            )
            contact, event_time, event_valid = interaction_labels(batch, targets, stages)
        result = interaction(
            reference_shape, rigid_position,
            batch["input_mask"], batch["floor_z"],
            global_output.physical_hidden,
        )
        return interaction_auxiliary_loss(
            result.contact_logits, result.event_time, contact, event_time,
            batch["input_mask"], event_valid,
        ).total

    return _fit(
        "shared_interaction", interaction, train_loader, validation_loader,
        device, step, Path(output), seed, config, options,
        (("shared_physical", physical),),
        {"rotation_policy": "identity" if use_identity_rotation else "learned"},
    )


def train_shared_deformation(root, manifest, output, seed, global_checkpoint, interaction_checkpoint, trunk_gradient_scale=0.0, use_identity_rotation=False, config=V5Config(), options=TrainOptions(), device="cpu"):
    if not 0 <= trunk_gradient_scale <= 1:
        raise ValueError("trunk_gradient_scale must lie in [0,1]")
    seed_all(seed)
    device = torch.device(device)
    train_loader, validation_loader = loaders(root, manifest, seed, ("soft_body",), options)
    physical, interaction, decoder = build_shared_modules(config, device)
    load_v5_stage(physical, global_checkpoint, "shared_global", device)
    load_v5_stage(interaction, interaction_checkpoint, "shared_interaction", device)
    freeze(physical); freeze(interaction)
    if trunk_gradient_scale > 0:
        for parameter in physical.trunk_parameters():
            parameter.requires_grad_(True)
    stage = SharedDeformationStage(
        physical, interaction, decoder, trunk_gradient_scale,
    ).to(device)

    def step(batch):
        targets, stages = targets_and_stages(batch)
        global_output = physical(**model_inputs(batch))
        reference_shape, _, _ = centered_reference(
            batch["x1"], batch["input_mask"],
        )
        rigid_position = (
            global_output.com[:, :, None] + reference_shape[:, None]
            if use_identity_rotation else global_output.position
        )
        # The shared physical model's local output is zero, so position is the
        # causal rigid trajectory produced by its COM and rotation paths.
        interaction_output = interaction(
            reference_shape, rigid_position.detach(),
            batch["input_mask"], batch["floor_z"],
            global_output.physical_hidden.detach(),
        )
        predicted = decoder(
            reference_shape, interaction_output.latent,
            batch["input_mask"], global_output.physical_hidden,
            trunk_gradient_scale=trunk_gradient_scale,
        )
        return event_normalized_deformation_mse(
            predicted, targets.displacement, batch["target_mask"],
            targets.valid_rotation, stages.labels[:, 1:], targets.radius,
        )

    protected = (("interaction", interaction),)
    if trunk_gradient_scale == 0:
        protected += (("shared_physical", physical),)
    else:
        protected += (
            ("com_head", physical.core.v42_com_head),
            ("rotation_head", physical.core.rotation_head),
        )
    return _fit(
        "shared_deformation", stage, train_loader, validation_loader, device,
        step, Path(output), seed, config, options, protected,
        {"rotation_policy": "identity" if use_identity_rotation else "learned",
         "trunk_gradient_scale": float(trunk_gradient_scale)},
    )


def train_rotation(root, manifest, output, seed, config=V5Config(), options=TrainOptions(), device="cpu"):
    seed_all(seed)
    device = torch.device(device)
    train_loader, validation_loader = loaders(root, manifest, seed, ("rigid", "soft_body"), options)
    _, rotation, _, _ = build_modules(config, device)

    def step(batch):
        targets, _ = targets_and_stages(batch)
        predicted = rotation(batch["x0"], batch["x1"], batch["input_mask"])
        return rotation_geodesic_mean(predicted, targets.rotation, targets.valid_rotation & batch["target_mask"].any(2))

    return _fit("rotation", rotation, train_loader, validation_loader, device, step, Path(output), seed, config, options)


def train_interaction(root, manifest, output, seed, com_checkpoint, rotation_checkpoint, use_identity_rotation=False, config=V5Config(), options=TrainOptions(), device="cpu"):
    seed_all(seed)
    device = torch.device(device)
    train_loader, validation_loader = loaders(root, manifest, seed, ("soft_body",), options)
    com, rotation, interaction, _ = build_modules(config, device)
    load_v5_stage(com, com_checkpoint, "com", device)
    freeze(com)
    if not use_identity_rotation:
        load_v5_stage(rotation, rotation_checkpoint, "rotation", device)
    freeze(rotation)

    def step(batch):
        targets, stages = targets_and_stages(batch)
        with torch.no_grad():
            predicted_com, _ = com(**model_inputs(batch))
            predicted_rotation = (
                rotation.identity(len(batch["x0"]), config.model.frames, device=device, dtype=batch["x0"].dtype)
                if use_identity_rotation else rotation(batch["x0"], batch["x1"], batch["input_mask"])
            )
            shape, _, _ = centered_reference(batch["x1"], batch["input_mask"])
            zero = torch.zeros(*predicted_com.shape[:2], shape.shape[1], 3, device=device)
            rigid = reconstruct_positions(predicted_com, predicted_rotation, shape, zero)
            contact, event_time, event_valid = interaction_labels(batch, targets, stages)
        result = interaction(shape, rigid, batch["input_mask"], batch["floor_z"])
        return interaction_auxiliary_loss(
            result.contact_logits, result.event_time, contact, event_time,
            batch["input_mask"], event_valid,
        ).total

    protected = (("com", com), ("rotation", rotation))
    return _fit("interaction", interaction, train_loader, validation_loader, device, step, Path(output), seed, config, options, protected)


def train_deformation(root, manifest, output, seed, com_checkpoint, rotation_checkpoint, interaction_checkpoint, use_identity_rotation=False, config=V5Config(), options=TrainOptions(), device="cpu"):
    seed_all(seed)
    device = torch.device(device)
    train_loader, validation_loader = loaders(root, manifest, seed, ("soft_body",), options)
    com, rotation, interaction, decoder = build_modules(config, device)
    load_v5_stage(com, com_checkpoint, "com", device)
    if not use_identity_rotation:
        load_v5_stage(rotation, rotation_checkpoint, "rotation", device)
    load_v5_stage(interaction, interaction_checkpoint, "interaction", device)
    freeze(com); freeze(rotation); freeze(interaction)

    def step(batch):
        targets, stages = targets_and_stages(batch)
        with torch.no_grad():
            predicted_com, _ = com(**model_inputs(batch))
            predicted_rotation = (
                rotation.identity(len(batch["x0"]), config.model.frames, device=device, dtype=batch["x0"].dtype)
                if use_identity_rotation else rotation(batch["x0"], batch["x1"], batch["input_mask"])
            )
            shape, _, _ = centered_reference(batch["x1"], batch["input_mask"])
            zero = torch.zeros(*predicted_com.shape[:2], shape.shape[1], 3, device=device)
            rigid = reconstruct_positions(predicted_com, predicted_rotation, shape, zero)
            latent = interaction(shape, rigid, batch["input_mask"], batch["floor_z"]).latent
        predicted = decoder(shape, latent, batch["input_mask"])
        return event_normalized_deformation_mse(
            predicted, targets.displacement, batch["target_mask"],
            targets.valid_rotation, stages.labels[:, 1:], targets.radius,
        )

    protected = (("com", com), ("rotation", rotation), ("interaction", interaction))
    return _fit("deformation", decoder, train_loader, validation_loader, device, step, Path(output), seed, config, options, protected)


def train_memory(root, manifest, output, seed, com_checkpoint, rotation_checkpoint, interaction_checkpoint, deformation_checkpoint, bank_path, use_identity_rotation=False, config=V5Config(), options=TrainOptions(), device="cpu"):
    if not config.memory.enabled:
        raise ValueError("set memory.enabled=true only after the causal base qualifies and misses the final target")
    seed_all(seed)
    device = torch.device(device)
    train_loader, validation_loader = loaders(root, manifest, seed, ("soft_body",), options)
    com, rotation, interaction, decoder = build_modules(config, device)
    load_v5_stage(com, com_checkpoint, "com", device)
    if not use_identity_rotation:
        load_v5_stage(rotation, rotation_checkpoint, "rotation", device)
    load_v5_stage(interaction, interaction_checkpoint, "interaction", device)
    load_v5_stage(decoder, deformation_checkpoint, "deformation", device)
    freeze(com); freeze(rotation); freeze(interaction); freeze(decoder)
    bank = V5MemoryBank.load(bank_path, train_loader.dataset.uids)
    memory = V5MemoryModule(
        query_dim=config.model.interaction_dim + 3,
        hidden_dim=config.model.hidden_dim,
        heads=config.model.heads,
        residual_bound=config.memory.residual_bound,
        gate_logit=config.memory.gate_logit,
    ).to(device)

    def step(batch):
        targets, stages = targets_and_stages(batch)
        with torch.no_grad():
            predicted_com, _ = com(**model_inputs(batch))
            predicted_rotation = (
                rotation.identity(len(batch["x0"]), config.model.frames, device=device, dtype=batch["x0"].dtype)
                if use_identity_rotation else rotation(batch["x0"], batch["x1"], batch["input_mask"])
            )
            shape, _, radius = centered_reference(batch["x1"], batch["input_mask"])
            zero = torch.zeros(*predicted_com.shape[:2], shape.shape[1], 3, device=device)
            rigid = reconstruct_positions(predicted_com, predicted_rotation, shape, zero)
            interaction_output = interaction(shape, rigid, batch["input_mask"], batch["floor_z"])
            base = decoder(shape, interaction_output.latent, batch["input_mask"])
            q = (shape / radius[:, None, None])[:, None].expand(-1, config.model.frames, -1, -1)
            query = torch.cat((q, interaction_output.latent), -1)
            selected = [bank.retrieve_uids(
                uid,
                batch["dino"][index],
                batch["dino_valid"][index] & batch["input_mask"][index],
                "train" if uid in bank.permitted_uids else "validation",
            ) for index, uid in enumerate(batch["uid"])]
        predicted = memory(
            base, query, interaction_output.event_time, batch["input_mask"],
            bank, selected,
        )
        return event_normalized_deformation_mse(
            predicted, targets.displacement, batch["target_mask"],
            targets.valid_rotation, stages.labels[:, 1:], targets.radius,
        )

    protected = (("com", com), ("rotation", rotation), ("interaction", interaction), ("deformation", decoder))
    return _fit("memory", memory, train_loader, validation_loader, device, step, Path(output), seed, config, options, protected)
