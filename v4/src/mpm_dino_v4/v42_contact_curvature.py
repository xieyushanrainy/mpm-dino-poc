from __future__ import annotations

import torch

from mpm_dino_v2.deformation import edge_validity, gather_neighbours

from .model import masked_mean
from .v42_gate2 import train_v42_gate2
from .v42_oracle import (
    LOSS_CONTRACT, TEMPORAL_DIM, event_normalized_canonical_mse,
    temporal_features,
)


CONTACT_DIM = 3
CURVATURE_DIM = 4
POINT_CONDITION_DIM = CONTACT_DIM + CURVATURE_DIM
DIRECT_CONDITION_DIM = POINT_CONDITION_DIM + TEMPORAL_DIM
VARIANTS = {
    "zero_control": (False, False),
    "oracle_contact": (True, False),
    "oracle_contact_curvature": (True, True),
    "curvature_only": (False, True),
}


def curvature_features(batch):
    """Four fixed local shape proxies from the reference neighbour graph."""
    points = batch["x1"]
    mask = batch["input_mask"]
    indices = batch["neighbour_indices"]
    neighbour_mask = batch["neighbour_mask"]
    valid = edge_validity(mask, indices, neighbour_mask)
    neighbours = gather_neighbours(points, indices)
    offsets = neighbours - points[:, :, None]
    weights = valid.to(points.dtype)
    count = weights.sum(-1).clamp_min(1)
    covariance = torch.einsum(
        "bnki,bnkj,bnk->bnij", offsets, offsets, weights,
    ) / count[..., None, None]
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    low, middle, high = eigenvalues.unbind(-1)
    scale = high.clamp_min(1e-10)
    linearity = (high - middle) / scale
    planarity = (middle - low) / scale
    scattering = low / scale
    normals = eigenvectors[..., 0]
    neighbour_normals = gather_neighbours(normals, indices)
    normal_difference = 1 - (
        normals[:, :, None] * neighbour_normals
    ).sum(-1).abs()
    normal_variation = (normal_difference * weights).sum(-1) / count
    result = torch.stack(
        (linearity, planarity, scattering, normal_variation), dim=-1,
    )
    return (result * mask[..., None]).nan_to_num().clamp(0, 1).detach()


def oracle_floor_contact_features(batch):
    """Future pointwise floor-contact cues; diagnostic oracle, not inference input."""
    positions = batch["target"]
    radius = torch.linalg.vector_norm(
        batch["x1"] - masked_mean(batch["x1"], batch["input_mask"])[:, None],
        dim=-1,
    ).masked_fill(~batch["input_mask"], 0).amax(1).clamp_min(1e-6)
    gap = (positions[..., 2] - batch["floor_z"][:, None, None]) / radius[:, None, None]
    signed_gap = (gap / 0.1).clamp(-1, 1)
    proximity = torch.exp(-gap.abs() / 0.02)
    previous = torch.cat((batch["x1"][:, None], positions[:, :-1]), dim=1)
    normal_velocity = (
        (positions[..., 2] - previous[..., 2])
        / batch["dt"][:, None, None]
        / radius[:, None, None]
    ).clamp(-10, 10) / 10
    valid = batch["target_mask"].to(positions.dtype)
    return torch.stack((signed_gap, proximity, normal_velocity), -1).mul(
        valid[..., None]
    ).detach()


class ContactCurvatureConditionBuilder:
    def __init__(self, contact: bool, curvature: bool):
        self.contact = bool(contact)
        self.curvature = bool(curvature)

    def __call__(self, batch, _stages):
        batch_size, frames, points = batch["target"].shape[:3]
        result = batch["target"].new_zeros(
            batch_size, frames, points, POINT_CONDITION_DIM,
        )
        if self.contact:
            result[..., :CONTACT_DIM] = oracle_floor_contact_features(batch)
        if self.curvature:
            static = curvature_features(batch)
            result[..., CONTACT_DIM:] = static[:, None]
        return result.detach()


class DirectProbeConditionBuilder:
    """Matched pointwise contact/curvature/time input for decoder probes."""

    def __init__(self, enabled=True):
        self.enabled = bool(enabled)

    def __call__(self, batch, stages):
        batch_size, frames, points = batch["target"].shape[:3]
        result = batch["target"].new_zeros(
            batch_size, frames, points, DIRECT_CONDITION_DIM,
        )
        if not self.enabled:
            return result
        result[..., :CONTACT_DIM] = oracle_floor_contact_features(batch)
        result[..., CONTACT_DIM:POINT_CONDITION_DIM] = (
            curvature_features(batch)[:, None]
        )
        temporal = temporal_features(stages, frames)
        result[..., POINT_CONDITION_DIM:] = temporal[:, :, None]
        return result.detach()


def train_contact_curvature_variant(
    dataset_root, manifest, checkpoint, output, seed, variant, **kwargs,
):
    if variant not in VARIANTS:
        raise ValueError(f"unknown contact-curvature variant: {variant}")
    contact, curvature = VARIANTS[variant]
    return train_v42_gate2(
        dataset_root, manifest, checkpoint, output, seed,
        loss_options=LOSS_CONTRACT, stage_weight_mode="total_mass",
        oracle_condition_dim=POINT_CONDITION_DIM, oracle_injection="adapter",
        condition_builder=ContactCurvatureConditionBuilder(contact, curvature),
        condition_name=variant, exploratory_control=True,
        objective_builder=event_normalized_canonical_mse,
        objective_name="event_frame_amplitude_normalized_canonical_mse_v1",
        dataset_families=("soft_body",), selection_mode="optimized",
        experiment_name="v42_contact_curvature_" + variant,
        model_contract_version="contact_curvature_point_adapter_v1",
        **kwargs,
    )


DIRECT_VARIANTS = {
    "adapter_full": ("adapter", True),
    "direct_zero": ("direct", False),
    "direct_full": ("direct", True),
}


def train_direct_decoder_probe(
    dataset_root, manifest, checkpoint, output, seed, variant, **kwargs,
):
    if variant not in DIRECT_VARIANTS:
        raise ValueError(f"unknown direct-decoder variant: {variant}")
    injection, enabled = DIRECT_VARIANTS[variant]
    return train_v42_gate2(
        dataset_root, manifest, checkpoint, output, seed,
        loss_options=LOSS_CONTRACT, stage_weight_mode="total_mass",
        oracle_condition_dim=DIRECT_CONDITION_DIM,
        oracle_injection=injection,
        condition_builder=DirectProbeConditionBuilder(enabled),
        condition_name=variant, exploratory_control=True,
        objective_builder=event_normalized_canonical_mse,
        objective_name="event_frame_amplitude_normalized_canonical_mse_v1",
        dataset_families=("soft_body",), selection_mode="optimized",
        experiment_name="v42_direct_decoder_probe_" + variant,
        model_contract_version="direct_point_decoder_probe_v1",
        **kwargs,
    )
