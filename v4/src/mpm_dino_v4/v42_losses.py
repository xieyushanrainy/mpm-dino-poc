from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F

from mpm_dino_v2.deformation import edge_validity, gather_neighbours
from .v42_geometry import CanonicalTargets, canonical_targets, rotation_chordal
from .v42_model import V42TrajectoryOutput


def _weighted_huber(values, mask, frame_weight=None, beta=0.01):
    raw = F.smooth_l1_loss(values, torch.zeros_like(values), reduction="none", beta=beta)
    while mask.ndim < raw.ndim:
        mask = mask[..., None]
    weight = mask.to(raw.dtype)
    if frame_weight is not None:
        extra = frame_weight
        while extra.ndim < raw.ndim:
            extra = extra[..., None]
        weight = weight * extra
    return (raw * weight).sum() / weight.expand_as(raw).sum().clamp_min(1)


@dataclass
class V42Losses:
    total: Tensor
    global_total: Tensor
    local_total: Tensor
    com_position: Tensor
    com_velocity: Tensor
    com_acceleration: Tensor
    com_key: Tensor
    rotation: Tensor
    rotation_key: Tensor
    rotation_event: Tensor
    rigid_fit: Tensor
    canonical: Tensor
    strain: Tensor
    edge_length: Tensor
    local_velocity: Tensor
    rigid_zero: Tensor


@dataclass
class V42GlobalLosses:
    total: Tensor
    com_position: Tensor
    com_velocity: Tensor
    com_acceleration: Tensor
    com_key: Tensor
    rotation: Tensor
    rotation_key: Tensor
    rotation_event: Tensor
    rigid_fit: Tensor


def compute_v42_global_losses(
    output: V42TrajectoryOutput,
    batch: dict,
    targets: CanonicalTargets | None = None,
    rotation_frame_weights: Tensor | None = None,
    beta: float = 0.01,
) -> V42GlobalLosses:
    mask = batch["target_mask"]
    if targets is None:
        targets = canonical_targets(
            batch["x1"], batch["target"], batch["input_mask"], mask
        )
    radius = targets.radius
    scale_com = radius[:, None, None]
    frame_valid = mask.any(2)
    com_position = _weighted_huber(
        (output.com - targets.com) / scale_com, frame_valid, beta=beta
    )
    com_velocity = _weighted_huber(
        (output.com[:, 1:] - output.com[:, :-1]
         - targets.com[:, 1:] + targets.com[:, :-1]) / scale_com,
        frame_valid[:, 1:] & frame_valid[:, :-1], beta=beta,
    )
    com_acceleration = _weighted_huber(
        (output.com[:, 2:] - 2 * output.com[:, 1:-1] + output.com[:, :-2]
         - targets.com[:, 2:] + 2 * targets.com[:, 1:-1] - targets.com[:, :-2])
        / scale_com,
        frame_valid[:, 2:] & frame_valid[:, 1:-1] & frame_valid[:, :-2],
        beta=beta,
    )
    key = [
        horizon - 1 for horizon in (1, 8, 16, 30, 40, 59)
        if horizon <= output.com.shape[1]
    ]
    com_key = _weighted_huber(
        (output.com[:, key] - targets.com[:, key]) / scale_com,
        frame_valid[:, key], beta=beta,
    )
    family_weight = torch.tensor(
        [1.0 if family == "rigid" else 0.25 for family in batch["family"]],
        device=output.com.device, dtype=output.com.dtype,
    )[:, None]
    rotation_errors = rotation_chordal(output.rotation, targets.rotation)
    rotation_mask = targets.valid_rotation & frame_valid
    rotation = (
        rotation_errors * rotation_mask * family_weight
    ).sum() / (rotation_mask * family_weight).sum().clamp_min(1)
    key_rotation_mask = rotation_mask[:, key]
    rotation_key = (
        rotation_errors[:, key] * key_rotation_mask * family_weight
    ).sum() / (
        key_rotation_mask * family_weight
    ).sum().clamp_min(1)
    if rotation_frame_weights is None:
        rotation_event = rotation.new_zeros(())
    else:
        event_weight = rotation_frame_weights * rotation_mask * family_weight
        rotation_event = (
            rotation_errors * event_weight
        ).sum() / event_weight.sum().clamp_min(1)
    rigid_objects = torch.tensor(
        [family == "rigid" for family in batch["family"]],
        device=output.com.device, dtype=torch.bool,
    )
    rigid_mask = mask & rigid_objects[:, None, None]
    predicted_rigid = torch.einsum(
        "bni,btij->btnj", targets.reference_shape, output.rotation
    )
    target_rigid = torch.einsum(
        "bni,btij->btnj", targets.reference_shape, targets.rotation
    )
    rigid_fit = _weighted_huber(
        (predicted_rigid - target_rigid) / radius[:, None, None, None],
        rigid_mask, beta=beta,
    )
    total = (
        com_position + 0.25 * com_velocity + 0.10 * com_acceleration
        + 0.25 * com_key + rotation + 0.25 * rotation_key
        + 0.50 * rotation_event + 0.25 * rigid_fit
    )
    return V42GlobalLosses(
        total, com_position, com_velocity, com_acceleration, com_key,
        rotation, rotation_key, rotation_event, rigid_fit,
    )


def compute_v42_losses(
    output: V42TrajectoryOutput,
    batch: dict,
    targets: CanonicalTargets | None = None,
    frame_weights: Tensor | None = None,
    rotation_frame_weights: Tensor | None = None,
    beta: float = 0.01,
) -> V42Losses:
    mask = batch["target_mask"]
    if targets is None:
        targets = canonical_targets(
            batch["x1"], batch["target"], batch["input_mask"], mask
        )
    radius = targets.radius
    global_losses = compute_v42_global_losses(
        output, batch, targets=targets,
        rotation_frame_weights=rotation_frame_weights, beta=beta,
    )
    rigid_objects = torch.tensor(
        [family == "rigid" for family in batch["family"]],
        device=output.com.device, dtype=torch.bool,
    )
    rigid_mask = mask & rigid_objects[:, None, None]

    canonical_mask = mask & targets.valid_rotation[:, :, None]
    canonical = _weighted_huber(
        (output.canonical_displacement - targets.displacement)
        / radius[:, None, None, None],
        canonical_mask, frame_weights, beta,
    )
    batch_size, frames, points, _ = output.canonical_shape.shape
    predicted = output.canonical_shape.reshape(batch_size * frames, points, 3)
    target_shape = (
        targets.reference_shape[:, None] + targets.displacement
    ).reshape(batch_size * frames, points, 3)
    flat_mask = mask.reshape(batch_size * frames, points)
    indices = batch["neighbour_indices"][:, None].expand(
        -1, frames, -1, -1
    ).reshape(batch_size * frames, points, -1)
    graph_mask = batch["neighbour_mask"][:, None].expand(
        -1, frames, -1, -1
    ).reshape(batch_size * frames, points, -1)
    valid_edges = edge_validity(flat_mask, indices, graph_mask)
    pred_vectors = gather_neighbours(predicted, indices) - predicted[:, :, None]
    target_vectors = gather_neighbours(target_shape, indices) - target_shape[:, :, None]
    pred_lengths = torch.linalg.vector_norm(pred_vectors, dim=-1)
    target_lengths = torch.linalg.vector_norm(target_vectors, dim=-1)
    rest = batch["rest_edge_lengths"][:, None].expand(
        -1, frames, -1, -1
    ).reshape_as(pred_lengths).clamp_min(1e-8)
    flat_weight = frame_weights.reshape(batch_size * frames) if frame_weights is not None else None
    strain = _weighted_huber(
        (pred_lengths - rest) / rest - (target_lengths - rest) / rest,
        valid_edges, flat_weight, beta,
    )
    edge_length = _weighted_huber(
        (pred_lengths - target_lengths)
        / radius.repeat_interleave(frames)[:, None, None],
        valid_edges, flat_weight, beta,
    )
    local_velocity = _weighted_huber(
        (
            output.canonical_displacement[:, 1:]
            - output.canonical_displacement[:, :-1]
            - targets.displacement[:, 1:] + targets.displacement[:, :-1]
        ) / radius[:, None, None, None],
        canonical_mask[:, 1:] & canonical_mask[:, :-1],
        frame_weights[:, 1:] if frame_weights is not None else None,
        beta,
    )
    rigid_zero = _weighted_huber(
        output.canonical_displacement / radius[:, None, None, None],
        rigid_mask, frame_weights, beta,
    )
    global_total = global_losses.total
    local_total = (
        canonical + 0.50 * strain + 0.25 * edge_length
        + 0.25 * local_velocity + 0.25 * rigid_zero
    )
    return V42Losses(
        global_total + local_total, global_total, local_total,
        global_losses.com_position, global_losses.com_velocity,
        global_losses.com_acceleration, global_losses.com_key,
        global_losses.rotation, global_losses.rotation_key,
        global_losses.rotation_event,
        global_losses.rigid_fit,
        canonical, strain, edge_length,
        local_velocity, rigid_zero,
    )
