from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F


EVENT_PHASES = (2, 3, 4)


def event_normalized_deformation_mse(
    predicted: Tensor,
    target: Tensor,
    target_mask: Tensor,
    valid_rotation: Tensor,
    stage_labels: Tensor,
    radius: Tensor,
) -> Tensor:
    """Episode-balanced objective for contact/compression/peak frames."""
    if stage_labels.shape[1] == predicted.shape[1] + 1:
        stage_labels = stage_labels[:, 1:]
    if stage_labels.shape != predicted.shape[:2]:
        raise ValueError("stage labels must align with predicted frames")
    selected = torch.zeros_like(stage_labels, dtype=torch.bool)
    for phase in EVENT_PHASES:
        selected |= stage_labels.eq(phase)
    valid = target_mask & valid_rotation[:, :, None] & selected[:, :, None]
    count = valid.sum((1, 2)).clamp_min(1)
    target_energy = (target.square().sum(-1) * valid).sum((1, 2)) / count
    scale = target_energy.maximum((1e-6 * radius).square()).detach()
    error = ((predicted - target).square().sum(-1) * valid).sum((1, 2)) / count
    return (error / scale).mean()


@dataclass
class InteractionLoss:
    total: Tensor
    contact: Tensor
    event_time: Tensor


def interaction_auxiliary_loss(
    contact_logits: Tensor,
    event_time: Tensor,
    contact_target: Tensor,
    event_time_target: Tensor,
    point_mask: Tensor,
    event_valid: Tensor | None = None,
    contact_weight: float = 1.0,
    event_time_weight: float = 1.0,
) -> InteractionLoss:
    valid_points = point_mask[:, None].expand_as(contact_target)
    contact_raw = F.binary_cross_entropy_with_logits(
        contact_logits, contact_target.to(contact_logits.dtype), reduction="none",
    )
    contact = (contact_raw * valid_points).sum() / valid_points.sum().clamp_min(1)
    if event_valid is None:
        event_valid = torch.ones_like(event_time_target, dtype=torch.bool)
    time_raw = F.smooth_l1_loss(event_time, event_time_target, reduction="none")
    timing = (time_raw * event_valid).sum() / event_valid.sum().clamp_min(1)
    return InteractionLoss(contact_weight * contact + event_time_weight * timing, contact, timing)


def com_mse(predicted: Tensor, target: Tensor, radius: Tensor, frame_valid: Tensor) -> Tensor:
    error = (predicted - target).square().sum(-1) / radius[:, None].square()
    return (error * frame_valid).sum() / frame_valid.sum().clamp_min(1)


@dataclass
class COMTrajectoryLoss:
    total: Tensor
    position: Tensor
    velocity: Tensor
    acceleration: Tensor
    key_horizons: Tensor


def _masked_smooth_l1(error: Tensor, valid: Tensor, beta: float = 0.01) -> Tensor:
    raw = F.smooth_l1_loss(error, torch.zeros_like(error), reduction="none", beta=beta)
    weight = valid.to(raw.dtype)
    while weight.ndim < raw.ndim:
        weight = weight[..., None]
    return (raw * weight).sum() / weight.expand_as(raw).sum().clamp_min(1)


def com_trajectory_loss(predicted: Tensor, target: Tensor, radius: Tensor, frame_valid: Tensor) -> COMTrajectoryLoss:
    """V4.2-style position, velocity, acceleration, and horizon objective."""
    scale = radius[:, None, None]
    position = _masked_smooth_l1((predicted - target) / scale, frame_valid)
    velocity = _masked_smooth_l1(
        (predicted[:, 1:] - predicted[:, :-1] - target[:, 1:] + target[:, :-1]) / scale,
        frame_valid[:, 1:] & frame_valid[:, :-1],
    )
    acceleration = _masked_smooth_l1(
        (predicted[:, 2:] - 2 * predicted[:, 1:-1] + predicted[:, :-2]
         - target[:, 2:] + 2 * target[:, 1:-1] - target[:, :-2]) / scale,
        frame_valid[:, 2:] & frame_valid[:, 1:-1] & frame_valid[:, :-2],
    )
    indices = [horizon - 1 for horizon in (1, 8, 16, 30, 40, 59) if horizon <= predicted.shape[1]]
    key = _masked_smooth_l1(
        (predicted[:, indices] - target[:, indices]) / scale,
        frame_valid[:, indices],
    )
    total = position + 0.25 * velocity + 0.10 * acceleration + 0.25 * key
    return COMTrajectoryLoss(total, position, velocity, acceleration, key)


def rotation_geodesic_mean(predicted: Tensor, target: Tensor, valid: Tensor) -> Tensor:
    relative = predicted.transpose(-1, -2) @ target
    cosine = (relative.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2
    angle = torch.acos(cosine.clamp(-1 + 1e-6, 1 - 1e-6))
    return (angle * valid).sum() / valid.sum().clamp_min(1)
