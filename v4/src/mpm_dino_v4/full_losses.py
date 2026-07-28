from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F

from mpm_dino_v2.deformation import edge_validity, gather_neighbours
from .full_model import FullTrajectoryOutput
from .model import masked_mean


def masked_smooth_l1(pred: Tensor, target: Tensor, mask: Tensor, beta: float = 0.001) -> Tensor:
    raw = F.smooth_l1_loss(pred, target, reduction="none", beta=beta).mean(-1)
    weight = mask.to(raw.dtype)
    return (raw * weight).sum() / weight.sum().clamp_min(1)


@dataclass
class FullTrajectoryLoss:
    total: Tensor
    residual: Tensor
    position: Tensor
    com: Tensor
    edge_vector: Tensor
    edge_length: Tensor
    key_horizons: Tensor


@dataclass
class ShapeBalancedTrajectoryLoss:
    total: Tensor
    world: Tensor
    com: Tensor
    shape: Tensor
    strain: Tensor
    key_horizons: Tensor


@dataclass
class LegacyShapeAuxTrajectoryLoss:
    """Stable Track-B objective plus a deformation-only auxiliary."""

    total: Tensor
    legacy: Tensor
    shape_auxiliary: Tensor
    residual: Tensor
    position: Tensor
    com: Tensor
    edge_vector: Tensor
    edge_length: Tensor
    legacy_key_horizons: Tensor
    shape: Tensor
    strain: Tensor
    shape_key_horizons: Tensor


@dataclass
class LocalShapeTrajectoryLoss:
    """COM-free objective for the Phase-2 soft-deformation experiment."""

    total: Tensor
    shape: Tensor
    edge_vector: Tensor
    strain: Tensor
    key_horizons: Tensor
    rigid_zero_local: Tensor
    mean_deformation_weight: Tensor


def compute_full_trajectory_loss(
    output: FullTrajectoryOutput,
    batch: dict,
    weights=(1.0, 1.0, 0.5, 0.25, 0.1, 0.25),
    beta: float = 0.001,
) -> FullTrajectoryLoss:
    target, mask = batch["target"], batch["target_mask"]
    target_residual = target - output.ballistic
    residual = masked_smooth_l1(output.residual, target_residual, mask, beta)
    position = masked_smooth_l1(output.position, target, mask, beta)
    predicted_com = masked_mean(output.position, mask, dim=2)
    target_com = masked_mean(target, mask, dim=2)
    frame_valid = mask.any(dim=2)
    com = masked_smooth_l1(predicted_com, target_com, frame_valid, beta)

    batch_size, frames, points, _ = target.shape
    flat_pred = output.position.reshape(batch_size * frames, points, 3)
    flat_target = target.reshape(batch_size * frames, points, 3)
    flat_mask = mask.reshape(batch_size * frames, points)
    indices = batch["neighbour_indices"][:, None].expand(-1, frames, -1, -1).reshape(
        batch_size * frames, points, -1
    )
    graph_mask = batch["neighbour_mask"][:, None].expand(-1, frames, -1, -1).reshape(
        batch_size * frames, points, -1
    )
    valid_edges = edge_validity(flat_mask, indices, graph_mask)
    pred_vectors = gather_neighbours(flat_pred, indices) - flat_pred[:, :, None]
    target_vectors = gather_neighbours(flat_target, indices) - flat_target[:, :, None]
    edge_vector = masked_smooth_l1(pred_vectors, target_vectors, valid_edges, beta)
    pred_lengths = torch.linalg.vector_norm(pred_vectors, dim=-1)
    target_lengths = torch.linalg.vector_norm(target_vectors, dim=-1)
    raw_length = F.smooth_l1_loss(pred_lengths, target_lengths, reduction="none", beta=beta)
    edge_length = (raw_length * valid_edges).sum() / valid_edges.sum().clamp_min(1)

    key_indices = [index for index in (3, 7, 15, frames - 1) if index < frames]
    key_horizons = masked_smooth_l1(
        output.position[:, key_indices], target[:, key_indices], mask[:, key_indices], beta
    )
    terms = (residual, position, com, edge_vector, edge_length, key_horizons)
    total = sum(weight * term for weight, term in zip(weights, terms))
    return FullTrajectoryLoss(total, *terms)


def compute_shape_balanced_trajectory_loss(
    output: FullTrajectoryOutput,
    batch: dict,
    weights=(1.0, 0.5, 1.0, 0.5, 0.25),
    beta: float = 0.01,
) -> ShapeBalancedTrajectoryLoss:
    """Radius-normalized trajectory loss with explicit deformation supervision.

    World, COM, centre-relative shape, and key-horizon errors are normalized by
    the fixed frame-0 reference radius. Edge strain is already dimensionless.
    H16/H30/H40 are the key horizons, matching guarded checkpoint selection.
    """
    target, mask = batch["target"], batch["target_mask"]
    reference, input_mask = batch["reference"], batch["input_mask"]
    reference_com = masked_mean(reference, input_mask)
    reference_radius = torch.linalg.vector_norm(
        reference - reference_com[:, None], dim=-1
    ).masked_fill(~input_mask, 0).amax(1).clamp_min(1e-6)
    coordinate_scale = reference_radius[:, None, None, None]

    world = masked_smooth_l1(
        output.position / coordinate_scale,
        target / coordinate_scale,
        mask,
        beta,
    )
    predicted_com = masked_mean(output.position, mask, dim=2)
    target_com = masked_mean(target, mask, dim=2)
    frame_valid = mask.any(dim=2)
    com = masked_smooth_l1(
        predicted_com / reference_radius[:, None, None],
        target_com / reference_radius[:, None, None],
        frame_valid,
        beta,
    )
    predicted_shape = output.position - predicted_com[:, :, None]
    target_shape = target - target_com[:, :, None]
    shape = masked_smooth_l1(
        predicted_shape / coordinate_scale,
        target_shape / coordinate_scale,
        mask,
        beta,
    )

    batch_size, frames, points, _ = target.shape
    flat_pred = output.position.reshape(batch_size * frames, points, 3)
    flat_target = target.reshape(batch_size * frames, points, 3)
    flat_mask = mask.reshape(batch_size * frames, points)
    indices = batch["neighbour_indices"][:, None].expand(
        -1, frames, -1, -1
    ).reshape(batch_size * frames, points, -1)
    graph_mask = batch["neighbour_mask"][:, None].expand(
        -1, frames, -1, -1
    ).reshape(batch_size * frames, points, -1)
    valid_edges = edge_validity(flat_mask, indices, graph_mask)
    predicted_vectors = gather_neighbours(flat_pred, indices) - flat_pred[:, :, None]
    target_vectors = gather_neighbours(flat_target, indices) - flat_target[:, :, None]
    rest_lengths = batch["rest_edge_lengths"][:, None].expand(
        -1, frames, -1, -1
    ).reshape(batch_size * frames, points, -1).clamp_min(1e-8)
    predicted_strain = (
        torch.linalg.vector_norm(predicted_vectors, dim=-1) - rest_lengths
    ) / rest_lengths
    target_strain = (
        torch.linalg.vector_norm(target_vectors, dim=-1) - rest_lengths
    ) / rest_lengths
    raw_strain = F.smooth_l1_loss(
        predicted_strain, target_strain, reduction="none", beta=beta
    )
    strain = (
        raw_strain * valid_edges.to(raw_strain.dtype)
    ).sum() / valid_edges.sum().clamp_min(1)

    key_indices = [horizon - 1 for horizon in (16, 30, 40) if horizon <= frames]
    key_horizons = masked_smooth_l1(
        output.position[:, key_indices] / coordinate_scale,
        target[:, key_indices] / coordinate_scale,
        mask[:, key_indices],
        beta,
    )
    terms = (world, com, shape, strain, key_horizons)
    total = sum(weight * term for weight, term in zip(weights, terms))
    return ShapeBalancedTrajectoryLoss(total, *terms)


def compute_legacy_shape_aux_trajectory_loss(
    output: FullTrajectoryOutput,
    batch: dict,
    auxiliary_weight: float = 0.2,
    shape_weights=(1.0, 0.5, 0.25),
    beta: float = 0.01,
) -> LegacyShapeAuxTrajectoryLoss:
    """Add shape supervision without duplicating Track-B world/COM terms.

    The legacy objective remains intact. The auxiliary contains only
    centre-relative shape, normalized edge strain, and shape at
    H16/H30/H40/H59. H59 is a training stability anchor, not a promotion
    horizon.
    """
    legacy = compute_full_trajectory_loss(output, batch)
    target, mask = batch["target"], batch["target_mask"]
    reference, input_mask = batch["reference"], batch["input_mask"]
    reference_com = masked_mean(reference, input_mask)
    reference_radius = torch.linalg.vector_norm(
        reference - reference_com[:, None], dim=-1
    ).masked_fill(~input_mask, 0).amax(1).clamp_min(1e-6)
    coordinate_scale = reference_radius[:, None, None, None]

    predicted_com = masked_mean(output.position, mask, dim=2)
    target_com = masked_mean(target, mask, dim=2)
    predicted_shape = output.position - predicted_com[:, :, None]
    target_shape = target - target_com[:, :, None]
    shape = masked_smooth_l1(
        predicted_shape / coordinate_scale,
        target_shape / coordinate_scale,
        mask,
        beta,
    )

    batch_size, frames, points, _ = target.shape
    flat_pred = output.position.reshape(batch_size * frames, points, 3)
    flat_target = target.reshape(batch_size * frames, points, 3)
    flat_mask = mask.reshape(batch_size * frames, points)
    indices = batch["neighbour_indices"][:, None].expand(
        -1, frames, -1, -1
    ).reshape(batch_size * frames, points, -1)
    graph_mask = batch["neighbour_mask"][:, None].expand(
        -1, frames, -1, -1
    ).reshape(batch_size * frames, points, -1)
    valid_edges = edge_validity(flat_mask, indices, graph_mask)
    predicted_vectors = gather_neighbours(flat_pred, indices) - flat_pred[:, :, None]
    target_vectors = gather_neighbours(flat_target, indices) - flat_target[:, :, None]
    rest_lengths = batch["rest_edge_lengths"][:, None].expand(
        -1, frames, -1, -1
    ).reshape(batch_size * frames, points, -1).clamp_min(1e-8)
    predicted_strain = (
        torch.linalg.vector_norm(predicted_vectors, dim=-1) - rest_lengths
    ) / rest_lengths
    target_strain = (
        torch.linalg.vector_norm(target_vectors, dim=-1) - rest_lengths
    ) / rest_lengths
    raw_strain = F.smooth_l1_loss(
        predicted_strain, target_strain, reduction="none", beta=beta
    )
    strain = (
        raw_strain * valid_edges.to(raw_strain.dtype)
    ).sum() / valid_edges.sum().clamp_min(1)

    key_indices = [
        horizon - 1 for horizon in (16, 30, 40, 59) if horizon <= frames
    ]
    shape_key_horizons = masked_smooth_l1(
        predicted_shape[:, key_indices] / coordinate_scale,
        target_shape[:, key_indices] / coordinate_scale,
        mask[:, key_indices],
        beta,
    )
    shape_auxiliary = sum(
        weight * term
        for weight, term in zip(
            shape_weights, (shape, strain, shape_key_horizons)
        )
    )
    total = legacy.total + auxiliary_weight * shape_auxiliary
    return LegacyShapeAuxTrajectoryLoss(
        total=total,
        legacy=legacy.total,
        shape_auxiliary=shape_auxiliary,
        residual=legacy.residual,
        position=legacy.position,
        com=legacy.com,
        edge_vector=legacy.edge_vector,
        edge_length=legacy.edge_length,
        legacy_key_horizons=legacy.key_horizons,
        shape=shape,
        strain=strain,
        shape_key_horizons=shape_key_horizons,
    )


def compute_local_shape_trajectory_loss(
    output: FullTrajectoryOutput,
    batch: dict,
    weights=(1.0, 0.25, 0.5, 0.25, 0.25),
    beta: float = 0.01,
) -> LocalShapeTrajectoryLoss:
    """Train only centre-relative shape with deformation-weighted frames.

    Ground-truth COM is used only to construct the supervision target. The
    model receives no future COM. The prediction baseline is the centred
    ballistic trajectory plus the model's zero-mean local residual.
    """
    target, mask = batch["target"], batch["target_mask"]
    reference, input_mask = batch["reference"], batch["input_mask"]
    reference_com = masked_mean(reference, input_mask)
    reference_shape = reference - reference_com[:, None]
    radius = torch.linalg.vector_norm(
        reference_shape, dim=-1
    ).masked_fill(~input_mask, 0).amax(1).clamp_min(1e-6)
    scale = radius[:, None, None, None]

    target_com = masked_mean(target, mask, dim=2)
    target_shape = target - target_com[:, :, None]
    ballistic_com = masked_mean(output.ballistic, mask, dim=2)
    ballistic_shape = output.ballistic - ballistic_com[:, :, None]
    predicted_shape = ballistic_shape + output.residual_local
    predicted_shape = (
        predicted_shape
        - masked_mean(predicted_shape, mask, dim=2)[:, :, None]
    )

    raw_deformation = torch.linalg.vector_norm(
        (target_shape - reference_shape[:, None]) / scale,
        dim=-1,
    )
    point_weight = mask.to(raw_deformation.dtype)
    frame_deformation = (
        (raw_deformation * point_weight).sum(2)
        / point_weight.sum(2).clamp_min(1)
    )
    frame_valid = mask.any(2)
    valid_weight = frame_valid.to(frame_deformation.dtype)
    mean_deformation = (
        (frame_deformation * valid_weight).sum(1, keepdim=True)
        / valid_weight.sum(1, keepdim=True).clamp_min(1)
    )
    frame_weight = 0.25 + frame_deformation / mean_deformation.clamp_min(1e-6)
    frame_weight = frame_weight / (
        (frame_weight * valid_weight).sum(1, keepdim=True)
        / valid_weight.sum(1, keepdim=True).clamp_min(1)
    ).clamp_min(1e-6)

    raw_shape = F.smooth_l1_loss(
        predicted_shape / scale, target_shape / scale,
        reduction="none", beta=beta,
    ).mean(-1)
    shape_weight = mask.to(raw_shape.dtype) * frame_weight[:, :, None]
    shape = (
        (raw_shape * shape_weight).sum()
        / shape_weight.sum().clamp_min(1)
    )

    batch_size, frames, points, _ = target.shape
    flat_pred = predicted_shape.reshape(batch_size * frames, points, 3)
    flat_target = target_shape.reshape(batch_size * frames, points, 3)
    flat_mask = mask.reshape(batch_size * frames, points)
    indices = batch["neighbour_indices"][:, None].expand(
        -1, frames, -1, -1
    ).reshape(batch_size * frames, points, -1)
    graph_mask = batch["neighbour_mask"][:, None].expand(
        -1, frames, -1, -1
    ).reshape(batch_size * frames, points, -1)
    valid_edges = edge_validity(flat_mask, indices, graph_mask)
    pred_vectors = gather_neighbours(flat_pred, indices) - flat_pred[:, :, None]
    target_vectors = (
        gather_neighbours(flat_target, indices) - flat_target[:, :, None]
    )
    flat_frame_weight = frame_weight.reshape(batch_size * frames, 1, 1)
    edge_weight = valid_edges.to(pred_vectors.dtype) * flat_frame_weight
    raw_edge = F.smooth_l1_loss(
        pred_vectors / radius.repeat_interleave(frames)[:, None, None, None],
        target_vectors / radius.repeat_interleave(frames)[:, None, None, None],
        reduction="none", beta=beta,
    ).mean(-1)
    edge_vector = (
        (raw_edge * edge_weight).sum()
        / edge_weight.sum().clamp_min(1)
    )

    rest = batch["rest_edge_lengths"][:, None].expand(
        -1, frames, -1, -1
    ).reshape(batch_size * frames, points, -1).clamp_min(1e-8)
    pred_strain = (
        torch.linalg.vector_norm(pred_vectors, dim=-1) - rest
    ) / rest
    target_strain = (
        torch.linalg.vector_norm(target_vectors, dim=-1) - rest
    ) / rest
    raw_strain = F.smooth_l1_loss(
        pred_strain, target_strain, reduction="none", beta=beta,
    )
    strain = (
        (raw_strain * edge_weight).sum()
        / edge_weight.sum().clamp_min(1)
    )

    key_indices = [
        horizon - 1 for horizon in (16, 30, 40, 59)
        if horizon <= frames
    ]
    key_horizons = masked_smooth_l1(
        predicted_shape[:, key_indices] / scale,
        target_shape[:, key_indices] / scale,
        mask[:, key_indices],
        beta,
    )
    rigid_objects = torch.tensor(
        [family == "rigid" for family in batch["family"]],
        device=mask.device,
        dtype=torch.bool,
    )
    if rigid_objects.any():
        rigid_mask = mask & rigid_objects[:, None, None]
        rigid_zero_local = masked_smooth_l1(
            output.residual_local / scale,
            torch.zeros_like(output.residual_local),
            rigid_mask,
            beta,
        )
    else:
        rigid_zero_local = output.residual_local.sum() * 0
    terms = (
        shape, edge_vector, strain, key_horizons, rigid_zero_local,
    )
    total = sum(weight * term for weight, term in zip(weights, terms))
    return LocalShapeTrajectoryLoss(
        total, *terms,
        mean_deformation_weight=frame_weight.mean(),
    )
