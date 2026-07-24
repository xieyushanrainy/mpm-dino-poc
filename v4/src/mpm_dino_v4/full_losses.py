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
