from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F

from mpm_dino_v2.deformation import edge_validity, gather_neighbours
from .model import V4Output, masked_mean


def masked_smooth_l1(pred: Tensor, target: Tensor, mask: Tensor, beta=0.001) -> Tensor:
    raw = F.smooth_l1_loss(pred, target, reduction="none", beta=beta).mean(-1)
    weight = mask.to(raw.dtype)
    return (raw * weight).sum() / weight.sum().clamp_min(1)


@dataclass
class V4Loss:
    total: Tensor
    residual: Tensor
    position: Tensor
    com: Tensor
    edge_vector: Tensor
    edge_length: Tensor


def compute_loss(output: V4Output, batch: dict, weights=(1.0, 1.0, 0.5, 0.25, 0.1), beta=0.001) -> V4Loss:
    mask, target = batch["target_mask"], batch["target"]
    target_residual = target - output.cv_position
    residual = masked_smooth_l1(output.residual, target_residual, mask, beta)
    position = masked_smooth_l1(output.position, target, mask, beta)
    target_com = masked_mean(target, mask)
    predicted_com = masked_mean(output.position, mask)
    com = F.smooth_l1_loss(predicted_com, target_com, beta=beta)
    valid_edges = edge_validity(mask, batch["neighbour_indices"], batch["neighbour_mask"])
    pred_vectors = gather_neighbours(output.position, batch["neighbour_indices"]) - output.position[:, :, None]
    true_vectors = gather_neighbours(target, batch["neighbour_indices"]) - target[:, :, None]
    edge_vector = masked_smooth_l1(pred_vectors, true_vectors, valid_edges, beta)
    pred_length, true_length = torch.linalg.vector_norm(pred_vectors, dim=-1), torch.linalg.vector_norm(true_vectors, dim=-1)
    raw = F.smooth_l1_loss(pred_length, true_length, reduction="none", beta=beta)
    edge_length = (raw * valid_edges).sum() / valid_edges.sum().clamp_min(1)
    terms = (residual, position, com, edge_vector, edge_length)
    total = sum(weight * term for weight, term in zip(weights, terms))
    return V4Loss(total, *terms)
