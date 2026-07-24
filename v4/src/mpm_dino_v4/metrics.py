from __future__ import annotations

import torch
from torch import Tensor

from mpm_dino_v2.deformation import edge_validity, gather_neighbours
from .model import masked_mean


def metric_values(pred: Tensor, target: Tensor, mask: Tensor, reference: Tensor,
                  indices: Tensor, neighbour_mask: Tensor, floor_z: Tensor, family: list[str]) -> dict[str, Tensor]:
    weight = mask.to(pred.dtype); count = weight.sum(1).clamp_min(1)
    distance = torch.linalg.vector_norm(pred - target, dim=-1)
    rmse = torch.sqrt(((pred - target).square().sum(-1) * weight).sum(1) / count)
    mae = (distance * weight).sum(1) / count
    pred_com, true_com = masked_mean(pred, mask), masked_mean(target, mask)
    com = torch.linalg.vector_norm(pred_com - true_com, dim=-1)
    shape_delta = (pred - pred_com[:, None]) - (target - true_com[:, None])
    shape = torch.sqrt((shape_delta.square().sum(-1) * weight).sum(1) / count)
    valid_edges = edge_validity(mask, indices, neighbour_mask)
    pred_edge = gather_neighbours(pred, indices) - pred[:, :, None]
    true_edge = gather_neighbours(target, indices) - target[:, :, None]
    edge_count = valid_edges.sum((1, 2)).clamp_min(1)
    edge_vector = (torch.linalg.vector_norm(pred_edge - true_edge, dim=-1) * valid_edges).sum((1, 2)) / edge_count
    edge_length = (torch.abs(torch.linalg.vector_norm(pred_edge, dim=-1) - torch.linalg.vector_norm(true_edge, dim=-1)) * valid_edges).sum((1, 2)) / edge_count
    penetration = torch.relu(floor_z[:, None] - pred[..., 2]) * mask
    floor_rate = ((penetration > 0).sum(1) / count)
    floor_depth = penetration.sum(1) / count
    coverage = mask.sum(1) / mask.shape[1]
    rigidity = torch.full_like(rmse, torch.nan)
    for b, label in enumerate(family):
        if label != "rigid" or int(mask[b].sum()) < 3: continue
        p, q = pred[b, mask[b]], reference[b, mask[b]]
        pc, qc = p - p.mean(0), q - q.mean(0)
        u, _, vh = torch.linalg.svd(qc.T @ pc)
        rotation = vh.T @ u.T
        if torch.linalg.det(rotation) < 0:
            vh[-1] *= -1; rotation = vh.T @ u.T
        aligned = qc @ rotation.T
        rigidity[b] = torch.sqrt((aligned - pc).square().sum(-1).mean())
    return {"rmse_m": rmse, "mae_m": mae, "com_m": com, "shape_m": shape,
            "edge_vector_m": edge_vector, "edge_length_m": edge_length,
            "floor_penetration_rate": floor_rate, "floor_penetration_depth_m": floor_depth,
            "active_coverage": coverage, "rigidity_residual_m": rigidity}
