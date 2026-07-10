"""Leakage-free local deformation descriptors over the fixed reference graph."""

from __future__ import annotations

import torch
from torch import Tensor


def gather_neighbours(values: Tensor, indices: Tensor) -> Tensor:
    """Gather (B,N,C) values using (B,N,K) indices."""
    b, n, channels = values.shape
    safe = indices.clamp_min(0)
    expanded = values[:, :, None, :].expand(-1, -1, safe.shape[-1], -1)
    return torch.gather(expanded, 1, safe[..., None].expand(-1, -1, -1, channels))


def edge_validity(particle_mask: Tensor, neighbour_indices: Tensor, neighbour_mask: Tensor) -> Tensor:
    neighbour_valid = gather_neighbours(particle_mask[..., None], neighbour_indices).squeeze(-1)
    return neighbour_mask & particle_mask[..., None] & neighbour_valid


def deformation_descriptors(
    positions: Tensor,
    particle_mask: Tensor,
    neighbour_indices: Tensor,
    neighbour_mask: Tensor,
    rest_edge_lengths: Tensor,
    eps: float = 1e-8,
) -> Tensor:
    """Return per-particle mean/std/max-absolute stretch, shaped (B,N,3)."""
    neighbours = gather_neighbours(positions, neighbour_indices)
    lengths = torch.linalg.vector_norm(neighbours - positions[:, :, None, :], dim=-1)
    valid = edge_validity(particle_mask, neighbour_indices, neighbour_mask)
    stretch = lengths / (rest_edge_lengths + eps) - 1
    weights = valid.to(stretch.dtype)
    count = weights.sum(-1).clamp_min(1)
    mean = (stretch * weights).sum(-1) / count
    variance = ((stretch - mean[..., None]).square() * weights).sum(-1) / count
    maximum = torch.where(valid, stretch.abs(), torch.zeros_like(stretch)).amax(-1)
    # sqrt has an infinite derivative at exactly zero. That is harmless for
    # teacher-forced inputs but poisons recurrent backpropagation through a
    # predicted state, especially at degree-1 particles. Keep the descriptor
    # numerically indistinguishable while making its gradient finite.
    standard_deviation = torch.sqrt(variance + eps * eps)
    result = torch.stack((mean, standard_deviation, maximum), dim=-1)
    return result * particle_mask[..., None]
