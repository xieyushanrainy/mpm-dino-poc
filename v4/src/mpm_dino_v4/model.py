from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from mpm_dino_v2.deformation import deformation_descriptors, edge_validity, gather_neighbours


def masked_mean(values: Tensor, mask: Tensor, dim: int = 1) -> Tensor:
    weight = mask.to(values.dtype)
    while weight.ndim < values.ndim: weight = weight.unsqueeze(-1)
    return (values * weight).sum(dim=dim) / weight.sum(dim=dim).clamp_min(1)


class GraphLayer(nn.Module):
    def __init__(self, hidden_dim: int, edge_dim: int = 11):
        super().__init__()
        self.message = nn.Sequential(nn.Linear(hidden_dim * 2 + edge_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim), nn.SiLU())
        self.update = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, hidden: Tensor, edges: Tensor, indices: Tensor, mask: Tensor) -> Tensor:
        neighbours = gather_neighbours(hidden, indices)
        centre = hidden[:, :, None].expand_as(neighbours)
        messages = self.message(torch.cat((centre, neighbours, edges), -1)) * mask[..., None]
        pooled = messages.sum(2) / mask.sum(2, keepdim=True).clamp_min(1)
        return self.norm(hidden + self.update(torch.cat((hidden, pooled), -1)))


@dataclass
class V4Output:
    residual_com: Tensor
    residual_local: Tensor
    residual: Tensor
    cv_position: Tensor
    position: Tensor


class V4ParticleSurrogate(nn.Module):
    def __init__(self, dino_dim=384, dino_embed_dim=16, hidden_dim=128, layers=3):
        super().__init__()
        self.dino = nn.Sequential(nn.LayerNorm(dino_dim), nn.Linear(dino_dim, 64), nn.SiLU(), nn.Linear(64, dino_embed_dim))
        self.node = nn.Sequential(nn.Linear(26 + dino_embed_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
        self.layers = nn.ModuleList(GraphLayer(hidden_dim) for _ in range(layers))
        self.local_head = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 3))
        self.com_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 3))
        nn.init.zeros_(self.local_head[-1].weight); nn.init.zeros_(self.local_head[-1].bias)
        nn.init.zeros_(self.com_head[-1].weight); nn.init.zeros_(self.com_head[-1].bias)

    def forward(self, x_prev: Tensor, x_curr: Tensor, mask_prev: Tensor, mask_curr: Tensor,
                reference: Tensor, dino: Tensor, dino_valid: Tensor, dt: Tensor, gravity: Tensor,
                floor_z: Tensor, neighbour_indices: Tensor, neighbour_mask: Tensor,
                rest_edge_vectors: Tensor, rest_edge_lengths: Tensor) -> V4Output:
        state_mask = mask_prev & mask_curr
        velocity = (x_curr - x_prev) / dt[:, None, None]
        centre = masked_mean(x_curr, state_mask)
        deformation = deformation_descriptors(x_curr, state_mask, neighbour_indices, neighbour_mask, rest_edge_lengths)
        valid_dino = dino * dino_valid[..., None].to(dino.dtype)
        conditions = torch.cat((
            dt[:, None, None].expand(-1, x_curr.shape[1], 1),
            gravity[:, None].expand(-1, x_curr.shape[1], -1),
            (x_curr[..., 2:3] - floor_z[:, None, None]),
        ), -1)
        features = torch.cat((x_curr, velocity, x_curr - centre[:, None], reference,
                              x_curr - reference, deformation,
                              mask_prev[..., None], mask_curr[..., None], conditions,
                              self.dino(valid_dino), dino_valid[..., None]), -1)
        hidden = self.node(features) * state_mask[..., None]
        neighbours = gather_neighbours(x_curr, neighbour_indices)
        neighbour_velocity = gather_neighbours(velocity, neighbour_indices)
        current_vector = neighbours - x_curr[:, :, None]
        current_length = torch.linalg.vector_norm(current_vector, dim=-1, keepdim=True)
        relative_velocity = neighbour_velocity - velocity[:, :, None]
        stretch = current_length / rest_edge_lengths[..., None].clamp_min(1e-8) - 1
        edges = torch.cat((rest_edge_vectors, current_vector, current_length, relative_velocity, stretch), -1)
        valid_edges = edge_validity(state_mask, neighbour_indices, neighbour_mask)
        for layer in self.layers:
            hidden = layer(hidden, edges, neighbour_indices, valid_edges) * state_mask[..., None]
        pooled = masked_mean(hidden, state_mask)
        residual_com = self.com_head(pooled)
        local_raw = self.local_head(hidden) * state_mask[..., None]
        residual_local = (local_raw - masked_mean(local_raw, state_mask)[:, None]) * state_mask[..., None]
        residual = (residual_com[:, None] + residual_local) * state_mask[..., None]
        cv = 2 * x_curr - x_prev
        return V4Output(residual_com, residual_local, residual, cv, cv + residual)
