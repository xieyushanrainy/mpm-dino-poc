from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from mpm_dino_v2.deformation import edge_validity, gather_neighbours
from .full_data import ballistic_trajectory
from .model import GraphLayer, masked_mean


class FiLM(nn.Module):
    def __init__(self, condition_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Linear(condition_dim, hidden_dim * 2)
        nn.init.zeros_(self.net.weight)
        nn.init.zeros_(self.net.bias)

    def forward(self, hidden: Tensor, condition: Tensor) -> Tensor:
        gamma, beta = self.net(condition).chunk(2, dim=-1)
        return hidden * (1 + gamma[:, None, None]) + beta[:, None, None]


class FactorizedGraphTemporalBlock(nn.Module):
    def __init__(self, hidden_dim: int, heads: int, dropout: float):
        super().__init__()
        self.spatial = GraphLayer(hidden_dim, edge_dim=8)
        self.temporal_norm = nn.LayerNorm(hidden_dim)
        self.temporal = nn.MultiheadAttention(hidden_dim, heads, dropout=dropout, batch_first=True)
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Dropout(dropout),
        )
        self.film = FiLM(hidden_dim, hidden_dim)

    def forward(
        self,
        hidden: Tensor,
        condition: Tensor,
        edge_features: Tensor,
        neighbour_indices: Tensor,
        edge_mask: Tensor,
        point_mask: Tensor,
    ) -> Tensor:
        batch, frames, points, width = hidden.shape
        flat_hidden = hidden.reshape(batch * frames, points, width)
        flat_edges = edge_features.reshape(batch * frames, points, edge_features.shape[-2], edge_features.shape[-1])
        flat_indices = neighbour_indices[:, None].expand(-1, frames, -1, -1).reshape(batch * frames, points, -1)
        flat_mask = edge_mask[:, None].expand(-1, frames, -1, -1).reshape(batch * frames, points, -1)
        hidden = self.spatial(flat_hidden, flat_edges, flat_indices, flat_mask).reshape(batch, frames, points, width)
        temporal_input = self.temporal_norm(hidden).permute(0, 2, 1, 3).reshape(batch * points, frames, width)
        temporal_output, _ = self.temporal(temporal_input, temporal_input, temporal_input, need_weights=False)
        hidden = hidden + temporal_output.reshape(batch, points, frames, width).permute(0, 2, 1, 3)
        hidden = hidden + self.ffn(self.ffn_norm(hidden))
        hidden = self.film(hidden, condition)
        return hidden * point_mask[:, None, :, None]


@dataclass
class FullTrajectoryOutput:
    residual_com: Tensor
    residual_local: Tensor
    residual: Tensor
    ballistic: Tensor
    position: Tensor


class FullTrajectorySurrogate(nn.Module):
    def __init__(
        self,
        dino_dim: int = 384,
        dino_embed_dim: int = 16,
        hidden_dim: int = 128,
        blocks: int = 4,
        heads: int = 4,
        dropout: float = 0.1,
        frames: int = 59,
        gradient_checkpointing: bool = True,
    ):
        super().__init__()
        self.frames = frames
        self.gradient_checkpointing = gradient_checkpointing
        self.dino = nn.Sequential(
            nn.LayerNorm(dino_dim), nn.Linear(dino_dim, 64), nn.SiLU(), nn.Linear(64, dino_embed_dim)
        )
        self.object_condition = nn.Sequential(
            nn.Linear(dino_embed_dim * 2 + 1, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.initial_node = nn.Sequential(nn.Linear(22, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
        self.initial_graph = nn.ModuleList(GraphLayer(hidden_dim, edge_dim=11) for _ in range(2))
        self.time_projection = nn.Sequential(nn.Linear(32, 32), nn.SiLU(), nn.Linear(32, 32))
        self.token = nn.Sequential(nn.Linear(hidden_dim + 3 + 3 + 1 + 32, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
        self.blocks = nn.ModuleList(
            FactorizedGraphTemporalBlock(hidden_dim, heads, dropout) for _ in range(blocks)
        )
        self.local_head = nn.Sequential(
            nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 3)
        )
        self.com_head = nn.Sequential(
            nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 3)
        )
        nn.init.zeros_(self.local_head[-1].weight)
        nn.init.zeros_(self.local_head[-1].bias)
        nn.init.zeros_(self.com_head[-1].weight)
        nn.init.zeros_(self.com_head[-1].bias)

    @staticmethod
    def _time_features(dt: Tensor, frames: int, dtype: torch.dtype) -> Tensor:
        offsets = torch.arange(1, frames + 1, device=dt.device, dtype=dtype)
        seconds = dt[:, None] * offsets[None]
        frequencies = (2.0 ** torch.arange(16, device=dt.device, dtype=dtype))[None, None]
        angles = math.pi * seconds[..., None] * frequencies
        return torch.cat((angles.sin(), angles.cos()), dim=-1)

    @staticmethod
    def _edge_features(
        positions: Tensor,
        neighbour_indices: Tensor,
        neighbour_mask: Tensor,
        rest_edge_vectors: Tensor,
        rest_edge_lengths: Tensor,
    ) -> tuple[Tensor, Tensor]:
        batch, frames, points, _ = positions.shape
        flat = positions.reshape(batch * frames, points, 3)
        indices = neighbour_indices[:, None].expand(-1, frames, -1, -1).reshape(batch * frames, points, -1)
        neighbours = gather_neighbours(flat, indices).reshape(batch, frames, points, indices.shape[-1], 3)
        vectors = neighbours - positions[:, :, :, None]
        lengths = torch.linalg.vector_norm(vectors, dim=-1, keepdim=True)
        rest_lengths = rest_edge_lengths[:, None, :, :, None]
        stretch = lengths / rest_lengths.clamp_min(1e-8) - 1
        rest = rest_edge_vectors[:, None].expand(-1, frames, -1, -1, -1)
        features = torch.cat((rest, vectors, lengths, stretch), dim=-1)
        mask = neighbour_mask
        return features, mask

    def forward(
        self,
        x0: Tensor,
        x1: Tensor,
        input_mask: Tensor,
        dino: Tensor,
        dino_valid: Tensor,
        dt: Tensor,
        gravity: Tensor,
        floor_z: Tensor,
        neighbour_indices: Tensor,
        neighbour_mask: Tensor,
        rest_edge_vectors: Tensor,
        rest_edge_lengths: Tensor,
    ) -> FullTrajectoryOutput:
        velocity = (x1 - x0) / dt[:, None, None]
        centre = masked_mean(x1, input_mask)
        valid_dino_mask = input_mask & dino_valid
        projected_dino = self.dino(dino * dino_valid[..., None])
        dino_mean = masked_mean(projected_dino, valid_dino_mask)
        maximum = projected_dino.masked_fill(~valid_dino_mask[..., None], -torch.inf).amax(1)
        maximum = torch.where(valid_dino_mask.any(1, keepdim=True), maximum, torch.zeros_like(maximum))
        valid_fraction = valid_dino_mask.float().mean(1, keepdim=True)
        condition = self.object_condition(torch.cat((dino_mean, maximum, valid_fraction), dim=-1))
        conditions = torch.cat(
            (
                dt[:, None, None].expand(-1, x1.shape[1], 1),
                gravity[:, None].expand(-1, x1.shape[1], -1),
                x1[..., 2:3] - floor_z[:, None, None],
            ),
            dim=-1,
        )
        node_features = torch.cat(
            (
                x1,
                velocity,
                x1 - centre[:, None],
                x0,
                x1 - x0,
                input_mask[..., None],
                input_mask[..., None],
                conditions,
            ),
            dim=-1,
        )
        initial = self.initial_node(node_features) * input_mask[..., None]
        neighbours = gather_neighbours(x1, neighbour_indices)
        neighbour_velocity = gather_neighbours(velocity, neighbour_indices)
        current_vectors = neighbours - x1[:, :, None]
        current_lengths = torch.linalg.vector_norm(current_vectors, dim=-1, keepdim=True)
        relative_velocity = neighbour_velocity - velocity[:, :, None]
        stretch = current_lengths / rest_edge_lengths[..., None].clamp_min(1e-8) - 1
        initial_edges = torch.cat(
            (rest_edge_vectors, current_vectors, current_lengths, relative_velocity, stretch), dim=-1
        )
        initial_edge_mask = edge_validity(input_mask, neighbour_indices, neighbour_mask)
        for layer in self.initial_graph:
            initial = layer(initial, initial_edges, neighbour_indices, initial_edge_mask) * input_mask[..., None]
        ballistic = ballistic_trajectory(x0, x1, gravity, dt, self.frames)
        time = self.time_projection(self._time_features(dt, self.frames, x0.dtype))
        token_input = torch.cat(
            (
                initial[:, None].expand(-1, self.frames, -1, -1),
                ballistic,
                ballistic - x1[:, None],
                ballistic[..., 2:3] - floor_z[:, None, None, None],
                time[:, :, None].expand(-1, -1, x1.shape[1], -1),
            ),
            dim=-1,
        )
        hidden = self.token(token_input) * input_mask[:, None, :, None]
        edge_features, spatial_mask = self._edge_features(
            ballistic, neighbour_indices, neighbour_mask, rest_edge_vectors, rest_edge_lengths
        )
        spatial_mask = edge_validity(input_mask, neighbour_indices, spatial_mask)
        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                hidden = checkpoint(
                    block,
                    hidden,
                    condition,
                    edge_features,
                    neighbour_indices,
                    spatial_mask,
                    input_mask,
                    use_reentrant=False,
                )
            else:
                hidden = block(
                    hidden, condition, edge_features, neighbour_indices, spatial_mask, input_mask
                )
        pooled = masked_mean(hidden, input_mask[:, None].expand(-1, self.frames, -1), dim=2)
        residual_com = self.com_head(pooled)
        local_raw = self.local_head(hidden) * input_mask[:, None, :, None]
        local_mean = masked_mean(
            local_raw, input_mask[:, None].expand(-1, self.frames, -1), dim=2
        )
        residual_local = (local_raw - local_mean[:, :, None]) * input_mask[:, None, :, None]
        residual = residual_com[:, :, None] + residual_local
        residual = residual * input_mask[:, None, :, None]
        return FullTrajectoryOutput(
            residual_com=residual_com,
            residual_local=residual_local,
            residual=residual,
            ballistic=ballistic,
            position=ballistic + residual,
        )
