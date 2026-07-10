from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor, nn

from .action import action_particle_features, initial_action_summary
from mpm_dino_v2.deformation import deformation_descriptors, edge_validity, gather_neighbours
from mpm_dino_v2.grid import GridSpec
from mpm_dino_v2.model import SurrogateOutput


V3Variant = Literal["graph_direct", "latent_graph", "action_token_graph"]


class FiLM(nn.Module):
    def __init__(self, condition_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(condition_dim, hidden_dim * 2), nn.SiLU(), nn.Linear(hidden_dim * 2, hidden_dim * 2))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, hidden: Tensor, condition: Tensor) -> Tensor:
        gamma, beta = self.net(condition).chunk(2, dim=-1)
        return (1 + gamma[:, None]) * hidden + beta[:, None]


class GraphMessageLayer(nn.Module):
    def __init__(self, hidden_dim: int, edge_dim: int, condition_dim: int = 0):
        super().__init__()
        self.message = nn.Sequential(
            nn.Linear(hidden_dim * 2 + edge_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
        )
        self.update = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
        self.film = FiLM(condition_dim, hidden_dim) if condition_dim else None

    def forward(self, hidden: Tensor, edge_features: Tensor, indices: Tensor, edge_mask: Tensor,
                condition: Tensor | None = None) -> Tensor:
        neighbours = gather_neighbours(hidden, indices)
        center = hidden[:, :, None].expand_as(neighbours)
        message = self.message(torch.cat((center, neighbours, edge_features), dim=-1))
        message = message * edge_mask[..., None].to(message.dtype)
        count = edge_mask.sum(dim=-1, keepdim=True).clamp_min(1).to(message.dtype)
        aggregated = message.sum(dim=2) / count
        updated = hidden + self.update(torch.cat((hidden, aggregated), dim=-1))
        if self.film is not None:
            if condition is None:
                raise ValueError("condition is required for FiLM-conditioned graph layers")
            updated = self.film(updated, condition)
        return updated


class ObjectLatentEncoder(nn.Module):
    def __init__(self, dino_dim: int, dino_embed_dim: int, hidden_dim: int, latent_dim: int):
        super().__init__()
        self.dino = nn.Sequential(nn.LayerNorm(dino_dim), nn.Linear(dino_dim, 64), nn.SiLU(), nn.Linear(64, dino_embed_dim))
        self.node = nn.Sequential(nn.Linear(3 + dino_embed_dim + 1, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
        self.out = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, latent_dim))

    def forward(self, x0: Tensor, dino: Tensor, dino_imputed: Tensor, particle_mask: Tensor) -> Tensor:
        features = torch.cat((x0, self.dino(dino), dino_imputed[..., None].to(x0.dtype)), dim=-1)
        hidden = self.node(features) * particle_mask[..., None].to(x0.dtype)
        weight = particle_mask[..., None].to(x0.dtype)
        mean = hidden.sum(dim=1) / weight.sum(dim=1).clamp_min(1)
        maximum = hidden.masked_fill(~particle_mask[..., None], -torch.inf).amax(dim=1)
        maximum = torch.where(particle_mask.any(dim=1, keepdim=True), maximum, torch.zeros_like(maximum))
        return self.out(torch.cat((mean, maximum), dim=-1))


class ActionTokenBlock(nn.Module):
    def __init__(self, hidden_dim: int, action_dim: int, heads: int):
        super().__init__()
        self.action = nn.Sequential(nn.Linear(action_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
        self.attention = nn.MultiheadAttention(hidden_dim, heads, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, hidden: Tensor, action_global: Tensor, contact_point: Tensor) -> Tensor:
        action_token = self.action(action_global)
        contact_token = self.action(torch.cat((contact_point, torch.zeros_like(action_global[:, 3:])), dim=-1))
        tokens = torch.stack((action_token, contact_token), dim=1)
        attended, _ = self.attention(hidden, tokens, tokens, need_weights=False)
        return self.norm(hidden + attended)


@dataclass
class V3Dimensions:
    node_dim: int
    edge_dim: int
    action_global_dim: int


class V3ParticleSurrogate(nn.Module):
    """DINO-centric particle-native V3 architecture candidates."""

    def __init__(
        self,
        dino_dim: int = 768,
        dino_embed_dim: int = 16,
        hidden_dim: int = 128,
        latent_dim: int = 32,
        layers: int = 3,
        variant: V3Variant = "graph_direct",
        attention_heads: int = 4,
        resolution: int = 32,
    ):
        super().__init__()
        if variant not in {"graph_direct", "latent_graph", "action_token_graph"}:
            raise ValueError(f"unsupported V3 variant: {variant}")
        self.variant = variant
        self.spec = GridSpec(resolution=resolution)
        if variant != "latent_graph":
            self.dino = nn.Sequential(nn.LayerNorm(dino_dim), nn.Linear(dino_dim, 64), nn.SiLU(), nn.Linear(64, dino_embed_dim))
        if variant == "latent_graph":
            self.latent_encoder = ObjectLatentEncoder(dino_dim, dino_embed_dim, hidden_dim, latent_dim)
        action_particle_dim = 3 + 1 + 3 + 1 + 1
        action_global_dim = 3 + 1 + 3 + 2 + 1
        geometry_node_dim = 3 + 3 + 3 + 3 + 3 + 1 + 2 + action_particle_dim
        node_dim = geometry_node_dim if variant == "latent_graph" else geometry_node_dim + dino_embed_dim
        edge_dim = 3 + 3 + 1 + 3 + 1 + 1 + 3
        condition_dim = latent_dim + action_global_dim if variant == "latent_graph" else 0
        self.dimensions = V3Dimensions(node_dim=node_dim, edge_dim=edge_dim, action_global_dim=action_global_dim)
        self.node = nn.Sequential(nn.Linear(node_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
        self.action_block = ActionTokenBlock(hidden_dim, action_global_dim, attention_heads) if variant == "action_token_graph" else None
        self.layers = nn.ModuleList(GraphMessageLayer(hidden_dim, edge_dim, condition_dim) for _ in range(layers))
        self.head = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 3))

    def _edge_features(
        self,
        positions: Tensor,
        velocities: Tensor,
        neighbour_indices: Tensor,
        neighbour_mask: Tensor,
        rest_edge_vectors: Tensor,
        rest_edge_lengths: Tensor,
    ) -> tuple[Tensor, Tensor]:
        neighbours = gather_neighbours(positions, neighbour_indices)
        neighbour_velocity = gather_neighbours(velocities, neighbour_indices)
        current_vector = neighbours - positions[:, :, None]
        current_length = torch.linalg.vector_norm(current_vector, dim=-1, keepdim=True)
        rest_length = rest_edge_lengths[..., None]
        stretch = current_length / rest_length.clamp_min(1e-8) - 1
        relative_velocity = neighbour_velocity - velocities[:, :, None]
        edge_features = torch.cat(
            (rest_edge_vectors, current_vector, current_length, relative_velocity, rest_length, stretch, relative_velocity),
            dim=-1,
        )
        return edge_features, neighbour_mask

    def forward(
        self, positions: Tensor, velocities: Tensor, dino: Tensor, particle_mask: Tensor,
        dino_imputed: Tensor, controller_positions: Tensor, controller_velocity: Tensor,
        controller_mask: Tensor, scale: Tensor, dt: Tensor, x0: Tensor,
        neighbour_indices: Tensor, neighbour_mask: Tensor, rest_edge_vectors: Tensor,
        rest_edge_lengths: Tensor, action_time: Tensor | None = None,
    ) -> SurrogateOutput:
        action_vector, contact_point = initial_action_summary(
            positions, particle_mask, controller_positions, controller_velocity, controller_mask,
        )
        action_features = action_particle_features(positions, action_vector, contact_point)
        magnitude = torch.linalg.vector_norm(action_vector, dim=-1, keepdim=True)
        if action_time is None:
            action_time = torch.zeros_like(dt)
        action_global = torch.cat((action_vector, magnitude, contact_point, scale[:, None], dt[:, None], action_time[:, None]), dim=-1)
        reference_displacement = positions - x0
        deformation = deformation_descriptors(
            positions, particle_mask, neighbour_indices, neighbour_mask, rest_edge_lengths,
        )
        conditions = torch.stack((scale, dt), -1)[:, None].expand(-1, positions.shape[1], -1)
        geometry = torch.cat(
            (positions, velocities, x0, reference_displacement, deformation,
             dino_imputed[..., None].to(positions.dtype), conditions, action_features),
            dim=-1,
        )
        if self.variant == "latent_graph":
            z_object = self.latent_encoder(x0, dino, dino_imputed, particle_mask)
            node_features = geometry
            layer_condition = torch.cat((z_object, action_global), dim=-1)
        else:
            node_features = torch.cat((geometry, self.dino(dino)), dim=-1)
            layer_condition = None
        hidden = self.node(node_features)
        if self.action_block is not None:
            hidden = self.action_block(hidden, action_global, contact_point)
        edge_features, valid_edges = self._edge_features(
            positions, velocities, neighbour_indices, neighbour_mask, rest_edge_vectors, rest_edge_lengths,
        )
        valid_edges = edge_validity(particle_mask, neighbour_indices, valid_edges)
        for layer in self.layers:
            hidden = layer(hidden, edge_features, neighbour_indices, valid_edges, layer_condition)
            hidden = hidden * particle_mask[..., None].to(hidden.dtype)
        displacement = self.head(hidden) * particle_mask[..., None].to(hidden.dtype)
        return SurrogateOutput(displacement, None, None, None)
