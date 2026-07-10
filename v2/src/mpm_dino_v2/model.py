from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor, nn

from .deformation import deformation_descriptors
from .grid import GridSpec, gather_grid, scatter_particles


class ConvBlock(nn.Module):
    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(cin, cout, 3, padding=1), nn.GroupNorm(4, cout), nn.SiLU(),
            nn.Conv3d(cout, cout, 3, padding=1), nn.GroupNorm(4, cout), nn.SiLU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class UNet3D(nn.Module):
    def __init__(self, cin: int, base: int = 24):
        super().__init__()
        self.e0, self.e1, self.mid = ConvBlock(cin, base), ConvBlock(base, base * 2), ConvBlock(base * 2, base * 4)
        self.down0 = nn.Conv3d(base, base, 3, stride=2, padding=1)
        self.down1 = nn.Conv3d(base * 2, base * 2, 3, stride=2, padding=1)
        self.u1, self.d1 = nn.ConvTranspose3d(base * 4, base * 2, 2, stride=2), ConvBlock(base * 4, base * 2)
        self.u0, self.d0 = nn.ConvTranspose3d(base * 2, base, 2, stride=2), ConvBlock(base * 2, base)

    def forward(self, x: Tensor) -> Tensor:
        e0 = self.e0(x)
        e1 = self.e1(self.down0(e0))
        mid = self.mid(self.down1(e1))
        d1 = self.d1(torch.cat((self.u1(mid), e1), 1))
        return self.d0(torch.cat((self.u0(d1), e0), 1))


@dataclass
class SurrogateOutput:
    displacement: Tensor
    next_occupancy: Tensor | None
    next_velocity: Tensor | None
    decoder_features: Tensor | None


class ParticleControllerEncoder(nn.Module):
    """Permutation-invariant per-particle context from prescribed controllers."""

    def __init__(self):
        super().__init__()
        self.pair_mlp = nn.Sequential(nn.Linear(7, 32), nn.SiLU(), nn.Linear(32, 32), nn.SiLU())

    def forward(self, positions: Tensor, controller_positions: Tensor,
                controller_velocity: Tensor, controller_mask: Tensor) -> Tensor:
        relative = controller_positions[:, None] - positions[:, :, None]
        velocity = controller_velocity[:, None].expand(-1, positions.shape[1], -1, -1)
        distance = torch.linalg.vector_norm(relative, dim=-1, keepdim=True)
        encoded = self.pair_mlp(torch.cat((relative, velocity, distance), dim=-1))
        valid = controller_mask[:, None, :, None]
        weight = valid.to(encoded.dtype)
        mean = (encoded * weight).sum(dim=2) / weight.sum(dim=2).clamp_min(1)
        maximum = encoded.masked_fill(~valid, -torch.inf).amax(dim=2)
        any_valid = controller_mask.any(dim=1)[:, None, None]
        maximum = torch.where(any_valid, maximum, torch.zeros_like(maximum))
        return torch.cat((mean, maximum), dim=-1)


class ParticleGridSurrogate(nn.Module):
    """V2-B surrogate with effective reference and local deformation state."""

    def __init__(self, dino_dim: int = 768, dino_grid_dim: int = 16, base: int = 24,
                 resolution: int = 32, variant: Literal["fused", "grid_only", "particle_only"] = "fused"):
        super().__init__()
        if variant not in {"fused", "grid_only", "particle_only"}:
            raise ValueError(f"unsupported model variant: {variant}")
        self.variant = variant
        self.spec = GridSpec(resolution=resolution)
        self.dino = nn.Sequential(nn.LayerNorm(dino_dim), nn.Linear(dino_dim, 64), nn.SiLU(), nn.Linear(64, dino_grid_dim))
        direct_particle_dim = 3 + 3 + dino_grid_dim + 1 + 2 + 9
        if variant != "particle_only":
            # V1 grid channels plus reference displacement(3) and stretch statistics(3).
            self.unet = UNet3D(1 + 3 + 1 + dino_grid_dim + 6 + 1 + 3, base)
            self.grid_head = nn.Conv3d(base, 4, 1)
        if variant == "particle_only":
            self.controller_encoder = ParticleControllerEncoder()
            particle_in = direct_particle_dim + 64
        elif variant == "grid_only":
            particle_in = base + 4
        else:
            particle_in = base + 4 + direct_particle_dim
        self.particle_input_dim = particle_in
        self.particle_head = nn.Sequential(
            nn.Linear(particle_in, 128), nn.SiLU(), nn.Linear(128, 128), nn.SiLU(), nn.Linear(128, 3),
        )

    def forward(
        self, positions: Tensor, velocities: Tensor, dino: Tensor, particle_mask: Tensor,
        dino_imputed: Tensor, controller_positions: Tensor, controller_velocity: Tensor,
        controller_mask: Tensor, scale: Tensor, dt: Tensor, x0: Tensor,
        neighbour_indices: Tensor, neighbour_mask: Tensor, rest_edge_lengths: Tensor,
    ) -> SurrogateOutput:
        embedding = self.dino(dino)
        reference_displacement = positions - x0
        deformation = deformation_descriptors(
            positions, particle_mask, neighbour_indices, neighbour_mask, rest_edge_lengths,
        )
        conditions = torch.stack((scale, dt), -1)[:, None].expand(-1, positions.shape[1], -1)
        direct_features = torch.cat(
            (positions, velocities, embedding, dino_imputed[..., None].to(positions.dtype),
             conditions, x0, reference_displacement, deformation), dim=-1)
        if self.variant == "particle_only":
            controller_context = self.controller_encoder(
                positions, controller_positions, controller_velocity, controller_mask)
            particle_features = torch.cat((direct_features, controller_context), dim=-1)
            next_occupancy = next_velocity = decoded = None
        else:
            object_features = torch.cat(
                (velocities, particle_mask[..., None].to(positions.dtype), embedding,
                 reference_displacement, deformation), dim=-1)
            object_grid, object_occupancy = scatter_particles(positions, object_features, particle_mask, self.spec)
            controller_grid, controller_occupancy = scatter_particles(
                controller_positions, controller_velocity, controller_mask, self.spec)
            decoded = self.unet(torch.cat((object_occupancy, object_grid, controller_occupancy, controller_grid), 1))
            raw = self.grid_head(decoded)
            next_occupancy = torch.nn.functional.softplus(raw[:, :1])
            next_velocity = raw[:, 1:]
            sampled = gather_grid(torch.cat((decoded, next_occupancy, next_velocity), 1), positions, self.spec)
            particle_features = sampled if self.variant == "grid_only" else torch.cat((sampled, direct_features), dim=-1)
        displacement = self.particle_head(particle_features) * particle_mask[..., None]
        return SurrogateOutput(displacement, next_occupancy, next_velocity, decoded)
