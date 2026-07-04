from dataclasses import dataclass

import torch
from torch import Tensor, nn

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
        self.e0 = ConvBlock(cin, base)
        self.e1 = ConvBlock(base, base * 2)
        self.mid = ConvBlock(base * 2, base * 4)
        # PyTorch MPS does not implement 3D pooling. Learned strided
        # convolutions are native on both MPS and CUDA and retain the same
        # 32 -> 16 -> 8 spatial contract.
        self.down0 = nn.Conv3d(base, base, 3, stride=2, padding=1)
        self.down1 = nn.Conv3d(base * 2, base * 2, 3, stride=2, padding=1)
        self.u1 = nn.ConvTranspose3d(base * 4, base * 2, 2, stride=2)
        self.d1 = ConvBlock(base * 4, base * 2)
        self.u0 = nn.ConvTranspose3d(base * 2, base, 2, stride=2)
        self.d0 = ConvBlock(base * 2, base)

    def forward(self, x: Tensor) -> Tensor:
        e0 = self.e0(x)
        e1 = self.e1(self.down0(e0))
        mid = self.mid(self.down1(e1))
        d1 = self.d1(torch.cat([self.u1(mid), e1], 1))
        return self.d0(torch.cat([self.u0(d1), e0], 1))


@dataclass
class SurrogateOutput:
    displacement: Tensor
    next_occupancy: Tensor
    next_velocity: Tensor
    decoder_features: Tensor


class ParticleGridSurrogate(nn.Module):
    """Grid-guided particle update with persistent particle-held DINO features."""

    def __init__(self, dino_dim: int = 768, dino_grid_dim: int = 16, base: int = 24, resolution: int = 32):
        super().__init__()
        self.spec = GridSpec(resolution=resolution)
        self.dino = nn.Sequential(nn.LayerNorm(dino_dim), nn.Linear(dino_dim, 64), nn.SiLU(), nn.Linear(64, dino_grid_dim))
        # object: occupancy, velocity(3), validity, dino; controller: occupancy, velocity(3)
        cin = 1 + 3 + 1 + dino_grid_dim + 1 + 3
        self.unet = UNet3D(cin, base)
        self.grid_head = nn.Conv3d(base, 4, 1)
        particle_in = base + 4 + 3 + 3 + dino_grid_dim + 1 + 2
        self.particle_head = nn.Sequential(nn.Linear(particle_in, 128), nn.SiLU(), nn.Linear(128, 128), nn.SiLU(), nn.Linear(128, 3))

    def forward(self, positions: Tensor, velocities: Tensor, dino: Tensor, particle_mask: Tensor,
                dino_imputed: Tensor, controller_positions: Tensor, controller_velocity: Tensor,
                controller_mask: Tensor, scale: Tensor, dt: Tensor) -> SurrogateOutput:
        emb = self.dino(dino)
        object_features = torch.cat([velocities, particle_mask[..., None].to(positions.dtype), emb], -1)
        object_grid, object_occ = scatter_particles(positions, object_features, particle_mask, self.spec)
        controller_grid, controller_occ = scatter_particles(controller_positions, controller_velocity, controller_mask, self.spec)
        grid_in = torch.cat([object_occ, object_grid, controller_occ, controller_grid], 1)
        decoded = self.unet(grid_in)
        raw = self.grid_head(decoded)
        next_occ = torch.nn.functional.softplus(raw[:, :1])
        next_velocity = raw[:, 1:]
        sampled = gather_grid(torch.cat([decoded, next_occ, next_velocity], 1), positions, self.spec)
        conditions = torch.stack([scale, dt], -1)[:, None].expand(-1, positions.shape[1], -1)
        particle_in = torch.cat([sampled, positions, velocities, emb, dino_imputed[..., None].to(positions.dtype), conditions], -1)
        displacement = self.particle_head(particle_in) * particle_mask[..., None]
        return SurrogateOutput(displacement, next_occ, next_velocity, decoded)
