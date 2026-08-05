from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from mpm_dino_v4.v42_geometry import identity_rotation_6d, rotation_6d_to_matrix


def masked_mean(value: Tensor, mask: Tensor, dim: int = 1) -> Tensor:
    weight = mask.to(value.dtype)
    while weight.ndim < value.ndim:
        weight = weight.unsqueeze(-1)
    return (value * weight).sum(dim) / weight.sum(dim).clamp_min(1)


def centered_reference(reference: Tensor, mask: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    centre = masked_mean(reference, mask)
    shape = reference - centre[:, None]
    radius = torch.linalg.vector_norm(shape, dim=-1).masked_fill(~mask, 0).amax(1).clamp_min(1e-6)
    return shape, centre, radius


def ballistic_com(
    x0: Tensor,
    x1: Tensor,
    mask: Tensor,
    dt: Tensor,
    gravity: Tensor,
    frames: int = 59,
) -> Tensor:
    """Analytic COM continuation from two observed frames."""
    c0 = masked_mean(x0, mask)
    c1 = masked_mean(x1, mask)
    step = dt.reshape(-1, 1).clamp_min(1e-8)
    velocity = (c1 - c0) / step
    horizon = torch.arange(1, frames + 1, device=x1.device, dtype=x1.dtype)[None, :, None]
    seconds = horizon * step[:, None]
    return c1[:, None] + velocity[:, None] * seconds + 0.5 * gravity[:, None] * seconds.square()


def reconstruct_positions(com: Tensor, rotation: Tensor, reference_shape: Tensor, deformation: Tensor) -> Tensor:
    canonical = reference_shape[:, None] + deformation
    return com[:, :, None] + torch.einsum("btni,btij->btnj", canonical, rotation)


class TemporalMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, frames: int, dropout: float):
        super().__init__()
        self.frames = frames
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim + 2),
            nn.Linear(input_dim + 2, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, context: Tensor) -> Tensor:
        phase = torch.linspace(0, 1, self.frames, device=context.device, dtype=context.dtype)
        time = torch.stack((phase, phase.square()), -1)[None].expand(len(context), -1, -1)
        return self.net(torch.cat((context[:, None].expand(-1, self.frames, -1), time), -1))


class V5COMModel(nn.Module):
    def __init__(self, hidden_dim: int = 128, frames: int = 59, dropout: float = 0.1):
        super().__init__()
        self.frames = frames
        self.residual = TemporalMLP(9, hidden_dim, 3, frames, dropout)

    def forward(self, x0: Tensor, x1: Tensor, mask: Tensor, dt: Tensor, gravity: Tensor) -> tuple[Tensor, Tensor]:
        base = ballistic_com(x0, x1, mask, dt, gravity, self.frames)
        c0 = masked_mean(x0, mask)
        c1 = masked_mean(x1, mask)
        velocity_step = c1 - c0
        context = torch.cat((c1, velocity_step, gravity), -1)
        correction = self.residual(context)
        correction = correction.clone()
        correction[:, 0] = 0
        return base + correction, base


class V5RotationModel(nn.Module):
    def __init__(self, hidden_dim: int = 128, frames: int = 59, dropout: float = 0.1):
        super().__init__()
        self.frames = frames
        self.head = TemporalMLP(9, hidden_dim, 6, frames, dropout)

    def forward(self, x0: Tensor, x1: Tensor, mask: Tensor) -> Tensor:
        q0, _, radius0 = centered_reference(x0, mask)
        q1, _, radius1 = centered_reference(x1, mask)
        radius = torch.maximum(radius0, radius1)
        shape_change = masked_mean((q1 - q0) / radius[:, None, None], mask)
        spread = masked_mean(q1.square() / radius[:, None, None].square(), mask)
        context = torch.cat((shape_change, spread, masked_mean(q1 / radius[:, None, None], mask)), -1)
        representation = self.head(context)
        # A random head is trained from scratch; adding identity keeps its initial
        # outputs finite without loading learned V4 state.
        representation = representation + identity_rotation_6d(
            len(x0), self.frames, device=x0.device, dtype=x0.dtype,
        )
        return rotation_6d_to_matrix(representation)

    @staticmethod
    def identity(batch: int, frames: int, *, device=None, dtype=None) -> Tensor:
        return torch.eye(3, device=device, dtype=dtype).expand(batch, frames, 3, 3).clone()


class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim * 2),
            nn.SiLU(), nn.Dropout(dropout), nn.Linear(hidden_dim * 2, hidden_dim),
        )

    def forward(self, value: Tensor) -> Tensor:
        return value + self.net(value)


@dataclass
class InteractionOutput:
    latent: Tensor
    contact_logits: Tensor
    event_time: Tensor


class V5InteractionEncoder(nn.Module):
    """Learned pointwise conditioning without analytic gap/proximity features."""

    def __init__(self, hidden_dim: int = 128, blocks: int = 4, dropout: float = 0.1):
        super().__init__()
        self.input = nn.Linear(10, hidden_dim)
        self.blocks = nn.ModuleList(ResidualBlock(hidden_dim, dropout) for _ in range(blocks))
        self.contact_head = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, 1))
        self.time_head = nn.Sequential(
            nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1), nn.Tanh(),
        )

    def forward(self, reference_shape: Tensor, rigid_position: Tensor, point_mask: Tensor, floor_z: Tensor) -> InteractionOutput:
        radius = torch.linalg.vector_norm(reference_shape, dim=-1).masked_fill(~point_mask, 0).amax(1).clamp_min(1e-6)
        q = reference_shape / radius[:, None, None]
        rigid = rigid_position / radius[:, None, None, None]
        floor = floor_z[:, None, None, None].expand(*rigid.shape[:-1], 1) / radius[:, None, None, None]
        q_time = q[:, None].expand(-1, rigid.shape[1], -1, -1)
        value = self.input(torch.cat((q_time, rigid, floor, q_time * rigid), -1))
        mask = point_mask[:, None, :, None].to(value.dtype)
        value = value * mask
        for block in self.blocks:
            value = block(value) * mask
        contact = self.contact_head(value).squeeze(-1).masked_fill(~point_mask[:, None], -20)
        pooled = masked_mean(value, point_mask[:, None].expand(-1, value.shape[1], -1), dim=2)
        event_time = self.time_head(pooled).squeeze(-1)
        return InteractionOutput(value, contact, event_time)


class V5DeformationDecoder(nn.Module):
    def __init__(self, interaction_dim: int = 128, hidden_dim: int = 128, blocks: int = 4, dropout: float = 0.1):
        super().__init__()
        self.input = nn.Linear(interaction_dim + 3, hidden_dim)
        self.blocks = nn.ModuleList(ResidualBlock(hidden_dim, dropout) for _ in range(blocks))
        self.output = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, 3))

    def forward(self, reference_shape: Tensor, latent: Tensor, point_mask: Tensor) -> Tensor:
        radius = torch.linalg.vector_norm(reference_shape, dim=-1).masked_fill(~point_mask, 0).amax(1).clamp_min(1e-6)
        q = (reference_shape / radius[:, None, None])[:, None].expand(-1, latent.shape[1], -1, -1)
        mask = point_mask[:, None, :, None].to(latent.dtype)
        value = self.input(torch.cat((q, latent), -1)) * mask
        for block in self.blocks:
            value = block(value) * mask
        raw = self.output(value) * radius[:, None, None, None] * mask
        mean = raw.sum(2, keepdim=True) / mask.sum(2, keepdim=True).clamp_min(1)
        return (raw - mean) * mask


@dataclass
class FactorizedMotionOutput:
    com: Tensor
    ballistic_com: Tensor
    rotation: Tensor
    reference_shape: Tensor
    interaction: InteractionOutput
    canonical_displacement: Tensor
    position: Tensor


class V5CausalDeformationModel(nn.Module):
    def __init__(self, com: V5COMModel, rotation: V5RotationModel, interaction: V5InteractionEncoder, decoder: V5DeformationDecoder):
        super().__init__()
        self.com_model = com
        self.rotation_model = rotation
        self.interaction_encoder = interaction
        self.deformation_decoder = decoder

    def forward(self, *, x0: Tensor, x1: Tensor, input_mask: Tensor, dt: Tensor, gravity: Tensor, floor_z: Tensor, use_identity_rotation: bool = False) -> FactorizedMotionOutput:
        com, ballistic = self.com_model(x0, x1, input_mask, dt, gravity)
        if use_identity_rotation:
            rotation = self.rotation_model.identity(len(x0), com.shape[1], device=x0.device, dtype=x0.dtype)
        else:
            rotation = self.rotation_model(x0, x1, input_mask)
        reference_shape, _, _ = centered_reference(x1, input_mask)
        rigid = reconstruct_positions(com, rotation, reference_shape, torch.zeros(
            *com.shape[:2], reference_shape.shape[1], 3, device=x0.device, dtype=x0.dtype,
        ))
        interaction = self.interaction_encoder(reference_shape, rigid, input_mask, floor_z)
        deformation = self.deformation_decoder(reference_shape, interaction.latent, input_mask)
        position = reconstruct_positions(com, rotation, reference_shape, deformation)
        return FactorizedMotionOutput(com, ballistic, rotation, reference_shape, interaction, deformation, position)

