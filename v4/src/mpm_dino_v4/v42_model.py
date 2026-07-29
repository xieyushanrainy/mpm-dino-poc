from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .model import masked_mean
from .v41_model import (
    GeometryAwareRegionTokens, RegionTokenLocalAdapter,
    V41TrajectorySurrogate,
)
from .v42_geometry import identity_rotation_6d, rotation_6d_to_matrix


@dataclass
class V42TrajectoryOutput:
    com: Tensor
    rotation_6d: Tensor
    rotation: Tensor
    canonical_displacement: Tensor
    canonical_shape: Tensor
    position: Tensor
    physical_hidden: Tensor
    local_hidden: Tensor


class V42RotationAwareSurrogate(V41TrajectorySurrogate):
    """Track-B physical trunk with explicit COM/rotation/canonical deformation.

    The inherited V4.1 heads are retained in the state dictionary only for
    historical initialization compatibility; V4.2 uses the heads below.
    """

    def __init__(
        self, local_mode="geometry", hidden_dim=128, blocks=4, heads=4,
        dropout=0.1, frames=59, local_trunk_alpha=0.0,
        gradient_checkpointing=True,
    ):
        if local_mode not in {"zero", "geometry", "real_dino"}:
            raise ValueError(local_mode)
        super().__init__(
            mechanism="none", hidden_dim=hidden_dim, blocks=blocks,
            heads=heads, dropout=dropout, frames=frames,
            gradient_checkpointing=gradient_checkpointing,
        )
        self.local_mode = local_mode
        self.local_trunk_alpha = float(local_trunk_alpha)
        self.v42_com_head = nn.Sequential(
            nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(), nn.Linear(hidden_dim, 3),
        )
        self.rotation_head = nn.Sequential(
            nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(), nn.Linear(hidden_dim, 6),
        )
        # Geometry-only and visual conditions are architecture-identical.
        # Geometry-only supplies zero DINO while retaining the real validity
        # mask; real/point-shuffled comparisons differ only in input tensors.
        self.region_encoder = GeometryAwareRegionTokens(
            32, hidden_dim, heads, region_tokens=4,
        )
        self.region_adapter = RegionTokenLocalAdapter(hidden_dim, heads)
        self.canonical_head = nn.Sequential(
            nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(), nn.Linear(hidden_dim, 3),
        )
        nn.init.zeros_(self.v42_com_head[-1].weight)
        nn.init.zeros_(self.v42_com_head[-1].bias)
        nn.init.zeros_(self.rotation_head[-1].weight)
        with torch.no_grad():
            self.rotation_head[-1].bias.copy_(identity_rotation_6d(
                device=self.rotation_head[-1].bias.device,
                dtype=self.rotation_head[-1].bias.dtype,
            ))
        nn.init.zeros_(self.canonical_head[-1].weight)
        nn.init.zeros_(self.canonical_head[-1].bias)

    def _physical_features(self, **inputs):
        captured = {}

        def capture(_module, _args, output):
            captured["hidden"] = output

        handle = self.blocks[-1].register_forward_hook(capture)
        try:
            legacy = super().forward(**inputs)
        finally:
            handle.remove()
        return legacy, captured["hidden"]

    def forward(self, **inputs):
        legacy, physical_hidden = self._physical_features(**inputs)
        input_mask = inputs["input_mask"]
        reference = inputs["x1"]
        reference_com = masked_mean(reference, input_mask)
        reference_shape = reference - reference_com[:, None]
        radius = torch.linalg.vector_norm(
            reference_shape, dim=-1
        ).masked_fill(~input_mask, 0).amax(1).clamp_min(1e-6)
        pooled = masked_mean(
            physical_hidden,
            input_mask[:, None].expand(-1, self.frames, -1),
            dim=2,
        )
        ballistic_com = masked_mean(
            legacy.ballistic,
            input_mask[:, None].expand(-1, self.frames, -1),
            dim=2,
        )
        com = ballistic_com + self.v42_com_head(pooled)
        rotation_6d = self.rotation_head(pooled)
        rotation = rotation_6d_to_matrix(rotation_6d)

        # alpha=0 is an exact stop-gradient; intermediate alpha scales only
        # local-loss gradients while leaving the forward value unchanged.
        protected = (
            physical_hidden.detach()
            + self.local_trunk_alpha
            * (physical_hidden - physical_hidden.detach())
        )
        dino = inputs["dino"]
        if self.local_mode in {"zero", "geometry"}:
            dino = torch.zeros_like(dino)
        visual = self.dino_projection(dino, inputs["dino_valid"])
        regions = self.region_encoder(
            visual, inputs["dino_valid"], inputs["reference"], input_mask,
            inputs["neighbour_mask"], inputs["rest_edge_lengths"],
        )
        local_hidden = self.region_adapter(protected, regions, input_mask)
        if self.local_mode == "zero":
            displacement = torch.zeros_like(legacy.residual_local)
        else:
            raw = self.canonical_head(local_hidden) * input_mask[:, None, :, None]
            displacement = (
                raw - masked_mean(
                    raw, input_mask[:, None].expand(-1, self.frames, -1), 2
                )[:, :, None]
            ) * input_mask[:, None, :, None]
        canonical_shape = reference_shape[:, None] + displacement
        rotated = torch.einsum("btni,btij->btnj", canonical_shape, rotation)
        position = com[:, :, None] + rotated
        return V42TrajectoryOutput(
            com=com, rotation_6d=rotation_6d, rotation=rotation,
            canonical_displacement=displacement,
            canonical_shape=canonical_shape, position=position,
            physical_hidden=physical_hidden, local_hidden=local_hidden,
        )
