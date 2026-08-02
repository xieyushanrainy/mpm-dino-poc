from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .model import masked_mean
from .v41_model import (
    GeometryAwareRegionTokens, RegionTokenLocalAdapter,
    V41TrajectorySurrogate,
)
from .v42_geometry import (
    identity_rotation_6d, rotation_6d_to_matrix, rotation_vector_to_matrix,
)


@dataclass
class V42TrajectoryOutput:
    com: Tensor
    ballistic_com: Tensor
    rotation_representation: Tensor
    rotation: Tensor
    angular_velocity: Tensor | None
    angular_velocity_change: Tensor | None
    contact_probability: Tensor | None
    contact_point: Tensor | None
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
        gradient_checkpointing=True, rotation_parameterization="6d",
        rotation_attention=False, rotation_dynamics=False,
        contact_rotation_mode=None, angular_damping=0.95,
        oracle_condition_dim=0, oracle_injection="decoder",
    ):
        if local_mode not in {"zero", "geometry", "real_dino"}:
            raise ValueError(local_mode)
        super().__init__(
            mechanism="none", hidden_dim=hidden_dim, blocks=blocks,
            heads=heads, dropout=dropout, frames=frames,
            gradient_checkpointing=gradient_checkpointing,
        )
        self.local_mode = local_mode
        self.oracle_condition_dim = int(oracle_condition_dim)
        if self.oracle_condition_dim < 0:
            raise ValueError("oracle_condition_dim must be non-negative")
        if oracle_injection not in {"decoder", "adapter"}:
            raise ValueError("oracle_injection must be decoder or adapter")
        self.oracle_injection = oracle_injection
        self.local_trunk_alpha = float(local_trunk_alpha)
        if rotation_parameterization not in {"6d", "axis_angle"}:
            raise ValueError(rotation_parameterization)
        self.rotation_parameterization = rotation_parameterization
        self.rotation_attention_enabled = bool(rotation_attention)
        self.rotation_dynamics_enabled = bool(rotation_dynamics)
        if contact_rotation_mode not in {None, "absolute", "impulse"}:
            raise ValueError(contact_rotation_mode)
        self.contact_rotation_mode = contact_rotation_mode
        self.angular_damping = float(angular_damping)
        if not 0 <= self.angular_damping <= 1:
            raise ValueError("angular_damping must lie in [0, 1]")
        if contact_rotation_mode is not None and (
            rotation_parameterization != "axis_angle"
            or not self.rotation_attention_enabled
            or self.rotation_dynamics_enabled
        ):
            raise ValueError(
                "contact rotation requires axis-angle attention without "
                "legacy rotation_dynamics"
            )
        if self.rotation_dynamics_enabled and (
            rotation_parameterization != "axis_angle"
            or not self.rotation_attention_enabled
        ):
            raise ValueError(
                "rotation dynamics requires axis-angle attention rotation"
            )
        self.v42_com_head = nn.Sequential(
            nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(), nn.Linear(hidden_dim, 3),
        )
        self.rotation_head = nn.Sequential(
            nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(), nn.Linear(
                hidden_dim, 6 if rotation_parameterization == "6d" else 3
            ),
        )
        if self.rotation_attention_enabled:
            self.rotation_contact_projection = nn.Sequential(
                nn.Linear(7, hidden_dim), nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.rotation_attention = nn.MultiheadAttention(
                hidden_dim, heads, dropout=dropout, batch_first=True,
            )
            self.rotation_adapter = nn.Sequential(
                nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(), nn.Linear(hidden_dim, hidden_dim),
            )
            if self.contact_rotation_mode is not None:
                self.rotation_contact_head = nn.Sequential(
                    nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, 1),
                )
                self.rotation_contact_score = nn.Sequential(
                    nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, 1),
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
        if self.oracle_condition_dim and self.oracle_injection == "decoder":
            self.oracle_canonical_head = nn.Sequential(
                nn.LayerNorm(hidden_dim + self.oracle_condition_dim),
                nn.Linear(hidden_dim + self.oracle_condition_dim, hidden_dim),
                nn.SiLU(), nn.Linear(hidden_dim, 3),
            )
        if self.oracle_condition_dim and self.oracle_injection == "adapter":
            self.oracle_adapter_projection = nn.Sequential(
                nn.LayerNorm(self.oracle_condition_dim),
                nn.Linear(self.oracle_condition_dim, hidden_dim), nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim, bias=False),
            )
            nn.init.zeros_(self.oracle_adapter_projection[1].bias)
        nn.init.zeros_(self.v42_com_head[-1].weight)
        nn.init.zeros_(self.v42_com_head[-1].bias)
        nn.init.zeros_(self.rotation_head[-1].weight)
        if rotation_parameterization == "6d":
            with torch.no_grad():
                self.rotation_head[-1].bias.copy_(identity_rotation_6d(
                    device=self.rotation_head[-1].bias.device,
                    dtype=self.rotation_head[-1].bias.dtype,
                ))
        else:
            nn.init.zeros_(self.rotation_head[-1].bias)
        if self.contact_rotation_mode is not None:
            nn.init.zeros_(self.rotation_contact_head[-1].weight)
            nn.init.constant_(self.rotation_contact_head[-1].bias, -4.0)
            nn.init.zeros_(self.rotation_contact_score[-1].weight)
            nn.init.zeros_(self.rotation_contact_score[-1].bias)
        nn.init.zeros_(self.canonical_head[-1].weight)
        nn.init.zeros_(self.canonical_head[-1].bias)
        if self.oracle_condition_dim and self.oracle_injection == "decoder":
            nn.init.zeros_(self.oracle_canonical_head[-1].weight)
            nn.init.zeros_(self.oracle_canonical_head[-1].bias)

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
        oracle_condition = inputs.pop("oracle_condition", None)
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
        com_correction = self.v42_com_head(pooled)
        # The first future frame is already determined very accurately by the
        # observed finite-difference velocity. Learn contact corrections after
        # H1 without allowing the decoder to damage that physical anchor.
        com_correction = com_correction.clone()
        com_correction[:, 0] = 0
        com = ballistic_com + com_correction
        rotation_features = pooled
        contact_probability = None
        contact_point = None
        if self.rotation_attention_enabled:
            batch, frames, points, hidden = physical_hidden.shape
            velocity = (
                (inputs["x1"] - inputs["x0"])
                / inputs["dt"][:, None, None].clamp_min(1e-8)
            )
            velocity_step = (
                velocity * inputs["dt"][:, None, None]
                / radius[:, None, None]
            )
            normalized_reference = reference_shape / radius[:, None, None]
            floor_gap = (
                legacy.ballistic[..., 2]
                - inputs["floor_z"][:, None, None]
            ) / radius[:, None, None]
            descriptor = torch.cat((
                normalized_reference[:, None].expand(-1, frames, -1, -1),
                floor_gap[..., None],
                velocity_step[:, None].expand(-1, frames, -1, -1),
            ), dim=-1)
            contact = self.rotation_contact_projection(descriptor)
            point_features = physical_hidden.detach() + contact
            query = pooled.detach().reshape(batch * frames, 1, hidden)
            attended, _ = self.rotation_attention(
                query,
                point_features.reshape(batch * frames, points, hidden),
                point_features.reshape(batch * frames, points, hidden),
                key_padding_mask=(
                    ~input_mask[:, None].expand(-1, frames, -1)
                ).reshape(batch * frames, points),
                need_weights=False,
            )
            rotation_features = (
                query[:, 0] + self.rotation_adapter(attended[:, 0])
            ).reshape(batch, frames, hidden)
            if self.contact_rotation_mode is not None:
                contact_probability = torch.sigmoid(
                    self.rotation_contact_head(rotation_features).squeeze(-1)
                )
                point_scores = self.rotation_contact_score(
                    point_features
                ).squeeze(-1).masked_fill(
                    ~input_mask[:, None], -torch.inf
                )
                point_weights = torch.softmax(point_scores, dim=2)
                contact_point = torch.einsum(
                    "btn,bni->bti", point_weights, reference_shape
                )
        rotation_representation = self.rotation_head(rotation_features)
        angular_velocity = None
        angular_velocity_change = None
        if self.contact_rotation_mode == "absolute":
            rotation_representation = rotation_representation.clone()
            rotation_representation[:, 0] = 0
            contact_probability = contact_probability.clone()
            contact_probability[:, 0] = 0
            # A differentiable "contact has happened" state keeps the
            # absolute residual enabled after a brief predicted onset.
            contact_state = 1 - torch.cumprod(
                1 - contact_probability.clamp(max=1 - 1e-6), dim=1
            )
            contact_state = contact_state.clone()
            contact_state[:, 0] = 0
            rotation = rotation_vector_to_matrix(
                rotation_representation * contact_state[..., None]
            )
        elif self.contact_rotation_mode == "impulse":
            rotation_representation = rotation_representation.clone()
            rotation_representation[:, 0] = 0
            contact_probability = contact_probability.clone()
            contact_probability[:, 0] = 0
            angular_velocity_change = (
                rotation_representation * contact_probability[..., None]
            )
            velocity = torch.zeros_like(rotation_representation[:, 0])
            accumulated = torch.eye(
                3, device=rotation_representation.device,
                dtype=rotation_representation.dtype,
            ).expand(rotation_representation.shape[0], 3, 3)
            velocities, rotations = [], []
            for frame in range(self.frames):
                if frame:
                    velocity = (
                        self.angular_damping * velocity
                        + angular_velocity_change[:, frame]
                    )
                    accumulated = accumulated @ rotation_vector_to_matrix(
                        velocity * inputs["dt"][:, None]
                    )
                velocities.append(velocity)
                rotations.append(accumulated)
            angular_velocity = torch.stack(velocities, dim=1)
            rotation = torch.stack(rotations, dim=1)
        elif self.rotation_dynamics_enabled:
            rotation_representation = rotation_representation.clone()
            rotation_representation[:, 0] = 0
            angular_velocity_change = rotation_representation
            angular_velocity = torch.cumsum(
                angular_velocity_change, dim=1,
            )
            angular_velocity = angular_velocity.clone()
            angular_velocity[:, 0] = 0
            step_rotation = rotation_vector_to_matrix(
                angular_velocity * inputs["dt"][:, None, None]
            )
            accumulated = torch.eye(
                3, device=step_rotation.device, dtype=step_rotation.dtype,
            ).expand(step_rotation.shape[0], 3, 3)
            rotations = []
            for frame in range(self.frames):
                if frame:
                    accumulated = accumulated @ step_rotation[:, frame]
                rotations.append(accumulated)
            rotation = torch.stack(rotations, dim=1)
        elif self.rotation_parameterization == "axis_angle":
            rotation_representation = rotation_representation.clone()
            rotation_representation[:, 0] = 0
            rotation = rotation_vector_to_matrix(rotation_representation)
        else:
            rotation = rotation_6d_to_matrix(rotation_representation)

        # alpha=0 is an exact stop-gradient; intermediate alpha scales only
        # local-loss gradients while leaving the forward value unchanged.
        protected = (
            physical_hidden.detach()
            + self.local_trunk_alpha
            * (physical_hidden - physical_hidden.detach())
        )
        if self.oracle_condition_dim and self.oracle_injection == "adapter":
            if oracle_condition is None:
                oracle_condition = protected.new_zeros(
                    protected.shape[0], self.frames,
                    self.oracle_condition_dim,
                )
            global_expected = (
                protected.shape[0], self.frames, self.oracle_condition_dim,
            )
            point_expected = (
                protected.shape[0], self.frames, protected.shape[2],
                self.oracle_condition_dim,
            )
            if tuple(oracle_condition.shape) not in {global_expected, point_expected}:
                raise ValueError(
                    "oracle_condition must have global or pointwise shape "
                    f"{global_expected} / {point_expected}, got "
                    f"{tuple(oracle_condition.shape)}"
                )
            projected = self.oracle_adapter_projection(oracle_condition)
            if projected.ndim == 3:
                projected = projected[:, :, None]
            protected = protected + projected
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
            if self.oracle_condition_dim and self.oracle_injection == "decoder":
                if oracle_condition is None:
                    oracle_condition = local_hidden.new_zeros(
                        local_hidden.shape[0], self.frames,
                        self.oracle_condition_dim,
                    )
                global_expected = (
                    local_hidden.shape[0], self.frames,
                    self.oracle_condition_dim,
                )
                point_expected = (
                    local_hidden.shape[0], self.frames, local_hidden.shape[2],
                    self.oracle_condition_dim,
                )
                if tuple(oracle_condition.shape) not in {global_expected, point_expected}:
                    raise ValueError(
                        "oracle_condition must have global or pointwise shape "
                        f"{global_expected} / {point_expected}, got "
                        f"{tuple(oracle_condition.shape)}"
                    )
                if oracle_condition.ndim == 3:
                    oracle_condition = oracle_condition[:, :, None].expand(
                        -1, -1, local_hidden.shape[2], -1,
                    )
                conditioned = torch.cat((
                    local_hidden, oracle_condition,
                ), dim=-1)
                raw = self.oracle_canonical_head(conditioned)
            else:
                raw = self.canonical_head(local_hidden)
            raw = raw * input_mask[:, None, :, None]
            displacement = (
                raw - masked_mean(
                    raw, input_mask[:, None].expand(-1, self.frames, -1), 2
                )[:, :, None]
            ) * input_mask[:, None, :, None]
        canonical_shape = reference_shape[:, None] + displacement
        rotated = torch.einsum("btni,btij->btnj", canonical_shape, rotation)
        position = com[:, :, None] + rotated
        return V42TrajectoryOutput(
            com=com, ballistic_com=ballistic_com,
            rotation_representation=rotation_representation,
            rotation=rotation, angular_velocity=angular_velocity,
            angular_velocity_change=angular_velocity_change,
            contact_probability=contact_probability,
            contact_point=contact_point,
            canonical_displacement=displacement,
            canonical_shape=canonical_shape, position=position,
            physical_hidden=physical_hidden, local_hidden=local_hidden,
        )
