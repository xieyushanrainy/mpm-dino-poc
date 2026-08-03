from __future__ import annotations

import torch

from .model import masked_mean
from .v41_data import MODEL_INPUT_KEYS
from .v42_contact_curvature import (
    CONTACT_DIM, DIRECT_CONDITION_DIM, POINT_CONDITION_DIM, curvature_features,
)
from .v42_gate2 import load_gate1e_source, train_v42_gate2
from .v42_oracle import LOSS_CONTRACT, event_normalized_canonical_mse


CAUSAL_VARIANTS = (
    "static_control", "causal_timing_only", "causal_continuous",
)


def rigid_proxy_features(batch, predictor, contact_threshold=0.01):
    """Causal point contact and event-time features from frozen rigid motion."""
    with torch.no_grad():
        output = predictor(**{key: batch[key] for key in MODEL_INPUT_KEYS})
        mask = batch["input_mask"]
        centre = masked_mean(batch["x1"], mask)
        reference_shape = batch["x1"] - centre[:, None]
        radius = torch.linalg.vector_norm(
            reference_shape, dim=-1,
        ).masked_fill(~mask, 0).amax(1).clamp_min(1e-6)
        positions = output.com[:, :, None] + torch.einsum(
            "bni,btij->btnj", reference_shape, output.rotation,
        )
        gap = (
            positions[..., 2] - batch["floor_z"][:, None, None]
        ) / radius[:, None, None]
        signed_gap = (gap / 0.1).clamp(-1, 1)
        # Smooth one-sided proximity: points below/near the surface remain high
        # rather than losing confidence as rigid penetration grows.
        proximity = torch.sigmoid(-gap / 0.02)
        previous = torch.cat((batch["x1"][:, None], positions[:, :-1]), 1)
        normal_velocity = (
            (positions[..., 2] - previous[..., 2])
            / batch["dt"][:, None, None].clamp_min(1e-8)
            / radius[:, None, None]
        ).clamp(-10, 10) / 10
        valid = mask[:, None].expand(gap.shape)
        contact = torch.stack(
            (signed_gap, proximity, normal_velocity), -1,
        ) * valid[..., None]
        near = (gap <= contact_threshold) & valid
        frames = gap.shape[1]
        frame_index = torch.arange(
            frames, device=gap.device, dtype=gap.dtype,
        )[None]
        onset = torch.full(
            (gap.shape[0],), -1, device=gap.device, dtype=torch.long,
        )
        for sample in range(gap.shape[0]):
            candidates = torch.where(near[sample].any(1))[0]
            if len(candidates):
                onset[sample] = candidates[0]
        relative_time = (
            frame_index - onset[:, None].to(gap.dtype)
        ) / max(frames - 1, 1)
        relative_time = relative_time.clamp(-1, 1)
        relative_time = torch.where(
            onset[:, None] >= 0, relative_time,
            torch.zeros_like(relative_time),
        )
    return contact.detach(), relative_time.detach()


class CausalContactConditionBuilder:
    """Matched 15-channel conditions with no future-target dependency."""

    def __init__(self, predictor, variant, contact_threshold=0.01):
        if variant not in CAUSAL_VARIANTS:
            raise ValueError(f"unknown causal-contact variant: {variant}")
        self.predictor = predictor
        self.variant = variant
        self.contact_threshold = float(contact_threshold)

    def __call__(self, batch, _stages):
        batch_size = batch["x1"].shape[0]
        frames = self.predictor.frames
        points = batch["x1"].shape[1]
        result = batch["x1"].new_zeros(
            batch_size, frames, points, DIRECT_CONDITION_DIM,
        )
        # Static curvature is causal and held identical in every arm.
        result[..., CONTACT_DIM:POINT_CONDITION_DIM] = (
            curvature_features(batch)[:, None]
        )
        if self.variant == "static_control":
            return result.detach()
        contact, relative_time = rigid_proxy_features(
            batch, self.predictor, self.contact_threshold,
        )
        if self.variant == "causal_continuous":
            result[..., :CONTACT_DIM] = contact
        # Seven discrete oracle-stage channels remain exactly zero. Only the
        # final continuous event-time channel is populated causally.
        result[..., -1] = relative_time[:, :, None]
        return result.detach()


def train_causal_contact_variant(
    dataset_root, manifest, checkpoint, output, seed, variant,
    contact_threshold=0.01, **kwargs,
):
    if variant not in CAUSAL_VARIANTS:
        raise ValueError(f"unknown causal-contact variant: {variant}")
    device = kwargs.get("device", "cuda")
    predictor, _ = load_gate1e_source(
        checkpoint, local_mode="zero", device=device,
    )
    predictor.eval()
    for parameter in predictor.parameters():
        parameter.requires_grad_(False)
    builder = CausalContactConditionBuilder(
        predictor, variant, contact_threshold,
    )
    return train_v42_gate2(
        dataset_root, manifest, checkpoint, output, seed,
        loss_options=LOSS_CONTRACT, stage_weight_mode="total_mass",
        oracle_condition_dim=DIRECT_CONDITION_DIM,
        oracle_injection="adapter", condition_builder=builder,
        condition_name=variant, exploratory_control=True,
        objective_builder=event_normalized_canonical_mse,
        objective_name="event_frame_amplitude_normalized_canonical_mse_v1",
        dataset_families=("soft_body",), selection_mode="optimized",
        experiment_name="v43_oracle_removal_" + variant,
        model_contract_version="causal_contact_adapter_v1",
        **kwargs,
    )
