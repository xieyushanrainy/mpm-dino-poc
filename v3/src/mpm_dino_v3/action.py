"""Future-compatible initial-action summaries derived from controller tracks.

The V3 models should not depend on continuous controller trajectories.  The
current POC caches still contain controller points, so this module extracts a
single window-initial action vector and contact point that can later be
replaced by true impulse metadata.
"""

from __future__ import annotations

import torch
from torch import Tensor


def masked_mean(values: Tensor, mask: Tensor) -> Tensor:
    weight = mask[..., None].to(values.dtype)
    return (values * weight).sum(dim=-2) / weight.sum(dim=-2).clamp_min(1)


def initial_action_summary(
    positions: Tensor,
    particle_mask: Tensor,
    controller_positions: Tensor,
    controller_velocity: Tensor,
    controller_mask: Tensor,
) -> tuple[Tensor, Tensor]:
    """Return ``(action_vector, contact_point)`` for a batch.

    ``action_vector`` is the masked mean initial controller velocity.  The
    ``contact_point`` is the valid controller point nearest to any valid object
    particle at the action frame.
    """

    action_vector = masked_mean(controller_velocity, controller_mask)
    distances = torch.cdist(controller_positions, positions)
    controller_valid = controller_mask[:, :, None]
    particle_valid = particle_mask[:, None, :]
    distances = distances.masked_fill(~(controller_valid & particle_valid), torch.inf)
    nearest_particle_distance = distances.amin(dim=-1)
    contact_index = nearest_particle_distance.argmin(dim=-1)
    contact_point = controller_positions[
        torch.arange(controller_positions.shape[0], device=controller_positions.device), contact_index
    ]
    any_controller = controller_mask.any(dim=-1, keepdim=True)
    return action_vector, torch.where(any_controller, contact_point, torch.zeros_like(contact_point))


def action_particle_features(
    positions: Tensor,
    action_vector: Tensor,
    contact_point: Tensor,
    eps: float = 1e-8,
) -> Tensor:
    """Build per-particle action/contact features from a fixed action summary."""

    relative_contact = positions - contact_point[:, None]
    contact_distance = torch.linalg.vector_norm(relative_contact, dim=-1, keepdim=True)
    magnitude = torch.linalg.vector_norm(action_vector, dim=-1, keepdim=True)
    direction = action_vector / magnitude.clamp_min(eps)
    projection = (relative_contact * direction[:, None]).sum(dim=-1, keepdim=True)
    repeated_action = action_vector[:, None].expand(-1, positions.shape[1], -1)
    repeated_magnitude = magnitude[:, None].expand(-1, positions.shape[1], -1)
    return torch.cat((repeated_action, repeated_magnitude, relative_contact, contact_distance, projection), dim=-1)
