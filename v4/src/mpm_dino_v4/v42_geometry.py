from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F

from .model import masked_mean


def rotation_6d_to_matrix(value: Tensor) -> Tensor:
    """Continuous 6D representation to a proper rotation matrix."""
    first = F.normalize(value[..., :3], dim=-1, eps=1e-8)
    second_raw = value[..., 3:6]
    second = F.normalize(
        second_raw - (first * second_raw).sum(-1, keepdim=True) * first,
        dim=-1,
        eps=1e-8,
    )
    third = torch.cross(first, second, dim=-1)
    return torch.stack((first, second, third), dim=-1)


def identity_rotation_6d(*shape: int, device=None, dtype=None) -> Tensor:
    value = torch.zeros(*shape, 6, device=device, dtype=dtype)
    value[..., 0] = 1
    value[..., 4] = 1
    return value


def rotation_geodesic(predicted: Tensor, target: Tensor) -> Tensor:
    relative = predicted.transpose(-1, -2) @ target
    cosine = ((relative.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2)
    # Avoid infinite acos gradients at exact identity.
    return torch.acos(cosine.clamp(-1 + 1e-6, 1 - 1e-6))


def rotation_chordal(predicted: Tensor, target: Tensor) -> Tensor:
    """Smooth squared chordal SO(3) loss; geodesic remains a report metric."""
    return 0.5 * (predicted - target).square().sum(dim=(-2, -1))


@dataclass
class CanonicalTargets:
    com: Tensor
    rotation: Tensor
    displacement: Tensor
    valid_rotation: Tensor
    singular_values: Tensor
    reference_shape: Tensor
    radius: Tensor


@torch.no_grad()
def canonical_targets(
    reference_frame: Tensor,
    target: Tensor,
    input_mask: Tensor,
    target_mask: Tensor,
    degeneracy_ratio: float = 1e-3,
) -> CanonicalTargets:
    """Build detached Kabsch-gauge targets for [B,T,N,3] trajectories."""
    batch, frames, _, _ = target.shape
    reference_com = masked_mean(reference_frame, input_mask)
    reference_shape = reference_frame - reference_com[:, None]
    radius = torch.linalg.vector_norm(
        reference_shape, dim=-1
    ).masked_fill(~input_mask, 0).amax(1).clamp_min(1e-6)
    com = masked_mean(target, target_mask, dim=2)
    centred_target = target - com[:, :, None]

    rotations, singular_values, valid_frames = [], [], []
    for batch_index in range(batch):
        batch_rotations, batch_singular, batch_valid = [], [], []
        for frame in range(frames):
            valid = input_mask[batch_index] & target_mask[batch_index, frame]
            source = reference_shape[batch_index, valid]
            destination = centred_target[batch_index, frame, valid]
            covariance = source.transpose(0, 1) @ destination
            u, singular, vh = torch.linalg.svd(covariance)
            rotation = u @ vh
            if torch.linalg.det(rotation) < 0:
                u = u.clone()
                u[:, -1] *= -1
                rotation = u @ vh
            ratio = singular[1] / singular[0].clamp_min(1e-12)
            batch_rotations.append(rotation)
            batch_singular.append(singular)
            batch_valid.append((valid.sum() >= 3) & (ratio >= degeneracy_ratio))
        rotations.append(torch.stack(batch_rotations))
        singular_values.append(torch.stack(batch_singular))
        valid_frames.append(torch.stack(batch_valid))
    rotation = torch.stack(rotations)
    singular = torch.stack(singular_values)
    valid_rotation = torch.stack(valid_frames)
    # The row-vector Kabsch convention is destination = source @ rotation,
    # hence canonical coordinates are destination @ rotation.T.
    canonical = torch.einsum(
        "btij,btnj->btni", rotation, centred_target
    )
    displacement = canonical - reference_shape[:, None]
    return CanonicalTargets(
        com=com.detach(),
        rotation=rotation.detach(),
        displacement=displacement.detach(),
        valid_rotation=valid_rotation.detach(),
        singular_values=singular.detach(),
        reference_shape=reference_shape.detach(),
        radius=radius.detach(),
    )
