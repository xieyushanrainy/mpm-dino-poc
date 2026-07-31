from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import torch
from torch import Tensor

from .model import masked_mean
from .v42_geometry import canonical_targets


class ImpactStage(IntEnum):
    UNLABELLED = 0
    FREE_FLIGHT = 1
    CONTACT_ONSET = 2
    COMPRESSION = 3
    PEAK_DEFORMATION = 4
    RECOVERY = 5
    POST_PRIMARY_EVENT = 6


STAGE_RAW_WEIGHTS = {
    ImpactStage.UNLABELLED: 1.0,
    ImpactStage.FREE_FLIGHT: 1.0,
    ImpactStage.CONTACT_ONSET: 2.0,
    ImpactStage.COMPRESSION: 3.0,
    ImpactStage.PEAK_DEFORMATION: 4.0,
    ImpactStage.RECOVERY: 3.0,
    ImpactStage.POST_PRIMARY_EVENT: 1.0,
}


def total_mass_stage_weights(labels: Tensor) -> Tensor:
    """Give each present stage total mass proportional to its importance.

    Within one episode, every frame in stage ``s`` receives ``alpha_s / n_s``
    before the complete trajectory is normalized to mean one. Unlike the
    historical per-frame weighting, a long stage cannot dominate merely by
    containing more saved frames.
    """
    if labels.ndim != 2:
        raise ValueError("stage labels must have shape [batch, frames]")
    weights = torch.zeros_like(labels, dtype=torch.float32)
    for stage, importance in STAGE_RAW_WEIGHTS.items():
        selected = labels.eq(int(stage))
        counts = selected.sum(1, keepdim=True)
        per_frame = float(importance) / counts.clamp_min(1).to(weights.dtype)
        weights = weights + selected.to(weights.dtype) * per_frame
    return weights / weights.mean(1, keepdim=True).clamp_min(1e-8)


@dataclass
class StageMetadata:
    labels: Tensor
    weights: Tensor
    contact_onset: Tensor
    peak_start: Tensor
    peak_end: Tensor
    recovery_end: Tensor
    floor_gap: Tensor
    excess_acceleration: Tensor
    deformation: Tensor


def _smooth_three(values: Tensor) -> Tensor:
    padded = torch.nn.functional.pad(values[:, None], (1, 1), mode="replicate")
    return torch.nn.functional.avg_pool1d(
        padded, kernel_size=3, stride=1
    ).squeeze(1)


def lowest_active_mean_gap(
    positions: Tensor,
    active: Tensor,
    floor_z: Tensor,
    tail_points: int = 4,
) -> Tensor:
    """Mean floor gap of the lowest fixed-count active surface points.

    A fixed count adapts to sparse contact patches better than a percentage
    quantile while remaining substantially more robust than a single minimum.
    """
    gaps = (
        positions[..., 2] - floor_z[:, None, None]
    ).masked_fill(~active, torch.inf)
    sorted_gaps = gaps.sort(dim=2).values
    available = active.sum(2).clamp(min=1, max=tail_points)
    ranks = torch.arange(
        tail_points, device=positions.device
    )[None, None]
    selected = ranks < available[:, :, None]
    values = sorted_gaps[:, :, :tail_points].masked_fill(~selected, 0)
    return values.sum(2) / available


@torch.no_grad()
def derive_impact_stages(
    x1: Tensor,
    target: Tensor,
    input_mask: Tensor,
    target_mask: Tensor,
    neighbour_indices: Tensor,
    neighbour_mask: Tensor,
    rest_edge_lengths: Tensor,
    dt: Tensor,
    gravity: Tensor,
    floor_z: Tensor,
) -> StageMetadata:
    """Derive first-impact metadata. This output must never enter the model."""
    from mpm_dino_v2.deformation import edge_validity, gather_neighbours

    all_positions = torch.cat((x1[:, None], target), dim=1)
    all_masks = torch.cat((input_mask[:, None], target_mask), dim=1)
    batch, frames, points, _ = all_positions.shape
    com = masked_mean(all_positions, all_masks, dim=2)
    floor_gap = lowest_active_mean_gap(
        all_positions, all_masks, floor_z, tail_points=4,
    )

    raw_acceleration = torch.zeros(
        batch, frames, device=target.device, dtype=target.dtype,
    )
    if frames >= 3:
        second = (
            com[:, 2:, 2] - 2 * com[:, 1:-1, 2] + com[:, :-2, 2]
        ) / dt[:, None].square()
        raw_acceleration[:, 1:-1] = (
            second - gravity[:, None, 2]
        ).abs()
    acceleration = _smooth_three(raw_acceleration)

    targets = canonical_targets(
        x1, all_positions, input_mask, all_masks
    )
    kabsch_nrmse = torch.sqrt(
        (
            targets.displacement.square().sum(-1)
            * all_masks.to(target.dtype)
        ).sum(2) / all_masks.sum(2).clamp_min(1)
    ) / targets.radius[:, None]

    flat = all_positions.reshape(batch * frames, points, 3)
    indices = neighbour_indices[:, None].expand(
        -1, frames, -1, -1
    ).reshape(batch * frames, points, -1)
    masks = all_masks.reshape(batch * frames, points)
    graph_mask = neighbour_mask[:, None].expand(
        -1, frames, -1, -1
    ).reshape(batch * frames, points, -1)
    valid_edges = edge_validity(masks, indices, graph_mask)
    vectors = gather_neighbours(flat, indices) - flat[:, :, None]
    lengths = torch.linalg.vector_norm(vectors, dim=-1)
    rest = rest_edge_lengths[:, None].expand(
        -1, frames, -1, -1
    ).reshape_as(lengths).clamp_min(1e-8)
    strain = (lengths - rest) / rest
    strain_rms = torch.sqrt(
        (strain.square() * valid_edges).sum((1, 2))
        / valid_edges.sum((1, 2)).clamp_min(1)
    ).reshape(batch, frames)
    deformation = _smooth_three(
        torch.sqrt(kabsch_nrmse.square() + strain_rms.square())
    )

    labels = torch.full(
        (batch, frames), int(ImpactStage.UNLABELLED),
        device=target.device, dtype=torch.long,
    )
    onset = torch.full((batch,), -1, device=target.device, dtype=torch.long)
    peak_start = onset.clone()
    peak_end = onset.clone()
    recovery_end = onset.clone()
    for b in range(batch):
        # The centred second difference identifies the transition interval.
        # The next saved frame is the first visible-contact frame.
        adaptive_gap = max(0.010, 0.25 * float(floor_gap[b, 0]))
        candidates = torch.where(
            (floor_gap[b] <= adaptive_gap)
            & (
                raw_acceleration[b]
                >= 0.2 * torch.linalg.vector_norm(gravity[b])
            )
        )[0]
        if not len(candidates):
            continue
        contact = min(int(candidates[0]) + 1, frames - 1)
        onset[b] = contact
        labels[b, :max(contact - 1, 0)] = int(ImpactStage.FREE_FLIGHT)
        labels[b, max(contact - 1, 0):min(contact + 2, frames)] = int(
            ImpactStage.CONTACT_ONSET
        )
        event = deformation[b, contact:]
        maximum = event.max()
        plateau = torch.where(event >= 0.95 * maximum)[0] + contact
        first, last = int(plateau[0]), int(plateau[-1])
        peak_start[b], peak_end[b] = first, last
        labels[b, min(contact + 2, frames):first] = int(ImpactStage.COMPRESSION)
        labels[b, first:last + 1] = int(ImpactStage.PEAK_DEFORMATION)
        pre = deformation[b, max(contact - 1, 0)]
        threshold = pre + 0.2 * (maximum - pre)
        end = frames - 1
        for frame in range(last + 1, frames):
            if deformation[b, frame] <= threshold:
                end = frame
                break
        recovery_end[b] = end
        labels[b, last + 1:end + 1] = int(ImpactStage.RECOVERY)
        labels[b, end + 1:] = int(ImpactStage.POST_PRIMARY_EVENT)

    raw = torch.ones_like(deformation)
    for stage, weight in STAGE_RAW_WEIGHTS.items():
        raw = torch.where(labels == int(stage), raw.new_tensor(weight), raw)
    weights = raw / raw.mean(1, keepdim=True).clamp_min(1e-8)
    weights = weights.clamp_max(4.0)
    return StageMetadata(
        labels=labels, weights=weights, contact_onset=onset,
        peak_start=peak_start, peak_end=peak_end, recovery_end=recovery_end,
        floor_gap=floor_gap, excess_acceleration=acceleration,
        deformation=deformation,
    )
