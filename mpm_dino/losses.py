from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F

from .grid import GridSpec, scatter_particles
from .model import SurrogateOutput


@dataclass
class Losses:
    total: Tensor
    particle: Tensor
    occupancy: Tensor
    grid_velocity: Tensor
    consistency: Tensor


def _masked_huber(pred: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    loss = F.huber_loss(pred, target, reduction="none").mean(-1)
    weight = mask.to(loss.dtype)
    return (loss * weight).sum() / weight.sum().clamp_min(1)


def _masked_smooth_l1(pred: Tensor, target: Tensor, mask: Tensor, beta: float) -> Tensor:
    """Huber-shaped loss normalized by beta so small physical errors are not numerically erased."""
    loss = F.smooth_l1_loss(pred, target, reduction="none", beta=beta).mean(-1)
    weight = mask.to(loss.dtype)
    return (loss * weight).sum() / weight.sum().clamp_min(1)


def one_step_loss(output: SurrogateOutput, batch: dict[str, Tensor], spec: GridSpec,
                  particle_beta: float = 0.01) -> Losses:
    mask = batch["target_mask"]
    particle = _masked_smooth_l1(
        output.displacement, batch["target_displacement"], mask, particle_beta
    )
    target_position = batch["positions"] + batch["target_displacement"]
    target_grid_velocity, target_occ = scatter_particles(target_position, batch["target_velocity"], mask, spec)
    active = target_occ[:, 0] > 1e-8
    occupancy = F.huber_loss(torch.log1p(output.next_occupancy), torch.log1p(target_occ))
    grid_velocity = _masked_huber(
        output.next_velocity.permute(0, 2, 3, 4, 1).reshape(output.next_velocity.shape[0], -1, 3),
        target_grid_velocity.permute(0, 2, 3, 4, 1).reshape(target_grid_velocity.shape[0], -1, 3),
        active.reshape(active.shape[0], -1),
    )
    predicted_position = batch["positions"] + output.displacement
    _, predicted_occ = scatter_particles(predicted_position, batch["target_velocity"], mask, spec)
    consistency = F.huber_loss(torch.log1p(output.next_occupancy), torch.log1p(predicted_occ))
    total = particle + 0.5 * grid_velocity + 0.25 * occupancy + 0.25 * consistency
    return Losses(total, particle, occupancy, grid_velocity, consistency)
