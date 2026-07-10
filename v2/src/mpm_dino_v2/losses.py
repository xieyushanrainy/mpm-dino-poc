from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F

from .deformation import edge_validity, gather_neighbours
from .grid import GridSpec, scatter_particles
from .model import SurrogateOutput


@dataclass
class Losses:
    total: Tensor
    particle: Tensor
    occupancy: Tensor
    grid_velocity: Tensor
    consistency: Tensor
    edge_vector: Tensor
    edge_length: Tensor


def _masked_vector_loss(pred: Tensor, target: Tensor, mask: Tensor, beta: float = 1.0) -> Tensor:
    loss = F.smooth_l1_loss(pred, target, reduction="none", beta=beta).mean(-1)
    weight = mask.to(loss.dtype)
    return (loss * weight).sum() / weight.sum().clamp_min(1)


def edge_deformation_losses(
    predicted_positions: Tensor,
    target_positions: Tensor,
    target_mask: Tensor,
    neighbour_indices: Tensor,
    neighbour_mask: Tensor,
    beta: float = 0.01,
) -> tuple[Tensor, Tensor]:
    valid = edge_validity(target_mask, neighbour_indices, neighbour_mask)
    predicted_vectors = gather_neighbours(predicted_positions, neighbour_indices) - predicted_positions[:, :, None]
    target_vectors = gather_neighbours(target_positions, neighbour_indices) - target_positions[:, :, None]
    vector = _masked_vector_loss(predicted_vectors, target_vectors, valid, beta)
    predicted_lengths = torch.linalg.vector_norm(predicted_vectors, dim=-1)
    target_lengths = torch.linalg.vector_norm(target_vectors, dim=-1)
    raw = F.smooth_l1_loss(predicted_lengths, target_lengths, reduction="none", beta=beta)
    weight = valid.to(raw.dtype)
    length = (raw * weight).sum() / weight.sum().clamp_min(1)
    return vector, length


def one_step_loss(
    output: SurrogateOutput, batch: dict[str, Tensor], spec: GridSpec,
    particle_beta: float = 0.01, edge_vector_weight: float = 0.25,
    edge_length_weight: float = 0.10,
) -> Losses:
    mask = batch["target_mask"]
    particle = _masked_vector_loss(output.displacement, batch["target_displacement"], mask, particle_beta)
    target_position = batch["positions"] + batch["target_displacement"]
    predicted_position = batch["positions"] + output.displacement
    if output.next_occupancy is None or output.next_velocity is None:
        occupancy = grid_velocity = consistency = particle.new_zeros(())
    else:
        target_grid_velocity, target_occupancy = scatter_particles(target_position, batch["target_velocity"], mask, spec)
        active = target_occupancy[:, 0] > 1e-8
        occupancy = F.huber_loss(torch.log1p(output.next_occupancy), torch.log1p(target_occupancy))
        grid_velocity = _masked_vector_loss(
            output.next_velocity.permute(0, 2, 3, 4, 1).reshape(output.next_velocity.shape[0], -1, 3),
            target_grid_velocity.permute(0, 2, 3, 4, 1).reshape(target_grid_velocity.shape[0], -1, 3),
            active.reshape(active.shape[0], -1), 1.0)
        _, predicted_occupancy = scatter_particles(predicted_position, batch["target_velocity"], mask, spec)
        consistency = F.huber_loss(torch.log1p(output.next_occupancy), torch.log1p(predicted_occupancy))
    edge_vector, edge_length = edge_deformation_losses(
        predicted_position, target_position, mask, batch["neighbour_indices"], batch["neighbour_mask"], particle_beta,
    )
    total = (particle + 0.5 * grid_velocity + 0.25 * occupancy + 0.25 * consistency
             + edge_vector_weight * edge_vector + edge_length_weight * edge_length)
    return Losses(total, particle, occupancy, grid_velocity, consistency, edge_vector, edge_length)
