from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class GridSpec:
    resolution: int = 32
    minimum: float = -1.0
    maximum: float = 1.0


def _corners(points: Tensor, spec: GridSpec):
    r = spec.resolution
    q = ((points - spec.minimum) / (spec.maximum - spec.minimum) * (r - 1)).clamp(0, r - 1 - 1e-6)
    lo, frac = q.floor().long(), q - q.floor()
    offsets = torch.tensor(
        [[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1],
         [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]], device=points.device,
    )
    ijk = (lo[..., None, :] + offsets).clamp(0, r - 1)
    f = frac[..., None, :]
    weights = torch.where(offsets.view(1, 1, 8, 3).bool(), f, 1 - f).prod(-1)
    flat = ijk[..., 0] * r * r + ijk[..., 1] * r + ijk[..., 2]
    return flat, weights


def scatter_particles(points: Tensor, features: Tensor, mask: Tensor, spec: GridSpec):
    if points.ndim != 3 or features.ndim != 3:
        raise ValueError("points and features must be (B,N,D)")
    b, n, _ = points.shape
    if features.shape[:2] != (b, n) or mask.shape != (b, n):
        raise ValueError("particle dimensions do not match")
    flat, weights = _corners(points, spec)
    weights = weights * mask.to(weights.dtype)[..., None]
    cells = spec.resolution ** 3
    occupancy = points.new_zeros((b, 1, cells))
    sums = points.new_zeros((b, features.shape[-1], cells))
    index = flat.reshape(b, 1, -1)
    occupancy.scatter_add_(2, index, weights.reshape(b, 1, -1))
    weighted = (features[..., None, :] * weights[..., None]).permute(0, 3, 1, 2)
    sums.scatter_add_(2, index.expand(-1, features.shape[-1], -1), weighted.reshape(b, features.shape[-1], -1))
    means = sums / occupancy.clamp_min(1e-8)
    shape = (b, -1, spec.resolution, spec.resolution, spec.resolution)
    return means.reshape(shape), occupancy.reshape(b, 1, *([spec.resolution] * 3))


def gather_grid(grid: Tensor, points: Tensor, spec: GridSpec) -> Tensor:
    b, channels = grid.shape[:2]
    flat, weights = _corners(points, spec)
    values = grid.reshape(b, channels, -1).gather(2, flat.reshape(b, 1, -1).expand(-1, channels, -1))
    values = values.reshape(b, channels, points.shape[1], 8)
    return (values * weights[:, None]).sum(-1).transpose(1, 2)
