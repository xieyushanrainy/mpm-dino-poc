"""Construction and validation of fixed effective-reference neighbour graphs."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class GraphValidationReport:
    particle_count: int
    directed_edge_count: int
    undirected_edge_count: int
    isolated_particle_count: int
    connected_components: int
    degree_min: int
    degree_median: float
    degree_max: int
    length_min: float
    length_median: float
    length_p95: float
    length_max: float

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


def build_mutual_knn_graph(
    x0: Tensor,
    particle_mask: Tensor,
    candidate_k: int = 12,
    max_neighbours: int = 8,
    eps: float = 1e-8,
) -> dict[str, Tensor]:
    """Build a reciprocal graph from mutual kNN candidates with a degree cap.

    Candidate pairs are considered shortest-first. An undirected pair is retained
    only while both endpoints have capacity, ensuring that every stored edge has a
    reciprocal slot and no particle has more than ``max_neighbours`` edges.
    """
    if x0.ndim != 2 or x0.shape[-1] != 3:
        raise ValueError(f"x0 must have shape (N, 3), got {tuple(x0.shape)}")
    if particle_mask.shape != x0.shape[:1] or particle_mask.dtype != torch.bool:
        raise ValueError("particle_mask must be boolean with shape (N,)")
    if candidate_k < 1 or max_neighbours < 1:
        raise ValueError("candidate_k and max_neighbours must be positive")

    valid_ids = torch.nonzero(particle_mask, as_tuple=False).flatten()
    count = int(valid_ids.numel())
    if count < 2:
        raise ValueError("at least two real particles are required")
    points = x0[valid_ids].to(dtype=torch.float64, device="cpu")
    if not torch.isfinite(points).all():
        raise ValueError("real reference positions must be finite")

    distances = torch.cdist(points, points)
    distances.fill_diagonal_(float("inf"))
    k = min(candidate_k, count - 1)
    nearest = torch.topk(distances, k=k, largest=False).indices
    candidate = torch.zeros((count, count), dtype=torch.bool)
    candidate.scatter_(1, nearest, True)
    mutual = candidate & candidate.T
    pairs = torch.nonzero(torch.triu(mutual, diagonal=1), as_tuple=False)
    if pairs.numel() == 0:
        raise ValueError("mutual-kNN construction produced no edges")
    pair_lengths = distances[pairs[:, 0], pairs[:, 1]]
    order = torch.argsort(pair_lengths, stable=True)

    adjacency: list[list[int]] = [[] for _ in range(count)]
    for pair_idx in order.tolist():
        i, j = pairs[pair_idx].tolist()
        if len(adjacency[i]) < max_neighbours and len(adjacency[j]) < max_neighbours:
            adjacency[i].append(j)
            adjacency[j].append(i)

    n = x0.shape[0]
    indices = torch.full((n, max_neighbours), -1, dtype=torch.int64)
    mask = torch.zeros((n, max_neighbours), dtype=torch.bool)
    for local_i, local_neighbours in enumerate(adjacency):
        global_i = int(valid_ids[local_i])
        ordered = sorted(local_neighbours, key=lambda j: (float(distances[local_i, j]), j))
        global_neighbours = valid_ids[torch.tensor(ordered, dtype=torch.int64)]
        degree = len(ordered)
        indices[global_i, :degree] = global_neighbours
        mask[global_i, :degree] = True

    safe_indices = indices.clamp_min(0)
    vectors = x0[safe_indices] - x0[:, None, :]
    vectors = torch.where(mask[..., None], vectors, torch.zeros_like(vectors))
    lengths = torch.linalg.vector_norm(vectors, dim=-1)
    if bool((lengths[mask] <= eps).any()):
        raise ValueError("reference graph contains a zero-length edge")
    return {
        "neighbour_indices": indices,
        "neighbour_mask": mask,
        "rest_edge_vectors": vectors.to(dtype=torch.float32),
        "rest_edge_lengths": lengths.to(dtype=torch.float32),
    }


def validate_neighbour_graph(
    x0: Tensor,
    particle_mask: Tensor,
    neighbour_indices: Tensor,
    neighbour_mask: Tensor,
    rest_edge_vectors: Tensor,
    rest_edge_lengths: Tensor,
    *,
    atol: float = 1e-6,
) -> GraphValidationReport:
    """Reject malformed graphs and return geometry statistics for audit logs."""
    n = x0.shape[0]
    expected = neighbour_indices.shape
    if x0.shape != (n, 3) or particle_mask.shape != (n,):
        raise ValueError("invalid x0 or particle_mask shape")
    if neighbour_indices.ndim != 2 or neighbour_mask.shape != expected:
        raise ValueError("neighbour indices and mask must have matching (N, K) shapes")
    if rest_edge_vectors.shape != (*expected, 3) or rest_edge_lengths.shape != expected:
        raise ValueError("rest edge tensor shapes do not match adjacency")
    if neighbour_indices.dtype != torch.int64 or neighbour_mask.dtype != torch.bool:
        raise ValueError("neighbour_indices must be int64 and neighbour_mask boolean")
    if bool(neighbour_mask[~particle_mask].any()):
        raise ValueError("padded particles must not have neighbours")
    if bool((neighbour_indices[~neighbour_mask] != -1).any()):
        raise ValueError("invalid neighbour slots must use index -1")

    rows = torch.arange(n)[:, None].expand_as(neighbour_indices)[neighbour_mask]
    cols = neighbour_indices[neighbour_mask]
    if cols.numel() == 0:
        raise ValueError("graph has no valid edges")
    if bool(((cols < 0) | (cols >= n)).any()):
        raise ValueError("neighbour index is out of bounds")
    if bool((~particle_mask[cols]).any()):
        raise ValueError("an edge targets a padded particle")
    if bool((rows == cols).any()):
        raise ValueError("self edges are not allowed")
    edge_set = set(zip(rows.tolist(), cols.tolist()))
    if any((j, i) not in edge_set for i, j in edge_set):
        raise ValueError("every edge must have a reciprocal edge")
    if len(edge_set) != int(rows.numel()):
        raise ValueError("duplicate neighbours are not allowed")

    expected_vectors = x0[cols] - x0[rows]
    stored_vectors = rest_edge_vectors[neighbour_mask]
    stored_lengths = rest_edge_lengths[neighbour_mask]
    if not torch.allclose(stored_vectors, expected_vectors, atol=atol, rtol=0):
        raise ValueError("stored rest vectors do not match x0")
    if not torch.allclose(stored_lengths, torch.linalg.vector_norm(expected_vectors, dim=-1), atol=atol, rtol=0):
        raise ValueError("stored rest lengths do not match x0")
    if not torch.isfinite(stored_lengths).all() or bool((stored_lengths <= 0).any()):
        raise ValueError("rest lengths must be finite and positive")

    valid_ids = torch.nonzero(particle_mask, as_tuple=False).flatten().tolist()
    adjacency = {i: set() for i in valid_ids}
    for i, j in edge_set:
        adjacency[i].add(j)
    seen: set[int] = set()
    components = 0
    for root in valid_ids:
        if root in seen:
            continue
        components += 1
        stack = [root]
        seen.add(root)
        while stack:
            for nxt in adjacency[stack.pop()]:
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)

    degrees = neighbour_mask[particle_mask].sum(dim=1)
    lengths = stored_lengths.float()
    return GraphValidationReport(
        particle_count=len(valid_ids),
        directed_edge_count=len(edge_set),
        undirected_edge_count=len(edge_set) // 2,
        isolated_particle_count=int((degrees == 0).sum()),
        connected_components=components,
        degree_min=int(degrees.min()),
        degree_median=float(degrees.float().median()),
        degree_max=int(degrees.max()),
        length_min=float(lengths.min()),
        length_median=float(lengths.median()),
        length_p95=float(torch.quantile(lengths, 0.95)),
        length_max=float(lengths.max()),
    )
