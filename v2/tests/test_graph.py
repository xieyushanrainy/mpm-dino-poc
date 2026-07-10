from __future__ import annotations

import pytest
import torch

from mpm_dino_v2.graph import build_mutual_knn_graph, validate_neighbour_graph


def test_graph_is_reciprocal_degree_capped_and_excludes_padding() -> None:
    generator = torch.Generator().manual_seed(7)
    x0 = torch.rand((24, 3), generator=generator)
    particle_mask = torch.zeros(24, dtype=torch.bool)
    particle_mask[:20] = True
    graph = build_mutual_knn_graph(x0, particle_mask, candidate_k=6, max_neighbours=4)
    report = validate_neighbour_graph(x0, particle_mask, **graph)

    assert report.particle_count == 20
    assert report.degree_max <= 4
    assert report.directed_edge_count == 2 * report.undirected_edge_count
    assert not graph["neighbour_mask"][20:].any()
    assert (graph["neighbour_indices"][~graph["neighbour_mask"]] == -1).all()


def test_validator_rejects_nonreciprocal_edge() -> None:
    x0 = torch.tensor([[0.0, 0, 0], [1.0, 0, 0], [2.0, 0, 0]])
    particle_mask = torch.ones(3, dtype=torch.bool)
    graph = build_mutual_knn_graph(x0, particle_mask, candidate_k=2, max_neighbours=2)
    graph["neighbour_mask"][1, 0] = False
    graph["neighbour_indices"][1, 0] = -1
    with pytest.raises(ValueError, match="reciprocal"):
        validate_neighbour_graph(x0, particle_mask, **graph)


def test_duplicate_reference_points_are_rejected() -> None:
    x0 = torch.tensor([[0.0, 0, 0], [0.0, 0, 0], [1.0, 0, 0]])
    with pytest.raises(ValueError, match="zero-length"):
        build_mutual_knn_graph(x0, torch.ones(3, dtype=torch.bool), candidate_k=2)
