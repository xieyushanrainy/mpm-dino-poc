from __future__ import annotations

import torch

from mpm_dino_v2.cache import SCHEMA_NAME, SCHEMA_VERSION, upgrade_v1_payload, validate_v2_cache


def make_v1_payload() -> dict:
    points = torch.rand((4, 12, 3), generator=torch.Generator().manual_seed(3))
    source_indices = torch.cat((torch.arange(10), torch.full((2,), -1)))
    points[:, 10:] = 0
    return {
        "points": points,
        "visible": torch.ones((4, 12), dtype=torch.bool),
        "motion_valid": torch.ones((4, 12), dtype=torch.bool),
        "controller": torch.zeros((4, 2, 3)),
        "dino": torch.zeros((12, 5)),
        "dino_imputed": torch.zeros(12, dtype=torch.bool),
        "scale": torch.tensor(1.0),
        "dt": torch.tensor(1 / 30),
        "source_indices": source_indices,
        "actual_points": torch.tensor(10),
    }


def test_upgrade_adds_versioned_reference_graph_without_mutating_v1() -> None:
    v1 = make_v1_payload()
    original = v1["points"].clone()
    payload, report = upgrade_v1_payload(v1, candidate_k=4, max_neighbours=3)

    assert payload["schema_name"] == SCHEMA_NAME
    assert payload["schema_version"] == SCHEMA_VERSION
    assert torch.equal(payload["x0"], payload["points"][0])
    assert torch.equal(v1["points"], original)
    assert report.degree_max <= 3
    validate_v2_cache(payload)
