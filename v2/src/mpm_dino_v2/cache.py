"""Versioned V2 scene-cache schema and V1-to-V2 conversion."""

from __future__ import annotations

from pathlib import Path

import torch

from .graph import GraphValidationReport, build_mutual_knn_graph, validate_neighbour_graph

SCHEMA_NAME = "mpm_dino_v2_scene"
SCHEMA_VERSION = 1
GRAPH_METHOD = "mutual_knn_greedy_degree_cap"

V1_KEYS = {
    "points", "visible", "motion_valid", "controller", "dino", "dino_imputed",
    "scale", "dt", "source_indices",
}
V2_KEYS = V1_KEYS | {"actual_points",
    "schema_name", "schema_version", "x0", "particle_mask", "neighbour_indices",
    "neighbour_mask", "rest_edge_vectors", "rest_edge_lengths", "graph_candidate_k",
    "graph_max_neighbours", "graph_method",
}


def validate_v2_cache(payload: dict) -> GraphValidationReport:
    missing = V2_KEYS - payload.keys()
    if missing:
        raise ValueError(f"V2 cache is missing keys: {sorted(missing)}")
    if payload["schema_name"] != SCHEMA_NAME or int(payload["schema_version"]) != SCHEMA_VERSION:
        raise ValueError("unsupported V2 cache schema")
    points = payload["points"]
    if points.ndim != 3 or points.shape[-1] != 3:
        raise ValueError("points must have shape (T, N, 3)")
    if payload["x0"].shape != points.shape[1:] or not torch.equal(payload["x0"], points[0]):
        raise ValueError("x0 must be an exact persistent copy of normalized frame 0")
    expected_particle_mask = payload["source_indices"] >= 0
    if not torch.equal(payload["particle_mask"], expected_particle_mask):
        raise ValueError("particle_mask must identify exactly the non-padded source indices")
    if int(payload["actual_points"]) != int(expected_particle_mask.sum()):
        raise ValueError("actual_points disagrees with particle_mask")
    return validate_neighbour_graph(
        payload["x0"], payload["particle_mask"], payload["neighbour_indices"],
        payload["neighbour_mask"], payload["rest_edge_vectors"], payload["rest_edge_lengths"],
    )


def upgrade_v1_payload(v1: dict, candidate_k: int = 12, max_neighbours: int = 8) -> tuple[dict, GraphValidationReport]:
    missing = V1_KEYS - v1.keys()
    if missing:
        raise ValueError(f"V1 cache is missing keys: {sorted(missing)}")
    payload = {key: value.clone() if isinstance(value, torch.Tensor) else value for key, value in v1.items()}
    particle_mask = payload["source_indices"] >= 0
    payload["actual_points"] = torch.tensor(int(particle_mask.sum()), dtype=torch.int64)
    x0 = payload["points"][0].clone()
    graph = build_mutual_knn_graph(x0, particle_mask, candidate_k, max_neighbours)
    payload.update({
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "x0": x0,
        "particle_mask": particle_mask,
        "graph_candidate_k": candidate_k,
        "graph_max_neighbours": max_neighbours,
        "graph_method": GRAPH_METHOD,
        **graph,
    })
    return payload, validate_v2_cache(payload)


def convert_v1_cache(source: str | Path, destination: str | Path, candidate_k: int = 12, max_neighbours: int = 8) -> GraphValidationReport:
    source, destination = Path(source), Path(destination)
    if source.resolve() == destination.resolve():
        raise ValueError("V2 destination must differ from the frozen V1 source")
    payload, report = upgrade_v1_payload(
        torch.load(source, map_location="cpu", weights_only=True), candidate_k, max_neighbours,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, destination)
    return report


def load_v2_cache(path: str | Path) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    validate_v2_cache(payload)
    return payload
