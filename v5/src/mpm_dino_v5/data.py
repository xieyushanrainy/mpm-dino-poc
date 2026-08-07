from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import Tensor

from mpm_dino_v4.v41_data import V41TrajectoryDataset, validate_v41_manifest
from mpm_dino_v4.v41_data import MODEL_INPUT_KEYS
from mpm_dino_v4.v42_geometry import CanonicalTargets, canonical_targets
from mpm_dino_v4.v42_stages import StageMetadata, derive_impact_stages

from .config import SPLIT_SHA256
from .provenance import validate_split_manifest


def load_manifest(path: str | Path) -> dict:
    manifest = json.loads(Path(path).read_text())
    validate_v41_manifest(manifest)
    validate_split_manifest(manifest, SPLIT_SHA256)
    return manifest


def dataset(root: str | Path, manifest: dict, split: str, *, families=None, seed: int = 42) -> V41TrajectoryDataset:
    if split == "test":
        raise ValueError("sealed test loading requires a separate explicit workflow")
    return V41TrajectoryDataset(
        root, manifest, split, dino_mode="real", seed=seed, families=families,
    )


@torch.no_grad()
def targets_and_stages(batch: dict) -> tuple[CanonicalTargets, StageMetadata]:
    targets = canonical_targets(
        batch["x1"], batch["target"], batch["input_mask"], batch["target_mask"],
    )
    stages = derive_impact_stages(
        batch["x1"], batch["target"], batch["input_mask"], batch["target_mask"],
        batch["neighbour_indices"], batch["neighbour_mask"],
        batch["rest_edge_lengths"], batch["dt"], batch["gravity"], batch["floor_z"],
    )
    return targets, stages


@torch.no_grad()
def interaction_labels(batch: dict, targets: CanonicalTargets, stages: StageMetadata, contact_radius_fraction: float = 0.01) -> tuple[Tensor, Tensor, Tensor]:
    radius = targets.radius[:, None, None]
    contact = (
        batch["target"][..., 2] - batch["floor_z"][:, None, None]
        <= contact_radius_fraction * radius
    ) & batch["target_mask"]
    frames = batch["target"].shape[1]
    saved_index = torch.arange(1, frames + 1, device=batch["target"].device)[None]
    onset = stages.contact_onset[:, None]
    peak = ((stages.peak_start + stages.peak_end) // 2)[:, None]
    duration = (peak - onset).clamp_min(1)
    event_time = (2 * (saved_index - onset) / duration - 1).clamp(-1, 1).to(batch["target"].dtype)
    event_valid = (onset >= 0) & (peak > onset)
    event_time = torch.where(event_valid, event_time, torch.zeros_like(event_time))
    return contact.detach(), event_time.detach(), event_valid.expand_as(event_time).detach()


def model_inputs(batch: dict) -> dict:
    return {key: batch[key] for key in MODEL_INPUT_KEYS}
