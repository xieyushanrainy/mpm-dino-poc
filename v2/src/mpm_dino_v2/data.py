"""Teacher-forced and recurrent datasets for versioned V2 caches."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset

from .cache import load_v2_cache


STATIC_KEYS = [
    "dino", "dino_imputed", "x0", "particle_mask", "neighbour_indices",
    "neighbour_mask", "rest_edge_vectors", "rest_edge_lengths", "scale", "dt",
]


class ScenePairDataset(Dataset):
    def __init__(self, paths: list[str | Path]):
        self.scenes = [load_v2_cache(path) for path in paths]
        self.index = [(scene_id, t) for scene_id, scene in enumerate(self.scenes)
                      for t in range(1, scene["points"].shape[0] - 1)]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, item):
        scene_id, t = self.index[item]
        scene = self.scenes[scene_id]
        dt = scene["dt"]
        mask = scene["particle_mask"] & scene["visible"][t] & scene["motion_valid"][t]
        target_mask = mask & scene["visible"][t + 1] & scene["motion_valid"][t + 1]
        displacement = scene["points"][t + 1] - scene["points"][t]
        result = {
            "positions": scene["points"][t],
            "velocities": (scene["points"][t] - scene["points"][t - 1]) / dt,
            "particle_mask": mask,
            "target_mask": target_mask,
            "controller_positions": scene["controller"][t],
            "controller_velocity": (scene["controller"][t + 1] - scene["controller"][t]) / dt,
            "controller_mask": torch.ones(scene["controller"].shape[1], dtype=torch.bool),
            "target_displacement": displacement,
            "target_velocity": displacement / dt,
        }
        result.update({key: scene[key] for key in STATIC_KEYS})
        return result


class SceneSequenceDataset(Dataset):
    def __init__(self, paths: list[str | Path], steps: int):
        if steps < 2:
            raise ValueError("rollout steps must be at least 2")
        self.steps = steps
        self.scenes = [load_v2_cache(path) for path in paths]
        self.index = [(scene_id, t) for scene_id, scene in enumerate(self.scenes)
                      for t in range(1, scene["points"].shape[0] - steps)]
        self.motion_scores = []
        for scene_id, t in self.index:
            scene = self.scenes[scene_id]
            valid = scene["particle_mask"] & scene["visible"][t:t + steps + 1] & scene["motion_valid"][t:t + steps + 1]
            displacement = scene["points"][t + 1:t + steps + 1] - scene["points"][t:t + steps]
            step_valid = valid[1:] & valid[:-1]
            magnitude = torch.linalg.vector_norm(displacement, dim=-1)
            self.motion_scores.append(float((magnitude * step_valid).sum() / step_valid.sum().clamp_min(1)))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, item):
        scene_id, t = self.index[item]
        scene = self.scenes[scene_id]
        end = t + self.steps + 1
        result = {
            "previous_points": scene["points"][t - 1],
            "points": scene["points"][t:end],
            "visible": scene["visible"][t:end],
            "motion_valid": scene["motion_valid"][t:end],
            "controller": scene["controller"][t:end],
        }
        result.update({key: scene[key] for key in STATIC_KEYS})
        return result
