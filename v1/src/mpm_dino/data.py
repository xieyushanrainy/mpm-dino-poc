from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset


def deterministic_indices(count: int, maximum: int, seed: int) -> np.ndarray:
    if count <= maximum:
        return np.arange(count)
    return np.sort(np.random.default_rng(seed).choice(count, maximum, replace=False))


def prepare_scene(final_data: str | Path, dino_features: str | Path, output: str | Path,
                  maximum_points: int = 2048, seed: int = 42, margin_factor: float = 2.0) -> None:
    """Convert one PhysTwin scene and precomputed per-track DINO features to a compact cache.

    `dino_features` must be `(N, D)` in the same track order as `final_data.pkl`.
    Visibility projection/imputation is deliberately kept in the DINO extraction stage so
    this function cannot silently attach a feature to the wrong particle.
    """
    with open(final_data, "rb") as handle:
        raw = pickle.load(handle)
    points = np.asarray(raw["object_points"], dtype=np.float32)
    visible = np.asarray(raw["object_visibilities"], dtype=bool)
    motion_valid = np.asarray(raw["object_motions_valid"], dtype=bool)
    controller = np.asarray(raw["controller_points"], dtype=np.float32)
    dino_file = np.load(dino_features)
    if isinstance(dino_file, np.lib.npyio.NpzFile):
        dino = dino_file["features"].astype(np.float32)
        dino_imputed = dino_file["imputed"].astype(bool)
    else:
        dino = dino_file.astype(np.float32)
        dino_imputed = np.zeros(dino.shape[0], dtype=bool)
    if dino.shape[0] != points.shape[1]:
        raise ValueError(f"DINO has {dino.shape[0]} tracks, expected {points.shape[1]}")
    idx = deterministic_indices(points.shape[1], maximum_points, seed)
    points, visible, motion_valid, dino = points[:, idx], visible[:, idx], motion_valid[:, idx], dino[idx]
    dino_imputed = dino_imputed[idx]

    initial = points[:2].reshape(-1, 3)
    center = (initial.min(0) + initial.max(0)) / 2
    side = float(np.max(initial.max(0) - initial.min(0)) * margin_factor)
    if not np.isfinite(side) or side <= 0:
        raise ValueError("invalid initial object extent")
    points_n = (points - center) / (side / 2)
    controller_n = (controller - center) / (side / 2)
    actual_points = points_n.shape[1]
    if actual_points < maximum_points:
        pad = maximum_points - actual_points
        points_n = np.pad(points_n, ((0, 0), (0, pad), (0, 0)))
        visible = np.pad(visible, ((0, 0), (0, pad)), constant_values=False)
        motion_valid = np.pad(motion_valid, ((0, 0), (0, pad)), constant_values=False)
        dino = np.pad(dino, ((0, pad), (0, 0)))
        dino_imputed = np.pad(dino_imputed, (0, pad), constant_values=False)
        idx = np.pad(idx, (0, pad), constant_values=-1)
    fps = 30.0
    metadata = Path(final_data).with_name("metadata.json")
    if metadata.exists():
        import json
        fps = float(json.loads(metadata.read_text())["fps"])
    payload = {
        "points": torch.from_numpy(points_n),
        "visible": torch.from_numpy(visible),
        "motion_valid": torch.from_numpy(motion_valid),
        "controller": torch.from_numpy(controller_n),
        "dino": torch.from_numpy(dino),
        "dino_imputed": torch.from_numpy(dino_imputed),
        "scale": torch.tensor(side / 2, dtype=torch.float32),
        "dt": torch.tensor(1.0 / fps, dtype=torch.float32),
        "source_indices": torch.from_numpy(idx),
        "actual_points": torch.tensor(actual_points, dtype=torch.int64),
    }
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)


class ScenePairDataset(Dataset):
    """Teacher-forced one-frame pairs from prepared scene caches."""

    def __init__(self, paths: list[str | Path]):
        self.scenes = [torch.load(path, map_location="cpu", weights_only=True) for path in paths]
        self.index = [(scene_id, t) for scene_id, s in enumerate(self.scenes) for t in range(1, s["points"].shape[0] - 1)]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, item):
        scene_id, t = self.index[item]
        s = self.scenes[scene_id]
        dt = s["dt"]
        velocity = (s["points"][t] - s["points"][t - 1]) / dt
        controller_velocity = (s["controller"][t + 1] - s["controller"][t]) / dt
        target_displacement = s["points"][t + 1] - s["points"][t]
        target_velocity = target_displacement / dt
        mask = s["visible"][t] & s["motion_valid"][t]
        target_mask = mask & s["visible"][t + 1] & s["motion_valid"][t + 1]
        return {
            "positions": s["points"][t], "velocities": velocity, "dino": s["dino"],
            "particle_mask": mask, "target_mask": target_mask, "dino_imputed": s["dino_imputed"],
            "controller_positions": s["controller"][t], "controller_velocity": controller_velocity,
            "controller_mask": torch.ones(s["controller"].shape[1], dtype=torch.bool),
            "scale": s["scale"], "dt": dt, "target_displacement": target_displacement,
            "target_velocity": target_velocity,
        }


class SceneSequenceDataset(Dataset):
    """Contiguous windows with one previous frame for recurrent initialization."""

    def __init__(self, paths: list[str | Path], steps: int):
        if steps < 2:
            raise ValueError("rollout steps must be at least 2")
        self.steps = steps
        self.scenes = [torch.load(path, map_location="cpu", weights_only=True) for path in paths]
        self.index = [
            (scene_id, t)
            for scene_id, scene in enumerate(self.scenes)
            for t in range(1, scene["points"].shape[0] - steps)
        ]
        self.motion_scores = []
        for scene_id, t in self.index:
            scene = self.scenes[scene_id]
            valid = scene["visible"][t:t + steps + 1] & scene["motion_valid"][t:t + steps + 1]
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
        return {
            "previous_points": scene["points"][t - 1],
            "points": scene["points"][t:end],
            "visible": scene["visible"][t:end],
            "motion_valid": scene["motion_valid"][t:end],
            "controller": scene["controller"][t:end],
            "dino": scene["dino"], "dino_imputed": scene["dino_imputed"],
            "scale": scene["scale"], "dt": scene["dt"],
        }
