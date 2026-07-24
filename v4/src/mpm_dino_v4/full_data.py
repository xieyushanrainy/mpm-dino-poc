from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset

from .data import donor_map


class FullTrajectoryDataset(Dataset):
    """One X[0], X[1] -> X[2:61] example per object UID."""

    def __init__(
        self,
        cache: str | Path,
        manifest: dict,
        split: str,
        families=("rigid", "soft_body"),
        dino_mode: str = "real",
        seed: int = 42,
    ):
        if dino_mode not in {"real", "zero", "scene_shuffled"}:
            raise ValueError(f"unsupported full-trajectory DINO mode: {dino_mode}")
        self.cache = Path(cache)
        self.split = split
        self.dino_mode = dino_mode
        self.seed = seed
        self.uids = [
            uid for uid in manifest["splits"][split]
            if manifest["strata"][uid]["family"] in families
        ]
        self.donors = donor_map(self.uids, seed)
        self._scenes: dict[str, dict] = {}

    def _load(self, uid: str) -> dict:
        if uid not in self._scenes:
            self._scenes[uid] = torch.load(
                self.cache / f"{uid}.pt", map_location="cpu", weights_only=False
            )
        return self._scenes[uid]

    def __len__(self) -> int:
        return len(self.uids)

    def __getitem__(self, item: int) -> dict:
        uid = self.uids[item]
        scene = self._load(uid)
        dino_scene = self._load(self.donors[uid]) if self.dino_mode == "scene_shuffled" else scene
        dino = dino_scene["dino"]
        dino_valid = dino_scene["dino_valid"]
        if self.dino_mode == "zero":
            dino = torch.zeros_like(dino)
        input_mask = scene["active"][0] & scene["active"][1]
        target_mask = scene["active"][2:] & input_mask[None]
        return {
            "uid": uid,
            "family": scene["family"],
            "x0": scene["positions"][0],
            "x1": scene["positions"][1],
            "target": scene["positions"][2:],
            "input_mask": input_mask,
            "target_mask": target_mask,
            "dino": dino,
            "dino_valid": dino_valid,
            "dt": torch.tensor(scene["dt"]),
            "gravity": scene["gravity"],
            "floor_z": torch.tensor(scene["floor_z"]),
            **{
                key: scene[key]
                for key in (
                    "neighbour_indices",
                    "neighbour_mask",
                    "rest_edge_vectors",
                    "rest_edge_lengths",
                )
            },
        }


def ballistic_trajectory(x0: torch.Tensor, x1: torch.Tensor, gravity: torch.Tensor, dt: torch.Tensor, frames: int = 59) -> torch.Tensor:
    """Discrete constant-acceleration extrapolation for X[2:2+frames]."""
    h = torch.arange(1, frames + 1, device=x0.device, dtype=x0.dtype)
    view = (1, frames, 1, 1)
    h = h.view(view)
    step = (x1 - x0)[:, None]
    acceleration = gravity[:, None, None] * dt[:, None, None, None].square()
    return x1[:, None] + h * step + 0.5 * h * (h + 1.0) * acceleration
