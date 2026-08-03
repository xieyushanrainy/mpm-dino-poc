from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from torch import Tensor, nn


ARMS = ("zero_memory", "geometry", "aligned_dino", "scene_shuffled", "point_shuffled")


def protected_snapshot(module: nn.Module) -> dict[str, Tensor]:
    """Bitwise snapshot used around rotation-reader optimization."""
    return {name: value.detach().cpu().clone() for name, value in module.state_dict().items()}


def assert_protected_identity(module: nn.Module, before: dict[str, Tensor]) -> None:
    after = module.state_dict()
    if set(after) != set(before) or any(
        not torch.equal(after[name].detach().cpu(), value) for name, value in before.items()
    ):
        raise RuntimeError("protected COM/deformation state changed")


def so3_exp(vector: Tensor) -> Tensor:
    """Differentiable Rodrigues exponential for row-vector rotation matrices."""
    angle = torch.linalg.vector_norm(vector, dim=-1, keepdim=True)
    axis = vector / angle.clamp_min(1e-8)
    x, y, z = axis.unbind(-1)
    zero = torch.zeros_like(x)
    skew = torch.stack((zero, -z, y, z, zero, -x, -y, x, zero), -1).reshape(*vector.shape[:-1], 3, 3)
    eye = torch.eye(3, dtype=vector.dtype, device=vector.device).expand_as(skew)
    a = torch.where(angle > 1e-6, torch.sin(angle) / angle, 1 - angle.square() / 6)
    b = torch.where(angle > 1e-6, (1 - torch.cos(angle)) / angle.square(), .5 - angle.square() / 24)
    raw = torch.stack((zero, -vector[..., 2], vector[..., 1], vector[..., 2], zero,
                       -vector[..., 0], -vector[..., 1], vector[..., 0], zero), -1).reshape_as(skew)
    return eye + a[..., None] * raw + b[..., None] * (raw @ raw)


def geodesic_radians(predicted: Tensor, target: Tensor) -> Tensor:
    relative = predicted.transpose(-1, -2) @ target
    cosine = ((relative.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2).clamp(-1, 1)
    return torch.acos(cosine)


@dataclass(frozen=True)
class RotationMemoryEntry:
    uid: str
    family: str
    split: str
    panel: str
    coordinates: Tensor
    dino: Tensor
    dino_valid: Tensor
    point_valid: Tensor
    pooled_dino: Tensor
    rotation_vectors: Tensor
    kabsch_valid: Tensor
    event_phase: Tensor
    geometry_scale: float
    representation: str
    target_provenance: str

    def validate(self) -> None:
        if self.split != "train": raise ValueError("rotation bank must be training-only")
        if self.family not in {"rigid", "soft_body"}: raise ValueError("fluid/family invalid")
        if self.rotation_vectors.shape != (59, 3) or self.kabsch_valid.shape != (59,):
            raise ValueError("rotation trajectory must have 59 frames")
        if self.coordinates.shape[-1] != 3 or self.coordinates.shape[0] != self.dino.shape[0]:
            raise ValueError("point arrays disagree")
        if self.representation != "proper_kabsch_rotvec_frame1": raise ValueError("representation")
        if not self.target_provenance or self.geometry_scale <= 0: raise ValueError("missing provenance")


class RotationBank:
    def __init__(self, entries: Iterable[RotationMemoryEntry]):
        self.entries = tuple(entries)
        if not self.entries: raise ValueError("empty rotation bank")
        for entry in self.entries: entry.validate()
        if len({e.uid for e in self.entries}) != len(self.entries): raise ValueError("duplicate UID")
        self.content_sha256 = self._hash()

    def _hash(self) -> str:
        digest = hashlib.sha256()
        for e in sorted(self.entries, key=lambda x: x.uid):
            digest.update(json.dumps({"uid": e.uid, "family": e.family, "split": e.split,
                "panel": e.panel, "scale": e.geometry_scale, "representation": e.representation,
                "provenance": e.target_provenance}, sort_keys=True).encode())
            for value in (e.coordinates, e.dino, e.dino_valid, e.point_valid, e.pooled_dino,
                          e.rotation_vectors, e.kabsch_valid, e.event_phase):
                value = value.detach().cpu().contiguous(); digest.update(str(value.dtype).encode())
                digest.update(str(tuple(value.shape)).encode()); digest.update(value.numpy().tobytes())
        return digest.hexdigest()

    def save(self, path: str | Path) -> None:
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"entries": self.entries, "content_sha256": self.content_sha256}, path)
        path.with_suffix(path.suffix + ".manifest.json").write_text(json.dumps({
            "version": "v43_rotation_bank_v1", "content_sha256": self.content_sha256,
            "uids": [e.uid for e in self.entries], "families": {f: sum(e.family == f for e in self.entries)
            for f in ("rigid", "soft_body")}, "test_data_used": False}, indent=2) + "\n")


def retrieve_rotation(bank: RotationBank, query_uid: str, query_split: str, query_geometry: Tensor,
                      query_dino: Tensor, query_valid: Tensor, query_dino_valid: Tensor, *, mode: str,
                      k: int = 3, seed: int = 0) -> list[RotationMemoryEntry]:
    if query_split not in {"train", "validation"}: raise ValueError("test retrieval is sealed")
    if mode not in ARMS: raise ValueError(mode)
    candidates = [e for e in bank.entries if e.uid != query_uid]
    if len(candidates) < k: raise ValueError("insufficient leave-one-UID-out sources")
    def score(e: RotationMemoryEntry) -> float:
        sv = e.coordinates[e.point_valid]; qv = query_geometry[query_valid]
        if mode in {"geometry", "zero_memory"}:
            d = torch.cdist(qv.float(), sv.float())
            return float(d.min(1).values.square().mean() + d.min(0).values.square().mean())
        valid = query_valid & query_dino_valid
        if not valid.any() or not e.dino_valid.any(): return float("inf")
        q = torch.nn.functional.normalize(query_dino[valid].float(), dim=-1).mean(0)
        return float(1 - q @ e.pooled_dino.float())
    ordered = sorted(candidates, key=lambda e: (score(e), e.uid))
    if mode == "scene_shuffled":
        shift = 1 + seed % (len(ordered) - k + 1)
        return [ordered[(i + shift) % len(ordered)] for i in range(k)]
    return ordered[:k]


class CompactRotationReader(nn.Module):
    """Identity-anchored compact reader; empty memory is exactly identity."""
    def __init__(self, query_dim: int, memory_dim: int, hidden_dim: int = 128,
                 heads: int = 4, max_degrees: float = 20.):
        super().__init__()
        self.query = nn.Linear(query_dim, hidden_dim)
        self.memory = nn.Linear(memory_dim, hidden_dim)
        self.attention = nn.MultiheadAttention(hidden_dim, heads, batch_first=True)
        self.residual = nn.Linear(hidden_dim, 3)
        self.gate = nn.Linear(hidden_dim, 1)
        self.max_radians = math.radians(max_degrees)

    def forward(self, query: Tensor, memory: Tensor, memory_valid: Tensor):
        # query [B,T,Q], memory [B,T,K,M]
        b, t, k, _ = memory.shape
        active = memory_valid.any(-1)
        safe_valid = memory_valid.clone(); safe_valid[..., 0] |= ~active
        q = self.query(query).reshape(b * t, 1, -1)
        m = self.memory(memory).reshape(b * t, k, -1)
        attended, _ = self.attention(q, m, m, key_padding_mask=~safe_valid.reshape(b * t, k))
        h = attended.reshape(b, t, -1)
        gate = torch.sigmoid(self.gate(h)) * active[..., None]
        delta = gate * self.max_radians * torch.tanh(self.residual(h))
        return so3_exp(delta), delta, gate
