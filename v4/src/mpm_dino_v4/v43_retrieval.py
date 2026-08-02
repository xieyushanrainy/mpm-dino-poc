from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from torch import Tensor, nn


CONTROL_MODES = {
    "zero_memory", "geometry", "aligned_dino",
    "scene_shuffled", "point_shuffled",
}


@dataclass(frozen=True)
class MemoryEntry:
    uid: str
    split: str
    stage: int
    event_time: float
    coordinates: Tensor
    dino: Tensor
    deformation: Tensor
    point_valid: Tensor
    dino_valid: Tensor
    contact: Tensor
    geometry_scale: float
    field_provenance: str

    def validate(self) -> None:
        if self.split != "train":
            raise ValueError(f"memory {self.uid} is not from train split")
        n = self.coordinates.shape[0]
        if self.coordinates.shape != (n, 3) or self.deformation.shape != (n, 3):
            raise ValueError("coordinates/deformation must be [N,3]")
        if self.point_valid.shape != (n,) or self.dino_valid.shape != (n,):
            raise ValueError("validity masks must be [N]")
        if self.dino.shape[0] != n or self.contact.shape[0] != n:
            raise ValueError("all memory arrays must share the point dimension")
        if not self.field_provenance:
            raise ValueError("field provenance is required")
        if self.geometry_scale <= 0:
            raise ValueError("geometry_scale must be positive")


def _tensor_bytes(value: Tensor) -> bytes:
    value = value.detach().cpu().contiguous()
    return value.numpy().tobytes()


class RetrievalBank:
    """Immutable, train-only memory bank with a reproducible content hash."""

    def __init__(self, entries: Iterable[MemoryEntry]):
        self.entries = tuple(entries)
        if not self.entries:
            raise ValueError("retrieval bank cannot be empty")
        for entry in self.entries:
            entry.validate()
        keys = [(e.uid, e.stage, e.event_time) for e in self.entries]
        if len(keys) != len(set(keys)):
            raise ValueError("duplicate memory key")
        self.content_sha256 = self._content_hash()

    def _content_hash(self) -> str:
        digest = hashlib.sha256()
        for entry in sorted(self.entries, key=lambda e: (e.uid, e.stage, e.event_time)):
            metadata = {
                "uid": entry.uid, "split": entry.split, "stage": entry.stage,
                "event_time": entry.event_time,
                "geometry_scale": entry.geometry_scale,
                "field_provenance": entry.field_provenance,
            }
            digest.update(json.dumps(metadata, sort_keys=True).encode())
            for value in (entry.coordinates, entry.dino, entry.deformation,
                          entry.point_valid, entry.dino_valid, entry.contact):
                digest.update(str(value.dtype).encode())
                digest.update(str(tuple(value.shape)).encode())
                digest.update(_tensor_bytes(value))
        return digest.hexdigest()

    def manifest(self) -> dict:
        return {
            "version": "v43_retrieval_bank_v1",
            "content_sha256": self.content_sha256,
            "splits": sorted({e.split for e in self.entries}),
            "uids": sorted({e.uid for e in self.entries}),
            "entries": [{
                "uid": e.uid, "split": e.split, "stage": e.stage,
                "event_time": e.event_time,
                "points": int(e.coordinates.shape[0]),
                "valid_points": int(e.point_valid.sum()),
                "valid_dino": int((e.point_valid & e.dino_valid).sum()),
                "geometry_scale": e.geometry_scale,
                "field_provenance": e.field_provenance,
            } for e in self.entries],
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"manifest": self.manifest(), "entries": self.entries}, path)
        path.with_suffix(path.suffix + ".manifest.json").write_text(
            json.dumps(self.manifest(), indent=2) + "\n"
        )


def _masked_chamfer(query: Tensor, source: MemoryEntry) -> float:
    q = query
    s = source.coordinates[source.point_valid]
    if not len(q) or not len(s):
        return float("inf")
    distances = torch.cdist(q.float(), s.float())
    return float(distances.min(1).values.square().mean()
                 + distances.min(0).values.square().mean())


def align_points(query_coordinates: Tensor, query_valid: Tensor,
                 source: MemoryEntry) -> tuple[Tensor, Tensor]:
    """Map every query point to its nearest valid source point."""
    source_indices = torch.nonzero(source.point_valid, as_tuple=False).flatten()
    mapping = torch.zeros(len(query_coordinates), dtype=torch.long,
                          device=query_coordinates.device)
    valid = query_valid.clone()
    if not len(source_indices):
        return mapping, torch.zeros_like(valid)
    query_indices = torch.nonzero(query_valid, as_tuple=False).flatten()
    if len(query_indices):
        nearest = torch.cdist(
            query_coordinates[query_indices].float(),
            source.coordinates[source_indices].to(query_coordinates).float(),
        ).argmin(1)
        mapping[query_indices] = source_indices.to(mapping.device)[nearest]
    valid &= source.point_valid.to(valid.device)[mapping]
    return mapping, valid


def _aligned_dino_distance(query_coordinates: Tensor, query_dino: Tensor,
                           query_valid: Tensor, query_dino_valid: Tensor,
                           source: MemoryEntry) -> float:
    mapping, aligned_valid = align_points(query_coordinates, query_valid, source)
    valid = aligned_valid & query_dino_valid & source.dino_valid.to(mapping.device)[mapping]
    if not valid.any():
        return float("inf")
    q = torch.nn.functional.normalize(query_dino[valid].float(), dim=-1)
    s = torch.nn.functional.normalize(
        source.dino.to(query_dino.device)[mapping[valid]].float(), dim=-1,
    )
    return float((1 - (q * s).sum(-1)).mean())


def retrieve(bank: RetrievalBank, query_uid: str, query_split: str,
             query_coordinates: Tensor, query_dino: Tensor,
             query_valid: Tensor, query_dino_valid: Tensor, *, stage: int,
             mode: str, k: int = 3, shuffle_seed: int = 0) -> list[MemoryEntry]:
    if mode not in CONTROL_MODES:
        raise ValueError(mode)
    if query_split not in {"train", "validation"}:
        raise ValueError("test retrieval is sealed")
    candidates = [e for e in bank.entries
                  if e.stage == stage and e.uid != query_uid]
    if len(candidates) < k:
        raise ValueError(f"only {len(candidates)} eligible memories for k={k}")
    scoring = "geometry" if mode in {"geometry", "zero_memory"} else "dino"
    scored = []
    for entry in candidates:
        distance = (_masked_chamfer(query_coordinates[query_valid], entry)
                    if scoring == "geometry" else _aligned_dino_distance(
                        query_coordinates, query_dino, query_valid,
                        query_dino_valid, entry))
        scored.append((distance, entry.uid, entry))
    scored.sort(key=lambda row: (row[0], row[1]))
    selected = [row[2] for row in scored[:k]]
    if mode == "scene_shuffled":
        ordered = [row[2] for row in scored]
        shift = 1 + (shuffle_seed % (len(ordered) - k + 1))
        selected = [ordered[(i + shift) % len(ordered)] for i in range(k)]
    return selected


def materialize_aligned(query_coordinates: Tensor, query_valid: Tensor,
                        entries: list[MemoryEntry], mode: str,
                        shuffle_seed: int = 0,
                        max_tokens: int | None = None) -> tuple[Tensor, Tensor]:
    """Return [K,N,F] matched tokens and [K,N] validity."""
    rows, masks = [], []
    for rank, entry in enumerate(entries):
        mapping, valid = align_points(query_coordinates, query_valid, entry)
        coordinates = entry.coordinates.to(query_coordinates.device)[mapping]
        dino = entry.dino.to(query_coordinates.device)[mapping]
        deformation = entry.deformation.to(query_coordinates.device)[mapping]
        contact = entry.contact.to(query_coordinates.device)[mapping]
        dino_valid = entry.dino_valid.to(query_coordinates.device)[mapping]
        if mode == "point_shuffled":
            generator = torch.Generator(device="cpu").manual_seed(
                shuffle_seed + rank * 1009
            )
            permutation = torch.randperm(len(mapping), generator=generator).to(mapping.device)
            dino = dino[permutation]
            dino_valid = dino_valid[permutation]
        if mode in {"geometry", "zero_memory"}:
            dino = torch.zeros_like(dino)
            dino_valid = torch.zeros_like(dino_valid)
        token = torch.cat((coordinates, dino, dino_valid[:, None].to(dino.dtype),
                           contact, deformation), dim=-1)
        if max_tokens is not None and len(token) > max_tokens:
            eligible = torch.nonzero(valid, as_tuple=False).flatten()
            if not len(eligible):
                chosen = torch.zeros(max_tokens, dtype=torch.long,
                                     device=token.device)
            else:
                positions = torch.linspace(
                    0, len(eligible) - 1, max_tokens, device=token.device
                ).round().long()
                chosen = eligible[positions]
            token, valid = token[chosen], valid[chosen]
        rows.append(token)
        masks.append(valid)
    values, mask = torch.stack(rows), torch.stack(masks)
    if mode == "zero_memory":
        values.zero_()
        mask.zero_()
    return values, mask


class AttendedMechanicalMemory(nn.Module):
    """Cross-attended, bounded residual that exactly ignores empty memory."""

    def __init__(self, query_dim: int, memory_dim: int, hidden_dim: int = 128,
                 heads: int = 4, residual_scale: float = 0.05):
        super().__init__()
        self.query_projection = nn.Linear(query_dim, hidden_dim)
        self.memory_projection = nn.Linear(memory_dim, hidden_dim)
        self.attention = nn.MultiheadAttention(
            hidden_dim, heads, batch_first=True
        )
        self.gate = nn.Sequential(nn.LayerNorm(hidden_dim * 2),
                                  nn.Linear(hidden_dim * 2, 1))
        self.residual = nn.Sequential(nn.LayerNorm(hidden_dim * 2),
                                      nn.Linear(hidden_dim * 2, 3))
        self.residual_scale = float(residual_scale)
        nn.init.constant_(self.gate[-1].bias, -4.0)
        nn.init.normal_(self.residual[-1].weight, std=1e-3)
        nn.init.zeros_(self.residual[-1].bias)

    def forward(self, base: Tensor, query: Tensor, memory: Tensor,
                memory_valid: Tensor, point_valid: Tensor,
                *, return_gate: bool = False):
        # base/query [B,T,N,*], memory [B,T,K,M,F]
        b, t, n, _ = query.shape
        if not memory_valid.any():
            gate = base.new_zeros(b, t, n, 1)
            return (base, gate) if return_gate else base
        k, m = memory.shape[2:4]
        q = self.query_projection(query).reshape(b * t, n, -1)
        mem = self.memory_projection(memory).reshape(b * t, k * m, -1)
        padding = ~memory_valid.reshape(b * t, k * m)
        attended, _ = self.attention(q, mem, mem,
                                     key_padding_mask=padding,
                                     need_weights=False)
        fused = torch.cat((q, attended), dim=-1)
        gate = torch.sigmoid(self.gate(fused))
        correction = torch.tanh(self.residual(fused)) * self.residual_scale
        mask = point_valid[:, None, :, None].to(base.dtype)
        raw = (base + gate * correction) * mask
        count = mask.sum(2, keepdim=True).clamp_min(1)
        output = (raw - raw.sum(2, keepdim=True) / count) * mask
        return (output, gate * mask) if return_gate else output


def parameter_snapshot(module: nn.Module, protected_prefixes: tuple[str, ...]) -> dict:
    return {name: value.detach().cpu().clone()
            for name, value in module.state_dict().items()
            if name.startswith(protected_prefixes)}


def parameters_identical(module: nn.Module, snapshot: dict) -> bool:
    state = module.state_dict()
    return all(name in state and torch.equal(value, state[name].detach().cpu())
               for name, value in snapshot.items())
