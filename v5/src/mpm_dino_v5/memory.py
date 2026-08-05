from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from torch import Tensor, nn
from torch.nn import functional as F


PHASES = (2, 3, 4)


@dataclass(frozen=True)
class SourcePhase:
    uid: str
    split: str
    phase: int
    coordinates: Tensor
    deformation: Tensor
    contact: Tensor
    point_valid: Tensor
    dino: Tensor
    dino_valid: Tensor

    def validate(self) -> None:
        if self.split != "train":
            raise ValueError("memory sources must come from train")
        if self.phase not in PHASES:
            raise ValueError(f"unsupported memory phase {self.phase}")
        n = self.coordinates.shape[0]
        if self.coordinates.shape != (n, 3) or self.deformation.shape != (n, 3):
            raise ValueError("coordinates and deformation must be [N,3]")
        if self.contact.shape != (n,) or self.point_valid.shape != (n,):
            raise ValueError("contact and point_valid must be [N]")
        if self.dino.shape[0] != n or self.dino_valid.shape != (n,):
            raise ValueError("DINO arrays must share the point dimension")

    def object_descriptor(self) -> Tensor:
        valid = self.point_valid & self.dino_valid
        if not valid.any():
            raise ValueError(f"source {self.uid} has no valid DINO")
        return F.normalize(self.dino[valid].float().mean(0), dim=0)


class V5MemoryBank:
    def __init__(self, entries: Iterable[SourcePhase], permitted_uids: Iterable[str]):
        self.entries = tuple(entries)
        self.permitted_uids = frozenset(permitted_uids)
        if len(self.permitted_uids) != 20:
            raise ValueError("V5 bank requires exactly the permitted 20 UIDs")
        if not self.entries:
            raise ValueError("memory bank cannot be empty")
        for entry in self.entries:
            entry.validate()
            if entry.uid not in self.permitted_uids:
                raise ValueError(f"out-of-scope bank UID: {entry.uid}")
        keys = {(entry.uid, entry.phase) for entry in self.entries}
        required = {(uid, phase) for uid in self.permitted_uids for phase in PHASES}
        if keys != required or len(self.entries) != len(required):
            raise ValueError("bank must contain exactly three phase entries per permitted UID")
        self.by_key = {(entry.uid, entry.phase): entry for entry in self.entries}
        self.content_sha256 = self._hash()

    def _hash(self) -> str:
        digest = hashlib.sha256()
        for entry in sorted(self.entries, key=lambda item: (item.uid, item.phase)):
            digest.update(json.dumps({"uid": entry.uid, "split": entry.split, "phase": entry.phase}, sort_keys=True).encode())
            for value in (entry.coordinates, entry.deformation, entry.contact, entry.point_valid, entry.dino, entry.dino_valid):
                tensor = value.detach().cpu().contiguous()
                digest.update(str(tensor.dtype).encode())
                digest.update(str(tuple(tensor.shape)).encode())
                digest.update(tensor.numpy().tobytes())
        return digest.hexdigest()

    def descriptor(self, uid: str) -> Tensor:
        return self.by_key[(uid, PHASES[0])].object_descriptor()

    def manifest(self) -> dict:
        return {
            "version": "v5_training_only_compact_bank_v1",
            "content_sha256": self.content_sha256,
            "splits": ["train"],
            "uids": sorted(self.permitted_uids),
            "phases": list(PHASES),
            "entries": [
                {
                    "uid": entry.uid,
                    "split": entry.split,
                    "phase": entry.phase,
                    "points": int(entry.coordinates.shape[0]),
                    "valid_points": int(entry.point_valid.sum()),
                    "valid_dino": int((entry.point_valid & entry.dino_valid).sum()),
                }
                for entry in sorted(self.entries, key=lambda item: (item.uid, item.phase))
            ],
            "test_data_used": False,
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save({"manifest": self.manifest(), "entries": self.entries}, temporary)
        temporary.replace(path)
        path.with_suffix(path.suffix + ".manifest.json").write_text(
            json.dumps(self.manifest(), indent=2) + "\n"
        )

    @classmethod
    def load(cls, path: str | Path, permitted_uids: Iterable[str]) -> "V5MemoryBank":
        payload = torch.load(path, map_location="cpu", weights_only=False)
        bank = cls(payload["entries"], permitted_uids)
        manifest = payload.get("manifest", {})
        if manifest.get("content_sha256") != bank.content_sha256:
            raise ValueError("memory bank content hash mismatch")
        return bank

    def retrieve_uids(self, query_uid: str, query_dino: Tensor, query_valid: Tensor, query_split: str, k: int = 3) -> list[str]:
        if query_split not in {"train", "validation", "test"}:
            raise ValueError("unknown query split")
        if k != 3:
            raise ValueError("V5 retrieval is fixed to top-3")
        if not query_valid.any():
            raise ValueError("query has no valid DINO")
        query = F.normalize(query_dino[query_valid].float().mean(0), dim=0)
        candidates = sorted(uid for uid in self.permitted_uids if uid != query_uid)
        if len(candidates) < k:
            raise ValueError("insufficient leave-one-UID-out sources")
        scored = [(float(1 - torch.dot(query, self.descriptor(uid).to(query))), uid) for uid in candidates]
        return [uid for _, uid in sorted(scored)[:k]]


class CompactSourceEncoder(nn.Module):
    """Compress a source phase into 32 latent mechanics tokens."""

    def __init__(self, hidden_dim: int = 128, heads: int = 4, tokens: int = 32):
        super().__init__()
        if tokens != 32:
            raise ValueError("V5 uses 32 compact tokens per source")
        self.tokens = nn.Parameter(torch.randn(tokens, hidden_dim) * 0.02)
        self.point_projection = nn.Linear(7, hidden_dim)
        self.attention = nn.MultiheadAttention(hidden_dim, heads, batch_first=True)
        self.output = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim), nn.SiLU())

    def forward(self, coordinates: Tensor, deformation: Tensor, contact: Tensor, point_valid: Tensor) -> Tensor:
        # Inputs are [B,N,*]. DINO is intentionally absent.
        points = self.point_projection(torch.cat((coordinates, deformation, contact[..., None]), -1))
        safe_valid = point_valid.clone()
        empty = ~safe_valid.any(1)
        if empty.any():
            safe_valid[empty, 0] = True
            points = points.clone()
            points[empty, 0] = 0
        queries = self.tokens[None].expand(len(points), -1, -1)
        tokens, _ = self.attention(queries, points, points, key_padding_mask=~safe_valid, need_weights=False)
        return self.output(tokens)


def interpolate_phase_tokens(tokens: Tensor, event_time: Tensor) -> Tensor:
    """Linear interpolation of [B,K,3,M,H] tokens at event_time [-1,1]."""
    if tokens.shape[2] != 3:
        raise ValueError("expected three ordered phase token sets")
    value = event_time.clamp(-1, 1)
    lower_half = value <= 0
    alpha_low = value + 1
    alpha_high = value
    first = (1 - alpha_low)[..., None, None, None] * tokens[:, None, :, 0] + alpha_low[..., None, None, None] * tokens[:, None, :, 1]
    second = (1 - alpha_high)[..., None, None, None] * tokens[:, None, :, 1] + alpha_high[..., None, None, None] * tokens[:, None, :, 2]
    return torch.where(lower_half[..., None, None, None], first, second)


class CompactMemoryResidual(nn.Module):
    def __init__(self, query_dim: int, hidden_dim: int = 128, heads: int = 4, residual_bound: float = 0.05, gate_logit: float = -4.0):
        super().__init__()
        if residual_bound <= 0 or gate_logit >= 0:
            raise ValueError("memory requires a positive bound and negative gate logit")
        self.query = nn.Linear(query_dim, hidden_dim)
        self.attention = nn.MultiheadAttention(hidden_dim, heads, batch_first=True)
        self.residual = nn.Sequential(nn.LayerNorm(hidden_dim * 2), nn.Linear(hidden_dim * 2, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 3))
        self.gate_logit = nn.Parameter(torch.tensor(float(gate_logit)))
        self.residual_bound = float(residual_bound)

    def forward(self, base: Tensor, query: Tensor, memory: Tensor | None, point_mask: Tensor, return_gate: bool = False):
        if memory is None:
            gate = base.new_zeros(())
            return (base, gate) if return_gate else base
        # query [B,T,N,Q], memory [B,T,K,M,H]
        b, t, n, _ = query.shape
        q = self.query(query).reshape(b * t, n, -1)
        mem = memory.reshape(b * t, memory.shape[2] * memory.shape[3], -1)
        attended, _ = self.attention(q, mem, mem, need_weights=False)
        residual = torch.tanh(self.residual(torch.cat((q, attended), -1))).reshape(b, t, n, 3)
        gate = torch.sigmoid(self.gate_logit)
        mask = point_mask[:, None, :, None].to(base.dtype)
        raw = (base + gate * self.residual_bound * residual) * mask
        output = (raw - raw.sum(2, keepdim=True) / mask.sum(2, keepdim=True).clamp_min(1)) * mask
        return (output, gate) if return_gate else output


class V5MemoryModule(nn.Module):
    """Trainable compact source encoder plus bounded residual reader."""

    def __init__(self, query_dim: int, hidden_dim: int = 128, heads: int = 4, residual_bound: float = 0.05, gate_logit: float = -4.0):
        super().__init__()
        self.source_encoder = CompactSourceEncoder(hidden_dim, heads, tokens=32)
        self.reader = CompactMemoryResidual(query_dim, hidden_dim, heads, residual_bound, gate_logit)

    def encode_sources(self, bank: V5MemoryBank, selected_uids: list[list[str]], device: torch.device) -> Tensor:
        rows = []
        for uids in selected_uids:
            objects = []
            for uid in uids:
                phases = []
                for phase in PHASES:
                    entry = bank.by_key[(uid, phase)]
                    phases.append(self.source_encoder(
                        entry.coordinates.to(device)[None],
                        entry.deformation.to(device)[None],
                        entry.contact.to(device)[None],
                        entry.point_valid.to(device)[None],
                    )[0])
                objects.append(torch.stack(phases))
            rows.append(torch.stack(objects))
        return torch.stack(rows)  # [B,K,3,32,H]

    def forward(self, base: Tensor, query: Tensor, event_time: Tensor, point_mask: Tensor, bank: V5MemoryBank, selected_uids: list[list[str]], return_gate: bool = False):
        source_tokens = self.encode_sources(bank, selected_uids, base.device)
        memory = interpolate_phase_tokens(source_tokens, event_time)
        return self.reader(base, query, memory, point_mask, return_gate=return_gate)
