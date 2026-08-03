from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .model import masked_mean
from .v41_data import MODEL_INPUT_KEYS
from .v42_oracle import EVENT_STAGES
from .v43_retrieval import align_points


FIELD_MEMORY_DIM = 3 + 384 + 1 + 3 + 3 + 1 + 1
FIELD_QUERY_DIM = 128 + 3 + 384 + 1 + 15


@dataclass(frozen=True)
class FieldEntry:
    uid: str
    split: str
    coordinates: Tensor
    dino: Tensor
    point_valid: Tensor
    dino_valid: Tensor
    displacement: Tensor
    contact: Tensor
    stages: Tensor
    event_time: Tensor
    provenance: str

    def validate(self):
        if self.split != "train":
            raise ValueError("field memory must be train-only")
        n = len(self.coordinates)
        if self.coordinates.shape != (n, 3) or self.dino.shape[0] != n:
            raise ValueError("invalid static field shapes")
        if self.displacement.ndim != 3 or self.displacement.shape[1:] != (n, 3):
            raise ValueError("displacement must be [T,N,3]")
        if self.contact.shape != self.displacement.shape:
            raise ValueError("contact and displacement shapes differ")
        if len(self.stages) != len(self.displacement) or len(self.event_time) != len(self.displacement):
            raise ValueError("temporal metadata length differs")


class FieldBank:
    def __init__(self, entries):
        self.entries = tuple(entries)
        if not self.entries:
            raise ValueError("empty field bank")
        for entry in self.entries: entry.validate()
        if len({e.uid for e in self.entries}) != len(self.entries):
            raise ValueError("duplicate field UID")
        digest = hashlib.sha256()
        for entry in sorted(self.entries, key=lambda e: e.uid):
            digest.update(json.dumps({"uid": entry.uid, "split": entry.split,
                                      "provenance": entry.provenance}, sort_keys=True).encode())
            for value in (entry.coordinates, entry.dino, entry.point_valid,
                          entry.dino_valid, entry.displacement, entry.contact,
                          entry.stages, entry.event_time):
                value = value.detach().cpu().contiguous()
                digest.update(str((value.dtype, tuple(value.shape))).encode())
                digest.update(value.numpy().tobytes())
        self.content_sha256 = digest.hexdigest()

    def manifest(self):
        return {"version": "v43_field_bank_v1", "content_sha256": self.content_sha256,
                "splits": sorted({e.split for e in self.entries}),
                "uids": sorted(e.uid for e in self.entries),
                "entries": [{"uid": e.uid, "frames": len(e.stages),
                             "points": len(e.coordinates),
                             "provenance": e.provenance} for e in self.entries]}


def _dino_distance(query_xyz, query_dino, query_valid, query_dino_valid, source):
    proxy = type("Proxy", (), {"coordinates": source.coordinates,
                                "point_valid": source.point_valid})()
    mapping, valid = align_points(query_xyz, query_valid, proxy)
    valid &= query_dino_valid & source.dino_valid[mapping]
    if not valid.any(): return float("inf")
    q = F.normalize(query_dino[valid].float(), dim=-1)
    s = F.normalize(source.dino[mapping[valid]].float(), dim=-1)
    return float((1 - (q * s).sum(-1)).mean())


def retrieve_field_entries(bank, query_uid, query_xyz, query_dino,
                           query_valid, query_dino_valid, k=3):
    rows = []
    for entry in bank.entries:
        if entry.uid == query_uid: continue
        distance = _dino_distance(query_xyz, query_dino, query_valid,
                                  query_dino_valid, entry)
        rows.append((distance, entry.uid, entry))
    rows.sort(key=lambda x: (x[0], x[1]))
    if len(rows) < k: raise ValueError("insufficient field neighbours")
    return [row[2] for row in rows[:k]]


def materialize_fields(query_xyz, query_dino, query_valid, query_dino_valid,
                       query_stages, query_event_time, sources, *,
                       shuffle=False, zero_field=False, seed=0):
    """Return point-matched [T,N,K,F] source tokens and validity."""
    frames, points, k = len(query_stages), len(query_xyz), len(sources)
    tokens = query_xyz.new_zeros(frames, points, k, FIELD_MEMORY_DIM)
    masks = torch.zeros(frames, points, k, dtype=torch.bool,
                        device=query_xyz.device)
    for rank, source in enumerate(sources):
        proxy = type("Proxy", (), {"coordinates": source.coordinates,
                                    "point_valid": source.point_valid})()
        mapping, valid = align_points(query_xyz, query_valid, proxy)
        if shuffle:
            generator = torch.Generator().manual_seed(seed + rank * 1009)
            mapping = mapping[torch.randperm(len(mapping), generator=generator)]
        source_xyz = source.coordinates[mapping]
        source_dino = source.dino[mapping]
        source_dino_valid = source.dino_valid[mapping]
        geom = torch.linalg.vector_norm(query_xyz - source_xyz, dim=-1)
        confidence = F.cosine_similarity(query_dino.float(), source_dino.float(), dim=-1)
        point_valid = valid & query_dino_valid & source_dino_valid
        for frame in range(frames):
            stage = int(query_stages[frame])
            candidates = torch.nonzero(source.stages.eq(stage), as_tuple=False).flatten()
            if not len(candidates) or stage not in [int(s) for s in EVENT_STAGES]:
                continue
            nearest = candidates[torch.argmin(
                (source.event_time[candidates] - query_event_time[frame]).abs()
            )]
            deformation = source.displacement[nearest, mapping]
            if zero_field: deformation = torch.zeros_like(deformation)
            token = torch.cat((source_xyz, source_dino,
                               source_dino_valid[:, None].to(source_dino.dtype),
                               source.contact[nearest, mapping], deformation,
                               geom[:, None], confidence[:, None]), dim=-1)
            tokens[frame, :, rank] = token
            masks[frame, :, rank] = point_valid
    return tokens, masks


class PointwiseFieldAttention(nn.Module):
    def __init__(self, hidden=139, heads=4, residual_scale=.05):
        super().__init__()
        self.query_projection = nn.Linear(FIELD_QUERY_DIM, hidden)
        self.key_projection = nn.Linear(FIELD_MEMORY_DIM, hidden)
        self.value_projection = nn.Linear(FIELD_MEMORY_DIM, hidden)
        self.bias = nn.Sequential(nn.Linear(2, hidden // 2), nn.SiLU(),
                                  nn.Linear(hidden // 2, 1))
        self.gate = nn.Sequential(nn.LayerNorm(hidden * 2), nn.Linear(hidden * 2, 1))
        self.residual = nn.Sequential(nn.LayerNorm(hidden * 2), nn.Linear(hidden * 2, 3))
        self.scale = hidden ** -0.5
        self.residual_scale = residual_scale
        nn.init.constant_(self.gate[-1].bias, -4.)
        nn.init.normal_(self.residual[-1].weight, std=1e-3)
        nn.init.zeros_(self.residual[-1].bias)

    def forward(self, base, query, memory, valid, point_valid, return_gate=False):
        if not valid.any():
            gate = base.new_zeros(*base.shape[:-1], 1)
            return (base, gate) if return_gate else base
        q = self.query_projection(query)
        keys, values = self.key_projection(memory), self.value_projection(memory)
        logits = (q[..., None, :] * keys).sum(-1) * self.scale
        logits = logits + self.bias(memory[..., -2:]).squeeze(-1)
        logits = logits.masked_fill(~valid, -1e4)
        weights = torch.softmax(logits, dim=-1) * valid.to(logits.dtype)
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-8)
        attended = (weights[..., None] * values).sum(-2)
        fused = torch.cat((q, attended), dim=-1)
        gate = torch.sigmoid(self.gate(fused))
        correction = torch.tanh(self.residual(fused)) * self.residual_scale
        mask = point_valid[:, None, :, None].to(base.dtype)
        raw = (base + gate * correction) * mask
        output = (raw - raw.sum(2, keepdim=True) / mask.sum(2, keepdim=True).clamp_min(1)) * mask
        return (output, gate * mask) if return_gate else output


class FieldAttentionModel(nn.Module):
    def __init__(self, base, bank, seed=42, top_k=3):
        super().__init__()
        self.base, self.bank, self.seed, self.top_k = base, bank, seed, top_k
        self.memory = PointwiseFieldAttention()

    def forward_batch(self, batch, stages, condition, split, ablation=None):
        ablation = ablation or {}
        inputs = {key: batch[key] for key in MODEL_INPUT_KEYS}
        inputs["oracle_condition"] = condition
        base = self.base(**inputs)
        com = masked_mean(batch["x1"], batch["input_mask"])
        shape = batch["x1"] - com[:, None]
        radius = torch.linalg.vector_norm(shape, dim=-1).masked_fill(
            ~batch["input_mask"], 0).amax(1).clamp_min(1e-6)
        xyz = shape / radius[:, None, None]
        sources = retrieve_field_entries(
            self.bank, batch["uid"][0], xyz[0].detach().cpu(),
            batch["dino"][0].detach().cpu(), batch["input_mask"][0].detach().cpu(),
            batch["dino_valid"][0].detach().cpu(), self.top_k,
        )
        # DirectProbeConditionBuilder broadcasts the temporal condition to
        # points, so one point carries the identical event-time scalar.
        event_time = condition[0, :, 0, 14].detach().cpu()
        memory, valid = materialize_fields(
            xyz[0].detach().cpu(), batch["dino"][0].detach().cpu(),
            batch["input_mask"][0].detach().cpu(), batch["dino_valid"][0].detach().cpu(),
            stages.labels[0, 1:].detach().cpu(), event_time, sources,
            shuffle=ablation.get("correspondence", False),
            zero_field=ablation.get("source_deformation", False), seed=self.seed,
        )
        memory, valid = memory[None].to(xyz), valid[None].to(xyz.device)
        if ablation.get("memory"):
            memory.zero_(); valid.zero_()
        query = torch.cat((base.physical_hidden.detach(),
            xyz[:, None].expand(-1, 59, -1, -1),
            batch["dino"][:, None].expand(-1, 59, -1, -1),
            batch["dino_valid"][:, None, :, None].expand(-1, 59, -1, -1).to(xyz.dtype),
            condition), dim=-1)
        displacement, gate = self.memory(base.canonical_displacement, query,
                                         memory, valid, batch["input_mask"], True)
        canonical_shape = shape[:, None] + displacement
        rotated = torch.einsum("btni,btij->btnj", canonical_shape, base.rotation)
        output = replace(base, canonical_displacement=displacement,
                         canonical_shape=canonical_shape,
                         position=base.com[:, :, None] + rotated)
        return output, gate, valid
