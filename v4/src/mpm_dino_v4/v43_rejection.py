from __future__ import annotations

import torch
from torch import nn

from .v43_retrieval import AttendedMechanicalMemory


class RejectingMechanicalMemory(AttendedMechanicalMemory):
    """Compact attention with per-source compatibility and a null memory."""

    def __init__(self, query_dim, memory_dim, hidden_dim=128, heads=4,
                 residual_scale=.05):
        super().__init__(query_dim, memory_dim, hidden_dim, heads, residual_scale)
        self.compatibility = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2), nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(), nn.Linear(hidden_dim, 1),
        )
        # Start from the established compact reader. Rejection must be learned
        # from evidence rather than winning immediately through a neutral
        # four-way softmax prior.
        self.register_buffer("null_logit", torch.tensor(-4.0))
        nn.init.zeros_(self.compatibility[-1].weight)
        nn.init.zeros_(self.compatibility[-1].bias)

    def forward(self, base, query, memory, memory_valid, point_valid,
                *, return_gate=False, return_compatibility=False):
        b, t, n, _ = query.shape
        k, m = memory.shape[2:4]
        if not memory_valid.any():
            gate = base.new_zeros(b, t, n, 1)
            weights = base.new_zeros(b, t, k)
            if return_compatibility:
                return base, gate, weights, base.new_ones(b, t)
            return (base, gate) if return_gate else base
        q = self.query_projection(query)
        projected = self.memory_projection(memory)
        q_mask = point_valid[:, None, :, None].to(q.dtype)
        q_summary = (q * q_mask).sum(2) / q_mask.sum(2).clamp_min(1)
        m_mask = memory_valid[..., None].to(projected.dtype)
        source_summary = (projected * m_mask).sum(3) / m_mask.sum(3).clamp_min(1)
        pair = torch.cat((q_summary[:, :, None].expand(-1, -1, k, -1),
                          source_summary), dim=-1)
        logits = self.compatibility(pair).squeeze(-1)
        source_available = memory_valid.any(3)
        available_float = source_available.to(logits.dtype)
        mean_logit = (logits * available_float).sum(-1, keepdim=True) / (
            available_float.sum(-1, keepdim=True).clamp_min(1)
        )
        # Remove the unidentifiable common shift. Compatibility may rank
        # neighbours, while the existing point gate decides to reject all.
        logits = logits - mean_logit
        logits = logits.masked_fill(~source_available, -1e4)
        null = self.null_logit.expand(b, t, 1)
        all_weights = torch.softmax(torch.cat((logits, null), dim=-1), dim=-1)
        source_weights, null_weight = all_weights[..., :k], all_weights[..., -1]
        # Uniform source compatibility must reproduce the established compact
        # reader rather than shrinking every value by 1/k.
        source_scale = source_weights * k / (
            (1 - null_weight).clamp_min(1e-8)[..., None]
        )
        weighted = projected * source_scale[..., None, None]
        flat_q = q.reshape(b * t, n, -1)
        flat_memory = weighted.reshape(b * t, k * m, -1)
        padding = ~memory_valid.reshape(b * t, k * m)
        attended, _ = self.attention(
            flat_q, flat_memory, flat_memory,
            key_padding_mask=padding, need_weights=False,
        )
        attended = attended.reshape(b, t, n, -1)
        fused = torch.cat((q, attended), dim=-1)
        gate = torch.sigmoid(self.gate(fused)) * (1 - null_weight[..., None, None])
        correction = torch.tanh(self.residual(fused)) * self.residual_scale
        mask = point_valid[:, None, :, None].to(base.dtype)
        raw = (base + gate * correction) * mask
        output = (raw - raw.sum(2, keepdim=True)
                  / mask.sum(2, keepdim=True).clamp_min(1)) * mask
        if return_compatibility:
            return output, gate * mask, source_weights, null_weight
        return (output, gate * mask) if return_gate else output
