from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from mpm_dino_v2.deformation import edge_validity, gather_neighbours
from .full_data import ballistic_trajectory
from .full_model import FullTrajectoryOutput, FullTrajectorySurrogate
from .model import GraphLayer, masked_mean


class PointDINOProjection(nn.Module):
    def __init__(self, dino_dim=384, visual_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dino_dim), nn.Linear(dino_dim, 64), nn.SiLU(),
            nn.Linear(64, visual_dim),
        )

    def forward(self, dino, valid):
        return self.net(dino * valid[..., None].to(dino.dtype))


class M1Fusion(nn.Module):
    def __init__(self, hidden_dim, visual_dim):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.update = nn.Sequential(
            nn.Linear(hidden_dim + visual_dim + 1, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        nn.init.zeros_(self.update[-1].weight)
        nn.init.zeros_(self.update[-1].bias)

    def forward(self, hidden, visual, valid, *_):
        v = visual[:, None].expand(-1, hidden.shape[1], -1, -1)
        m = valid[:, None, :, None].expand(-1, hidden.shape[1], -1, -1).to(hidden.dtype)
        return hidden + self.update(torch.cat((self.norm(hidden), v, m), -1))


class M2LocalMemory(nn.Module):
    def __init__(self, hidden_dim, visual_dim, heads=4):
        super().__init__()
        if hidden_dim % heads:
            raise ValueError("hidden width must divide heads")
        self.heads, self.head_dim = heads, hidden_dim // heads
        self.q = nn.Linear(hidden_dim, hidden_dim)
        self.k = nn.Linear(visual_dim + 4, hidden_dim)
        self.v = nn.Linear(visual_dim + 4, hidden_dim)
        self.no_visual = nn.Parameter(torch.zeros(hidden_dim))
        self.out = nn.Linear(hidden_dim, hidden_dim)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        self.last_attention_shape = None

    def forward(self, hidden, visual, valid, indices, neighbour_mask, rest_vectors):
        batch, frames, points, width = hidden.shape
        local_v = torch.cat((visual[:, :, None], gather_neighbours(visual, indices)), dim=2)
        local_valid = torch.cat((valid[:, :, None], gather_neighbours(valid[..., None], indices).squeeze(-1)), 2)
        local_mask = torch.cat((torch.ones_like(valid[:, :, None]), neighbour_mask), 2) & local_valid
        zero_vec = torch.zeros_like(rest_vectors[:, :, :1])
        local_rel = torch.cat((zero_vec, rest_vectors), 2)
        token = torch.cat((local_v, local_rel, local_valid[..., None].to(local_v.dtype)), -1)
        memory_k = self.k(token).view(batch, points, -1, self.heads, self.head_dim)
        memory_v = self.v(token).view(batch, points, -1, self.heads, self.head_dim)
        q = self.q(hidden).view(batch, frames, points, self.heads, self.head_dim)
        scores = torch.einsum("btphd,bpmhd->btphm", q, memory_k) / math.sqrt(self.head_dim)
        self.last_attention_shape = tuple(scores.shape)
        scores = scores.masked_fill(~local_mask[:, None, :, None, :], -torch.inf)
        all_invalid = ~local_mask.any(-1)
        safe_scores = torch.where(all_invalid[:, None, :, None, None], torch.zeros_like(scores), scores)
        weights = safe_scores.softmax(-1)
        attended = torch.einsum("btphm,bpmhd->btphd", weights, memory_v).reshape(batch, frames, points, width)
        attended = torch.where(
            all_invalid[:, None, :, None],
            self.no_visual[None, None, None].expand(batch, frames, points, -1),
            attended,
        )
        return hidden + self.out(attended)


class M6Adapter(nn.Module):
    def __init__(self, hidden_dim, visual_dim, gate_max=0.1):
        super().__init__()
        self.gate_max = gate_max
        self.norm = nn.LayerNorm(hidden_dim)
        self.adapter = nn.Sequential(
            nn.Linear(hidden_dim + visual_dim + 1, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.gate = nn.Linear(hidden_dim + visual_dim + 1, 1)
        nn.init.zeros_(self.adapter[-1].weight)
        nn.init.zeros_(self.adapter[-1].bias)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)
        self.last_gate = None

    def forward(self, hidden, visual, valid, *_):
        v = visual[:, None].expand(-1, hidden.shape[1], -1, -1)
        m = valid[:, None, :, None].expand(-1, hidden.shape[1], -1, -1).to(hidden.dtype)
        features = torch.cat((self.norm(hidden), v, m), -1)
        gate = self.gate_max * torch.sigmoid(self.gate(features))
        self.last_gate = gate.detach()
        return hidden + gate * torch.tanh(self.adapter(features))


class PhysicalBlock(nn.Module):
    def __init__(self, hidden_dim, heads, dropout):
        super().__init__()
        self.spatial = GraphLayer(hidden_dim, edge_dim=8)
        self.temporal_norm = nn.LayerNorm(hidden_dim)
        self.temporal = nn.MultiheadAttention(hidden_dim, heads, dropout=dropout, batch_first=True)
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim), nn.Dropout(dropout),
        )

    def forward(self, hidden, edges, indices, edge_mask, point_mask):
        b, t, n, d = hidden.shape
        flat_i = indices[:, None].expand(-1, t, -1, -1).reshape(b * t, n, -1)
        flat_m = edge_mask[:, None].expand(-1, t, -1, -1).reshape(b * t, n, -1)
        hidden = self.spatial(hidden.reshape(b*t, n, d), edges.reshape(b*t, n, edges.shape[-2], 8), flat_i, flat_m).reshape(b,t,n,d)
        q = self.temporal_norm(hidden).permute(0,2,1,3).reshape(b*n,t,d)
        attended, _ = self.temporal(q, q, q, need_weights=False)
        hidden = hidden + attended.reshape(b,n,t,d).permute(0,2,1,3)
        hidden = hidden + self.ffn(self.ffn_norm(hidden))
        return hidden * point_mask[:, None, :, None]


class V41TrajectorySurrogate(nn.Module):
    """DINO-free Track-B physical trunk with one pluggable aligned visual path."""

    def __init__(self, mechanism="m1", dino_dim=384, visual_dim=32, hidden_dim=128,
                 blocks=4, heads=4, dropout=0.1, frames=59, gradient_checkpointing=True):
        super().__init__()
        if mechanism not in {"m1", "m2", "m6", "none"}:
            raise ValueError(mechanism)
        self.mechanism, self.frames = mechanism, frames
        self.gradient_checkpointing = gradient_checkpointing
        self.dino_projection = PointDINOProjection(dino_dim, visual_dim)
        self.initial_node = nn.Sequential(nn.Linear(22, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
        self.initial_graph = nn.ModuleList(GraphLayer(hidden_dim, 11) for _ in range(2))
        self.time_projection = nn.Sequential(nn.Linear(32, 32), nn.SiLU(), nn.Linear(32, 32))
        self.token = nn.Sequential(nn.Linear(hidden_dim + 39, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
        self.blocks = nn.ModuleList(PhysicalBlock(hidden_dim, heads, dropout) for _ in range(blocks))
        if mechanism == "m1":
            self.visual = nn.ModuleList(M1Fusion(hidden_dim, visual_dim) for _ in range(blocks))
        elif mechanism == "m2":
            self.visual = nn.ModuleList(M2LocalMemory(hidden_dim, visual_dim, heads) for _ in range(blocks))
        elif mechanism == "m6":
            self.visual = M6Adapter(hidden_dim, visual_dim)
        else:
            self.visual = None
        self.local_head = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 3))
        self.com_head = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 3))
        for head in (self.local_head, self.com_head):
            nn.init.zeros_(head[-1].weight); nn.init.zeros_(head[-1].bias)

    @staticmethod
    def _time(dt, frames, dtype):
        offsets = torch.arange(1, frames + 1, device=dt.device, dtype=dtype)
        angle = math.pi * dt[:,None,None] * offsets[None,:,None] * (2.0 ** torch.arange(16, device=dt.device, dtype=dtype))[None,None]
        return torch.cat((angle.sin(), angle.cos()), -1)

    def trunk_state_dict(self):
        return {k: v for k, v in self.state_dict().items() if not k.startswith(("visual.", "dino_projection."))}

    def forward(self, x0, x1, input_mask, reference, dino, dino_valid, dt, gravity,
                floor_z, neighbour_indices, neighbour_mask, rest_edge_vectors, rest_edge_lengths):
        velocity = (x1-x0)/dt[:,None,None]
        centre = masked_mean(x1, input_mask)
        conditions = torch.cat((dt[:,None,None].expand(-1,x1.shape[1],1), gravity[:,None].expand(-1,x1.shape[1],-1), x1[...,2:3]-floor_z[:,None,None]), -1)
        nodes = torch.cat((x1,velocity,x1-centre[:,None],x0,x1-x0,input_mask[...,None],input_mask[...,None],conditions),-1)
        initial = self.initial_node(nodes)*input_mask[...,None]
        neighbours = gather_neighbours(x1,neighbour_indices)
        neighbour_v = gather_neighbours(velocity,neighbour_indices)
        vectors = neighbours-x1[:,:,None]; lengths=torch.linalg.vector_norm(vectors,dim=-1,keepdim=True)
        edges0=torch.cat((rest_edge_vectors,vectors,lengths,neighbour_v-velocity[:,:,None],lengths/rest_edge_lengths[...,None].clamp_min(1e-8)-1),-1)
        edge_mask=edge_validity(input_mask,neighbour_indices,neighbour_mask)
        for layer in self.initial_graph: initial=layer(initial,edges0,neighbour_indices,edge_mask)*input_mask[...,None]
        ballistic=ballistic_trajectory(x0,x1,gravity,dt,self.frames)
        time=self.time_projection(self._time(dt,self.frames,x0.dtype))
        token=torch.cat((initial[:,None].expand(-1,self.frames,-1,-1),ballistic,ballistic-x1[:,None],ballistic[...,2:3]-floor_z[:,None,None,None],time[:,:,None].expand(-1,-1,x1.shape[1],-1)),-1)
        hidden=self.token(token)*input_mask[:,None,:,None]
        b,t,n,_=ballistic.shape
        flat=ballistic.reshape(b*t,n,3); idx=neighbour_indices[:,None].expand(-1,t,-1,-1).reshape(b*t,n,-1)
        vec=(gather_neighbours(flat,idx).reshape(b,t,n,-1,3)-ballistic[:,:,:,None])
        leng=torch.linalg.vector_norm(vec,dim=-1,keepdim=True)
        edge_features=torch.cat((rest_edge_vectors[:,None].expand(-1,t,-1,-1,-1),vec,leng,leng/rest_edge_lengths[:,None,:,:,None].clamp_min(1e-8)-1),-1)
        visual=self.dino_projection(dino,dino_valid)
        for i, block in enumerate(self.blocks):
            hidden=block(hidden,edge_features,neighbour_indices,edge_mask,input_mask)
            if self.mechanism in {"m1","m2"}:
                hidden=self.visual[i](hidden,visual,dino_valid,neighbour_indices,neighbour_mask,rest_edge_vectors)
        if self.mechanism=="m6":
            hidden=self.visual(hidden,visual,dino_valid,neighbour_indices,neighbour_mask,rest_edge_vectors)
        pooled=masked_mean(hidden,input_mask[:,None].expand(-1,self.frames,-1),2)
        residual_com=self.com_head(pooled)
        raw=self.local_head(hidden)*input_mask[:,None,:,None]
        residual_local=(raw-masked_mean(raw,input_mask[:,None].expand(-1,self.frames,-1),2)[:,:,None])*input_mask[:,None,:,None]
        residual=(residual_com[:,:,None]+residual_local)*input_mask[:,None,:,None]
        return FullTrajectoryOutput(residual_com,residual_local,residual,ballistic,ballistic+residual)


class V41PooledTrackBSurrogate(FullTrajectorySurrogate):
    """Exact V4 Track B pooled-DINO/FiLM model with the V4.1 input contract.

    The original architecture does not consume the explicit reference tensor.
    It remains in the shared V4.1 loader/model contract for graph construction,
    normalization, and metrics, and is deliberately ignored here.
    """

    mechanism = "track_b_pooled"

    def forward(
        self, x0, x1, input_mask, reference, dino, dino_valid, dt, gravity,
        floor_z, neighbour_indices, neighbour_mask, rest_edge_vectors,
        rest_edge_lengths,
    ):
        del reference
        return super().forward(
            x0=x0,
            x1=x1,
            input_mask=input_mask,
            dino=dino,
            dino_valid=dino_valid,
            dt=dt,
            gravity=gravity,
            floor_z=floor_z,
            neighbour_indices=neighbour_indices,
            neighbour_mask=neighbour_mask,
            rest_edge_vectors=rest_edge_vectors,
            rest_edge_lengths=rest_edge_lengths,
        )


def build_v41_model(
    mechanism="m1", hidden_dim=128, blocks=4, heads=4, dropout=0.1,
    frames=59, gradient_checkpointing=True,
):
    """Build a reviewed V4.1 mechanism or the exact pooled V4 Track B bridge."""
    if mechanism == "track_b_pooled":
        return V41PooledTrackBSurrogate(
            dino_dim=384,
            dino_embed_dim=16,
            hidden_dim=hidden_dim,
            blocks=blocks,
            heads=heads,
            dropout=dropout,
            frames=frames,
            gradient_checkpointing=gradient_checkpointing,
        )
    return V41TrajectorySurrogate(
        mechanism=mechanism,
        hidden_dim=hidden_dim,
        blocks=blocks,
        heads=heads,
        dropout=dropout,
        frames=frames,
        gradient_checkpointing=gradient_checkpointing,
    )
