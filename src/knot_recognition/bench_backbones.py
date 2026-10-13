"""
bench_backbones.py - Batched graph backbones for the KnotBench evaluation.

All backbones consume the same coordinate-free signed 4-valent diagram graph
(node features [sign, is_pos, is_neg, f1..f4], edge features
[port_src, port_dst, is_over_src, is_over_dst]) and share an identical budget:
hidden width, depth, optimiser, epochs and pooling (mean || max per graph).

Backbones
---------
mpnn      Signed message passing (the original R-GNO layer).            1-WL bounded
gin       GINE-style sum aggregation with edge features (Xu et al. 2019).  1-WL maximal
gps       GraphGPS-style: local signed MPNN + global multi-head attention
          per layer (Rampasek et al. 2022).
subgraph  DS-GNN with node-marking policy (Bevilacqua et al. 2022):
          one copy of the graph per node, the root node marked; MPNN on all
          copies, pooled per copy then averaged. Strictly more expressive than 1-WL.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from knot_recognition.r_gno import SignedMessagePassingLayer

GraphT = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]  # (node_feats, edge_index, edge_feats)

BACKBONES = ("mpnn", "gin", "gps", "subgraph")


# ----------------------------------------------------------------------------
# Batching
# ----------------------------------------------------------------------------

@dataclass
class GraphBatch:
    x: torch.Tensor          # (N, F)
    edge_index: torch.Tensor  # (2, E)
    edge_attr: torch.Tensor   # (E, 4)
    batch: torch.Tensor       # (N,) graph id of each node
    num_graphs: int


def collate(graphs: Sequence[GraphT]) -> GraphBatch:
    xs, eis, eas, bs = [], [], [], []
    offset = 0
    for gi, (x, ei, ea) in enumerate(graphs):
        xs.append(x)
        eis.append(ei + offset)
        eas.append(ea)
        bs.append(torch.full((x.shape[0],), gi, dtype=torch.long))
        offset += x.shape[0]
    return GraphBatch(
        x=torch.cat(xs, 0),
        edge_index=torch.cat(eis, 1) if eis else torch.empty((2, 0), dtype=torch.long),
        edge_attr=torch.cat(eas, 0) if eas else torch.empty((0, 4)),
        batch=torch.cat(bs, 0),
        num_graphs=len(graphs),
    )


def node_marked_copies(graphs: Sequence[GraphT]) -> Tuple[GraphBatch, torch.Tensor]:
    """Expand every graph into one copy per node with a root-marker channel.

    Returns the batch of all copies and a (num_copies,) tensor mapping each copy
    to its parent graph.
    """
    xs, eis, eas, bs, parent = [], [], [], [], []
    offset, copy_id = 0, 0
    for gi, g in enumerate(graphs):
        key = id(g[0])
        hit = _COPY_CACHE.get(key)
        if hit is None or hit[0] is not g[0]:
            x, ei, ea = g
            n = x.shape[0]
            xx = torch.cat([x.repeat(n, 1), torch.eye(n, dtype=x.dtype).reshape(n * n, 1)], dim=1)
            shifts = (torch.arange(n) * n).repeat_interleave(ei.shape[1])
            eie = ei.repeat(1, n) + shifts.unsqueeze(0)
            eae = ea.repeat(n, 1)
            hit = (g[0], n, xx, eie, eae)
            _COPY_CACHE[key] = hit
        _, n, xx, eie, eae = hit
        xs.append(xx)
        eis.append(eie + offset)
        eas.append(eae)
        bs.append(torch.arange(copy_id, copy_id + n).repeat_interleave(n))
        parent.extend([gi] * n)
        offset += n * n
        copy_id += n
    gb = GraphBatch(x=torch.cat(xs, 0), edge_index=torch.cat(eis, 1), edge_attr=torch.cat(eas, 0),
                    batch=torch.cat(bs, 0), num_graphs=copy_id)
    return gb, torch.tensor(parent, dtype=torch.long)


_COPY_CACHE: dict = {}


def scatter_mean_max(h: torch.Tensor, batch: torch.Tensor, num_graphs: int) -> torch.Tensor:
    d = h.shape[1]
    s = torch.zeros((num_graphs, d), dtype=h.dtype).index_add_(0, batch, h)
    cnt = torch.zeros((num_graphs,), dtype=h.dtype).index_add_(0, batch, torch.ones_like(batch, dtype=h.dtype))
    mean = s / cnt.clamp(min=1.0).unsqueeze(1)
    mx = torch.full((num_graphs, d), -1e9, dtype=h.dtype).scatter_reduce(
        0, batch.unsqueeze(1).expand(-1, d), h, reduce="amax", include_self=True
    )
    return torch.cat([mean, mx], dim=1)


# ----------------------------------------------------------------------------
# Layers
# ----------------------------------------------------------------------------

class GINELayer(nn.Module):
    """h_i' = h_i + MLP((1+eps) h_i + sum_j ReLU(h_j + W e_ij))."""

    def __init__(self, hidden: int, edge_dim: int = 4):
        super().__init__()
        self.eps = nn.Parameter(torch.zeros(1))
        self.edge = nn.Linear(edge_dim, hidden)
        self.mlp = nn.Sequential(
            nn.Linear(hidden, hidden * 2), nn.LayerNorm(hidden * 2), nn.GELU(), nn.Linear(hidden * 2, hidden)
        )

    def forward(self, h, ei, ea, batch=None, num_graphs=None):
        if ei.shape[1] == 0:
            return h + self.mlp((1 + self.eps) * h)
        src, dst = ei
        msg = F.relu(h[src] + self.edge(ea))
        agg = torch.zeros_like(h).index_add_(0, dst, msg)
        return h + self.mlp((1 + self.eps) * h + agg)


class MPNNLayer(nn.Module):
    def __init__(self, hidden: int, edge_dim: int = 4):
        super().__init__()
        self.inner = SignedMessagePassingLayer(hidden, edge_in_dim=edge_dim)

    def forward(self, h, ei, ea, batch=None, num_graphs=None):
        return self.inner(h, ei, ea)


class GPSLayer(nn.Module):
    """Local signed MPNN + global multi-head self-attention within each graph + FFN."""

    def __init__(self, hidden: int, edge_dim: int = 4, heads: int = 4):
        super().__init__()
        self.local = SignedMessagePassingLayer(hidden, edge_in_dim=edge_dim)
        self.attn = nn.MultiheadAttention(hidden, heads, batch_first=True)
        self.norm_attn = nn.LayerNorm(hidden)
        self.ffn = nn.Sequential(nn.Linear(hidden, hidden * 2), nn.GELU(), nn.Linear(hidden * 2, hidden))
        self.norm_ffn = nn.LayerNorm(hidden)

    def forward(self, h, ei, ea, batch, num_graphs):
        h_local = self.local(h, ei, ea)
        # Dense padding per graph for attention
        counts = torch.bincount(batch, minlength=num_graphs)
        max_n = int(counts.max().item())
        pos = torch.zeros_like(batch)
        starts = torch.cumsum(counts, 0) - counts
        pos = torch.arange(h.shape[0]) - starts[batch]
        dense = torch.zeros((num_graphs, max_n, h.shape[1]), dtype=h.dtype)
        dense[batch, pos] = h
        pad_mask = torch.ones((num_graphs, max_n), dtype=torch.bool)
        pad_mask[batch, pos] = False
        att, _ = self.attn(dense, dense, dense, key_padding_mask=pad_mask, need_weights=False)
        h_global = att[batch, pos]
        h = self.norm_attn(h_local + h_global)
        return self.norm_ffn(h + self.ffn(h))


# ----------------------------------------------------------------------------
# Encoders
# ----------------------------------------------------------------------------

class GraphEncoder(nn.Module):
    """Backbone -> per-graph pooled representation of size 2*hidden."""

    def __init__(self, backbone: str = "mpnn", in_dim: int = 7, hidden: int = 64, layers: int = 4):
        super().__init__()
        if backbone not in BACKBONES:
            raise ValueError(f"unknown backbone {backbone}")
        self.backbone = backbone
        self.subgraph = backbone == "subgraph"
        eff_in = in_dim + (1 if self.subgraph else 0)
        self.embed = nn.Sequential(nn.Linear(eff_in, hidden), nn.LayerNorm(hidden), nn.GELU())
        layer_cls = {"mpnn": MPNNLayer, "gin": GINELayer, "gps": GPSLayer, "subgraph": MPNNLayer}[backbone]
        self.layers = nn.ModuleList([layer_cls(hidden) for _ in range(layers)])
        self.out_dim = hidden * 2

    def forward(self, graphs: Sequence[GraphT]) -> torch.Tensor:
        if self.subgraph:
            gb, parent = node_marked_copies(graphs)
        else:
            gb, parent = collate(graphs), None
        h = self.embed(gb.x)
        for layer in self.layers:
            h = layer(h, gb.edge_index, gb.edge_attr, gb.batch, gb.num_graphs)
        pooled = scatter_mean_max(h, gb.batch, gb.num_graphs)
        if parent is not None:
            # average copy representations per original graph
            g = len(graphs)
            s = torch.zeros((g, pooled.shape[1]), dtype=pooled.dtype).index_add_(0, parent, pooled)
            cnt = torch.bincount(parent, minlength=g).clamp(min=1).to(pooled.dtype)
            pooled = s / cnt.unsqueeze(1)
        return pooled


class RegressionModel(nn.Module):
    def __init__(self, backbone: str, num_targets: int, hidden: int = 64, layers: int = 4):
        super().__init__()
        self.enc = GraphEncoder(backbone, hidden=hidden, layers=layers)
        self.head = nn.Sequential(nn.Linear(self.enc.out_dim, hidden), nn.ReLU(), nn.Linear(hidden, num_targets))

    def forward(self, graphs):
        return self.head(self.enc(graphs))


class EmbeddingModel(nn.Module):
    """Unit-norm embedding for verification; optional classifier head for supervised ID training."""

    def __init__(self, backbone: str, out_dim: int = 64, num_classes: int = 0, hidden: int = 64, layers: int = 4):
        super().__init__()
        self.enc = GraphEncoder(backbone, hidden=hidden, layers=layers)
        self.proj = nn.Linear(self.enc.out_dim, out_dim)
        self.cls = nn.Linear(out_dim, num_classes) if num_classes > 0 else None

    def embed(self, graphs):
        return F.normalize(self.proj(self.enc(graphs)), dim=-1)

    def forward(self, graphs):
        z = self.embed(graphs)
        return z, (self.cls(z) if self.cls is not None else None)


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
