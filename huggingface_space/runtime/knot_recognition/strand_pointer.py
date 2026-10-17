"""
strand_pointer.py - The network learns the order in which a strand visits its pixels.

The parser orders a strand with hand-written rules: it traces the skeleton pixel by pixel,
bridges short gaps, and at each crossing pairs the four arms by straightness. Here the network
learns the order itself.

Points are placed along the strand centre-lines, about every ``spacing`` pixels of the 256 x 256
input. Around every crossing a disc of radius R = 0.9 w + 1.5 (the parser's cut; w = cable
width) is removed from both strands, so each crossing leaves four arm ends. A small transformer
reads the UNetFormer features at every point and gives, for each point i, a distribution over
all other points j plus a 'none' slot: the probability that j is next to i along the strand.
Reading order and strand order differ:

      1 2 3        points numbered in reading (raster) order. The strand visits 1, 2, 3, 6,
      4 5 6        passes under the strand that covers 5, and comes out at 4: next(6) = 4, not 5.

An image does not show which way a strand runs, so the target is symmetric: each point's two
neighbours along its strand get probability 1/2 each. A strand end (an open cable) gives 1/2
to its one neighbour and 1/2 to 'none'; a point off the cable (clutter) gives 1 to 'none'.
Following the pointers from any point returns the strand in order, across the crossings.

Targets come from the exact strand paths of the renders (scripts/make_strand_sidecar.py).
Points come either from the true paths (jittered) or from the network's own centre-line ridge,
labelled by the nearest true path, so the head also trains on what it sees at test time.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

INPUT = 256
SPACING = 4.0          # point spacing along a strand, pixels of the 256 input
MAX_LINK = 64.0        # longest link the head can predict (a crossing jump is about 2R + spacing; R <= 22 px)


def cut_radius(width: float) -> float:
    """The parser's crossing cut, R = 0.9 w + 3 pixels of a 512 image, on the 256 input grid."""
    return 0.9 * float(width) + 1.5


# --------------------------------------------------------------------------- points and targets
@dataclass
class Points:
    xy: np.ndarray                      # (N, 2) float32, x right, y down, on the 256 grid
    comp: np.ndarray                    # (N,) strand component, -1 = off the cable
    s: np.ndarray                       # (N,) arc length along the component (nan off the cable)
    edges: List[Tuple[int, int, bool]]  # consecutive points along a strand; True = across a crossing
    spacing: float = SPACING
    source: str = "true"                # "true" (sampled on the true paths) or "ridge" (predicted)

    @property
    def n(self) -> int:
        return len(self.xy)

    def neighbours(self) -> List[List[int]]:
        nb: List[List[int]] = [[] for _ in range(self.n)]
        for i, j, _ in self.edges:
            nb[i].append(j)
            nb[j].append(i)
        return nb


def _arc(P: np.ndarray, closed: bool):
    Q = np.vstack([P, P[:1]]) if closed else P
    cum = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(Q, axis=0), axis=1))])
    return Q, cum


def _at(Q: np.ndarray, cum: np.ndarray, s: np.ndarray) -> np.ndarray:
    return np.stack([np.interp(s, cum, Q[:, 0]), np.interp(s, cum, Q[:, 1])], 1)


def _crossing_xy(rec) -> np.ndarray:
    return np.array([c["pos"] for c in rec["crossings"]], np.float64).reshape(-1, 2)


def _chain(idx: np.ndarray, s: np.ndarray, xy: np.ndarray, L: float, closed: bool, Q, cum,
           C: np.ndarray, R: float, max_link: float) -> List[Tuple[int, int, bool]]:
    """Edges between consecutive points of one strand (sorted by arc length)."""
    idx = idx[np.argsort(s[idx], kind="stable")]
    pairs = list(zip(idx[:-1], idx[1:]))
    if closed and len(idx) >= 3:
        pairs.append((idx[-1], idx[0]))
    out = []
    for i, j in pairs:
        if np.linalg.norm(xy[i] - xy[j]) > max_link:
            continue                            # out of reach: both become strand ends
        a, b = s[i], s[j]
        if b < a:                               # the closing edge of a loop wraps around
            b += L
        mid = _at(Q, cum, np.array([((a + b) / 2) % L if closed else (a + b) / 2]))[0]
        jump = bool(len(C)) and bool((np.linalg.norm(C - mid, axis=1) < R).any())
        out.append((int(i), int(j), jump))
    return out


def true_points(rec: Dict[str, Any], spacing: float = SPACING, jitter: float = 0.0,
                rng: Optional[np.random.Generator] = None, max_link: float = MAX_LINK) -> Points:
    """Points every ``spacing`` px along the true strand paths, with the crossing discs removed."""
    R = cut_radius(rec["width"])
    C = _crossing_xy(rec)
    xy_l, comp_l, s_l, meta = [], [], [], []
    for ci, P in enumerate(rec["components"]):
        closed = bool(rec["closed"][ci])
        Q, cum = _arc(np.asarray(P, np.float64), closed)
        L = float(cum[-1])
        off = float(rng.uniform(0, spacing)) if rng is not None else 0.5 * spacing
        s = np.arange(off, L, spacing)
        if len(s) == 0:
            continue
        xy = _at(Q, cum, s)
        keep = np.ones(len(s), bool) if not len(C) else (np.linalg.norm(xy[:, None] - C[None], axis=2) > R).all(1)
        keep &= (xy >= 0).all(1) & (xy < INPUT).all(1)
        xy_l.append(xy[keep]); s_l.append(s[keep]); comp_l.append(np.full(int(keep.sum()), ci))
        meta.append((ci, L, closed, Q, cum))
    xy = np.concatenate(xy_l) if xy_l else np.zeros((0, 2))
    s = np.concatenate(s_l) if s_l else np.zeros(0)
    comp = np.concatenate(comp_l) if comp_l else np.zeros(0, int)
    if jitter > 0 and rng is not None and len(xy):       # before linking: reach is checked on the final points
        xy = xy + rng.normal(0, jitter, xy.shape)
    edges: List[Tuple[int, int, bool]] = []
    for ci, L, closed, Q, cum in meta:
        idx = np.nonzero(comp == ci)[0]
        edges += _chain(idx, s, xy, L, closed, Q, cum, C, R, max_link)
    return Points(xy.astype(np.float32), comp.astype(int), s, edges, spacing, "true")


def _skeleton(prob: np.ndarray, width: float, thr: float = 0.5) -> np.ndarray:
    """The parser's skeleton on the 256 ridge: threshold, drop specks, skeletonise, prune spurs."""
    import cv2
    from skimage.morphology import skeletonize

    from .cable.neural_parser import _prune

    band = prob > thr
    n, lab, st, _ = cv2.connectedComponentsWithStats(band.astype(np.uint8), connectivity=8)
    keep = np.zeros(n, bool)
    keep[1:] = st[1:, cv2.CC_STAT_AREA] >= (0.6 * width) ** 2
    sk = skeletonize(keep[lab])
    k = max(2, int(0.8 * width))
    return _prune(_prune(sk, k), k)


def ridge_points(prob: np.ndarray, crossings: np.ndarray, width: float, spacing: float = SPACING,
                 thr: float = 0.5) -> np.ndarray:
    """Points on the predicted centre-line ridge (256 grid): skeleton pixels at least 0.8 spacing
    apart, with the crossing discs removed. Returns (N, 2) pixel centres (x, y)."""
    sk = _skeleton(prob, width, thr)
    C = np.asarray(crossings, np.float64).reshape(-1, 2)
    R = cut_radius(width)
    ys, xs = np.nonzero(sk)
    P = np.stack([xs + 0.5, ys + 0.5], 1)
    if len(C) and len(P):
        P = P[(np.linalg.norm(P[:, None] - C[None], axis=2) > R).all(1)]
    r = 0.8 * spacing
    ri = int(math.ceil(r))
    dy, dx = np.mgrid[-ri:ri + 1, -ri:ri + 1]
    disc = dy ** 2 + dx ** 2 <= r * r
    dy, dx = dy[disc], dx[disc]
    blocked = np.zeros((INPUT + 2 * ri, INPUT + 2 * ri), bool)
    out = []
    for x, y in P:
        i, j = int(y) + ri, int(x) + ri
        if blocked[i, j]:
            continue
        out.append((x, y))
        blocked[i + dy, j + dx] = True
    return np.array(out, np.float32).reshape(-1, 2)


def label_points(xy: np.ndarray, rec: Dict[str, Any], spacing: float = SPACING, tol: Optional[float] = None,
                 max_link: float = MAX_LINK, source: str = "ridge") -> Points:
    """True strand order for arbitrary points: each point takes the arc position of the nearest
    true path (off the cable if farther than ``tol``); consecutive points along a strand are
    neighbours, which across a crossing disc is the next arm of the same strand."""
    from scipy.spatial import cKDTree

    R = cut_radius(rec["width"])
    C = _crossing_xy(rec)
    tol = tol if tol is not None else float(np.clip(0.5 * rec["width"], 1.5, 4.0))
    dense, dcomp, ds, meta = [], [], [], []
    for ci, P in enumerate(rec["components"]):
        closed = bool(rec["closed"][ci])
        Q, cum = _arc(np.asarray(P, np.float64), closed)
        s = np.arange(0, cum[-1], 0.5)
        dense.append(_at(Q, cum, s)); dcomp.append(np.full(len(s), ci)); ds.append(s)
        meta.append((ci, float(cum[-1]), closed, Q, cum))
    xy = np.asarray(xy, np.float64).reshape(-1, 2)
    if not dense or not len(xy):
        return Points(xy.astype(np.float32), np.full(len(xy), -1), np.full(len(xy), np.nan), [], spacing, source)
    D, dcomp, ds = np.concatenate(dense), np.concatenate(dcomp), np.concatenate(ds)
    dist, k = cKDTree(D).query(xy)
    comp = np.where(dist <= tol, dcomp[k], -1)
    s = np.where(comp >= 0, ds[k], np.nan)
    edges: List[Tuple[int, int, bool]] = []
    for ci, L, closed, Q, cum in meta:
        idx = np.nonzero(comp == ci)[0]
        edges += _chain(idx, s, xy, L, closed, Q, cum, C, R, max_link)
    return Points(xy.astype(np.float32), comp.astype(int), s, edges, spacing, source)


def with_distractors(pts: Points, extra_xy: np.ndarray) -> Points:
    """Append off-cable points (target: 'none')."""
    extra_xy = np.asarray(extra_xy, np.float32).reshape(-1, 2)
    if not len(extra_xy):
        return pts
    return Points(np.concatenate([pts.xy, extra_xy]), np.concatenate([pts.comp, np.full(len(extra_xy), -1)]),
                  np.concatenate([pts.s, np.full(len(extra_xy), np.nan)]), list(pts.edges), pts.spacing, pts.source)


def load_records(path) -> List[Dict[str, Any]]:
    """Strand records (scripts/make_strand_sidecar.py), in the grid_sample pixel convention.
    Files written before the convention was recorded hold OpenCV coordinates of 512 renders
    scaled by 1/2; they are shifted by (0.5 px) / 2 here."""
    import pickle

    recs = pickle.loads(Path(path).read_bytes())
    for r in recs:
        if r.get("coords") != "edge":
            r["components"] = [np.asarray(P, np.float32) + 0.25 for P in r["components"]]
            r["crossings"] = [{**c, "pos": [c["pos"][0] + 0.25, c["pos"][1] + 0.25]} for c in r["crossings"]]
            r["coords"] = "edge"
    return recs


def transform_xy(P, hflip: bool, rot: int, size: int = INPUT) -> np.ndarray:
    """Points (..., 2) under the image transform ``torch.rot90(x.flip(-1) if hflip else x, rot)``
    (x right, y down): a flip maps (x, y) -> (S - x, y), a quarter turn (x, y) -> (y, S - x)."""
    P = np.asarray(P, np.float64).copy()
    if hflip:
        P[..., 0] = size - P[..., 0]
    for _ in range(rot % 4):
        P = np.stack([P[..., 1], size - P[..., 0]], -1)
    return P


def transform_record(rec: Dict[str, Any], hflip: bool, rot: int, size: int = INPUT) -> Dict[str, Any]:
    """The strand record under the same image transform as transform_xy."""
    f = lambda P: transform_xy(P, hflip, rot, size)
    out = dict(rec)
    out["components"] = [f(P).astype(np.float32) for P in rec["components"]]
    out["crossings"] = [{**c, "pos": f(np.array(c["pos"])).tolist()} for c in rec["crossings"]]
    return out


def targets(batch: Sequence[Points], jump_weight: float = 3.0, max_link: float = MAX_LINK):
    """Padded tensors: xy (B, N, 2), valid (B, N), target (B, N, N + 1) with the last column
    'none', and a per-point weight (points next to a crossing jump count ``jump_weight`` times).
    A neighbour beyond the head's reach counts as 'none' (the head masks it)."""
    B, N = len(batch), max(1, max(p.n for p in batch))
    xy = torch.zeros(B, N, 2)
    valid = torch.zeros(B, N, dtype=torch.bool)
    tgt = torch.zeros(B, N, N + 1)
    wgt = torch.zeros(B, N)
    for b, p in enumerate(batch):
        xy[b, : p.n] = torch.from_numpy(p.xy)
        valid[b, : p.n] = True
        nb = p.neighbours()
        jumpers = {i for e in p.edges if e[2] for i in e[:2]}
        for i in range(p.n):
            js = [j for j in nb[i][:2] if j != i and np.linalg.norm(p.xy[i] - p.xy[j]) <= max_link]
            if not js:
                tgt[b, i, N] = 1.0
            else:
                for j in js:
                    tgt[b, i, j] += 0.5
                if len(js) == 1:
                    tgt[b, i, N] += 0.5
            wgt[b, i] = jump_weight if i in jumpers else 1.0
    return xy, valid, tgt, wgt


def neighbour_index(batch: Sequence[Points], jump_weight: float = 3.0, max_link: float = MAX_LINK, pad_to: int = 1):
    """The targets of ``targets`` in compact form, cheap to move between processes: xy (B, N, 2),
    valid (B, N), nbr (B, N, 2) indices of each point's neighbours (-1 = the 'none' slot), weight
    (B, N). ``target_from_index`` rebuilds the (B, N, N + 1) target on the device. ``pad_to`` rounds N
    up (a few fixed shapes let the GPU reuse its memory between steps)."""
    B, N = len(batch), max(1, max(p.n for p in batch))
    N = -(-N // pad_to) * pad_to
    xy = torch.zeros(B, N, 2)
    valid = torch.zeros(B, N, dtype=torch.bool)
    nbr = torch.full((B, N, 2), -1, dtype=torch.long)
    wgt = torch.zeros(B, N)
    for b, p in enumerate(batch):
        xy[b, : p.n] = torch.from_numpy(np.asarray(p.xy, np.float32))
        valid[b, : p.n] = True
        nb = p.neighbours()
        jumpers = {i for e in p.edges if e[2] for i in e[:2]}
        for i in range(p.n):
            js = [j for j in nb[i][:2] if j != i and np.linalg.norm(p.xy[i] - p.xy[j]) <= max_link]
            for k, j in enumerate(js):
                nbr[b, i, k] = j
            wgt[b, i] = jump_weight if i in jumpers else 1.0
    return xy, valid, nbr, wgt


def target_from_index(nbr: torch.Tensor) -> torch.Tensor:
    """(B, N, 2) neighbour indices -> (B, N, N + 1) target: 1/2 per neighbour slot, an empty slot
    counts for 'none' (the last column)."""
    B, N, _ = nbr.shape
    idx = torch.where(nbr >= 0, nbr, torch.full_like(nbr, N))
    tgt = torch.zeros(B, N, N + 1, device=nbr.device)
    return tgt.scatter_add_(2, idx, torch.full(idx.shape, 0.5, device=nbr.device))


# --------------------------------------------------------------------------- the head
@dataclass
class PointerConfig:
    grids: Tuple[int, ...] = (16, 32, 64, 128, 256)       # UNetFormer feature maps read at each point
    feat_dims: Tuple[int, ...] = (192, 128, 64, 32, 16)
    d: int = 128
    depth: int = 3
    heads: int = 4
    n_freq: int = 8
    spacing: float = SPACING
    max_link: float = MAX_LINK
    tangent: bool = True          # local strand direction from neighbouring points, for the pair cues
    ends: bool = True             # signed open-end direction: which way a strand end points
    radius: bool = True           # the crossing cut radius R is an input (jump lengths scale with it)
    candidates: int = 64          # score each point against its nearest K points only (0 = all). The true
                                  # neighbour is outside the 64 nearest for < 0.05% of links (13.8)


def point_tangents(xy: torch.Tensor, valid: torch.Tensor, radius: float):
    """Axial direction of the strand at each point, from the points within ``radius``:
    (cos t, sin t) of the principal axis (sign-free) and its coherence in [0, 1]
    (1 = all neighbours on a line, 0 = no direction, e.g. an isolated clutter point)."""
    rel = xy[:, None, :, :] - xy[:, :, None, :]                     # p_j - p_i
    w = ((rel.norm(dim=-1) <= radius) & valid[:, None, :] & valid[:, :, None]).float()
    a = (w * rel[..., 0] ** 2).sum(-1)
    b = (w * rel[..., 0] * rel[..., 1]).sum(-1)
    c = (w * rel[..., 1] ** 2).sum(-1)
    th = 0.5 * torch.atan2(2 * b, a - c)
    coh = torch.sqrt((a - c) ** 2 + 4 * b ** 2) / (a + c).clamp_min(1e-6)
    return torch.stack([th.cos(), th.sin()], -1), coh * ((a + c) > 0)


def open_ends(xy: torch.Tensor, valid: torch.Tensor, radius: float) -> torch.Tensor:
    """Signed direction in which a strand is open at each point: minus the mean unit vector to the
    points within ``radius``. About 0 inside a strand (neighbours on both sides), a unit vector
    pointing out of a strand end (one neighbour), 0 for an isolated point."""
    rel = xy[:, None, :, :] - xy[:, :, None, :]
    d = rel.norm(dim=-1)
    w = ((d <= radius) & (d > 0) & valid[:, None, :] & valid[:, :, None]).float()
    u = rel / d.clamp_min(1e-6)[..., None]
    return -(w[..., None] * u).sum(2) / w.sum(2).clamp_min(1.0)[..., None]


class _Block(nn.Module):
    def __init__(self, d: int, h: int):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, h, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))

    def forward(self, x, bias):
        y = self.n1(x)
        x = x + self.attn(y, y, y, attn_mask=bias, need_weights=False)[0]
        return x + self.mlp(self.n2(x))


class StrandPointerHead(nn.Module):
    """Points -> for each point, logits over every other point (its neighbours along the strand)
    and a final 'none' slot. Equivariant to the order in which points are listed."""

    def __init__(self, cfg: Optional[PointerConfig] = None):
        super().__init__()
        self.cfg = cfg = cfg or PointerConfig()
        din = sum(cfg.feat_dims) + 4 * cfg.n_freq + (3 if cfg.tangent else 0) + (3 if cfg.ends else 0) + (1 if cfg.radius else 0)
        self.embed = nn.Sequential(nn.Linear(din, cfg.d), nn.GELU(), nn.Linear(cfg.d, cfg.d))
        self.blocks = nn.ModuleList([_Block(cfg.d, cfg.heads) for _ in range(cfg.depth)])
        # attention falls off with distance, -a_h * dist / spacing; heads start from near to global
        a0 = torch.logspace(math.log10(0.5), math.log10(0.01), cfg.heads)
        self.dist_a = nn.Parameter(torch.log(torch.expm1(a0)).repeat(cfg.depth, 1))
        self.norm = nn.LayerNorm(cfg.d)
        self.q, self.k = nn.Linear(cfg.d, cfg.d), nn.Linear(cfg.d, cfg.d)
        self.u = nn.Linear(cfg.d, 8)
        npair = 8 + 8 + 4 + (5 if cfg.tangent else 0) + (2 if cfg.ends else 0) + (1 if cfg.radius else 0)
        self.pair = nn.Sequential(nn.Linear(npair, 32), nn.GELU(), nn.Linear(32, 1))
        self.none = nn.Linear(cfg.d, 1)
        self.register_buffer("freq", (2.0 ** torch.arange(cfg.n_freq)) * math.pi, persistent=False)

    def _fourier(self, xy):
        p = (xy / INPUT)[..., None] * self.freq                       # (B, N, 2, F)
        return torch.cat([p.sin(), p.cos()], -1).flatten(-2)

    def sample(self, feats: Dict[int, torch.Tensor], xy: torch.Tensor) -> torch.Tensor:
        g = (xy / INPUT * 2 - 1)[:, :, None, :]                         # align_corners=False
        return torch.cat([F.grid_sample(feats[s], g, mode="bilinear", align_corners=False).squeeze(-1).transpose(1, 2)
                          for s in self.cfg.grids], -1)

    def forward(self, feats: Dict[int, torch.Tensor], xy: torch.Tensor, valid: torch.Tensor,
                radius: Optional[torch.Tensor] = None) -> torch.Tensor:
        """``radius``: (B,) crossing cut radius of each image (cut_radius of the cable width),
        required when the head was built with ``radius``."""
        cfg = self.cfg
        B, N, _ = xy.shape
        if cfg.radius:
            if radius is None:
                raise ValueError("this head needs the crossing cut radius of each image")
            radius = torch.as_tensor(radius, dtype=xy.dtype, device=xy.device).reshape(B)
        node = [self.sample(feats, xy), self._fourier(xy)]
        if cfg.tangent:
            t, coh = point_tangents(xy, valid, 1.6 * cfg.spacing)
            t2 = torch.stack([t[..., 0] ** 2 - t[..., 1] ** 2, 2 * t[..., 0] * t[..., 1]], -1)   # doubled angle
            node.append(torch.cat([t2 * coh[..., None], coh[..., None]], -1))
        if cfg.ends:
            o = open_ends(xy, valid, 1.6 * cfg.spacing)
            node.append(torch.cat([o, o.norm(dim=-1, keepdim=True)], -1))
        if cfg.radius:
            node.append((radius / (4 * cfg.spacing))[:, None, None].expand(B, N, 1))
        h = self.embed(torch.cat(node, -1))
        dist = (xy[:, None, :, :] - xy[:, :, None, :]).norm(dim=-1)    # dist[b, i, j] = |p_j - p_i|
        keyoff = ~valid[:, None, None, :]
        for li, blk in enumerate(self.blocks):
            a = F.softplus(self.dist_a[li])[None, :, None, None]
            bias = (-a * (dist / cfg.spacing)[:, None]).masked_fill(keyoff, -1e4)
            h = blk(h, bias.reshape(B * cfg.heads, N, N))
        h = self.norm(h)
        qk = self.q(h) @ self.k(h).transpose(1, 2) / math.sqrt(cfg.d)
        u = self.u(h)
        eye = torch.eye(N, dtype=torch.bool, device=xy.device)[None]
        impossible = eye | (dist > cfg.max_link) | ~valid[:, None, :]
        # the pairs that are scored: every point (dense), or each point's nearest candidates
        sparse = 0 < cfg.candidates < N
        if sparse:
            dk, J = dist.masked_fill(impossible, float("inf")).topk(cfg.candidates, dim=-1, largest=False)
            ok = torch.isfinite(dk)
        else:
            J = torch.arange(N, device=xy.device)[None, None].expand(B, N, N)
            ok = ~impossible
        bi = torch.arange(B, device=xy.device)[:, None, None]
        K = J.shape[-1]
        rel = xy[bi, J] - xy[:, :, None, :]                               # p_j - p_i, (B, N, K, 2)
        dj = rel.norm(dim=-1)
        geo = [rel[..., 0] / cfg.max_link, rel[..., 1] / cfg.max_link, dj / cfg.max_link, torch.log1p(dj / cfg.spacing)]
        if cfg.tangent:
            # does j lie straight on along i's strand, and do the two strands line up? (sign-free)
            uij = rel / dj.clamp_min(1e-6)[..., None]
            ti, tj = t[:, :, None, :], t[bi, J]
            geo += [(ti * uij).sum(-1).abs(), (tj * uij).sum(-1).abs(), (ti * tj).sum(-1).abs(),
                    coh[:, :, None].expand(B, N, K), coh[bi, J]]
        if cfg.ends:
            # is j ahead of i's open end, and does j's open end face back at i?
            uij = rel / dj.clamp_min(1e-6)[..., None]
            geo += [(o[:, :, None, :] * uij).sum(-1), -(o[bi, J] * uij).sum(-1)]
        if cfg.radius:
            geo += [dj / (2 * radius[:, None, None])]
        geo = torch.stack(geo, -1)
        pair_in = torch.cat([u[:, :, None].expand(B, N, K, 8), u[bi, J], geo], -1)
        scored = (torch.gather(qk, 2, J) + self.pair(pair_in).squeeze(-1)).masked_fill(~ok, -1e4)
        if sparse:
            logits = torch.full((B, N, N), -1e4, dtype=scored.dtype, device=xy.device).scatter(2, J, scored)
        else:
            logits = scored
        return torch.cat([logits, self.none(h)], -1)


def pointer_loss(logits, tgt, valid, wgt) -> torch.Tensor:
    """Soft cross-entropy between each point's pointer distribution and its true neighbours. A true
    neighbour the head cannot choose (not among the candidates it scored) counts as 'none', as a
    neighbour out of reach does; otherwise its target would sit on a masked logit."""
    blocked = logits[..., :-1] <= -1e3
    if bool((tgt[..., :-1] * blocked).any()):
        moved = (tgt[..., :-1] * blocked).sum(-1)
        tgt = torch.cat([tgt[..., :-1].masked_fill(blocked, 0.0), tgt[..., -1:] + moved[..., None]], -1)
    nll = -(tgt * F.log_softmax(logits, -1)).sum(-1)
    w = wgt * valid
    return (nll * w).sum() / w.sum().clamp_min(1.0)


def save_head(head: StrandPointerHead, path, extra: Optional[Dict[str, Any]] = None) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sd = {k: v.detach().cpu() for k, v in head.state_dict().items()}     # portable across cpu / mps / cuda
    torch.save({"config": asdict(head.cfg), "state_dict": sd, "extra": extra or {}}, path)


def load_head(path, map_location="cpu") -> StrandPointerHead:
    ck = torch.load(path, map_location=map_location, weights_only=False)
    cfg = {"tangent": False, "ends": False, "radius": False, "candidates": 0, **ck["config"]}   # heads saved before these
    head = StrandPointerHead(PointerConfig(**{k: tuple(v) if isinstance(v, list) else v for k, v in cfg.items()}))
    head.load_state_dict(ck["state_dict"])
    head.extra = ck.get("extra", {})
    return head.eval()


# --------------------------------------------------------------------------- decoding
def decode(prob: np.ndarray, thr: float = 0.15) -> List[Tuple[int, int]]:
    """Pointer probabilities (N, N + 1) -> strand edges. Each point gets at most two neighbours;
    pairs are taken greedily by their mutual score (P[i, j] + P[j, i]) / 2."""
    N = prob.shape[0]
    if N < 2:
        return []
    P = prob[:, :N]
    S = 0.5 * (P + P.T)
    iu, ju = np.triu_indices(N, 1)
    sc = S[iu, ju]
    order = np.argsort(-sc, kind="stable")
    deg = np.zeros(N, int)
    out = []
    for o in order:
        if sc[o] < thr:
            break
        i, j = int(iu[o]), int(ju[o])
        if deg[i] < 2 and deg[j] < 2:
            out.append((i, j))
            deg[i] += 1
            deg[j] += 1
    return out


def walks(edges: Sequence[Tuple[int, int]], n: int) -> List[Dict[str, Any]]:
    """Edges (each point degree <= 2) -> strands in order: open paths first, then closed loops."""
    adj: List[List[int]] = [[] for _ in range(n)]
    for i, j in edges:
        adj[i].append(j)
        adj[j].append(i)
    seen = np.zeros(n, bool)
    out = []

    def run(start):
        seq, prev, cur = [start], -1, start
        seen[start] = True
        while True:
            nxt = [v for v in adj[cur] if v != prev and not seen[v]]
            if not nxt:
                return seq, len(seq) > 2 and start in adj[cur]
            prev, cur = cur, nxt[0]
            seen[cur] = True
            seq.append(cur)

    for v in range(n):
        if not seen[v] and len(adj[v]) == 1:
            seq, _ = run(v)
            out.append({"order": seq, "closed": False})
    for v in range(n):
        if not seen[v] and adj[v]:
            seq, closed = run(v)
            out.append({"order": seq, "closed": closed})
    return out


def straightest(xy: np.ndarray, width: float, spacing: float = SPACING, max_link: float = MAX_LINK,
                crossings: Optional[np.ndarray] = None, band: Optional[np.ndarray] = None) -> List[Tuple[int, int]]:
    """Baseline without learning, the parser's rules on the same points.
    (1) Chain nearby points, closest first (each point at most two links). With ``band`` (the
        predicted ridge band, 256 x 256 bool), a link must stay on it, as the parser's skeleton
        tracing does: two strands that run close, or a strand that doubles back, stay apart.
    (2) At each crossing (``crossings``, (K, 2)), take the free ends that face it within the cut
        radius plus 1.5 spacings; with four ends, keep the one of the three pairings whose strands
        run straightest (the parser's rule); with two, join them.
    (3) Join the remaining free ends whose strands run straightest (both end tangents within 45
        degrees of the joining segment), best first."""
    xy = np.asarray(xy, np.float64)
    N = len(xy)
    if N < 2:
        return []
    D = np.linalg.norm(xy[:, None] - xy[None], axis=2)
    iu, ju = np.triu_indices(N, 1)
    near = D[iu, ju] <= 1.5 * spacing
    cand = sorted(zip(D[iu, ju][near], iu[near], ju[near]))
    deg = np.zeros(N, int)
    adj: List[List[int]] = [[] for _ in range(N)]
    out = []

    def link(i, j):
        out.append((int(i), int(j))); deg[i] += 1; deg[j] += 1; adj[i].append(j); adj[j].append(i)

    def on_band(i, j):
        if band is None:
            return True
        f = np.linspace(0.0, 1.0, 9)[:, None]
        q = np.clip((xy[i] * (1 - f) + xy[j] * f).astype(int), 0, np.array(band.shape[::-1]) - 1)
        return band[q[:, 1], q[:, 0]].mean() >= 0.9

    for _, i, j in cand:
        if deg[i] < 2 and deg[j] < 2 and on_band(i, j):
            link(i, j)
    ends = [i for i in range(N) if deg[i] == 1]
    tan = {i: (xy[i] - xy[adj[i][0]]) / max(np.linalg.norm(xy[i] - xy[adj[i][0]]), 1e-6) for i in ends}

    def straight(i, j):                       # 2 = perfectly straight through, < 1.4 = a turn
        u = (xy[j] - xy[i]) / max(D[i, j], 1e-6)
        return float(tan[i] @ u) + float(-tan[j] @ u)

    R = cut_radius(width)
    C = np.asarray(crossings if crossings is not None else np.zeros((0, 2)), np.float64).reshape(-1, 2)
    for c in C:
        free = [i for i in ends if deg[i] == 1 and np.linalg.norm(xy[i] - c) <= R + 1.5 * spacing
                and float(tan[i] @ (c - xy[i])) > 0]                    # the arm points into the disc
        if len(free) == 4:
            a, b, cc, d = free
            best = max(((a, b, cc, d), (a, cc, b, d), (a, d, b, cc)),
                       key=lambda p: straight(p[0], p[1]) + straight(p[2], p[3]))
            link(best[0], best[1]); link(best[2], best[3])
        elif len(free) == 2 and straight(*free) > 1.0:
            link(*free)
    reach = min(max_link, 2 * R + 3 * spacing)
    sc = []
    ends = [i for i in ends if deg[i] == 1]
    for a in range(len(ends)):
        for b in range(a + 1, len(ends)):
            i, j = ends[a], ends[b]
            if D[i, j] > reach or j in adj[i]:
                continue
            u = (xy[j] - xy[i]) / max(D[i, j], 1e-6)
            ci, cj = float(tan[i] @ u), float(-tan[j] @ u)
            if ci > 0.7 and cj > 0.7:
                sc.append((ci + cj, i, j))
    for _, i, j in sorted(sc, reverse=True):
        if deg[i] < 2 and deg[j] < 2:
            link(i, j)
    return out


JUMP_CHECKS = {"balanced": (70.0, 0.3, 0.0), "strict": (40.0, 0.4, 0.1)}   # (max angle, ridge mean, ridge min), fixed on dev


def check_jumps(edges: Sequence[Tuple[int, int]], xy: np.ndarray, ridge: np.ndarray, spacing: float = SPACING,
                max_angle: float = 70.0, ridge_mean: float = 0.3, ridge_min: float = 0.0):
    """Re-check every jump (a link longer than two spacings) by a separate, non-jump linear test:
    walk the straight segment through the crossing and require that
      direction: each end's strand arrives pointing along the segment (angle to the segment, from
                 the end's other neighbour, at most ``max_angle``), and
      ridge:     the centre-line ridge (256 grid; trained to run through crossings, under the top
                 strand too) stays high along it (mean >= ``ridge_mean``, min >= ``ridge_min``).
    Returns (kept, rejected) edges; a rejected jump leaves its two ends open (the parser abstains
    there rather than read a wrong strand)."""
    xy = np.asarray(xy, np.float64)
    adj: Dict[int, List[int]] = {}
    for a, b in edges:
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)
    cos = math.cos(math.radians(max_angle))
    f = np.linspace(0.08, 0.92, 15)[:, None]

    def arrives(i, j, u):
        o = [v for v in adj.get(i, []) if v != j]
        if not o:
            return True
        d = xy[i] - xy[o[0]]
        return float(d @ u) / (np.linalg.norm(d) + 1e-9) >= cos

    kept, rejected = [], []
    for a, b in edges:
        u = xy[b] - xy[a]
        L = float(np.linalg.norm(u))
        if L <= 2 * spacing:
            kept.append((a, b))
            continue
        u = u / L
        q = xy[a] * (1 - f) + xy[b] * f
        v = ridge[np.clip(q[:, 1].astype(int), 0, ridge.shape[0] - 1), np.clip(q[:, 0].astype(int), 0, ridge.shape[1] - 1)]
        ok = arrives(a, b, u) and arrives(b, a, -u) and v.mean() >= ridge_mean and v.min() >= ridge_min
        (kept if ok else rejected).append((a, b))
    return kept, rejected


# --------------------------------------------------------------------------- re-checking jumps
def _bilinear(img: np.ndarray, P: np.ndarray) -> np.ndarray:
    x = np.clip(P[:, 0] - 0.5, 0, img.shape[1] - 1.001)          # grid_sample convention -> array index
    y = np.clip(P[:, 1] - 0.5, 0, img.shape[0] - 1.001)
    x0, y0 = x.astype(int), y.astype(int)
    fx, fy = x - x0, y - y0
    return (img[y0, x0] * (1 - fx) * (1 - fy) + img[y0, x0 + 1] * fx * (1 - fy)
            + img[y0 + 1, x0] * (1 - fx) * fy + img[y0 + 1, x0 + 1] * fx * fy)


def _arm_direction(xy, adj, i, partner, spacing) -> Optional[np.ndarray]:
    """Direction in which point i's arm leaves it, away from its other (nearby) neighbours."""
    nb = [v for v in adj[i] if v != partner and np.linalg.norm(xy[v] - xy[i]) <= 2.5 * spacing]
    if not nb:
        return None
    d = xy[i] - np.mean(xy[nb], 0)
    n = np.linalg.norm(d)
    return d / n if n > 0 else None


def ridge_check(ridge: np.ndarray, xy: np.ndarray, adj: Sequence[Sequence[int]], i: int, j: int,
                spacing: float = SPACING, n: int = 24) -> float:
    """Check a link without the jump: follow the smooth curve that leaves i along its arm and
    enters j along j's arm (cubic Hermite) and read the centre-line ridge (256 grid) along it.
    The ridge runs through every crossing on both strands (an X), so a true jump follows one bar of
    the X while a turn onto the other strand cuts the corner, where the ridge is low. Returns the
    10th percentile of the ridge along the curve (its weakest stretch), nan if an arm has no
    direction."""
    ti, tj = _arm_direction(xy, adj, i, j, spacing), _arm_direction(xy, adj, j, i, spacing)
    if ti is None or tj is None:
        return float("nan")
    p0, p1 = np.asarray(xy[i], float), np.asarray(xy[j], float)
    L = np.linalg.norm(p1 - p0)
    s = np.linspace(0, 1, n)[:, None]
    curve = ((2 * s**3 - 3 * s**2 + 1) * p0 + (s**3 - 2 * s**2 + s) * (ti * L)
             + (-2 * s**3 + 3 * s**2) * p1 + (s**3 - s**2) * (-tj * L))
    return float(np.percentile(_bilinear(ridge, curve), 10))


def verify_relink(edges: Sequence[Tuple[int, int]], prob: np.ndarray, xy: np.ndarray, ridge: np.ndarray,
                  radius: float, spacing: float = SPACING, tau: float = 0.2, min_prob: float = 0.005) -> List[Tuple[int, int]]:
    """Re-check every jump the head took (links longer than 2.5 spacings) with ridge_check; drop the
    ones below ``tau``. Then link the free strand ends again: among pairs within reach that the head
    gives at least ``min_prob`` (mutual) and that pass the check, best (head x check) first.
    tau = 0.2 was chosen on validation images 0-149 (protocol 13.7)."""
    xy = np.asarray(xy, np.float64)
    N = len(xy)
    adj: List[List[int]] = [[] for _ in range(N)]
    for a, b in edges:
        adj[a].append(b); adj[b].append(a)
    keep = []
    for a, b in edges:
        if np.linalg.norm(xy[a] - xy[b]) > 2.5 * spacing:
            c = ridge_check(ridge, xy, adj, a, b, spacing)
            if np.isfinite(c) and c < tau:
                continue
        keep.append((int(a), int(b)))
    adj = [[] for _ in range(N)]
    for a, b in keep:
        adj[a].append(b); adj[b].append(a)
    deg = np.array([len(v) for v in adj])
    free = np.nonzero(deg == 1)[0]
    reach = min(MAX_LINK, 2 * radius + 3 * spacing)
    M = np.sqrt(prob[:, :N] * prob[:, :N].T)
    cand = []
    for x in range(len(free)):
        for y in range(x + 1, len(free)):
            a, b = free[x], free[y]
            if b in adj[a] or np.linalg.norm(xy[a] - xy[b]) > reach or M[a, b] < min_prob:
                continue
            c = ridge_check(ridge, xy, adj, a, b, spacing)
            if np.isfinite(c) and c >= tau:
                cand.append((M[a, b] * c, int(a), int(b)))
    for _, a, b in sorted(cand, reverse=True):
        if deg[a] < 2 and deg[b] < 2:
            keep.append((a, b)); deg[a] += 1; deg[b] += 1
    return keep


def recheck(edges: Sequence[Tuple[int, int]], prob: np.ndarray, xy: np.ndarray, ridge: np.ndarray, radius: float,
            spacing: float = SPACING) -> List[Tuple[int, int]]:
    """The default re-check of the head's links (protocol 13.7): the curved ridge check with
    re-linking of free ends (verify_relink), then the straight check at its balanced setting
    (check_jumps: 70 degrees, ridge mean 0.3). Held-out: half the wrong long links of the head
    alone, for 1.5-3 points of jumps found."""
    kept, _ = check_jumps(verify_relink(edges, prob, xy, ridge, radius, spacing), xy, ridge, spacing,
                          max_angle=70.0, ridge_mean=0.3, ridge_min=0.0)
    return [tuple(map(int, e)) for e in kept]


def edge_report(pred: Sequence[Tuple[int, int]], pts: Points) -> Dict[str, Any]:
    T = {frozenset(e[:2]) for e in pts.edges}
    J = {frozenset(e[:2]) for e in pts.edges if e[2]}
    Pset = {frozenset(e) for e in pred}
    off = set(np.nonzero(pts.comp < 0)[0].tolist())
    return {"true": len(T), "pred": len(Pset), "hit": len(T & Pset), "jump_true": len(J), "jump_hit": len(J & Pset),
            "to_clutter": sum(1 for e in Pset if e & off), "exact": T == Pset}


# --------------------------------------------------------------------------- inference
def heat_peaks(heat: np.ndarray, thr: float = 0.35) -> np.ndarray:
    """Crossing positions (x, y) on the 256 grid from a 128-grid heatmap (the parser's rule)."""
    import cv2

    pk = (heat == cv2.dilate(heat, np.ones((3, 3), np.uint8))) & (heat > thr)
    cell = INPUT / heat.shape[0]
    out = []
    for i, j in zip(*np.nonzero(pk)):
        i0, i1, j0, j1 = max(i - 1, 0), min(i + 2, heat.shape[0]), max(j - 1, 0), min(j + 2, heat.shape[1])
        w = heat[i0:i1, j0:j1]
        yy, xx = np.mgrid[i0:i1, j0:j1]
        out.append(((w * (xx + 0.5)).sum() / w.sum() * cell, (w * (yy + 0.5)).sum() / w.sum() * cell))
    return np.array(out, np.float32).reshape(-1, 2)


def mask_width(mask128: np.ndarray) -> float:
    """Cable width on the 256 grid, as the parser measures it (twice the median distance to the
    mask edge along its skeleton)."""
    import cv2
    from skimage.morphology import skeletonize

    m = cv2.resize(mask128.astype(np.float32), (INPUT, INPUT), interpolation=cv2.INTER_CUBIC) > 0.5
    dt = cv2.distanceTransform(m.astype(np.uint8), cv2.DIST_L2, 5)
    sk = skeletonize(m)
    return float(2 * np.median(dt[sk])) if sk.any() else 4.0


@torch.no_grad()
def order_strands(trunk, head: StrandPointerHead, img: np.ndarray, thr: float = 0.15) -> Dict[str, Any]:
    """Image -> points on the predicted centre-line and the strands as ordered point sequences."""
    from .knot_aware import preprocess

    x = preprocess(np.asarray(img))
    dev = next(head.parameters()).device
    t = torch.from_numpy(x).permute(2, 0, 1).float()[None].to(dev) / 255.0
    out = trunk(t, return_feats=True)
    prob = torch.sigmoid(out["center_hi"])[0].cpu().numpy()
    heat = torch.sigmoid(out["heat"])[0].cpu().numpy()
    w = mask_width(torch.sigmoid(out["mask"])[0].cpu().numpy())
    C = heat_peaks(heat)
    xy = ridge_points(prob, C, w, head.cfg.spacing)
    if len(xy) < 2:
        return {"xy": xy, "crossings": C, "width": w, "edges": [], "strands": []}
    v = torch.ones(1, len(xy), dtype=torch.bool, device=dev)
    p = torch.softmax(head(out["feats"], torch.from_numpy(xy)[None].to(dev), v, torch.tensor([cut_radius(w)])), -1)[0].cpu().numpy()
    edges = decode(p, thr)
    return {"xy": xy, "crossings": C, "width": w, "prob": p, "edges": edges, "strands": walks(edges, len(xy))}
