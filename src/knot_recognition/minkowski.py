"""
minkowski.py - Minkowski (L_p) attention and a Lebesgue-measure (layer-cake) loss for the
knot-aware NanoTransformer.

1. Minkowski position bias (attn_pos_bias="minkowski")
   Each head adds  -alpha_h * ||r_i - r_j||_{p_h}  to its attention logits, where r is a token's
   (x, y) cell on the 16 x 16 grid. Both p_h = 1 + softplus(rho_h) (initial value 2) and
   alpha_h = softplus(a_h) are learned, per head and per block. The initial slopes are spread
   across heads (0.2 ... 0.01), so some heads start local and others nearly global. p
   sets the shape of the neighbourhood a head favours: diamond (p = 1), disc (p = 2) or
   square (p -> inf). The count token gets no bias.
   (|dx|^p + |dy|^p)^(1/p) is unchanged by the 8 flips and quarter turns of the square, so
   the bias keeps the model's behaviour consistent across the test-time views. These
   views are what the over/under confidence shrink measures.

2. Minkowski content attention (attn_content="minkowski")
   Logits are a negative distance between query and key, not a dot product:
       logit_ij = -t_h * ||q_i - k_j||_{p_h}^{p_h} / (p_h * sqrt(d_h))
   with p_h = 1 (Manhattan) on the first half of the heads and p_h = 2 on the rest, and a
   learned temperature t_h (initial value 1). For p = 2 this equals dot-product attention
   minus a key-norm term (q.k - |k|^2/2), so those heads start close to the original model. L1
   heads compare features coordinate by coordinate, which is less dominated by a few large
   coordinates. p is fixed in these heads: a learnable p needs a per-coordinate |q - k|^p
   tensor, about 400x slower than a dot product on CPU (measured: 4.6 s vs 0.011 s per
   block, batch 16).

3. Layer-cake (Lebesgue) loss
   The parser thresholds the mask over the levels t in [0.15, 0.85] (stable segmentation)
   and the centre-line ridge at 0.5. By the layer-cake formula, the mean over t in [a, b]
   of the Lebesgue measure (area) of the symmetric difference of the super-level sets
       (1 / (b - a)) * integral_a^b  mu({p > t} xor {m > t}) dt
   equals  mean |clip(p, a, b) - clip(m, a, b)| / (b - a)  per pixel. The term is the
   expected area that the thresholded prediction gets wrong when the threshold is drawn
   uniformly from the band the parser uses. Its gradient is nonzero only where the
   prediction is inside the band, so it pushes ambiguous pixels out of the band. BCE
   already handles confident errors.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .nanotransformer import MLPBlock


def _inv_softplus(y: float) -> float:
    return math.log(math.expm1(y))


class MinkowskiPositionBias(nn.Module):
    """-alpha_h * ||r_i - r_j||_{p_h} on a square token grid, with learnable p_h and alpha_h.
    Tokens before the grid (``n_prefix``, e.g. the count token) get zero bias."""

    def __init__(self, grid: int, num_heads: int, p_init: float = 2.0,
                 alphas: Optional[Sequence[float]] = None, n_prefix: int = 1):
        super().__init__()
        ys, xs = torch.meshgrid(torch.arange(grid), torch.arange(grid), indexing="ij")
        r = torch.stack([xs.flatten(), ys.flatten()], -1).float()
        self.register_buffer("absdiff", (r[:, None] - r[None]).abs(), persistent=False)  # (N, N, 2)
        self.n_prefix = n_prefix
        alphas = list(alphas) if alphas is not None else \
            [0.2 * (0.05 ** (h / max(1, num_heads - 1))) for h in range(num_heads)]   # 0.2 ... 0.01
        self.rho = nn.Parameter(torch.full((num_heads,), _inv_softplus(p_init - 1.0)))
        self.a = nn.Parameter(torch.tensor([_inv_softplus(v) for v in alphas]))

    @property
    def p(self) -> torch.Tensor:
        return 1.0 + F.softplus(self.rho)

    @property
    def alpha(self) -> torch.Tensor:
        return F.softplus(self.a)

    def distance(self) -> torch.Tensor:
        p = self.p[:, None, None, None]
        # clamp keeps d/dp (x^p) = x^p log x finite where |dx| or |dy| is 0
        return (self.absdiff.clamp_min(1e-6)[None] ** p).sum(-1) ** (1.0 / p[..., 0])   # (H, N, N)

    def forward(self) -> torch.Tensor:
        b = -self.alpha[:, None, None] * self.distance()
        return F.pad(b, (self.n_prefix, 0, self.n_prefix, 0))                           # (H, N+P, N+P)


def lp_distance_pow(q: torch.Tensor, k: torch.Tensor, p: float) -> torch.Tensor:
    """||q_i - k_j||_p^p for (..., N, d) tensors -> (..., N, N)."""
    if p == 2:
        d2 = q.pow(2).sum(-1, keepdim=True) + k.pow(2).sum(-1).unsqueeze(-2) - 2 * q @ k.transpose(-2, -1)
        return d2.clamp_min(0)
    if p == 1 and q.device.type != "mps":
        return torch.cdist(q, k, p=1.0)
    # explicit form (any p; also MPS, where cdist with p != 2 is not available on every version)
    rows = []
    for i in range(0, q.shape[-2], 64):
        rows.append(((q[..., i:i + 64, None, :] - k[..., None, :, :]).abs() ** p).sum(-1))
    return torch.cat(rows, -2)


class MinkowskiSelfAttention(nn.Module):
    """Drop-in for nanotransformer.MultiHeadSelfAttention (same qkv / proj parameter names, so
    dot-product weights load into it), with an optional Minkowski position bias and optional
    L_p content logits."""

    def __init__(self, embed_dim: int, num_heads: int, content: str = "dot", pos_bias: bool = False,
                 grid: int = 16, n_prefix: int = 1, dropout: float = 0.0, attention_dropout: float = 0.0,
                 content_p: Optional[Sequence[float]] = None):
        super().__init__()
        self.embed_dim, self.num_heads = embed_dim, num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=True)
        self.attn_drop = nn.Dropout(attention_dropout)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(dropout)
        if content not in ("dot", "minkowski"):
            raise ValueError(content)
        self.content = content
        if content == "minkowski":
            half = num_heads // 2
            self.content_p = list(content_p) if content_p is not None else [1.0] * half + [2.0] * (num_heads - half)
            self.log_temp = nn.Parameter(torch.zeros(num_heads))
        self.pos = MinkowskiPositionBias(grid, num_heads, n_prefix=n_prefix) if pos_bias else None

    def logits(self, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        if self.content == "dot":
            a = (q @ k.transpose(-2, -1)) * self.scale
        else:
            parts, h0 = [], 0
            ps = self.content_p
            while h0 < len(ps):                       # consecutive heads sharing one p in one call
                h1 = h0
                while h1 < len(ps) and ps[h1] == ps[h0]:
                    h1 += 1
                p = ps[h0]
                parts.append(-lp_distance_pow(q[:, h0:h1], k[:, h0:h1], p) / (p * math.sqrt(self.head_dim)))
                h0 = h1
            a = torch.cat(parts, 1) * self.log_temp.exp()[None, :, None, None]
        if self.pos is not None:
            a = a + self.pos()[None]
        return a

    def forward(self, x: torch.Tensor, return_attn: bool = False):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = F.softmax(self.logits(q, k), dim=-1)
        out = (self.attn_drop(attn) @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj_drop(self.proj(out))
        return (out, attn) if return_attn else out


class MinkowskiEncoderBlock(nn.Module):
    """Pre-LN block with the same parameter names as nanotransformer.TransformerEncoderBlock."""

    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float = 2.0, dropout: float = 0.0,
                 attention_dropout: float = 0.0, content: str = "dot", pos_bias: bool = False, grid: int = 16):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MinkowskiSelfAttention(embed_dim, num_heads, content, pos_bias, grid,
                                           dropout=dropout, attention_dropout=attention_dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLPBlock(embed_dim, int(embed_dim * mlp_ratio), dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


# ----------------------------------------------------------------------------- Lebesgue measure
def layer_cake_loss(prob: torch.Tensor, target: torch.Tensor, a: float, b: float) -> torch.Tensor:
    """Mean over thresholds t ~ U[a, b] of the area fraction of {prob > t} xor {target > t}
    (layer-cake formula; exact, no sampling of t)."""
    return ((prob.clamp(a, b) - target.clamp(a, b)).abs() / (b - a)).mean()


def layer_cake_loss_sampled(prob: torch.Tensor, target: torch.Tensor, a: float, b: float, n: int = 401) -> torch.Tensor:
    """The same quantity by explicit integration over thresholds (used only in tests)."""
    ts = torch.linspace(a, b, n)
    vals = torch.stack([((prob > t) ^ (target > t)).float().mean() for t in ts])
    return torch.trapezoid(vals, ts) / (b - a)


# the bands the parser thresholds in (cable.measure.stable_segmentation levels; ridge threshold 0.5)
LAYER_CAKE_BANDS = {"mask": (0.15, 0.85), "center": (0.3, 0.7), "center_hi": (0.3, 0.7)}
