"""
unetformer.py - A U-Net with a transformer at its bottleneck, predicting every knot-aware head.

Why: against the U-Net pair, the NanoTransformer loses on topology, not on crossings. On
KnotBench-Photo both models find crossings and read over/under almost perfectly. Of the 43 images
only the U-Net answers, 29 fail in the NanoTransformer as "free cable ends": its centre-line
breaks. The NanoTransformer reaches its dense outputs through a light decoder (skips at
strides 4 and 2 only), and its 256-grid ridge branch was trained on a frozen trunk. This model
keeps the U-Net's dense path, with skip connections at every resolution down from 256 x 256
(TransUNet-style), and adds global attention where the U-Net has none: on the 16 x 16 grid at
the bottleneck, plus a count token. Attention lets the two arms of a strand on either side of a
crossing, and a strand followed through clutter, inform each other.

One network predicts everything the parser needs (the U-Net baseline needs two):
mask, heat, over, orient and center on the 128 grid, center_hi on the 256 grid, and count.
It has about the same number of parameters as the U-Net pair.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .nanotransformer import TransformerEncoderBlock


def _block(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 3, 1, 1), nn.GroupNorm(min(8, o // 4), o), nn.GELU(),
                         nn.Conv2d(o, o, 3, 1, 1), nn.GroupNorm(min(8, o // 4), o), nn.GELU())


@dataclass
class UNetFormerConfig:
    arch: str = "unetformer"
    in_channels: int = 4
    widths: List[int] = field(default_factory=lambda: [16, 32, 64, 128, 192])
    embed_dim: int = 192
    depth: int = 4
    num_heads: int = 6
    mlp_ratio: float = 2.0
    attn_pos_bias: str = "none"      # "minkowski": learnable-p position bias (knot_recognition.minkowski)
    hi_res_center: bool = True       # always: read by knot_aware.predict
    layer_cake: float = 0.0          # recorded training-loss weights
    topo_loss: float = 0.0


class UNetFormer(nn.Module):
    def __init__(self, cfg: UNetFormerConfig = None):
        super().__init__()
        self.cfg = cfg = cfg or UNetFormerConfig()
        w, d = cfg.widths, cfg.embed_dim
        self.enc = nn.ModuleList([_block(cfg.in_channels, w[0])] + [_block(w[i], w[i + 1]) for i in range(len(w) - 1)])
        self.grid = 256 // 2 ** (len(w) - 1)
        self.to_tok = nn.Conv2d(w[-1], d, 1)
        self.pos = nn.Parameter(torch.zeros(1, self.grid * self.grid + 1, d))
        self.count_token = nn.Parameter(torch.zeros(1, 1, d))
        nn.init.trunc_normal_(self.pos, std=0.02)
        nn.init.trunc_normal_(self.count_token, std=0.02)
        if cfg.attn_pos_bias == "none":
            self.blocks = nn.ModuleList([TransformerEncoderBlock(d, cfg.num_heads, cfg.mlp_ratio, 0.0, 0.0) for _ in range(cfg.depth)])
        else:
            from .minkowski import MinkowskiEncoderBlock
            self.blocks = nn.ModuleList([MinkowskiEncoderBlock(d, cfg.num_heads, cfg.mlp_ratio, pos_bias=True, grid=self.grid)
                                         for _ in range(cfg.depth)])
        self.norm = nn.LayerNorm(d)
        self.from_tok = nn.Conv2d(d, w[-1], 1)
        nn.init.zeros_(self.from_tok.weight)      # starts as a plain U-Net (residual), attention is learned on top
        nn.init.zeros_(self.from_tok.bias)
        self.up = nn.ModuleList([nn.ConvTranspose2d(w[i + 1], w[i], 2, 2) for i in reversed(range(len(w) - 1))])
        self.dec = nn.ModuleList([_block(2 * w[i], w[i]) for i in reversed(range(len(w) - 1))])
        self.head_mask = nn.Conv2d(w[1], 1, 1)
        self.head_heat = nn.Conv2d(w[1], 1, 1)
        self.head_over = nn.Conv2d(w[1], 2, 1)
        self.head_orient = nn.Conv2d(w[1], 2, 1)
        self.head_center = nn.Conv2d(w[1], 1, 1)
        self.head_center_hi = nn.Conv2d(w[0], 1, 1)
        self.head_count = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
        nn.init.constant_(self.head_heat.bias, -2.19)

    def forward(self, x: torch.Tensor, return_feats: bool = False) -> Dict[str, torch.Tensor]:
        """``return_feats``: also return the decoder feature maps by grid size (16 = bottleneck
        after attention, then 32, 64, 128, 256), read by the strand pointer head."""
        x = x[:, : self.cfg.in_channels]
        skips = []
        for i, e in enumerate(self.enc):
            x = e(x if i == 0 else F.max_pool2d(x, 2))
            skips.append(x)
        B, C, h, w_ = x.shape
        tok = self.to_tok(x).flatten(2).transpose(1, 2)
        tok = torch.cat([self.count_token.expand(B, -1, -1), tok], 1) + self.pos
        for blk in self.blocks:
            tok = blk(tok)
        tok = self.norm(tok)
        cnt = self.head_count(tok[:, 0]).squeeze(-1)
        x = x + self.from_tok(tok[:, 1:].transpose(1, 2).reshape(B, -1, h, w_))
        feats = {x.shape[-1]: x}
        for k, (u, dblk) in enumerate(zip(self.up, self.dec)):
            x = dblk(torch.cat([u(x), skips[-2 - k]], 1))
            feats[x.shape[-1]] = x
        f128, f256 = feats[128], feats[256]
        out = {"mask": self.head_mask(f128).squeeze(1), "heat": self.head_heat(f128).squeeze(1),
               "over": self.head_over(f128), "orient": self.head_orient(f128), "center": self.head_center(f128).squeeze(1),
               "center_hi": self.head_center_hi(f256).squeeze(1), "count": cnt}
        if return_feats:
            out["feats"] = feats
        return out
