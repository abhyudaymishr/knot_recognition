"""
knot_aware.py - Knot-aware NanoTransformer: dense structure instead of a 250-way label.

The original NanoTransformer maps a whole image to one of 250 prime knots. That
cannot generalise past the census, cannot say *where* the crossings are, and
cannot abstain. The knot-aware variant keeps the same transformer encoder blocks
but predicts the structure a knot diagram is made of, so exact topology can be
computed downstream (cable.neural_parser):

  mask     P(pixel is cable)
  heat     crossing-centre heatmap
  over     axial direction (cos 2a, sin 2a) of the strand on top at each crossing
  orient   axial direction of the strand everywhere else (used to pair arms)
  count    number of crossings (global token; checked against the Euler count)

Input: 4 channels at 256 x 256 = RGB + an FFT-cleaned grey channel
(homomorphic illumination removal + Wiener shrinkage, cable.denoise).
Outputs are on a 128 x 128 grid (stride 2 at the model input, 4 pixels at 512):
parallel cable strands a few widths apart must stay separate for the Euler check.

Architecture: convolutional stem (stride 4) -> 4x4 patch embedding (stride 16,
16 x 16 tokens) + [COUNT] token -> NanoTransformer encoder blocks (global
attention lets a strand's evidence on both sides of a crossing meet) -> light
U-Net style decoder with a stride-4 skip connection -> four dense heads.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .nanotransformer import TransformerEncoderBlock

INPUT_SIZE = 256


def preprocess(img: np.ndarray, size: int = INPUT_SIZE) -> np.ndarray:
    """RGB image (any size) -> uint8 array (size, size, 4): RGB + FFT-cleaned grey."""
    import cv2

    from .cable.denoise import homomorphic, to_gray, wiener

    rgb = cv2.resize(np.asarray(img)[..., :3], (size, size), interpolation=cv2.INTER_AREA)
    g = to_gray(rgb)
    x = homomorphic(g, d0=1 / 256, g_low=0.5)
    x, _ = wiener(x)
    lo, hi = np.percentile(x, [1, 99])
    f = np.clip((x - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)
    return np.concatenate([rgb, f[..., None]], axis=2)


@dataclass
class KnotAwareConfig:
    in_channels: int = 4
    stem_dim: int = 48
    embed_dim: int = 128
    depth: int = 4
    num_heads: int = 4
    mlp_ratio: float = 2.0
    dropout: float = 0.0


def _conv(i, o, s=1):
    return nn.Sequential(nn.Conv2d(i, o, 3, s, 1), nn.GroupNorm(min(8, o // 4), o), nn.GELU())


class KnotAwareNanoTransformer(nn.Module):
    def __init__(self, cfg: Optional[KnotAwareConfig] = None):
        super().__init__()
        self.cfg = cfg = cfg or KnotAwareConfig()
        d = cfg.embed_dim
        self.stem2 = _conv(cfg.in_channels, 24, 2)          # stride 2  (128 x 128)
        self.stem = _conv(24, cfg.stem_dim, 2)               # stride 4  (64 x 64)
        self.patch = nn.Sequential(_conv(cfg.stem_dim, 96, 2), nn.Conv2d(96, d, 2, 2))
        self.grid = INPUT_SIZE // 16
        self.pos = nn.Parameter(torch.zeros(1, self.grid * self.grid + 1, d))
        self.count_token = nn.Parameter(torch.zeros(1, 1, d))
        nn.init.trunc_normal_(self.pos, std=0.02)
        nn.init.trunc_normal_(self.count_token, std=0.02)
        self.blocks = nn.ModuleList([TransformerEncoderBlock(d, cfg.num_heads, cfg.mlp_ratio, cfg.dropout, cfg.dropout)
                                     for _ in range(cfg.depth)])
        self.norm = nn.LayerNorm(d)
        self.up = nn.Sequential(nn.ConvTranspose2d(d, 96, 2, 2), nn.GELU(), nn.ConvTranspose2d(96, cfg.stem_dim, 2, 2))
        self.fuse = nn.Sequential(_conv(2 * cfg.stem_dim, cfg.stem_dim), _conv(cfg.stem_dim, cfg.stem_dim))
        self.up2 = nn.ConvTranspose2d(cfg.stem_dim, 32, 2, 2)
        self.fuse2 = nn.Sequential(_conv(32 + 24, 32), _conv(32, 32))
        self.head_mask = nn.Conv2d(32, 1, 1)
        self.head_heat = nn.Conv2d(32, 1, 1)
        self.head_over = nn.Conv2d(32, 2, 1)
        self.head_orient = nn.Conv2d(32, 2, 1)
        self.head_count = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
        nn.init.constant_(self.head_heat.bias, -2.19)  # prior P(crossing) ~ 0.1

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        f2 = self.stem2(x)                             # (B, 24, 128, 128)
        f4 = self.stem(f2)                             # (B, C, 64, 64)
        t = self.patch(f4)                             # (B, d, 16, 16)
        B, d, h, w = t.shape
        tok = t.flatten(2).transpose(1, 2)
        tok = torch.cat([self.count_token.expand(B, -1, -1), tok], 1) + self.pos
        for blk in self.blocks:
            tok = blk(tok)
        tok = self.norm(tok)
        cnt = self.head_count(tok[:, 0]).squeeze(-1)
        g = tok[:, 1:].transpose(1, 2).reshape(B, d, h, w)
        y = self.fuse(torch.cat([self.up(g), f4], 1))  # (B, C, 64, 64)
        y = self.fuse2(torch.cat([self.up2(y), f2], 1))  # (B, 32, 128, 128)
        return {"mask": self.head_mask(y).squeeze(1), "heat": self.head_heat(y).squeeze(1),
                "over": self.head_over(y), "orient": self.head_orient(y), "count": cnt}


# ----------------------------------------------------------------------------- losses

def focal_heat_loss(logit, target, alpha=2.0, beta=4.0):
    p = torch.sigmoid(logit).clamp(1e-4, 1 - 1e-4)
    pos = target > 0.99
    neg = ~pos
    lp = -((1 - p) ** alpha * torch.log(p))[pos].sum()
    ln = -(((1 - target) ** beta) * p ** alpha * torch.log(1 - p))[neg].sum()
    return (lp + ln) / pos.sum().clamp(min=1)


def axial_loss(pred, target, weight):
    p = pred / pred.norm(dim=1, keepdim=True).clamp(min=1e-6)
    err = ((p - target) ** 2).sum(1)
    return (err * weight).sum() / weight.sum().clamp(min=1)


def total_loss(out, tg):
    m = tg["mask"]
    bce = F.binary_cross_entropy_with_logits(out["mask"], m)
    pm = torch.sigmoid(out["mask"])
    dice = 1 - (2 * (pm * m).sum((1, 2)) + 1) / (pm.sum((1, 2)) + m.sum((1, 2)) + 1)
    parts = {
        "mask": bce + dice.mean(),
        "heat": focal_heat_loss(out["heat"], tg["heat"]),
        "over": axial_loss(out["over"], tg["over"], tg["over_w"]),
        "orient": axial_loss(out["orient"], tg["orient"], tg["orient_w"]),
        "count": F.smooth_l1_loss(out["count"], tg["count"]) * 0.1,
    }
    return sum(parts.values()), {k: float(v) for k, v in parts.items()}


# ----------------------------------------------------------------------------- io

def save(model: KnotAwareNanoTransformer, path, extra: Optional[dict] = None):
    torch.save({"config": asdict(model.cfg), "state_dict": model.state_dict(), "extra": extra or {}}, path)


def load(path, device: str = "cpu") -> KnotAwareNanoTransformer:
    ck = torch.load(path, map_location=device, weights_only=False)
    m = KnotAwareNanoTransformer(KnotAwareConfig(**ck["config"]))
    m.load_state_dict(ck["state_dict"])
    m.extra = ck.get("extra", {})
    return m.eval()


def _axial_inverse(v: torch.Tensor, hflip: bool, rot: int) -> torch.Tensor:
    """Map axial vectors predicted on a transformed image back to the original frame.

    Forward transform = optional horizontal flip, then ``rot`` quarter turns
    (torch.rot90, dims (-2, -1)); each quarter turn maps 2a -> 2a - pi (negation)
    and a flip maps 2a -> -2a (negate the sine).
    """
    if rot % 2:
        v = -v
    if hflip:
        v = torch.stack([v[:, 0], -v[:, 1]], 1)
    return v


@torch.no_grad()
def predict(model: KnotAwareNanoTransformer, img: np.ndarray, tta: bool = True) -> Dict[str, np.ndarray]:
    """Run on one RGB image; maps on the 128 x 128 grid.

    With ``tta`` the image is also seen under the 8 flips/rotations of the square.
    Maps are averaged after undoing each transform. For the over-strand axis the
    *unnormalised* average is kept: where the network's answers disagree between
    views (evidence it cannot actually see, e.g. a silhouette), the averaged vector
    shrinks, the over/under margin shrinks, and the reported confidence drops.
    """
    x = preprocess(img)
    t = torch.from_numpy(x).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    views = [(False, 0)] + ([(f, r) for f in (False, True) for r in range(4)][1:] if tta else [])
    acc = {"mask": 0, "heat": 0, "over": 0, "orient": 0, "count": 0.0}
    for hf, rot in views:
        xi = t.flip(-1) if hf else t
        xi = torch.rot90(xi, rot, (-2, -1))
        o = model(xi)
        back = lambda m: (torch.rot90(m, -rot, (-2, -1)).flip(-1) if hf else torch.rot90(m, -rot, (-2, -1)))
        acc["mask"] = acc["mask"] + back(torch.sigmoid(o["mask"]))
        acc["heat"] = acc["heat"] + back(torch.sigmoid(o["heat"]))
        ov = o["over"] / o["over"].norm(dim=1, keepdim=True).clamp(min=1e-6)
        acc["over"] = acc["over"] + _axial_inverse(back(ov), hf, rot)
        orn = o["orient"] / o["orient"].norm(dim=1, keepdim=True).clamp(min=1e-6)
        acc["orient"] = acc["orient"] + _axial_inverse(back(orn), hf, rot)
        acc["count"] += float(o["count"][0])
    n = len(views)
    return {
        "mask": (acc["mask"] / n)[0].numpy(),
        "heat": (acc["heat"] / n)[0].numpy(),
        "over": (acc["over"] / n)[0].numpy(),      # length < 1 where views disagree
        "orient": (acc["orient"] / n)[0].numpy(),
        "count": acc["count"] / n,
        "cell_px": np.asarray(img).shape[0] / 128.0,
        "tta_views": n,
    }
