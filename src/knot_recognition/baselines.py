"""
baselines.py - Standard dense CNN baselines that feed the SAME parser and gates.

  unet_seg   U-Net segmentation model [Ronneberger et al. 2015]: cable mask (128 grid) and
             strand centre-line (128 and 256 grids). No crossing or over/under output, so the
             parser takes crossings from skeleton junctions and has no over/under evidence
             (every crossing p = 1/2): it measures what segmentation alone can decide.
  unet_kp    CenterNet-style keypoint detector [Zhou et al. 2019]: crossing-centre heatmap
             with a per-keypoint attribute, the over-strand axis (128 grid). It is combined
             with unet_seg (strands) in `SegKeypointPair`, so the pair has every input the
             parser needs, from two standard CNNs instead of one transformer.

Same input (RGB + FFT channel at 256 x 256), same training data (data/knot_aware/*_v3.npz),
same targets and losses (knot_aware.total_loss), same test-time views.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .knot_aware import _axial_inverse, preprocess


def _block(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 3, 1, 1), nn.GroupNorm(min(8, o // 4), o), nn.GELU(),
                         nn.Conv2d(o, o, 3, 1, 1), nn.GroupNorm(min(8, o // 4), o), nn.GELU())


@dataclass
class UNetConfig:
    arch: str = "unet_seg"
    in_channels: int = 4
    widths: List[int] = field(default_factory=lambda: [12, 32, 64, 128, 192])


class UNet(nn.Module):
    """Encoder 256 -> 16 (5 levels), decoder back to 256 with skip connections."""

    def __init__(self, cfg: UNetConfig):
        super().__init__()
        self.cfg = cfg
        w = cfg.widths
        self.enc = nn.ModuleList([_block(cfg.in_channels, w[0])] + [_block(w[i], w[i + 1]) for i in range(len(w) - 1)])
        self.up = nn.ModuleList([nn.ConvTranspose2d(w[i + 1], w[i], 2, 2) for i in reversed(range(len(w) - 1))])
        self.dec = nn.ModuleList([_block(2 * w[i], w[i]) for i in reversed(range(len(w) - 1))])
        if cfg.arch == "unet_seg":
            self.head_mask = nn.Conv2d(w[1], 1, 1)
            self.head_center = nn.Conv2d(w[1], 1, 1)
            self.head_center_hi = nn.Conv2d(w[0], 1, 1)
        else:
            self.head_heat = nn.Conv2d(w[1], 1, 1)
            self.head_over = nn.Conv2d(w[1], 2, 1)
            nn.init.constant_(self.head_heat.bias, -2.19)

    def forward(self, x):
        x = x[:, : self.cfg.in_channels]
        skips = []
        for i, e in enumerate(self.enc):
            x = e(x if i == 0 else F.max_pool2d(x, 2))
            skips.append(x)
        feats = {}
        for k, (u, d) in enumerate(zip(self.up, self.dec)):
            x = d(torch.cat([u(x), skips[-2 - k]], 1))
            feats[x.shape[-1]] = x
        f128, f256 = feats[128], feats[256]
        if self.cfg.arch == "unet_seg":
            return {"mask": self.head_mask(f128).squeeze(1), "center": self.head_center(f128).squeeze(1),
                    "center_hi": self.head_center_hi(f256).squeeze(1)}
        return {"heat": self.head_heat(f128).squeeze(1), "over": self.head_over(f128)}


def save_unet(model: UNet, path, extra=None):
    sd = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    torch.save({"config": asdict(model.cfg), "state_dict": sd, "extra": extra or {}}, path)


def load_unet(path) -> UNet:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = UNet(UNetConfig(**ck["config"]))
    m.load_state_dict(ck["state_dict"])
    m.extra = ck.get("extra", {})
    return m.eval()


@torch.no_grad()
def predict_maps(models, img: np.ndarray, tta: bool = True) -> Dict[str, np.ndarray]:
    """Test-time views exactly as knot_aware.predict, for any set of dense heads."""
    x = preprocess(img)
    t = torch.from_numpy(x).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    views = [(False, 0)] + ([(f, r) for f in (False, True) for r in range(4)][1:] if tta else [])
    acc: Dict[str, torch.Tensor] = {}
    for hf, rot in views:
        xi = torch.rot90(t.flip(-1) if hf else t, rot, (-2, -1))
        back = lambda m: (torch.rot90(m, -rot, (-2, -1)).flip(-1) if hf else torch.rot90(m, -rot, (-2, -1)))
        for model in models:
            for k, v in model(xi).items():
                if k in ("over", "orient"):
                    v = _axial_inverse(back(v / v.norm(dim=1, keepdim=True).clamp(min=1e-6)), hf, rot)
                else:
                    v = back(torch.sigmoid(v))
                acc[k] = acc.get(k, 0) + v
    out = {k: (v / len(views))[0].numpy() for k, v in acc.items()}
    out["cell_px"] = np.asarray(img).shape[0] / 128.0
    out["tta_views"] = len(views)
    return out


class Predictor:
    """Wraps one or two baseline models so evaluation code can call `.predict_maps(img, tta)`."""

    def __init__(self, models, name: str, kappa: float = 26.5, gamma: float = 2.0):
        self.models, self.name = models, name
        self.extra = {"kappa": kappa, "gamma": gamma}

    def predict_maps(self, img, tta=True):
        return predict_maps(self.models, img, tta)
