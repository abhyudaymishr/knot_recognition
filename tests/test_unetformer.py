"""UNetFormer and the clDice topology loss."""

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from knot_recognition.knot_aware import load, predict, save, total_loss
from knot_recognition.topology_loss import cldice_loss, soft_skeleton
from knot_recognition.unetformer import UNetFormer, UNetFormerConfig


def _ring(n=64, r=20, w=3):
    yy, xx = np.mgrid[0:n, 0:n]
    d = np.abs(np.hypot(yy - n / 2, xx - n / 2) - r)
    return torch.from_numpy((d <= w / 2).astype(np.float32))


def test_soft_skeleton_of_a_thick_ring_is_thin_and_inside_it():
    g = _ring(w=5)[None, None]
    sk = soft_skeleton(g, 8)
    assert sk.sum() < 0.6 * g.sum() and float((sk * (1 - g)).sum()) == 0.0


def test_cldice_rises_when_a_ring_is_cut():
    g = _ring(w=3)
    broken = g.clone()
    broken[:, 30:34] = 0                        # cut the ring at two places, 4 cells each
    cl_perfect = float(cldice_loss((g * 0.98 + 0.01)[None], g[None], 5))
    cl_broken = float(cldice_loss((broken * 0.98 + 0.01)[None], g[None], 5))
    assert cl_perfect < 0.05 and cl_broken > cl_perfect + 0.02


def test_unetformer_heads_shapes_and_round_trip(tmp_path):
    torch.manual_seed(0)
    m = UNetFormer(UNetFormerConfig(topo_loss=1.0))
    x = torch.rand(2, 4, 256, 256)
    out = m(x)
    assert out["mask"].shape == (2, 128, 128) and out["over"].shape == (2, 2, 128, 128)
    assert out["center_hi"].shape == (2, 256, 256) and out["count"].shape == (2,)
    tg = {"mask": (torch.rand(2, 128, 128) > 0.8).float(), "heat": torch.zeros(2, 128, 128),
          "over": torch.randn(2, 2, 128, 128), "over_w": torch.zeros(2, 128, 128),
          "orient": torch.randn(2, 2, 128, 128), "orient_w": torch.ones(2, 128, 128), "count": torch.zeros(2),
          "center": torch.zeros(2, 128, 128), "center_hi": torch.zeros(2, 256, 256)}
    loss, parts = total_loss(out, tg, topo=1.0)
    assert {"cldice_mask", "cldice_center", "cldice_center_hi"} <= set(parts)
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
    save(m, tmp_path / "uf.pth")
    m2 = load(tmp_path / "uf.pth")
    assert isinstance(m2, UNetFormer) and m2.has_center
    m.eval()
    with torch.no_grad():
        assert torch.allclose(m(x)["center_hi"], m2(x)["center_hi"], atol=1e-5)
    pred = predict(m2, (np.random.rand(512, 512, 3) * 255).astype(np.uint8))
    assert pred["center_hi"].shape == (256, 256) and pred["over"].shape == (2, 128, 128)


def test_unetformer_parameters_match_the_unet_pair():
    from knot_recognition.baselines import UNet, UNetConfig
    n = lambda m: sum(p.numel() for p in m.parameters())
    pair = n(UNet(UNetConfig("unet_seg"))) + n(UNet(UNetConfig("unet_kp")))
    assert abs(n(UNetFormer()) - pair) / pair < 0.1
