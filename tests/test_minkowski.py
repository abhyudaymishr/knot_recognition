"""Minkowski attention and the Lebesgue (layer-cake) loss."""

import math
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from knot_recognition.knot_aware import KnotAwareConfig, KnotAwareNanoTransformer, load, save, total_loss
from knot_recognition.minkowski import (MinkowskiPositionBias, MinkowskiSelfAttention, layer_cake_loss,
                                        layer_cake_loss_sampled, lp_distance_pow)
from knot_recognition.nanotransformer import MultiHeadSelfAttention


def test_layer_cake_equals_integral_of_symmetric_difference_measure():
    g = torch.Generator().manual_seed(0)
    p = torch.rand(2, 32, 32, generator=g)
    m = (torch.rand(2, 32, 32, generator=g) > 0.7).float()
    soft = torch.rand(2, 32, 32, generator=g)
    for tgt in (m, soft):
        for a, b in ((0.15, 0.85), (0.3, 0.7)):
            assert abs(float(layer_cake_loss(p, tgt, a, b)) - float(layer_cake_loss_sampled(p, tgt, a, b, 2001))) < 2e-3


def test_layer_cake_is_zero_outside_band_and_has_no_gradient_there():
    m = torch.tensor([0.0, 1.0, 0.0, 1.0])
    p = torch.tensor([0.05, 0.95, 0.5, 0.5], requires_grad=True)
    loss = layer_cake_loss(p, m, 0.15, 0.85)
    loss.backward()
    assert p.grad[0] == 0 and p.grad[1] == 0 and p.grad[2] > 0 and p.grad[3] < 0
    assert abs(float(loss) - 0.25) < 1e-6   # two pixels at 0.35 / 0.7 = 0.5 each, mean over 4


def test_dot_weights_load_into_minkowski_attention_and_match_with_no_bias():
    torch.manual_seed(0)
    ref = MultiHeadSelfAttention(64, 4, dropout=0, attention_dropout=0).eval()
    mk = MinkowskiSelfAttention(64, 4, content="dot", pos_bias=False).eval()
    mk.load_state_dict(ref.state_dict())
    x = torch.randn(2, 17, 64)
    assert torch.allclose(ref(x), mk(x), atol=1e-6)


def test_l2_content_logits_are_dot_product_minus_norms():
    torch.manual_seed(1)
    q, k = torch.randn(2, 3, 10, 8), torch.randn(2, 3, 10, 8)
    d2 = lp_distance_pow(q, k, 2.0)
    expect = q.pow(2).sum(-1, keepdim=True) + k.pow(2).sum(-1).unsqueeze(-2) - 2 * q @ k.transpose(-1, -2)
    assert torch.allclose(d2, expect, atol=1e-4)
    # softmax over keys: -d2/2 and q.k - |k|^2/2 give the same attention
    a1 = torch.softmax(-d2 / 2, -1)
    a2 = torch.softmax(q @ k.transpose(-1, -2) - k.pow(2).sum(-1).unsqueeze(-2) / 2, -1)
    assert torch.allclose(a1, a2, atol=1e-5)


@pytest.mark.parametrize("p", [1.0, 2.0, 3.0])
def test_lp_distance_matches_explicit_sum_with_gradients(p):
    torch.manual_seed(2)
    q = torch.randn(2, 2, 70, 6, requires_grad=True)
    k = torch.randn(2, 2, 70, 6, requires_grad=True)
    fast = lp_distance_pow(q, k, p)
    slow = ((q[..., :, None, :] - k[..., None, :, :]).abs() ** p).sum(-1)
    assert torch.allclose(fast, slow, atol=1e-3, rtol=1e-4)
    gf = torch.autograd.grad(fast.sin().sum(), (q, k))
    gs = torch.autograd.grad(slow.sin().sum(), (q, k))
    for a, b in zip(gf, gs):
        assert torch.allclose(a, b, atol=1e-2, rtol=1e-3)


def test_position_bias_is_invariant_under_the_eight_test_time_views():
    """B[g(i), g(j)] = B[i, j] for every flip / quarter turn g of the token grid, at any p."""
    grid = 6
    pb = MinkowskiPositionBias(grid, 3, p_init=1.7)
    with torch.no_grad():
        pb.rho[1] = 2.5
    B = pb()[:, 1:, 1:]
    idx = torch.arange(grid * grid).reshape(grid, grid)
    for hf in (False, True):
        for r in range(4):
            g = torch.rot90(idx.flip(-1) if hf else idx, r, (0, 1)).flatten()
            perm = torch.empty_like(g)
            perm[g] = torch.arange(len(g))   # token at original index i moves to perm[i]
            assert torch.allclose(B[:, perm][:, :, perm], B, atol=1e-5)
    assert torch.all(pb()[:, 0] == 0) and torch.all(pb()[:, :, 0] == 0)   # count token unbiased


def test_position_bias_distance_is_minkowski_and_p_is_learnable():
    pb = MinkowskiPositionBias(4, 2, p_init=2.0)
    D = pb.distance()
    # cell (0,0) -> (3,3): L2 = 3 sqrt 2 ; cell (0,0) -> (3,0): 3
    assert abs(float(D[0, 0, 15]) - 3 * math.sqrt(2)) < 1e-4 and abs(float(D[0, 0, 3]) - 3) < 1e-4
    with torch.no_grad():
        pb.rho[1] = -20.0                         # p -> 1: Manhattan
    assert abs(float(pb.distance()[1, 0, 15]) - 6.0) < 1e-3
    pb().sum().backward()
    assert torch.isfinite(pb.rho.grad).all() and torch.isfinite(pb.a.grad).all() and pb.rho.grad.abs().sum() > 0


@pytest.mark.parametrize("bias,content", [("minkowski", "dot"), ("minkowski", "minkowski"), ("none", "minkowski")])
def test_knot_aware_variants_train_step_and_round_trip(tmp_path, bias, content):
    torch.manual_seed(3)
    cfg = KnotAwareConfig(hi_res_center=True, attn_pos_bias=bias, attn_content=content, layer_cake=0.5)
    m = KnotAwareNanoTransformer(cfg)
    x = torch.rand(2, 4, 256, 256)
    out = m(x)
    tg = {"mask": (torch.rand(2, 128, 128) > 0.8).float(), "heat": torch.zeros(2, 128, 128),
          "over": torch.randn(2, 2, 128, 128), "over_w": torch.zeros(2, 128, 128),
          "orient": torch.randn(2, 2, 128, 128), "orient_w": torch.ones(2, 128, 128),
          "count": torch.zeros(2), "center": torch.zeros(2, 128, 128), "center_hi": torch.zeros(2, 256, 256)}
    loss, parts = total_loss(out, tg, layer_cake=0.5)
    assert {"lebesgue_mask", "lebesgue_center", "lebesgue_center_hi"} <= set(parts)
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
    save(m, tmp_path / "m.pth")
    m2 = load(tmp_path / "m.pth")
    m.eval()
    with torch.no_grad():
        assert torch.allclose(m(x)["mask"], m2(x)["mask"], atol=1e-5)


def test_default_config_is_unchanged_for_old_checkpoints():
    m = KnotAwareNanoTransformer(KnotAwareConfig(hi_res_center=True))
    names = {n for n, _ in m.named_parameters()}
    assert not any(".pos." in n or "log_temp" in n for n in names)
    ck = ROOT / "checkpoints" / "knot_aware_nanotransformer.pth"
    if ck.exists():
        sd = torch.load(ck, map_location="cpu", weights_only=False)["state_dict"]
        missing, unexpected = m.load_state_dict(sd, strict=False)
        assert not missing and not unexpected
