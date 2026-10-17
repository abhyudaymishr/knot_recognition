"""Next-point pointer: targets, decoding, the head, and its link to the UNetFormer."""

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from knot_recognition.strand_pointer import (PointerConfig, Points, StrandPointerHead, decode, label_points,
                                             load_head, pointer_loss, save_head, straightest, targets,
                                             transform_record, true_points, walks)
from knot_recognition.unetformer import UNetFormer


def _figure_eight(width=6.0):
    """One closed strand that crosses itself at the centre (128, 128), passing along (1, 1)
    at t = 0 and along (-1, 1) at t = pi."""
    t = np.linspace(0, 2 * np.pi, 800, endpoint=False)
    P = np.stack([128 + 80 * np.sin(t), 128 + 40 * np.sin(2 * t)], 1).astype(np.float32)
    return {"components": [P], "closed": [True], "width": width,
            "crossings": [{"pos": [128.0, 128.0], "over": 0, "under": 0, "over_tangent": [1.0, 1.0]}]}


def _order(strand, start):
    o = strand["order"]
    k = o.index(start)
    o = o[k:] + o[:k]
    return o if len(o) < 2 or o[1] != o[-1] else o


def test_reading_order_is_not_strand_order():
    # points in reading order 1 2 3 / 4 5 6 (0-based 0..5); the strand runs 1, 2, 3, 6, 4, 5
    xy = np.array([[0, 0], [1, 0], [2, 0], [0, 1], [1, 1], [2, 1]], np.float32) * 8 + 100
    strand = [0, 1, 2, 5, 3, 4]
    edges = [(a, b, a == 5 and b == 3) for a, b in zip(strand[:-1], strand[1:])]
    pts = Points(xy, np.zeros(6, int), np.arange(6.0), edges)
    _, _, tgt, _ = targets([pts])
    assert tgt[0, 5, 3] == 0.5 and tgt[0, 5, 4] == 0.0      # next(6) = 4, not its raster neighbour 5
    found = walks(decode(tgt[0].numpy()), 6)
    assert len(found) == 1 and not found[0]["closed"]
    o = found[0]["order"]
    assert o in (strand, strand[::-1])


def test_true_points_jump_straight_across_the_crossing():
    rec = _figure_eight()
    pts = true_points(rec)
    jumps = [(i, j) for i, j, jp in pts.edges if jp]
    assert len(jumps) == 2                                   # each pass of the strand jumps once
    for i, j in jumps:
        v = pts.xy[j] - pts.xy[i]
        v = v / np.linalg.norm(v)
        # straight on along one of the two passes, never a turn onto the other pass
        assert max(abs(v @ np.array([1, 1])), abs(v @ np.array([-1, 1]))) / np.sqrt(2) > 0.95
    assert all(len(n) == 2 for n in pts.neighbours())        # a closed strand: two neighbours each
    found = walks([e[:2] for e in pts.edges], pts.n)
    assert len(found) == 1 and found[0]["closed"] and len(found[0]["order"]) == pts.n


def test_labelling_points_recovers_the_true_order():
    rec = _figure_eight()
    pts = true_points(rec)
    lab = label_points(pts.xy, rec)
    assert {frozenset(e[:2]) for e in lab.edges} == {frozenset(e[:2]) for e in pts.edges}
    assert sum(e[2] for e in lab.edges) == 2
    off = label_points(np.array([[10.0, 10.0]]), rec)
    assert off.comp[0] == -1 and not off.edges


def test_straightest_baseline_is_exact_on_clean_points():
    rec = _figure_eight()
    pts = true_points(rec)
    assert {frozenset(e) for e in straightest(pts.xy, rec["width"])} == {frozenset(e[:2]) for e in pts.edges}


def test_record_transform_matches_the_image_transform():
    import cv2

    rec = _figure_eight()
    rec["components"] = [rec["components"][0] * np.array([0.9, 1.3], np.float32) + np.array([5, -20], np.float32)]

    def draw(r):
        m = np.zeros((256, 256), np.uint8)
        # records use grid_sample's convention (pixel j spans [j, j + 1)); OpenCV draws at centres
        P = np.round((r["components"][0] - 0.5) * 16).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(m, [P], True, 1, 3, shift=4)
        return torch.from_numpy(m).float()

    for hf in (False, True):
        for rot in range(4):
            img = draw(rec)
            img = torch.rot90(img.flip(-1) if hf else img, rot, (-2, -1))
            ref = draw(transform_record(rec, hf, rot))
            iou = float((img * ref).sum() / ((img + ref) > 0).sum())
            assert iou > 0.85, (hf, rot, iou)


RAD = torch.tensor([7.0])          # crossing cut radius of the test strand (width 6)


def _feats(B=1, seed=0):
    g = torch.Generator().manual_seed(seed)
    cfg = PointerConfig()
    return {s: torch.randn(B, c, s, s, generator=g) for s, c in zip(cfg.grids, cfg.feat_dims)}


def test_head_is_equivariant_to_point_order_and_learns():
    torch.manual_seed(0)
    head = StrandPointerHead().eval()
    pts = true_points(_figure_eight())
    xy, valid, tgt, wgt = targets([pts])
    feats = _feats()
    perm = torch.randperm(pts.n)
    with torch.no_grad():
        a = head(feats, xy, valid, RAD)[0]
        b = head(feats, xy[:, perm], valid, RAD)[0]
    full = torch.cat([perm, torch.tensor([pts.n])])
    assert torch.allclose(a[perm][:, full], b, atol=1e-4)
    head.train()
    opt = torch.optim.Adam(head.parameters(), 3e-3)
    first = None
    for _ in range(40):
        loss = pointer_loss(head(feats, xy, valid, RAD), tgt, valid, wgt)
        first = first if first is not None else float(loss)
        opt.zero_grad(); loss.backward(); opt.step()
    assert all(torch.isfinite(p.grad).all() for p in head.parameters() if p.grad is not None)
    assert float(loss) < 0.5 * first
    with torch.no_grad():
        p = torch.softmax(head(feats, xy, valid, RAD), -1)[0].numpy()
    hits = {frozenset(e) for e in decode(p)} & {frozenset(e[:2]) for e in pts.edges}
    assert len(hits) >= 0.9 * len(pts.edges)


def test_padding_does_not_change_the_points_that_are_there():
    torch.manual_seed(0)
    head = StrandPointerHead().eval()
    pts = true_points(_figure_eight())
    short = Points(pts.xy[:30], pts.comp[:30], pts.s[:30], [e for e in pts.edges if e[0] < 30 and e[1] < 30])
    xy, valid, _, _ = targets([short, pts])
    feats = {k: v.repeat(2, 1, 1, 1) for k, v in _feats().items()}
    with torch.no_grad():
        both = head(feats, xy, valid, RAD.repeat(2))
        alone = head({k: v[:1] for k, v in feats.items()}, xy[:1, :30], valid[:1, :30], RAD)
    keep = torch.cat([torch.arange(30), torch.tensor([xy.shape[1]])])
    assert torch.allclose(both[0, :30][:, keep], alone[0], atol=1e-4)


def test_unetformer_feature_maps_feed_the_head(tmp_path):
    torch.manual_seed(0)
    trunk = UNetFormer().eval()
    with torch.no_grad():
        out = trunk(torch.rand(1, 4, 256, 256), return_feats=True)
    cfg = PointerConfig()
    for s, c in zip(cfg.grids, cfg.feat_dims):
        assert tuple(out["feats"][s].shape) == (1, c, s, s)
    head = StrandPointerHead()
    pts = true_points(_figure_eight())
    xy, valid, _, _ = targets([pts])
    with torch.no_grad():
        lg = head(out["feats"], xy, valid, RAD)
    assert lg.shape == (1, pts.n, pts.n + 1)
    save_head(head, tmp_path / "ptr.pth", {"trunk": "x"})
    h2 = load_head(tmp_path / "ptr.pth")
    with torch.no_grad():
        assert torch.allclose(h2(out["feats"], xy, valid, RAD), head.eval()(out["feats"], xy, valid, RAD), atol=1e-5)
    assert h2.extra == {"trunk": "x"}


def test_point_tangents_follow_the_strand_and_ignore_isolated_points():
    from knot_recognition.strand_pointer import point_tangents

    line = [[100 + 4 * k, 50 + 4 * k] for k in range(6)]              # along (1, 1)
    xy = torch.tensor([line + [[10.0, 200.0]]], dtype=torch.float32)  # plus one isolated point
    t, coh = point_tangents(xy, torch.ones(1, 7, dtype=torch.bool), 6.4)
    assert torch.allclose((t[0, :6] @ torch.tensor([1.0, 1.0])).abs() / 2 ** 0.5, torch.ones(6), atol=1e-4)
    assert (coh[0, :6] > 0.99).all() and coh[0, 6] == 0


def test_a_neighbour_out_of_reach_is_a_strand_end():
    xy = np.array([[10, 10], [14, 10], [90, 10]], np.float32)            # 76 px > the 64 px reach
    pts = Points(xy, np.zeros(3, int), np.arange(3.0), [(0, 1, False), (1, 2, True)])
    _, _, tgt, _ = targets([pts])
    assert tgt[0, 1, 2] == 0 and tgt[0, 1, 0] == 0.5 and tgt[0, 1, 3] == 0.5


def test_open_ends_point_out_of_strand_ends():
    from knot_recognition.strand_pointer import open_ends

    xy = torch.tensor([[[0.0, 0.0], [4.0, 0.0], [8.0, 0.0]]]) + 50
    o = open_ends(xy, torch.ones(1, 3, dtype=torch.bool), 6.4)[0]
    assert torch.allclose(o[0], torch.tensor([-1.0, 0.0])) and torch.allclose(o[2], torch.tensor([1.0, 0.0]))
    assert o[1].norm() < 1e-6


def test_the_head_requires_the_cut_radius():
    import pytest

    head = StrandPointerHead()
    xy, valid, _, _ = targets([true_points(_figure_eight())])
    with pytest.raises(ValueError):
        head(_feats(), xy, valid)


def test_linear_check_keeps_true_jumps_and_rejects_a_wrong_turn():
    import cv2

    from knot_recognition.strand_pointer import JUMP_CHECKS, check_jumps

    rec = _figure_eight()
    pts = true_points(rec)
    ridge = np.zeros((256, 256), np.float32)       # the centre-line runs on through the crossing
    P = np.round((rec["components"][0] - 0.5) * 16).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(ridge, [P], True, 1.0, 3, shift=4)
    true_edges = [e[:2] for e in pts.edges]
    for name, th in JUMP_CHECKS.items():
        kept, rejected = check_jumps(true_edges, pts.xy, ridge, 4.0, *th)
        assert not rejected and len(kept) == len(true_edges), name
    # swap partners at the crossing: each arm turns onto the other pass instead of going straight on
    (a, b), (c, d) = [e[:2] for e in pts.edges if e[2]]
    wrong = [e for e in true_edges if {e[0], e[1]} not in ({a, b}, {c, d})] + [(a, c), (b, d)]
    _, rejected = check_jumps(wrong, pts.xy, ridge, 4.0, *JUMP_CHECKS["strict"])
    assert {frozenset(e) for e in rejected} == {frozenset((a, c)), frozenset((b, d))}
    # at a right-angle crossing a turn is only 45 degrees off each arm and mostly on the ridge:
    # the balanced setting (70 degrees, no ridge minimum) lets it through, by design of its trade-off
    _, rejected = check_jumps(wrong, pts.xy, ridge, 4.0, *JUMP_CHECKS["balanced"])
    assert not rejected


def test_the_ridge_check_passes_a_jump_and_fails_a_turn():
    from knot_recognition.strand_pointer import ridge_check, verify_relink

    # two straight strands crossing at (128, 128); the ridge is drawn through the crossing on both
    yy, xx = np.mgrid[0:256, 0:256] + 0.5
    d1 = np.abs((xx - 128) - (yy - 128)) / np.sqrt(2)                 # along (1, 1)
    d2 = np.abs((xx - 128) + (yy - 128)) / np.sqrt(2)                 # along (1, -1)
    ridge = np.maximum(np.exp(-d1 ** 2 / 2), np.exp(-d2 ** 2 / 2))
    u, v = np.array([1.0, 1.0]) / np.sqrt(2), np.array([1.0, -1.0]) / np.sqrt(2)
    c = np.array([128.0, 128.0])
    # each arm: an end 12 px from the crossing and one point 4 px further out
    arms = [c - 12 * u, c - 16 * u, c + 12 * u, c + 16 * u, c - 12 * v, c - 16 * v, c + 12 * v, c + 16 * v]
    xy = np.array(arms)
    adj = [[1], [0], [3], [2], [5], [4], [7], [6]]                     # ends 0, 2, 4, 6
    assert ridge_check(ridge, xy, adj, 0, 2) > 0.8                     # straight across: on the ridge
    assert ridge_check(ridge, xy, adj, 0, 4) < 0.2                     # a turn onto the other strand
    # the head took the turns; re-checking drops them and links straight across
    prob = np.full((8, 9), 0.05)
    out = verify_relink([(0, 1), (2, 3), (4, 5), (6, 7), (0, 4), (2, 6)], prob, xy, ridge, radius=10.0)
    assert {frozenset(e) for e in out} >= {frozenset((0, 2)), frozenset((4, 6))}
    assert frozenset((0, 4)) not in {frozenset(e) for e in out}


def test_the_default_recheck_keeps_straight_jumps():
    from knot_recognition.strand_pointer import recheck

    yy, xx = np.mgrid[0:256, 0:256] + 0.5
    d1 = np.abs((xx - 128) - (yy - 128)) / np.sqrt(2)
    d2 = np.abs((xx - 128) + (yy - 128)) / np.sqrt(2)
    ridge = np.maximum(np.exp(-d1 ** 2 / 2), np.exp(-d2 ** 2 / 2))
    u, v = np.array([1.0, 1.0]) / np.sqrt(2), np.array([1.0, -1.0]) / np.sqrt(2)
    c = np.array([128.0, 128.0])
    xy = np.array([c - 12 * u, c - 16 * u, c + 12 * u, c + 16 * u, c - 12 * v, c - 16 * v, c + 12 * v, c + 16 * v])
    out = {frozenset(e) for e in recheck([(0, 1), (2, 3), (4, 5), (6, 7), (0, 4), (2, 6)], np.full((8, 9), 0.05), xy, ridge, 10.0)}
    assert frozenset((0, 2)) in out and frozenset((4, 6)) in out and frozenset((0, 4)) not in out


def test_nearest_candidates_score_like_all_pairs():
    torch.manual_seed(0)
    dense = StrandPointerHead(PointerConfig(candidates=0)).eval()
    sparse = StrandPointerHead(PointerConfig(candidates=16)).eval()
    sparse.load_state_dict(dense.state_dict())
    pts = true_points(_figure_eight())
    xy, valid, _, _ = targets([pts])
    feats = _feats()
    with torch.no_grad():
        a, b = dense(feats, xy, valid, RAD)[0], sparse(feats, xy, valid, RAD)[0]
    scored = b[:, :-1] > -1e3
    assert (scored.sum(1) == 16).all()                                    # each point: its 16 nearest
    assert torch.allclose(a[:, :-1][scored], b[:, :-1][scored], atol=1e-4)
    assert torch.allclose(a[:, -1], b[:, -1], atol=1e-5)                  # the 'none' slot is unchanged
    d = (xy[0, :, None] - xy[0, None]).norm(dim=-1) + torch.eye(pts.n) * 1e9
    assert (d[scored].max() <= d.sort(1).values[:, 15].max() + 1e-4)       # the scored ones are the nearest


def test_a_neighbour_the_head_cannot_choose_counts_as_none():
    logits = torch.tensor([[[0.0, -1e4, 0.0, 0.0]]])                      # point 0: candidate 1 masked
    tgt = torch.tensor([[[0.0, 0.5, 0.5, 0.0]]])                           # neighbours 1 (masked) and 2
    loss = pointer_loss(logits, tgt, torch.ones(1, 1, dtype=torch.bool), torch.ones(1, 1))
    ref = -(0.5 * torch.log_softmax(logits, -1)[0, 0, 2] + 0.5 * torch.log_softmax(logits, -1)[0, 0, 3])
    assert loss < 10 and torch.allclose(loss, ref)
