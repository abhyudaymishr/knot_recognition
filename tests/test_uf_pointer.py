"""Joint UNetFormer + pointer training: the views of images, dense targets and strand paths agree,
and the compact targets equal the full ones."""

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src"), str(ROOT)]

from knot_recognition.cable.targets import centerline_target, dense_targets
from knot_recognition.strand_pointer import (neighbour_index, target_from_index, targets, transform_record,
                                             transform_xy, true_points)
from train_unetformer_pointer import view_dense, view_image


def _record():
    t = np.linspace(0, 2 * np.pi, 800, endpoint=False)
    P = np.stack([120 + 80 * np.sin(t), 136 + 45 * np.sin(2 * t) + 10 * np.cos(t)], 1).astype(np.float32)
    return {"components": [P], "closed": [True], "width": 6.0, "coords": "edge",
            "crossings": [{"pos": [120.0, 136.0], "over": 0, "under": 0, "over_tangent": [1.0, 1.0]}]}


def _maps(rec):
    """Dense targets of a record, as the training data were made (512 render, stride 4 / 2)."""
    ann = {"image_size": [512, 512], "cable_width_px": rec["width"] * 2,
           "components": [(np.asarray(P, np.float64) * 2 - 0.5).tolist() for P in rec["components"]],
           "crossings": [{"pos_px": [c["pos"][0] * 2 - 0.5, c["pos"][1] * 2 - 0.5], "over_tangent_px": [1.0, 0.0],
                          "under_tangent_px": [0.0, 1.0], "over_component": 0, "under_component": 0, "cid": 0}
                         for c in rec["crossings"]]}
    t = dense_targets(ann, stride=4)
    t["center_hi"] = centerline_target(ann, stride=2, sigma_cells=1.0)
    return t


def test_dense_targets_and_strand_paths_transform_together():
    rec = _record()
    base = _maps(rec)
    for hf in (False, True):
        for rot in range(4):
            moved = view_dense({k: np.asarray(v, np.float32) for k, v in base.items()}, hf, rot)
            ref = _maps(transform_record(rec, hf, rot))
            for k in ("mask", "center_hi", "center", "heat"):
                assert np.abs(moved[k] - ref[k]).mean() < 0.02, (k, hf, rot)
            w = (moved["orient_w"] > 0) & (ref["orient_w"] > 0)
            agree = (moved["orient"][:, w] * ref["orient"][:, w]).sum(0)          # cos of the doubled-angle error
            assert w.sum() > 50 and np.mean(agree > 0.9) > 0.95, (hf, rot)


def test_image_view_matches_torch_and_the_point_transform():
    img = np.zeros((1, 256, 256), np.float32)
    img[0, 40, 200] = 1.0                                                  # pixel (x=200, y=40)
    for hf in (False, True):
        for rot in range(4):
            v = view_image(img, hf, rot)
            t = torch.rot90(torch.from_numpy(img).flip(-1) if hf else torch.from_numpy(img), rot, (-2, -1)).numpy()
            assert np.array_equal(v, t)
            y, x = np.argwhere(v[0])[0]
            px = transform_xy(np.array([200.5, 40.5]), hf, rot)
            assert np.allclose(px, [x + 0.5, y + 0.5])


def test_compact_targets_equal_the_full_ones():
    p1 = true_points(_record())
    p2 = true_points(transform_record(_record(), True, 1))
    xy, valid, tgt, wgt = targets([p1, p2])
    xy2, valid2, nbr, wgt2 = neighbour_index([p1, p2])
    full = target_from_index(nbr)
    assert torch.equal(xy, xy2) and torch.equal(valid, valid2) and torch.equal(wgt, wgt2)
    assert torch.allclose(full[valid], tgt[valid])
