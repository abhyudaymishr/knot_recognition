"""Knot-aware NanoTransformer, FFT/measure denoising and the neural parser."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "src"), str(ROOT), str(ROOT / "scripts")]

from knot_recognition.cable import denoise
from knot_recognition.cable.diagram import braid_closure_curves
from knot_recognition.cable.identify import identify_diagram, type_key
from knot_recognition.cable.neural_parser import parse
from knot_recognition.cable.synth import gt_mask, render
from knot_recognition.cable.targets import dense_targets
from knot_recognition.knot_aware import KnotAwareNanoTransformer, preprocess

DB = ROOT / "data" / "knotinfo" / "knotinfo_full_dataset.json"
needs_db = pytest.mark.skipif(not DB.exists(), reason="KnotInfo dataset not extracted")


def _render(word, seed=0, session="S1_studio", width=10):
    return render(braid_closure_curves(word, samples_per_step=40), session, "black_coax", width, np.random.default_rng(seed))


# ----------------------------------------------------------------------------- FFT / measure
def test_homomorphic_flattens_illumination_gradient():
    yy, xx = np.mgrid[0:128, 0:128]
    scene = np.where((xx // 16 + yy // 16) % 2 == 0, 120.0, 160.0)
    lit = scene * (0.4 + 1.2 * xx / 128)              # strong left-to-right lighting
    out = denoise.homomorphic(lit, d0=1 / 64, g_low=0.0)
    left, right = out[:, :32].mean(), out[:, -32:].mean()
    raw_l, raw_r = np.log1p(lit[:, :32]).mean(), np.log1p(lit[:, -32:]).mean()
    assert abs(right - left) < 0.25 * abs(raw_r - raw_l)


def test_wiener_reduces_white_noise():
    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[0:128, 0:128]
    clean = np.sin(xx / 9.0) * 40
    noisy = clean + rng.normal(0, 25, clean.shape)
    out, _ = denoise.wiener(noisy)
    assert np.mean((out - clean) ** 2) < 0.6 * np.mean((noisy - clean) ** 2)


def test_stable_segmentation_recovers_trefoil_topology():
    img, ann, _ = _render([1, 1, 1], session="S0_silhouette")
    g = 255 - denoise.to_gray(np.asarray(img))
    mask, info = denoise.stable_segmentation(g, ann["cable_width_px"])
    assert info["stable"] and info["holes"] - info["pieces"] == len(ann["crossings"]) == 3


# ----------------------------------------------------------------------------- model
def test_model_shapes_and_preprocess_channels():
    img, ann, _ = _render([1, 1, 1])
    x = preprocess(np.asarray(img))
    assert x.shape == (256, 256, 4) and x.dtype == np.uint8
    m = KnotAwareNanoTransformer().eval()
    with torch.no_grad():
        out = m(torch.from_numpy(x).permute(2, 0, 1).float()[None] / 255)
    assert out["mask"].shape == (1, 128, 128) and out["over"].shape == (1, 2, 128, 128) and out["count"].shape == (1,)


def test_targets_mark_every_crossing_and_axial_direction():
    img, ann, _ = _render([1, -2, 1, -2])
    t = dense_targets(ann, stride=4)
    assert (t["heat"] > 0.99).sum() >= len(ann["crossings"])
    assert t["count"] == len(ann["crossings"])
    v = t["over"][:, t["over_w"] > 0]
    assert np.allclose(np.linalg.norm(v, axis=0), 1, atol=1e-5)


# ----------------------------------------------------------------------------- parser on oracle maps
@needs_db
@pytest.mark.parametrize("word", [[1, 1, 1], [-1, -1, -1], [1, -2, 1, -2], [1, 1]])
def test_parser_on_ground_truth_maps_gives_correct_type(word):
    """With perfect network outputs the parser must produce a geometrically valid diagram of the right type."""
    import cv2

    from knot_recognition.cable.diagram import from_curves_3d

    img, ann, D = _render(word, session="S1_studio", width=11)
    t = dense_targets(ann, stride=4)
    soft = cv2.resize(gt_mask(ann).astype(np.float32), (128, 128), interpolation=cv2.INTER_AREA)
    pred = {"mask": soft, "heat": t["heat"], "over": t["over"], "orient": t["orient"], "count": float(t["count"])}
    Dp, rep = parse(np.asarray(img), None, pred=pred, kappa=8)
    assert not rep["abstain_reasons"], rep["abstain_reasons"]
    assert type_key(identify_diagram(Dp)) == type_key(identify_diagram(D))


def test_parser_abstains_when_a_crossing_is_missed():
    import cv2

    img, ann, _ = _render([1, 1, 1], session="S1_studio", width=11)
    t = dense_targets(ann, stride=4)
    c = ann["crossings"][0]["pos_px"]
    yy, xx = np.mgrid[0:128, 0:128]
    heat = t["heat"] * (((xx + 0.5) * 4 - c[0]) ** 2 + ((yy + 0.5) * 4 - c[1]) ** 2 > 30 ** 2)
    soft = cv2.resize(gt_mask(ann).astype(np.float32), (128, 128), interpolation=cv2.INTER_AREA)
    pred = {"mask": soft, "heat": heat, "over": t["over"], "orient": t["orient"], "count": 2.0}
    _, rep = parse(np.asarray(img), None, pred=pred, kappa=8)
    assert rep["abstain_reasons"]  # a hidden crossing is never silently dropped


# ----------------------------------------------------------------------------- augmentation
def test_dihedral_augmentation_keeps_axial_targets_consistent():
    import cv2
    from train_knot_aware import axial_hflip, axial_rot90, axial_vflip

    def axial_of(a2):
        gx = cv2.Sobel(a2, cv2.CV_64F, 1, 0)
        gy = cv2.Sobel(a2, cv2.CV_64F, 0, 1)
        v = -np.array([(gx * gx).sum() - (gy * gy).sum(), 2 * (gx * gy).sum()])
        return v / np.linalg.norm(v)

    for deg in (15, 50, 110):
        a = np.radians(deg)
        img = np.zeros((101, 101), np.float32)
        cv2.line(img, (int(50 - 40 * np.cos(a)), int(50 - 40 * np.sin(a))),
                 (int(50 + 40 * np.cos(a)), int(50 + 40 * np.sin(a))), 1.0, 3)
        v = torch.tensor([np.cos(2 * a), np.sin(2 * a)]).float().reshape(1, 2, 1, 1)
        t = torch.from_numpy(img)[None]
        for f_img, f_vec in ((lambda x: x.flip(-1), axial_hflip), (lambda x: x.flip(-2), axial_vflip),
                             (lambda x: torch.rot90(x, 1, (-2, -1)), axial_rot90)):
            assert np.dot(axial_of(f_img(t)[0].numpy()), f_vec(v)[0, :, 0, 0].numpy()) > 0.95
