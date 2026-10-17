"""
targets.py - Dense training targets for the knot-aware NanoTransformer, from exact synthetic geometry.

All maps are on a grid with ``stride`` pixels per cell (relative to the rendered image):

  mask      fraction of each cell covered by cable (soft target)
  heat      Gaussian bumps (sigma = 1 cell) at crossing centres
  over      axial direction of the upper strand, (cos 2a, sin 2a), on cells near a crossing
  over_w    weight map for ``over`` (1 near crossings, else 0)
  orient    axial direction of the strand, (cos 2a, sin 2a), on cable cells away from crossings
  orient_w  weight map for ``orient``
  center    centre-line ridge: exp(-d^2 / (2 sigma^2)), d = distance to the nearest
            strand centre-line in cells (sigma = 0.8 cell). Two strands closer than a cable
            width merge in ``mask`` but keep separate ridges here.
  count     number of crossings

Axial (doubled-angle) encoding makes a strand direction and its reverse identical,
which is what an image can show.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from .synth import gt_mask


def dense_targets(ann: Dict[str, Any], stride: int = 8) -> Dict[str, np.ndarray]:
    import cv2

    W, H = ann["image_size"]
    gh, gw = H // stride, W // stride
    m = gt_mask(ann).astype(np.float32)
    # soft coverage (fraction of each cell covered by cable): upsampled smoothly it
    # reconstructs boundaries well enough to keep near-parallel strands apart
    mask = cv2.resize(m, (gw, gh), interpolation=cv2.INTER_AREA)

    yy, xx = np.mgrid[0:gh, 0:gw].astype(np.float32) + 0.5
    heat = np.zeros((gh, gw), np.float32)
    over = np.zeros((2, gh, gw), np.float32)
    over_w = np.zeros((gh, gw), np.float32)
    near = np.zeros((gh, gw), bool)
    for c in ann["crossings"]:
        cx, cy = c["pos_px"][0] / stride, c["pos_px"][1] / stride
        d2 = (xx - cx) ** 2 + (yy - cy) ** 2
        heat = np.maximum(heat, np.exp(-d2 / 2.0))
        ci, cj = min(int(cy), gh - 1), min(int(cx), gw - 1)
        heat[ci, cj] = 1.0  # the crossing's own cell is the positive (CenterNet convention)
        a = np.arctan2(c["over_tangent_px"][1], c["over_tangent_px"][0])
        sel = d2 <= 1.6 ** 2
        over[0][sel], over[1][sel] = np.cos(2 * a), np.sin(2 * a)
        over_w[sel] = 1.0
        near |= d2 <= (ann["cable_width_px"] / stride + 1.5) ** 2

    orient = np.zeros((2, gh, gw), np.float32)
    acc = np.zeros((gh, gw), np.float32)
    for comp in ann["components"]:
        P = np.asarray(comp, dtype=np.float64)
        Q = np.vstack([P, P[:1]])
        for a, b in zip(Q[:-1], Q[1:]):
            v = b - a
            L = np.linalg.norm(v)
            if L == 0:
                continue
            ang = np.arctan2(v[1], v[0])
            for f in np.linspace(0, 1, max(2, int(L / (stride / 2)))):
                p = a + f * v
                i, j = int(p[1] / stride), int(p[0] / stride)
                if 0 <= i < gh and 0 <= j < gw:
                    orient[0, i, j] += np.cos(2 * ang)
                    orient[1, i, j] += np.sin(2 * ang)
                    acc[i, j] += 1
    n = np.sqrt(orient[0] ** 2 + orient[1] ** 2)
    ok = (acc > 0) & (n > 0.5 * acc)  # cells crossed by one strand direction only
    orient[:, ok] /= n[ok]
    orient[:, ~ok] = 0
    orient_w = (ok & ~near).astype(np.float32)
    center = centerline_target(ann, stride)
    return {"mask": mask.astype(np.float32), "center": center, "heat": heat, "over": over, "over_w": over_w,
            "orient": orient, "orient_w": orient_w, "count": np.float32(len(ann["crossings"]))}


def centerline_target(ann: Dict[str, Any], stride: int = 4, sigma_cells: float = 0.8) -> np.ndarray:
    import cv2

    W, H = ann["image_size"]
    line = np.full((H, W), 255, np.uint8)
    for comp in ann["components"]:
        P = np.round(np.asarray(comp) * 16).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(line, [P], True, 0, 1, lineType=cv2.LINE_8, shift=4)
    d = cv2.distanceTransform(line, cv2.DIST_L2, 5)
    c = stride // 2
    dc = d[c::stride, c::stride][: H // stride, : W // stride] / stride
    return np.exp(-dc ** 2 / (2 * sigma_cells ** 2)).astype(np.float32)
