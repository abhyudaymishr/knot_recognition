"""
synth.py - Photo-like renders of cables with exact ground truth (KnotBench-Photo, synthetic split).

A 3D curve (from a braid closure plus smooth random deformation) is projected
with a random rotation, so crossings, over/under and the knot type are known
exactly. The cable is drawn as a shaded tube; at each crossing the upper strand
is redrawn on top with its outline, as a real cable occludes the one below.

Sessions vary background, lighting, blur, noise and JPEG quality; cables vary
colour, thickness and 3D shape. These images exercise the pipeline and the
metrics; they are not a substitute for real photographs.
"""

from __future__ import annotations

import io
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from .diagram import Diagram, braid_closure_curves, from_curves_3d, rotation_matrix

DIAGRAM_SESSION = {
    "S0_diagram": {"bg": "plain_white", "light": 0.0, "blur": 0.0, "noise": 0.0, "jpeg": 98},     # textbook gaps
    "S0_silhouette": {"bg": "plain_white", "light": 0.0, "blur": 0.0, "noise": 0.0, "jpeg": 98},  # solid cable mask
}
SESSIONS = {
    "S1_studio": {"bg": "plain_light", "light": 0.0, "blur": 0.0, "noise": 2.0, "jpeg": 95},
    "S2_desk": {"bg": "wood", "light": 0.25, "blur": 0.6, "noise": 4.0, "jpeg": 85},
    "S3_dim": {"bg": "gradient_dark", "light": 0.45, "blur": 1.0, "noise": 7.0, "jpeg": 75},
    "S4_clutter": {"bg": "texture", "light": 0.3, "blur": 0.8, "noise": 5.0, "jpeg": 70},
}
CABLE_COLOURS = {
    "black_coax": (35, 35, 38), "white_coax": (225, 225, 220), "blue_patch": (40, 80, 170),
    "red_rope": (170, 40, 40), "grey_thread": (120, 120, 125),
}


def _background(kind: str, W: int, H: int, rng: np.random.Generator) -> np.ndarray:
    yy, xx = np.mgrid[0:H, 0:W].astype(float)
    if kind == "plain_white":
        img = np.full((H, W, 3), 255.0)
    elif kind == "plain_light":
        img = np.full((H, W, 3), 235.0)
    elif kind == "wood":
        stripes = 30 * np.sin(xx / rng.uniform(6, 14) + 3 * np.sin(yy / 60.0))
        img = np.stack([170 + stripes, 120 + 0.8 * stripes, 75 + 0.5 * stripes], -1)
    elif kind == "gradient_dark":
        g = 70 + 60 * (xx / W)
        img = np.stack([g, g, g + 10], -1)
    else:  # texture
        n = rng.normal(0, 1, (H // 8 + 1, W // 8 + 1))
        n = np.kron(n, np.ones((8, 8)))[:H, :W]
        img = np.stack([150 + 25 * n, 145 + 25 * n, 135 + 25 * n], -1)
    return img


def _shade(c: Tuple[int, int, int], f: float) -> Tuple[int, int, int]:
    return tuple(int(np.clip(v * f, 0, 255)) for v in c)


def render(curves3d: Sequence[np.ndarray], session: str, colour: str, width_px: int, rng: np.random.Generator,
           size: int = 512, view: Optional[np.ndarray] = None) -> Tuple[Image.Image, Dict[str, Any], Diagram]:
    S = {**SESSIONS, **DIAGRAM_SESSION}[session]
    diagram_style = session == "S0_diagram"
    silhouette = session == "S0_silhouette"
    R = rotation_matrix(rng) if view is None else view
    D = from_curves_3d(curves3d, view=R)
    allp = np.vstack([c.points for c in D.components])
    lo, hi = allp.min(0), allp.max(0)
    scale = (size * 0.78) / float(np.max(hi - lo))
    off = (size - scale * (hi - lo)) / 2

    def px(p):
        x = off[0] + scale * (p[0] - lo[0])
        y = size - (off[1] + scale * (p[1] - lo[1]))
        return float(x), float(y)

    import cv2

    bg = _background(S["bg"], size, size, rng)
    canvas = np.ascontiguousarray(np.clip(bg, 0, 255).astype(np.uint8))
    base = CABLE_COLOURS[colour]
    edge = _shade(base, 0.45) if sum(base) > 300 else _shade(base, 0.35)
    hl = _shade(base, 1.35) if sum(base) < 600 else _shade(base, 0.85)
    w = int(width_px)
    ow = max(2, w // 4)

    def stroke(pts, closed=False):
        P = np.array([px(p) for p in pts], dtype=np.float64)
        if silhouette:  # solid mask: crossings visible, over/under not
            P16 = np.round(P * 16).astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(canvas, [P16], closed, (20, 20, 20), int(w), lineType=cv2.LINE_AA, shift=4)
            return
        if diagram_style:  # textbook knot diagram: thin line, white gap on the under strand
            P16 = np.round(P * 16).astype(np.int32).reshape(-1, 1, 2)
            if getattr(stroke, "halo", False):
                cv2.polylines(canvas, [P16], closed, (255, 255, 255), int(3 * max(2, w // 3)), lineType=cv2.LINE_AA, shift=4)
            cv2.polylines(canvas, [P16], closed, (0, 0, 0), int(max(2, w // 3)), lineType=cv2.LINE_AA, shift=4)
            return
        P16 = np.round(P * 16).astype(np.int32).reshape(-1, 1, 2)  # 4 bits sub-pixel precision
        for col, th in ((edge, w + 2 * ow), (base, w)):
            cv2.polylines(canvas, [P16], closed, col, int(th), lineType=cv2.LINE_AA, shift=4)
        # specular highlight offset towards the light
        Hl = np.round((P + np.array([-0.18 * w, -0.18 * w])) * 16).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(canvas, [Hl], closed, hl, max(1, w // 5), lineType=cv2.LINE_AA, shift=4)

    for c in D.components:
        stroke(c.points, closed=c.closed)
    half = 2.2 * (w + 2 * ow) / scale  # local window around each crossing (in curve units)
    stroke.halo = diagram_style
    for cr in D.crossings:
        comp = D.components[cr.over.component]
        ts = np.linspace(cr.over.t - half, cr.over.t + half, 15)
        stroke([comp.point_at(t) for t in ts])
    stroke.halo = False
    img = Image.fromarray(canvas)
    if S["light"] > 0:
        a = rng.uniform(0, 2 * math.pi)
        yy, xx = np.mgrid[0:size, 0:size].astype(float)
        grad = 1 - S["light"] * (0.5 + 0.5 * np.cos(a) * (xx / size - 0.5) * 2 + 0.5 * np.sin(a) * (yy / size - 0.5) * 2)
        img = Image.fromarray(np.clip(np.asarray(img, float) * grad[..., None], 0, 255).astype(np.uint8))
    if S["blur"] > 0:
        img = img.filter(ImageFilter.GaussianBlur(S["blur"]))
    arr = np.asarray(img, float) + rng.normal(0, S["noise"], (size, size, 3))
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=int(S["jpeg"]))
    img = Image.open(io.BytesIO(buf.getvalue())).convert("RGB")

    def tangent_px(comp, t):
        a, b = px(comp.point_at(t - 1e-3)), px(comp.point_at(t + 1e-3))
        v = np.array([b[0] - a[0], b[1] - a[1]])
        return (v / (np.linalg.norm(v) or 1)).tolist()

    ann = {
        "image_size": [size, size],
        "cable_width_px": (3 * max(2, w // 3)) if diagram_style else (w if silhouette else w + 2 * ow),
        "components": [[px(p) for p in c.points[:: max(1, len(c.points) // 400)]] for c in D.components],
        "crossings": [{
            "cid": cr.cid, "pos_px": list(px(cr.pos)),
            "over_tangent_px": tangent_px(D.components[cr.over.component], cr.over.t),
            "under_tangent_px": tangent_px(D.components[cr.under.component], cr.under.t),
            "over_component": cr.over.component, "under_component": cr.under.component,
        } for cr in D.crossings],
    }
    Dpx = Diagram([type(c)(np.array([[px(p)[0], size - px(p)[1]] for p in c.points]), c.closed) for c in D.components],
                  D.crossings, {"image_height": size})
    # crossing parameters were measured on the unscaled curve: rescale t
    for cr in Dpx.crossings:
        cr.over.t *= scale
        cr.under.t *= scale
        cr.pos = (px(cr.pos)[0], size - px(cr.pos)[1])
    return img, ann, Dpx


def chaikin(c: np.ndarray, iters: int = 3) -> np.ndarray:
    """Corner-cutting smoothing of a closed 3D polyline (keeps the knot type for small cuts)."""
    for _ in range(iters):
        nxt = np.roll(c, -1, axis=0)
        c = np.stack([0.75 * c + 0.25 * nxt, 0.25 * c + 0.75 * nxt], axis=1).reshape(-1, c.shape[1])
    return c


def make_instance(word: Sequence[int], seed: int, wobble: float = 0.25) -> List[np.ndarray]:
    rng = np.random.default_rng(seed)
    curves = braid_closure_curves(word, rng=rng, wobble=wobble, samples_per_step=10)
    return [chaikin(c, 3) for c in curves]
