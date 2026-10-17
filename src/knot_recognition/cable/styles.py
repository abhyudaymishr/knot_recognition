"""
styles.py - Broader rendering styles for training the knot-aware parser (not used by the benchmark).

KnotBench-Photo (synth.render) draws one style: a shaded tube on four backgrounds.
Real knot pictures also come as outlined ribbons, textbook diagrams with gaps,
twisted rope, shaded 3D tubes, multicoloured links and small line drawings. This
module renders the same exact-ground-truth curves in those styles, with the same
annotation format as synth.render, so dense targets are computed identically.

Backgrounds exclude the benchmark's held-out sessions (gradient_dark, texture), so
S3/S4 stay unseen.

Styles
  tube      synth-like shaded cable with outline and highlight
  ribbon    flat fill with a dark outline (common in vector drawings)
  diagram   thin line, under-strand interrupted by a gap
  rope      tube with periodic diagonal twist stripes
  shaded    3D tube: concentric strokes from dark rim to bright core
"""

from __future__ import annotations

import io
import math
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageFilter

from .diagram import Diagram, from_curves_3d, rotation_matrix

STYLES = ("tube", "ribbon", "diagram", "rope", "shaded")
BACKGROUNDS = ("white", "plain", "wood", "paper", "blobs")


def _bg(kind: str, size: int, rng) -> np.ndarray:
    yy, xx = np.mgrid[0:size, 0:size].astype(float)
    if kind == "white":
        return np.full((size, size, 3), 255.0)
    if kind == "plain":
        return np.ones((size, size, 3)) * rng.uniform(40, 250, 3)
    if kind == "wood":
        s = 30 * np.sin(xx / rng.uniform(6, 14) + 3 * np.sin(yy / 60.0))
        return np.stack([170 + s, 120 + 0.8 * s, 75 + 0.5 * s], -1)
    if kind == "paper":
        base = rng.uniform(215, 250)
        n = rng.normal(0, 3, (size, size))
        return np.stack([base + n, base - 4 + n, base - 12 + n], -1)
    # blobs: smooth low-frequency colour field (unlike the benchmark's blocky texture)
    import cv2

    f = rng.normal(0, 1, (6, 6, 3))
    f = cv2.resize(f, (size, size), interpolation=cv2.INTER_CUBIC)
    return np.clip(rng.uniform(80, 200, 3) + 35 * f, 0, 255)


def _col(rng, bg_mean: float) -> Tuple[int, int, int]:
    """A cable colour with enough contrast to the background."""
    for _ in range(20):
        c = rng.uniform(0, 255, 3)
        if abs(c.mean() - bg_mean) > 60 or np.ptp(c) > 120:
            return tuple(int(v) for v in c)
    return (20, 20, 20) if bg_mean > 128 else (235, 235, 235)


def parametric_instance(rng: np.random.Generator, n: int = 720):
    """Round curves unlike braid closures: torus knots, Lissajous knots, random Fourier knots.

    The knot type is not assumed; it is computed from the projected diagram downstream."""
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    kind = str(rng.choice(["torus", "lissajous", "fourier"], p=[0.35, 0.25, 0.4]))
    if kind == "torus":
        p, q = [(2, 3), (2, 5), (2, 7), (3, 4), (3, 5), (2, 9)][int(rng.integers(6))]
        R, r0 = 1.0, float(rng.uniform(0.35, 0.6))
        rr = R + r0 * np.cos(q * t)
        c = np.stack([rr * np.cos(p * t), rr * np.sin(p * t), r0 * np.sin(q * t)], 1)
    elif kind == "lissajous":
        nx, ny, nz = [(3, 2, 5), (2, 3, 7), (3, 2, 7), (3, 4, 5), (2, 5, 3)][int(rng.integers(5))]
        ph = rng.uniform(0, 2 * np.pi, 3)
        c = np.stack([np.cos(nx * t + ph[0]), np.cos(ny * t + ph[1]), np.cos(nz * t + ph[2])], 1)
    else:
        K = int(rng.integers(2, 5))
        c = np.zeros((n, 3))
        for k in range(1, K + 1):
            a, b = rng.normal(0, 1 / k, (2, 3))
            c += np.outer(np.cos(k * t), a) + np.outer(np.sin(k * t), b)
    # gentle random 3D deformation keeps it round but not perfectly symmetric
    c = c + 0.04 * np.stack([np.sin(t * int(rng.integers(1, 4)) + rng.uniform(0, 6)) for _ in range(3)], 1)
    return [c], kind


def _shade(c, f):
    return tuple(int(np.clip(v * f, 0, 255)) for v in c)


def render_styled(curves3d: Sequence[np.ndarray], rng: np.random.Generator, style: Optional[str] = None,
                  size: int = 512, width_px: Optional[int] = None,
                  gap_mult_range: Optional[Tuple[float, float]] = None) -> Tuple[Image.Image, Dict[str, Any], Diagram, Dict]:
    """``gap_mult_range`` (diagram style only) overrides the erased-gap width, in line widths,
    e.g. (1.05, 1.4) for nearly invisible gaps (the random stream is unchanged)."""
    import cv2

    style = style or str(rng.choice(STYLES))
    bgk = "white" if (style == "diagram" and rng.random() < 0.6) else str(rng.choice(BACKGROUNDS))
    R = rotation_matrix(rng)
    D = from_curves_3d(curves3d, view=R)
    allp = np.vstack([c.points for c in D.components])
    lo, hi = allp.min(0), allp.max(0)
    fill = float(rng.uniform(0.6, 0.86))
    scale = (size * fill) / float(np.max(hi - lo))
    room = (size - scale * (hi - lo)) / 2
    off = room + rng.uniform(-0.5, 0.5, 2) * np.maximum(room - 0.03 * size, 0)  # stays inside the frame

    def px(p):
        return float(off[0] + scale * (p[0] - lo[0])), float(size - (off[1] + scale * (p[1] - lo[1])))

    canvas = np.ascontiguousarray(np.clip(_bg(bgk, size, rng), 0, 255).astype(np.uint8))
    bg_mean = float(canvas.mean())
    fill_col = tuple(int(v) for v in np.median(canvas.reshape(-1, 3), 0))
    # log-uniform width 5..32 px at 512: thin threads up to thick drawn ribbons
    w = int(width_px or round(float(np.exp(rng.uniform(np.log(5), np.log(32))))))
    per_comp = rng.random() < 0.5
    cols = [_col(rng, bg_mean) for _ in D.components]
    if not per_comp:
        cols = [cols[0]] * len(cols)
    if style == "shaded":  # a shaded tube needs a mid-tone base to show its rim and core
        cols = [tuple(int(min(255, v + max(0, 110 - np.mean(c)))) for v in c) for c in cols]
    ink = (0, 0, 0) if bg_mean > 110 else (245, 245, 245)
    ow = max(1, int(round(w * rng.uniform(0.12, 0.3))))
    line_w = int(rng.integers(2, max(3, w) + 1))
    gap_mult = float(rng.uniform(1.6, 3.0)) if line_w > 6 else float(rng.uniform(2.2, 4.0))
    if gap_mult_range is not None:
        gap_mult = float(np.random.default_rng(int(gap_mult * 1e6)).uniform(*gap_mult_range))
    hollow = style == "ribbon" and rng.random() < 0.45   # outline only, fill = background
    stripe_period = float(rng.uniform(1.2, 2.5)) * w

    def P16(P):
        return np.round(np.asarray(P) * 16).astype(np.int32).reshape(-1, 1, 2)

    def stroke(pts, col, closed=False, over=False):
        P = np.array([px(p) for p in pts], dtype=np.float64)
        if style == "diagram":
            if over:  # erase a gap in the strand below
                cv2.polylines(canvas, [P16(P)], closed, tuple(int(v) for v in canvas[0, 0]) if bgk != "white" else (255, 255, 255),
                              int(gap_mult * line_w), lineType=cv2.LINE_AA, shift=4)
            cv2.polylines(canvas, [P16(P)], closed, ink if not per_comp else col, line_w, lineType=cv2.LINE_AA, shift=4)
            return
        if style == "ribbon":
            cv2.polylines(canvas, [P16(P)], closed, ink, w + 2 * ow, lineType=cv2.LINE_AA, shift=4)
            cv2.polylines(canvas, [P16(P)], closed, fill_col if hollow else col, w, lineType=cv2.LINE_AA, shift=4)
            return
        if style == "shaded":
            n = 5
            for k in range(n):
                f = 0.45 + 0.85 * k / (n - 1)
                cv2.polylines(canvas, [P16(P)], closed, _shade(col, f), max(1, int(round((w + 2 * ow) * (1 - k / n)))),
                              lineType=cv2.LINE_AA, shift=4)
            return
        edge = _shade(col, 0.4)
        cv2.polylines(canvas, [P16(P)], closed, edge, w + 2 * ow, lineType=cv2.LINE_AA, shift=4)
        cv2.polylines(canvas, [P16(P)], closed, col, w, lineType=cv2.LINE_AA, shift=4)
        if style == "rope":  # diagonal twist stripes at a fixed arc-length period
            seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
            cum = np.concatenate([[0], np.cumsum(seg)])
            for s in np.arange(0, cum[-1], stripe_period):
                i = min(int(np.searchsorted(cum, s)), len(P) - 2)
                t = P[i + 1] - P[i]
                t = t / (np.linalg.norm(t) + 1e-9)
                nrm = np.array([-t[1], t[0]])
                a = P[i] + nrm * w / 2 - t * w * 0.35
                b = P[i] - nrm * w / 2 + t * w * 0.35
                cv2.line(canvas, (int(a[0] * 16), int(a[1] * 16)), (int(b[0] * 16), int(b[1] * 16)), edge,
                         max(1, w // 5), lineType=cv2.LINE_AA, shift=4)
        else:
            hl = _shade(col, 1.4) if sum(col) < 600 else _shade(col, 0.85)
            Hl = P + np.array([-0.18 * w, -0.18 * w])
            cv2.polylines(canvas, [P16(Hl)], closed, hl, max(1, w // 5), lineType=cv2.LINE_AA, shift=4)

    for ci, c in enumerate(D.components):
        stroke(c.points, cols[ci], closed=c.closed)
    vis_w = line_w if style == "diagram" else w + 2 * ow
    half = 2.2 * max(vis_w * (gap_mult if style == "diagram" else 1), 6) / scale
    for cr in D.crossings:
        comp = D.components[cr.over.component]
        ts = np.linspace(cr.over.t - half, cr.over.t + half, 15)
        stroke([comp.point_at(t) for t in ts], cols[cr.over.component], over=True)

    img = canvas.astype(float)
    light = float(rng.uniform(0, 0.35))
    if light > 0 and bgk != "white":
        a = rng.uniform(0, 2 * math.pi)
        yy, xx = np.mgrid[0:size, 0:size].astype(float) / size - 0.5
        img *= (1 - light * (0.5 + np.cos(a) * xx + np.sin(a) * yy))[..., None]
    pil = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))
    blur = float(rng.uniform(0, 1.0)) if style != "diagram" else float(rng.uniform(0, 0.5))
    if blur > 0.05:
        pil = pil.filter(ImageFilter.GaussianBlur(blur))
    noise = float(rng.uniform(0, 6))
    arr = np.asarray(pil, float) + rng.normal(0, noise, (size, size, 3))
    pil = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=int(rng.integers(70, 97)))
    pil = Image.open(io.BytesIO(buf.getvalue())).convert("RGB")

    def tangent_px(comp, t):
        a, b = px(comp.point_at(t - 1e-3)), px(comp.point_at(t + 1e-3))
        v = np.array([b[0] - a[0], b[1] - a[1]])
        return (v / (np.linalg.norm(v) or 1)).tolist()

    # annotated width = visible width; diagram lines are thin, so the target mask uses
    # at least 3 px so the gap region is still learned as cable
    ann = {
        "image_size": [size, size],
        "cable_width_px": max(3, line_w) if style == "diagram" else w + 2 * ow,
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
    for cr in Dpx.crossings:
        cr.over.t *= scale
        cr.under.t *= scale
        cr.pos = (px(cr.pos)[0], size - px(cr.pos)[1])
    return pil, ann, Dpx, {"style": style, "background": bgk, "width": w, "hollow": hollow, "gap_mult": gap_mult,
                           "line_w": line_w}
