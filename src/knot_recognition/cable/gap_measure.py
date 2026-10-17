"""
gap_measure.py - Image evidence for over/under at a crossing, independent of the network.

The strand on top is continuous, and the strand underneath is interrupted. Its boundary
disappears under the other strand, and in drawings also into a gap. This module measures
that interruption directly on the full-resolution picture (512 px, while the network sees
256 px). Two measures of "the strand is not visible here" are implemented:

  edge  (used) boundary continuity. The contrast across each of the strand's two edges is
        |I(just outside) - I(just inside)| at +-w/2 on the normal, averaged over the two
        sides. The edge counts as missing at t if the contrast falls below tau times its
        median on the free part of the strand. The over strand keeps its outline through
        the crossing. The under strand's outline would lie inside the other strand (or in
        a gap), where there is no edge.
  xsec  (rejected) cross-section correlation. The RGB profile across the strand (+-0.75 w)
        is correlated with its mean free profile, and the profile counts as missing where
        the correlation is below tau. On shaded 3D renders, both strands' profiles change
        at a crossing, and this measure read the wrong strand (see the paper).

For each strand through a crossing (traced centre-line arms joined by a chord), with samples
every 0.5 px of arc length, t = 0 at the crossing, window |t| <= T = 0.5 w / sin(theta) + w,
and reference zone T <= |t| <= T + 1.75 w:
      G_s = Lebesgue measure { t in [-T, T] : strand s not visible at t } / w.
For the strand on top, G is about 0. For the strand underneath, G is about 1/sin(theta) plus
the gaps. D = G(strand read as under) - G(strand read as over) is the picture's evidence for
the network's reading.
  fuse mode: logit p' = logit p + lambda * D. If p' < 1/2, the reading flips.
  veto mode: p -> 1/2 where D < delta (ablation).
tau, lambda (and the veto's delta) are fixed on development data only
(scripts/calibrate_gap_check.py), never on a held-out set.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

# fixed by scripts/calibrate_gap_check.py on development images (results/gap_check/calibration_*.json)
GAP_KIND = "edge"
GAP_TAU = 0.4          # fraction of the free strand's edge contrast below which the edge is missing
GAP_LAMBDA = 1.7       # weight of D in the over/under log-odds (maximum likelihood, development crossings)
GAP_VETO_TAU = 0.4     # veto mode (rejected on development data: it only removed correct answers)
GAP_DELTA = -1.0
PROFILE_POINTS = 13
PROFILE_HALF_WIDTH = 0.75     # in cable widths (xsec)
STEP_PX = 0.5


def strand_path(arcs: Sequence[Sequence[Tuple[int, int]]], arm_a: Tuple[int, int], arm_b: Tuple[int, int],
                max_len: float) -> np.ndarray:
    """Polyline (x, y) through a crossing: arm a traced inwards, then arm b traced outwards.
    Each arm is (arc index, end); end 0 means the arc starts at the crossing. The two arm ends
    next to the crossing are joined by a straight chord."""
    def outward(arm):
        ai, end = arm
        pts = np.asarray(arcs[ai], float)[:, ::-1]          # (y, x) -> (x, y)
        pts = pts if end == 0 else pts[::-1]
        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        cum = np.concatenate([[0.0], np.cumsum(seg)])
        return pts[cum <= max_len]
    a, b = outward(arm_a), outward(arm_b)
    return np.vstack([a[::-1], b])


def _resample(path: np.ndarray, step: float) -> Tuple[np.ndarray, np.ndarray]:
    seg = np.linalg.norm(np.diff(path, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    s = np.arange(0.0, cum[-1] + 1e-9, step)
    return np.stack([np.interp(s, cum, path[:, 0]), np.interp(s, cum, path[:, 1])], 1), s


def _smooth(xy: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0 or len(xy) < 3:
        return xy
    r = int(3 * sigma)
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    pad = np.pad(xy, ((r, r), (0, 0)), mode="edge")
    return np.stack([np.convolve(pad[:, i], k, mode="valid") for i in range(2)], 1)


def _sample(img_f, pts):
    import cv2

    pts = pts.astype(np.float32)
    return cv2.remap(img_f, pts[..., 0], pts[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def profiles(img_f: np.ndarray, path: np.ndarray, w: float):
    """Along a polyline: arc length s, centre xy, cross-section profiles (n, 3K) and boundary
    edge contrast (n,)."""
    xy, s = _resample(path, STEP_PX)
    if len(xy) < 5:
        return s, xy, None, None
    sm = _smooth(xy, sigma=max(1.0, 0.5 * w) / STEP_PX)
    tan = np.gradient(sm, axis=0)
    tan /= np.linalg.norm(tan, axis=1, keepdims=True).clip(1e-9)
    nrm = np.stack([-tan[:, 1], tan[:, 0]], 1)
    v = np.linspace(-PROFILE_HALF_WIDTH * w, PROFILE_HALF_WIDTH * w, PROFILE_POINTS)
    vals = _sample(img_f, xy[:, None, :] + v[None, :, None] * nrm[:, None, :])       # (n, K, 3)
    P = vals.transpose(0, 2, 1).reshape(len(xy), -1)
    a = max(0.25 * w, 1.0)
    off = np.array([-(0.5 * w + a), -(0.5 * w - a), 0.5 * w - a, 0.5 * w + a])
    e = _sample(img_f, xy[:, None, :] + off[None, :, None] * nrm[:, None, :]).astype(np.float64)   # (n, 4, 3)
    edge = 0.5 * (np.linalg.norm(e[:, 0] - e[:, 1], axis=1) + np.linalg.norm(e[:, 3] - e[:, 2], axis=1))
    return s, xy, P, edge


def _corr(P: np.ndarray, ref: np.ndarray) -> np.ndarray:
    a = P - P.mean(1, keepdims=True)
    b = ref - ref.mean()
    sa, sb = np.linalg.norm(a, axis=1), np.linalg.norm(b)
    c = (a @ b) / np.maximum(sa * sb, 1e-9)
    c[sa < 2.0 * math.sqrt(a.shape[1])] = 0.0        # a flat cross-section (spread < 2 grey levels) is not the strand
    return c


def strand_evidence(img_f: np.ndarray, path: np.ndarray, center: Tuple[float, float], w: float,
                    theta_deg: Optional[float]) -> Optional[Dict[str, object]]:
    """Visibility curves of one strand around a crossing, or None when the traced strand is too
    short on either side to give a reference."""
    s, xy, P, edge = profiles(img_f, path, w)
    if P is None:
        return None
    sin_t = max(math.sin(math.radians(theta_deg if theta_deg is not None else 90.0)), 0.35)
    T = 0.5 * w / sin_t + w
    i0 = int(np.argmin(np.linalg.norm(xy - np.asarray(center, float), axis=1)))
    t = s - s[i0]
    ref_zone = (np.abs(t) >= T) & (np.abs(t) <= T + 1.75 * w)
    if (t[ref_zone] < 0).sum() < 0.5 * w / STEP_PX or (t[ref_zone] > 0).sum() < 0.5 * w / STEP_PX:
        return None
    win = np.abs(t) <= T
    edge_ref = float(np.median(edge[ref_zone]))
    return {"corr": _corr(P[win], P[ref_zone].mean(0)).astype(np.float32),
            "edge_rel": (edge[win] / max(edge_ref, 1e-9)).astype(np.float32), "edge_ref": edge_ref,
            "step": STEP_PX, "w": float(w), "T": float(T)}


def gap_length(ev: Optional[Dict[str, object]], tau: float = None, kind: str = None) -> Optional[float]:
    """Lebesgue measure of {t : strand not visible at t} in the crossing window, in cable widths."""
    if ev is None:
        return None
    kind = kind or GAP_KIND
    tau = GAP_TAU if tau is None else tau
    if kind == "edge":
        if "edge_rel" not in ev or ev["edge_ref"] < 8.0:     # no visible outline even on the free strand
            return None
        missing = np.asarray(ev["edge_rel"]) < tau
    else:
        missing = np.asarray(ev["corr"]) < tau
    return float(missing.sum() * ev["step"] / ev["w"])


def evidence_margin(ev_over, ev_under, tau: float = None, kind: str = None) -> Optional[float]:
    """D = G(strand read as under) - G(strand read as over); None if either is unmeasurable."""
    go, gu = gap_length(ev_over, tau, kind), gap_length(ev_under, tau, kind)
    return None if go is None or gu is None else gu - go
