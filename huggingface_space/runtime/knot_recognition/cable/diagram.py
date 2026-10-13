"""
diagram.py - One geometric diagram model shared by every cable tool.

A ``Diagram`` is a set of oriented components (2D polylines in math coordinates:
x to the right, y up) plus a list of crossings. Each crossing records where the
over and under passages sit on their components (component index + arc-length
parameter ``t``) and a confidence that the over/under reading is correct.

Everything else is derived from it:
  * the signed Gauss order along each component,
  * a PD code in SnapPy/spherogram convention (each 4-tuple starts at the
    incoming under-strand and runs counter-clockwise),
  * crossing changes (``flip``) used by uncertainty propagation,
  * pixel positions used by photo overlays (``to_pixel``).

Image coordinates (row down) must be converted with ``from_image_xy`` first;
using them directly mirrors the knot.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class Passage:
    component: int
    t: float  # arc-length parameter along the component polyline


@dataclass
class Crossing:
    cid: int
    pos: Tuple[float, float]          # math coordinates (x, y up)
    over: Passage
    under: Passage
    confidence: float = 1.0           # P(over/under reading is correct)
    meta: Dict = field(default_factory=dict)


@dataclass
class Component:
    points: np.ndarray                # (N, 2) polyline, math coordinates
    closed: bool = True
    name: str = ""

    def __post_init__(self):
        self.points = np.asarray(self.points, dtype=float)
        seg = np.diff(self._ring(), axis=0)
        self._seglen = np.linalg.norm(seg, axis=1)
        self._cum = np.concatenate([[0.0], np.cumsum(self._seglen)])

    def _ring(self) -> np.ndarray:
        return np.vstack([self.points, self.points[:1]]) if self.closed else self.points

    @property
    def length(self) -> float:
        return float(self._cum[-1])

    def point_at(self, t: float) -> np.ndarray:
        ring = self._ring()
        t = t % self.length if self.closed else min(max(t, 0.0), self.length)
        i = int(np.clip(np.searchsorted(self._cum, t, side="right") - 1, 0, len(self._seglen) - 1))
        f = 0.0 if self._seglen[i] == 0 else (t - self._cum[i]) / self._seglen[i]
        return ring[i] + f * (ring[i + 1] - ring[i])

    def tangent_at(self, t: float, h: Optional[float] = None) -> np.ndarray:
        """Unit direction of travel at ``t``: the exact direction of the polyline segment
        containing ``t`` (a finite difference can straddle a kink and flip a crossing sign).
        Pass ``h`` to average over a window instead (noisy pixel polylines)."""
        if h is None:
            ring = self._ring()
            tt = t % self.length if self.closed else min(max(t, 0.0), self.length)
            i = int(np.clip(np.searchsorted(self._cum, tt, side="right") - 1, 0, len(self._seglen) - 1))
            while self._seglen[i] == 0 and i + 1 < len(self._seglen):
                i += 1
            v = ring[i + 1] - ring[i]
        else:
            v = self.point_at(t + h) - self.point_at(t - h)
        n = np.linalg.norm(v)
        return v / n if n > 0 else np.array([1.0, 0.0])

    def project(self, p: Sequence[float]) -> float:
        """Arc-length parameter of the closest point on the polyline to ``p``."""
        ring = self._ring()
        a, b = ring[:-1], ring[1:]
        ab = b - a
        denom = np.einsum("ij,ij->i", ab, ab)
        denom[denom == 0] = 1.0
        f = np.clip(np.einsum("ij,ij->i", np.asarray(p) - a, ab) / denom, 0, 1)
        q = a + f[:, None] * ab
        i = int(np.argmin(np.linalg.norm(q - np.asarray(p), axis=1)))
        return float(self._cum[i] + f[i] * self._seglen[i])


@dataclass
class Diagram:
    components: List[Component]
    crossings: List[Crossing]
    meta: Dict = field(default_factory=dict)

    # ------------------------------------------------------------------ basics
    @property
    def is_closed(self) -> bool:
        return all(c.closed for c in self.components)

    def crossing(self, cid: int) -> Crossing:
        return next(c for c in self.crossings if c.cid == cid)

    def copy(self) -> "Diagram":
        return copy.deepcopy(self)

    def flip(self, cids: Iterable[int]) -> "Diagram":
        """Crossing change at every crossing in ``cids`` (swap over and under)."""
        d = self.copy()
        ids = set(cids)
        for c in d.crossings:
            if c.cid in ids:
                c.over, c.under = c.under, c.over
                c.confidence = 1.0 - c.confidence
        return d

    def sign(self, c: Crossing) -> int:
        """Right-handed (+1) / left-handed (-1) crossing from the passage tangents."""
        h = self.meta.get("tangent_window")
        to = self.components[c.over.component].tangent_at(c.over.t, h)
        tu = self.components[c.under.component].tangent_at(c.under.t, h)
        # Right-hand rule: positive when (over x under) points at the viewer (+z).
        return 1 if (to[0] * tu[1] - to[1] * tu[0]) > 0 else -1

    def gauss_order(self) -> List[List[Tuple[int, bool]]]:
        """Per component: crossings in travel order as (cid, is_over)."""
        events: List[List[Tuple[float, int, bool]]] = [[] for _ in self.components]
        for c in self.crossings:
            events[c.over.component].append((c.over.t, c.cid, True))
            events[c.under.component].append((c.under.t, c.cid, False))
        return [[(cid, ov) for _, cid, ov in sorted(ev)] for ev in events]

    # ------------------------------------------------------------------ PD code
    def to_pd(self) -> List[Tuple[int, int, int, int]]:
        """PD code (spherogram convention). Requires every component to be closed.

        Components without crossings are returned in ``meta['free_loops']`` and do
        not appear in the PD (they are split unknotted components).
        """
        if not self.is_closed:
            raise ValueError("Diagram has open components; apply a closure first (cable.closure).")
        order = self.gauss_order()
        arc_in: Dict[Tuple[int, bool], int] = {}
        arc_out: Dict[Tuple[int, bool], int] = {}
        label = 0
        self.meta["free_loops"] = 0
        for comp_events in order:
            n = len(comp_events)
            if n == 0:
                self.meta["free_loops"] += 1
                continue
            first = label
            for k, (cid, ov) in enumerate(comp_events):
                arc_in[(cid, ov)] = label if k > 0 else first + n - 1
                arc_out[(cid, ov)] = first + k
                label = first + k + 1
            # incoming arc of the first event is the last arc of this component
            cid0, ov0 = comp_events[0]
            arc_in[(cid0, ov0)] = first + n - 1
            for k in range(1, n):
                cid, ov = comp_events[k]
                arc_in[(cid, ov)] = first + k - 1
            label = first + n
        pd = []
        for c in self.crossings:
            a, cc = arc_in[(c.cid, False)], arc_out[(c.cid, False)]
            b_in, b_out = arc_in[(c.cid, True)], arc_out[(c.cid, True)]
            # Counter-clockwise from incoming under: for a right-handed crossing the
            # outgoing over-strand comes next; for a left-handed one the incoming does.
            if self.sign(c) > 0:
                pd.append((a, b_out, cc, b_in))
            else:
                pd.append((a, b_in, cc, b_out))
        return pd

    def to_link(self):
        import snappy
        pd = self.to_pd()
        return snappy.Link(pd) if pd else snappy.Link([])

    def crossing_index(self) -> List[int]:
        """cid of each PD row, in PD order (PD row i == crossings[i])."""
        return [c.cid for c in self.crossings]

    # ------------------------------------------------------------- conversions
    def to_pixel(self, pos: Tuple[float, float]) -> Tuple[float, float]:
        """Math (x, y-up) -> image (col, row) using the stored image height."""
        h = self.meta.get("image_height", 0)
        return float(pos[0]), float(h - pos[1]) if h else float(-pos[1])


def from_image_xy(points_xy: np.ndarray, image_height: int) -> np.ndarray:
    """Image (col, row-down) -> math (x, y-up)."""
    p = np.asarray(points_xy, dtype=float)
    return np.stack([p[:, 0], image_height - p[:, 1]], axis=1)


# ============================================================================
# Building diagrams from 3D curves (synthetic renders, 3D scans, protein-like data)
# ============================================================================

def _segment_intersections(P: np.ndarray, Q: np.ndarray, same: bool, closed_p: bool, closed_q: bool):
    """All proper intersections between polyline P and polyline Q (2D). Vectorised per segment of P."""
    Pr = np.vstack([P, P[:1]]) if closed_p else P
    Qr = np.vstack([Q, Q[:1]]) if closed_q else Q
    a0, a1 = Pr[:-1], Pr[1:]
    b0, b1 = Qr[:-1], Qr[1:]
    nb = len(b0)
    out = []
    d2 = b1 - b0
    for i in range(len(a0)):
        p, r = a0[i], a1[i] - a0[i]
        denom = r[0] * d2[:, 1] - r[1] * d2[:, 0]
        ok = np.abs(denom) > 1e-12
        qp = b0 - p
        with np.errstate(divide="ignore", invalid="ignore"):
            s = (qp[:, 0] * d2[:, 1] - qp[:, 1] * d2[:, 0]) / denom
            u = (qp[:, 0] * r[1] - qp[:, 1] * r[0]) / denom
        hit = ok & (s > 0) & (s <= 1) & (u > 0) & (u <= 1)
        for j in np.nonzero(hit)[0]:
            if same:
                if j <= i:
                    continue
                if abs(i - j) <= 1 or (closed_p and {i, j} == {0, len(a0) - 1}):
                    continue
            out.append((i, float(s[j]), int(j), float(u[j])))
    return out


def from_curves_3d(curves: Sequence[np.ndarray], closed: Sequence[bool] = None,
                   view: Optional[np.ndarray] = None) -> Diagram:
    """Project 3D polylines along ``view`` (default: +z towards the viewer) and record exact crossings.

    Over/under comes from the depth of the two passages at the intersection, so
    the result is ground truth by construction.
    """
    closed = list(closed) if closed is not None else [True] * len(curves)
    R = np.eye(3) if view is None else np.asarray(view, dtype=float)
    curves3 = [np.asarray(c, dtype=float) @ R.T for c in curves]
    comps = [Component(c[:, :2], closed=cl) for c, cl in zip(curves3, closed)]

    def depth(ci, seg, f):
        c = curves3[ci]
        n = len(c)
        a = c[seg]
        b = c[(seg + 1) % n] if closed[ci] else c[seg + 1]
        return a[2] + f * (b[2] - a[2])

    crossings: List[Crossing] = []
    cid = 1
    for i in range(len(comps)):
        for j in range(i, len(comps)):
            hits = _segment_intersections(comps[i].points, comps[j].points, i == j, closed[i], closed[j])
            for si, fs, sj, fu in hits:
                ti = comps[i]._cum[si] + fs * comps[i]._seglen[si]
                tj = comps[j]._cum[sj] + fu * comps[j]._seglen[sj]
                pos = tuple(comps[i].point_at(ti))
                zi, zj = depth(i, si, fs), depth(j, sj, fu)
                pi, pj = Passage(i, float(ti)), Passage(j, float(tj))
                over, under = (pi, pj) if zi > zj else (pj, pi)
                crossings.append(Crossing(cid, (float(pos[0]), float(pos[1])), over, under, 1.0,
                                          {"depth_gap": float(abs(zi - zj))}))
                cid += 1
    return Diagram(comps, crossings)


# ============================================================================
# 3D curves for test and synthetic data
# ============================================================================

def braid_closure_curves(word: Sequence[int], n_strands: Optional[int] = None, samples_per_step: int = 24,
                         rng: Optional[np.random.Generator] = None, wobble: float = 0.0) -> List[np.ndarray]:
    """3D closed curves of the closure of a braid word (generators +/-i, 1-based).

    Strands run upward; generator i swaps strands at positions i and i+1 with the
    over strand lifted in z according to the generator's sign. Return arcs loop
    around the right-hand side without crossings. ``wobble`` adds smooth noise.
    """
    word = [int(g) for g in word]
    n = n_strands or (max(abs(g) for g in word) + 1 if word else 1)
    steps = max(len(word), 1)
    H = float(steps)
    # position[k] = which strand sits at slot k ; track per-strand x(t), z(t)
    slot_of = list(range(n))  # slot_of[strand]
    paths = {s: [] for s in range(n)}
    for step, g in enumerate(word):
        i = abs(g) - 1
        for m in range(samples_per_step):
            f = m / samples_per_step
            y = step + f
            sm = 0.5 - 0.5 * math.cos(math.pi * f)  # smooth 0..1
            for s in range(n):
                k = slot_of[s]
                x, z = float(k), 0.0
                if k == i:
                    x = i + sm
                    z = (1 if g > 0 else -1) * 0.5 * math.sin(math.pi * f)
                elif k == i + 1:
                    x = i + 1 - sm
                    z = -(1 if g > 0 else -1) * 0.5 * math.sin(math.pi * f)
                paths[s].append((x, y, z))
        a = [s for s in range(n) if slot_of[s] == i][0]
        b = [s for s in range(n) if slot_of[s] == i + 1][0]
        slot_of[a], slot_of[b] = i + 1, i
    # permutation: strand starting at slot k ends at slot slot_of[k]
    end_slot = {s: slot_of[s] for s in range(n)}
    start_strand = {s: s for s in range(n)}  # strand s starts at slot s
    curves, seen = [], set()
    for s0 in range(n):
        if s0 in seen:
            continue
        pts = []
        s = s0
        while True:
            seen.add(s)
            pts.extend(paths[s] if paths[s] else [(float(s), y, 0.0) for y in np.linspace(0, H, samples_per_step, endpoint=False)])
            k_end = end_slot[s]
            # return path: nested rectangles on the right, slot 0 outermost (no crossings)
            d = 0.5 + (n - 1 - k_end) * 0.4
            X = n + (n - 1 - k_end) * 0.4
            corners = [(k_end, H), (k_end, H + d), (X, H + d), (X, -d), (k_end, -d), (k_end, 0.0)]
            for (x0, y0), (x1, y1) in zip(corners[:-1], corners[1:]):
                for f in np.linspace(0, 1, samples_per_step, endpoint=False)[1:]:
                    pts.append((x0 + f * (x1 - x0), y0 + f * (y1 - y0), 0.0))
            s = start_strand[k_end]
            if s == s0:
                break
        c = np.asarray(pts, dtype=float)
        if wobble and rng is not None:
            t = np.linspace(0, 2 * math.pi, len(c), endpoint=False)
            for d in range(3):
                amp = wobble * (0.15 if d == 2 else 1.0)
                c[:, d] += amp * (rng.normal() * np.sin(t * rng.integers(1, 4) + rng.uniform(0, 6)) +
                                  rng.normal() * np.cos(t * rng.integers(1, 4) + rng.uniform(0, 6)))
        curves.append(c)
    return curves


def rotation_matrix(rng: np.random.Generator, max_tilt_deg: float = 15.0) -> np.ndarray:
    """Random in-plane rotation plus a small out-of-plane tilt (keeps the projection generic)."""
    a = rng.uniform(0, 2 * math.pi)
    tx, ty = np.radians(rng.uniform(-max_tilt_deg, max_tilt_deg, 2))
    Rz = np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])
    Rx = np.array([[1, 0, 0], [0, math.cos(tx), -math.sin(tx)], [0, math.sin(tx), math.cos(tx)]])
    Ry = np.array([[math.cos(ty), 0, math.sin(ty)], [0, 1, 0], [-math.sin(ty), 0, math.cos(ty)]])
    return Ry @ Rx @ Rz
