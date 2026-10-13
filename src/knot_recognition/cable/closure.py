"""
closure.py - Make the closure of an open cable a recorded measurement.

A cable with free ends is not knotted until its ends are joined or fixed. The
knot type depends on *how* they are joined, so a closure has to be declared,
recorded with evidence, and replayed computationally the same way every time.

Computational closures of an open component (ends A ... B, viewed from the camera):

  outer   the closing arc runs around the outside of the picture. Unique and
          crossing-free when both ends lie on the outer boundary; otherwise it
          would have to cross strands and is reported as ambiguous.
  behind  the closing arc runs behind everything (a back panel, a wall socket,
          a cable tray under the tangle): every new crossing has the arc under.
  front   the closing arc runs in front of everything: every new crossing has the arc over.

``behind`` and ``front`` do not depend on the arc's route in the picture,
because an arc lying entirely behind (or in front of) the tangle can be moved
freely. ``closure_sensitivity`` reports whether the knot type changes between them.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .diagram import Component, Crossing, Diagram, Passage, _segment_intersections

PHYSICAL_METHODS = ("fixture", "connectors", "joined_loop")
DECLARED_METHODS = ("declared_outer", "declared_behind", "declared_front")
SIDES = ("outer", "behind", "front")


@dataclass
class ClosureRecord:
    """How the ends of one open cable were joined or fixed, with evidence."""
    method: str                                  # one of PHYSICAL_METHODS + DECLARED_METHODS
    closing_arc_side: str                        # outer | behind | front (what the closure realises)
    component: int = 0
    endpoint_a_px: Optional[Tuple[float, float]] = None
    endpoint_b_px: Optional[Tuple[float, float]] = None
    fixture_id: Optional[str] = None             # e.g. "panel-01", "clamp-board-B"
    description: str = ""
    evidence_images: List[str] = field(default_factory=list)
    operator: str = ""
    recorded_at: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    notes: str = ""

    def validate(self) -> List[str]:
        errs = []
        if self.method not in PHYSICAL_METHODS + DECLARED_METHODS:
            errs.append(f"unknown method {self.method!r}")
        if self.closing_arc_side not in SIDES:
            errs.append(f"closing_arc_side must be one of {SIDES}")
        if self.method in PHYSICAL_METHODS and not (self.fixture_id or self.evidence_images):
            errs.append("a physical closure needs a fixture_id or at least one evidence image")
        if self.method.startswith("declared_") and self.method != f"declared_{self.closing_arc_side}":
            errs.append("declared method and closing_arc_side disagree")
        return errs

    @property
    def is_measurement(self) -> bool:
        return self.method in PHYSICAL_METHODS and not self.validate()

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @staticmethod
    def from_json(s: str) -> "ClosureRecord":
        d = json.loads(s)
        for k in ("endpoint_a_px", "endpoint_b_px"):
            if d.get(k) is not None:
                d[k] = tuple(d[k])
        return ClosureRecord(**d)


# ----------------------------------------------------------------------------- geometry

def _closing_path(diagram: Diagram, ci: int, side: str, margin: float = 0.08) -> Tuple[np.ndarray, int]:
    """Polyline from the end of component ``ci`` back to its start.

    For ``outer`` the path leaves each end along the outward direction that
    meets the fewest strands, then follows the bounding box. Returns
    (path, number of strands the path crosses).
    """
    comp = diagram.components[ci]
    A, B = comp.points[0], comp.points[-1]
    if side in ("behind", "front"):
        return np.vstack([B, A]), -1
    allpts = np.vstack([c.points for c in diagram.components])
    lo, hi = allpts.min(0), allpts.max(0)
    pad = margin * float(np.max(hi - lo) or 1.0)
    lo, hi = lo - pad, hi + pad

    def exit_ray(P):
        best = None
        for k in range(32):
            th = 2 * math.pi * k / 32
            d = np.array([math.cos(th), math.sin(th)])
            ts = []
            for ax in (0, 1):
                if abs(d[ax]) > 1e-9:
                    for bound in (lo[ax], hi[ax]):
                        t = (bound - P[ax]) / d[ax]
                        if t > 0:
                            ts.append(t)
            Q = P + min(ts) * d
            n = _count_hits(diagram, np.vstack([P + 1e-6 * d * np.max(hi - lo), Q]))
            if best is None or n < best[0]:
                best = (n, Q)
        return best

    nb, QB = exit_ray(B)
    na, QA = exit_ray(A)
    path = [B, QB] + _perimeter(QB, QA, lo, hi) + [QA, A]
    return np.asarray(path, dtype=float), nb + na


def _perimeter(P, Q, lo, hi) -> List[np.ndarray]:
    """Corners of the bounding box visited going counter-clockwise from P to Q (both on the box)."""
    corners = [np.array([hi[0], lo[1]]), np.array([hi[0], hi[1]]), np.array([lo[0], hi[1]]), np.array([lo[0], lo[1]])]

    def s(X):  # perimeter coordinate, counter-clockwise from (lo,lo)
        w, h = hi[0] - lo[0], hi[1] - lo[1]
        if abs(X[1] - lo[1]) < 1e-9:
            return X[0] - lo[0]
        if abs(X[0] - hi[0]) < 1e-9:
            return w + X[1] - lo[1]
        if abs(X[1] - hi[1]) < 1e-9:
            return w + h + hi[0] - X[0]
        return 2 * w + h + hi[1] - X[1]

    w, h = hi[0] - lo[0], hi[1] - lo[1]
    cs = [w, w + h, 2 * w + h, 2 * w + 2 * h]
    sp, sq = s(P), s(Q)
    if sq < sp:
        sq += 2 * (w + h)
    out = []
    for k in range(8):
        c = cs[k % 4] + (2 * (w + h) if k >= 4 else 0)
        if sp < c < sq:
            out.append(corners[k % 4])
    return out


def _count_hits(diagram: Diagram, seg: np.ndarray) -> int:
    return sum(len(_segment_intersections(seg, c.points, False, False, c.closed)) for c in diagram.components)


def close_component(diagram: Diagram, ci: int, side: str) -> Tuple[Diagram, Dict[str, Any]]:
    """Return a new diagram with component ``ci`` closed according to ``side``."""
    if side not in SIDES:
        raise ValueError(side)
    comp = diagram.components[ci]
    if comp.closed:
        return diagram.copy(), {"note": "component already closed"}
    path, outer_hits = _closing_path(diagram, ci, side)
    info: Dict[str, Any] = {"side": side}
    if side == "outer":
        info["strands_crossed_by_outer_path"] = outer_hits
        if outer_hits:
            info["ambiguous"] = True
            info["note"] = ("An end lies inside the tangle: an outer closure must cross strands, so it is "
                            "the same as choosing behind or front. Record a physical closure instead.")
    L0 = comp.length
    new_pts = np.vstack([comp.points, path[1:-1]]) if len(path) > 2 else comp.points
    newd = diagram.copy()
    newc = Component(new_pts, closed=True, name=comp.name)
    newd.components[ci] = newc
    # crossings between the closing arc and every strand
    closing = np.vstack([path[0], *path[1:-1], path[-1]])
    seglen = np.linalg.norm(np.diff(closing, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seglen)])
    next_id = max([c.cid for c in diagram.crossings], default=0) + 1
    added = 0
    for cj, other in enumerate(diagram.components):
        hits = _segment_intersections(closing, other.points, False, False, other.closed)
        for si, fs, sj, fu in hits:
            t_close = L0 + cum[si] + fs * seglen[si]
            t_other = other._cum[sj] + fu * other._seglen[sj]
            if cj == ci and (t_other < 1e-6 or t_other > L0 - 1e-6):
                continue
            pos = closing[si] + fs * (closing[si + 1] - closing[si])
            arc = Passage(ci, float(t_close))
            strand = Passage(cj, float(t_other))
            over, under = (arc, strand) if side == "front" else (strand, arc)
            newd.crossings.append(Crossing(next_id, (float(pos[0]), float(pos[1])), over, under, 1.0,
                                           {"from_closure": side}))
            next_id += 1
            added += 1
    info["crossings_added"] = added
    return newd, info


def apply_record(diagram: Diagram, record: ClosureRecord) -> Tuple[Diagram, Dict[str, Any]]:
    errs = record.validate()
    if errs:
        raise ValueError("invalid closure record: " + "; ".join(errs))
    d, info = close_component(diagram, record.component, record.closing_arc_side)
    info["record"] = json.loads(record.to_json())
    info["is_measurement"] = record.is_measurement
    return d, info


def closure_sensitivity(diagram: Diagram) -> Dict[str, Any]:
    """Knot/link type of every open component under each closure, and whether they agree."""
    from .identify import identify_diagram, type_key

    open_ids = [i for i, c in enumerate(diagram.components) if not c.closed]
    results: Dict[str, Any] = {}
    for side in SIDES:
        d = diagram
        notes = []
        for ci in open_ids:
            d, info = close_component(d, ci, side)
            notes.append(info)
        if side == "outer" and any(n.get("ambiguous") for n in notes):
            results[side] = {"type": None, "closure": notes}
            continue
        r = identify_diagram(d)
        results[side] = {"type": type_key(r), "result": r, "closure": notes}
    types = {v["type"] for v in results.values() if v["type"] is not None}
    return {
        "open_components": open_ids,
        "by_closure": results,
        "closure_independent": len(types) == 1,
        "verdict": ("Every closure gives the same type: " + next(iter(types))) if len(types) == 1 else
                   ("The type depends on how the ends are joined: " + ", ".join(sorted(types)) +
                    ". Record a physical closure (ClosureRecord) before reporting a knot type."),
    }
