"""
from_traversal.py - Build a ``Diagram`` from the photo traverser's output.

The traverser's own PD code lists the four arcs at a crossing by angle, which
does not encode which strand is on top or the travel direction. This adapter
walks the skeleton graph instead, records for every crossing which passage
runs over, and lets ``Diagram.to_pd`` derive a PD code whose crossing signs
come from the actual geometry.

Ambiguous crossings are kept with their provisional reading (the more
continuous strand on top) and a low confidence, so ``cable.uncertainty`` can
enumerate both readings. Rejected junctions are treated as non-crossings.
Free ends make a component open; ``cable.closure`` must close it before a type
is reported.

Known limitation: a curl that leaves and re-enters the same crossing is a
self-loop edge in the traverser's graph, and its two ends share one pairing key,
so traversal through it is not recoverable from the traverser output.

Confidence mapping: P(correct) = 0.5 + 0.5 * traverser_confidence. The
traverser's score is an evidence margin, not a calibrated probability.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .diagram import Component, Crossing, Diagram, Passage


from ..gauss_pd import _canon_edge  # must match the traverser's edge canonicalisation exactly


def _oriented_path(SG, edge, from_node, incoming: Optional[np.ndarray] = None) -> np.ndarray:
    """Edge path (row, col) oriented away from ``from_node``.

    A self-loop (a curl that leaves and re-enters the same crossing) has both ends
    at the node; its orientation is chosen so the path continues the incoming
    direction as straight as possible, matching how the strand pairing was made.
    """
    u, v, k = edge
    data = SG.get_edge_data(u, v, k) or SG.get_edge_data(v, u, k)
    path = np.asarray(data["path"], dtype=float)  # (row, col)
    if u == v and incoming is not None and len(path) > 2:
        h = min(len(path) - 1, 6)
        d_fwd = path[h] - path[0]
        d_bwd = path[-1 - h] - path[-1]
        if float(np.dot(d_bwd, incoming)) > float(np.dot(d_fwd, incoming)):
            path = path[::-1]
        return path
    p_from = np.asarray(SG.nodes[from_node]["pos"], dtype=float)
    if np.linalg.norm(path[0] - p_from) > np.linalg.norm(path[-1] - p_from):
        path = path[::-1]
    return path


def diagram_from_traversal(result) -> Tuple[Diagram, Dict[str, Any]]:
    SG, pairing = result.skeleton_graph, result.pairing
    if SG is None or pairing is None:
        raise ValueError("TraversalResult lacks skeleton_graph/pairing (re-run with the current traverser)")
    H = result.image_shape[0]
    inters = {i.junction_node: i for i in result.intersections
              if i.junction_node is not None and i.status in ("resolved", "ambiguous")}
    all_edges = [_canon_edge(u, v, k) for u, v, k in SG.edges(keys=True)]
    used = set()
    deg = dict(SG.degree())
    # start from free ends first so open cables are walked end to end
    order = sorted(all_edges, key=lambda e: -int(deg.get(e[0], 0) == 1 or deg.get(e[1], 0) == 1))
    comps: List[Component] = []
    events: Dict[Any, Dict[str, Passage]] = {}
    report = {"components": 0, "open_components": 0, "dropped_crossings": []}
    for edge in order:
        if edge in used:
            continue
        u, v, _ = edge
        start = u if deg.get(u, 0) == 1 else (v if deg.get(v, 0) == 1 else u)
        cur_edge, cur_node = edge, start
        pts: List[np.ndarray] = []
        local_events: List[Tuple[Any, bool, int]] = []
        closed = False
        guard = 0
        while True:
            used.add(cur_edge)
            inc = None
            if len(pts) >= 2:
                h = min(len(pts) - 1, 6)
                inc = np.asarray(pts[-1]) - np.asarray(pts[-1 - h])
            path = _oriented_path(SG, cur_edge, cur_node, inc)
            pts.extend(path if not pts else path[1:])
            a, b, _ = cur_edge
            nxt = b if cur_node == a else a
            if SG.nodes[nxt].get("type") != "junction":
                break  # free end
            inter = inters.get(nxt)
            if inter is not None:
                is_over = cur_edge in inter.over_edges
                is_under = cur_edge in inter.under_edges
                if is_over or is_under:
                    local_events.append((nxt, is_over, len(pts) - 1))
            partner = pairing.get((nxt, cur_edge))
            if partner is None:
                break
            if partner == edge and nxt == start:
                closed = True
                break
            if partner in used:
                closed = partner == edge
                break
            cur_edge, cur_node = partner, nxt
            guard += 1
            if guard > 4 * len(all_edges):
                break
        P = np.asarray(pts, dtype=float)
        if len(P) < 3:
            continue
        xy = np.stack([P[:, 1], H - P[:, 0]], axis=1)
        if closed and np.linalg.norm(xy[0] - xy[-1]) < 1e-6:
            xy = xy[:-1]
        comp = Component(xy, closed=closed)
        ci = len(comps)
        comps.append(comp)
        report["open_components"] += int(not closed)
        cum = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(np.vstack([xy, xy[:1]]) if closed else xy, axis=0), axis=1))])
        for node, is_over, idx in local_events:
            slot = events.setdefault(node, {})
            slot["over" if is_over else "under"] = Passage(ci, float(cum[min(idx, len(cum) - 1)]))
    report["components"] = len(comps)
    crossings = []
    for node, inter in inters.items():
        e = events.get(node, {})
        if "over" not in e or "under" not in e:
            report["dropped_crossings"].append(inter.crossing_id)
            continue
        r, c = inter.pos
        crossings.append(Crossing(inter.crossing_id, (float(c), float(H - r)), e["over"], e["under"],
                                  0.5 + 0.5 * float(inter.confidence),
                                  {"status": inter.status, "traverser_confidence": float(inter.confidence)}))
    D = Diagram(comps, crossings, {"image_height": H, "source": "traverser", "tangent_window": 6.0,
                                   "confidence_calibrated": False})
    return D, report
