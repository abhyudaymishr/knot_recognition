"""
neural_parser.py - Photo -> geometric knot diagram with the knot-aware NanoTransformer.

Steps
  1. Network maps: cable probability, crossing heatmap, over-strand axis, strand axis.
  2. Mask = most persistent superlevel set of the cable probability (cable.denoise,
     measure-theoretic: parts of measure below (0.6 w)^2 are null).
  3. Crossings = heatmap peaks. The skeleton is cut on a disc around each crossing,
     leaving arcs whose ends are the crossing's four arms.
  4. Pairing: of the three ways to split four arms into two strands, take the one
     whose strands run straightest (arms most opposed).
  5. Over/under: the strand whose axis agrees best with the predicted over-axis
     is on top; P(correct) = sigmoid(kappa * margin), kappa calibrated on validation.
  6. Tracing arcs through crossings gives oriented polylines and passages, i.e. a
     ``Diagram`` whose PD signs come from geometry (never from arm angle order).
  7. Gates (abstain with a reason): a crossing without exactly four arms, a skeleton
     branch away from any crossing, free ends without a closure record, and the
     Euler check (holes of the stable mask = crossings + pieces).
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .denoise import stable_segmentation
from .diagram import Component, Crossing, Diagram, Passage

KAPPA_DEFAULT = 4.0


def _axial(v) -> np.ndarray:
    a = math.atan2(v[1], v[0])
    return np.array([math.cos(2 * a), math.sin(2 * a)])


def _neighbors(sk, y, x):
    H, W = sk.shape
    out = []
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if (dy or dx) and 0 <= y + dy < H and 0 <= x + dx < W and sk[y + dy, x + dx]:
                out.append((y + dy, x + dx))
    return out


_RING = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]


def _is_junction(sk, y, x) -> bool:
    """Crossing number >= 3: at least three separate skeleton branches leave this pixel.
    (Counting 8-neighbours instead flags every staircase step of a diagonal line.)"""
    H, W = sk.shape
    v = [bool(sk[y + dy, x + dx]) if 0 <= y + dy < H and 0 <= x + dx < W else False for dy, dx in _RING]
    return sum((not v[i]) and v[(i + 1) % 8] for i in range(8)) >= 3


def _prune(sk: np.ndarray, length: int) -> np.ndarray:
    """Remove spur branches shorter than ``length`` (endpoint-to-junction)."""
    sk = sk.copy()
    ys, xs = np.nonzero(sk)
    ends = [(y, x) for y, x in zip(ys, xs) if len(_neighbors(sk, y, x)) == 1]
    for e in ends:
        path, cur, prev = [e], e, None
        while len(path) <= length:
            nb = [p for p in _neighbors(sk, *cur) if p != prev and p not in path]
            if len(nb) >= 2:
                # next to a junction several pixels are adjacent at once: the spur ends here
                for p in path:
                    sk[p] = False
                break
            if len(nb) == 0:
                # an isolated fragment shorter than ``length`` is noise
                for p in path:
                    sk[p] = False
                break
            prev, cur = cur, nb[0]
            if _is_junction(sk, *cur) or len(_neighbors(sk, *cur)) >= 3 and _is_junction(sk, *cur):
                for p in path:
                    sk[p] = False
                break
            path.append(cur)
    return sk


def _trace_arc(sk, start, visited) -> List[Tuple[int, int]]:
    path, cur, prev = [start], start, None
    visited.add(start)
    while True:
        nb = [p for p in _neighbors(sk, *cur) if p not in visited]
        if not nb:
            break
        # prefer 4-neighbours to keep the path thin and ordered
        nb.sort(key=lambda p: abs(p[0] - cur[0]) + abs(p[1] - cur[1]))
        prev, cur = cur, nb[0]
        visited.add(cur)
        path.append(cur)
    return path


def parse(img: np.ndarray, model, heat_thr: float = 0.35, kappa: Optional[float] = None,
          pred: Optional[Dict[str, np.ndarray]] = None, gamma: Optional[float] = None,
          prune_frac: float = 0.8, euler_one_sided: bool = False,
          euler_min_area_frac: float = 0.6) -> Tuple[Optional[Diagram], Dict[str, Any]]:
    import cv2
    from skimage.morphology import skeletonize

    from ..knot_aware import predict
    from .consistency import euler_check

    kappa = kappa or getattr(model, "extra", {}).get("kappa", KAPPA_DEFAULT)
    gamma = gamma if gamma is not None else getattr(model, "extra", {}).get("gamma", 4.0)
    pred = pred or predict(model, img)
    H, W = np.asarray(img).shape[:2]
    cell = H / pred["heat"].shape[0]
    rep: Dict[str, Any] = {"count_head": pred["count"], "abstain_reasons": []}

    # 2. measure-stable mask from the probability map
    prob = np.clip(cv2.resize(pred["mask"].astype(np.float32), (W, H), interpolation=cv2.INTER_CUBIC), 0, 1)
    m0 = prob > 0.5
    dt = cv2.distanceTransform(m0.astype(np.uint8), cv2.DIST_L2, 5)
    sk0 = skeletonize(m0)
    w = float(2 * np.median(dt[sk0])) if sk0.any() else 8.0
    mask, info = stable_segmentation(prob, w, levels=np.linspace(0.15, 0.85, 29))
    rep["mask"] = {k: info[k] for k in ("stable", "threshold", "area_fraction") if k in info}
    rep["width_px"] = w

    # 3. crossings from heatmap peaks (sub-cell refinement by local centroid)
    heat = pred["heat"]
    peaks = (heat == cv2.dilate(heat, np.ones((3, 3), np.uint8))) & (heat > heat_thr)
    cr: List[Dict[str, Any]] = []
    for i, j in zip(*np.nonzero(peaks)):
        i0, i1, j0, j1 = max(i - 1, 0), min(i + 2, heat.shape[0]), max(j - 1, 0), min(j + 2, heat.shape[1])
        wgt = heat[i0:i1, j0:j1]
        yy, xx = np.mgrid[i0:i1, j0:j1]
        cy = float((wgt * (yy + 0.5)).sum() / wgt.sum()) * cell
        cx = float((wgt * (xx + 0.5)).sum() / wgt.sum()) * cell
        # weighted mean of the over-axis; NOT renormalised, so disagreement between
        # test-time views (a shorter mean vector) lowers the over/under margin
        ov = (pred["over"][:, i0:i1, j0:j1] * wgt).sum((1, 2)) / wgt.sum()
        cr.append({"pos": (cx, cy), "score": float(heat[i, j]), "over_axial": ov,
                   "over_consistency": float(np.linalg.norm(ov))})
    rep["crossings_detected"] = len(cr)

    # skeleton, pruned, cut around crossings
    sk = skeletonize(mask)
    # corner spurs run into the mask boundary, where the distance to the edge is small;
    # the true centreline sits about w/2 from both edges
    dtm = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    sk &= dtm > 0.25 * w
    k = max(2, int(prune_frac * w))
    sk = _prune(_prune(sk, k), k)
    R = 0.9 * w + 3
    cut = sk.copy()
    for c in cr:
        cv2.circle(cut.view(np.uint8), (int(round(c["pos"][0])), int(round(c["pos"][1]))), int(math.ceil(R)), 0, -1)
    cut = cut.astype(bool)
    # unexplained branching away from crossings
    ys, xs = np.nonzero(cut)
    branch = [(y, x) for y, x in zip(ys, xs) if _is_junction(cut, y, x)]
    # several pixels of one junction count once
    clusters = []
    for p in branch:
        if not any(abs(p[0] - q[0]) + abs(p[1] - q[1]) <= 4 for q in clusters):
            clusters.append(p)
    rep["unexplained_branch_points"] = len(clusters)
    rep["branch_points_px"] = [(int(x), int(y)) for y, x in clusters]

    # 4. arcs and arms
    visited: set = set()
    arcs: List[List[Tuple[int, int]]] = []
    ends = [(y, x) for y, x in zip(ys, xs) if len(_neighbors(cut, y, x)) == 1]
    def near_crossing(p):
        return any(math.hypot(p[1] - c["pos"][0], p[0] - c["pos"][1]) < R + 4 for c in cr)

    for e in ends:
        if e not in visited:
            a = _trace_arc(cut, e, visited)
            # short arcs are kept when both ends sit at crossings (tightly packed braids)
            if len(a) >= max(3, int(0.5 * w)) or (len(a) >= 2 and near_crossing(a[0]) and near_crossing(a[-1])):
                arcs.append(a)
    loops = 0
    for y, x in zip(ys, xs):
        if (y, x) not in visited:
            a = _trace_arc(cut, (y, x), visited)
            if len(a) > 4 * w:
                arcs.append(a + [a[0]])
                loops += 1
    arms: Dict[int, List[Tuple[int, int, np.ndarray]]] = {i: [] for i in range(len(cr))}
    attach_r = R + 3
    free_ends = 0
    arc_end_cross: Dict[Tuple[int, int], Optional[int]] = {}
    for ai, a in enumerate(arcs):
        if a[0] == a[-1] and len(a) > 4 * w:
            arc_end_cross[(ai, 0)] = arc_end_cross[(ai, 1)] = None
            continue
        for end, p, q in ((0, a[0], a[min(len(a) - 1, int(1.5 * w))]), (1, a[-1], a[max(0, len(a) - 1 - int(1.5 * w))])):
            best, bd = None, attach_r
            for ci, c in enumerate(cr):
                d = math.hypot(p[1] - c["pos"][0], p[0] - c["pos"][1])
                if d < bd:
                    best, bd = ci, d
            arc_end_cross[(ai, end)] = best
            if best is None:
                free_ends += 1
            else:
                c = cr[best]
                v = np.array([q[1] - c["pos"][0], q[0] - c["pos"][1]], float)
                arms[best].append((ai, end, v / (np.linalg.norm(v) + 1e-9)))
    rep["free_ends"] = free_ends
    bad = [ci for ci, a in arms.items() if len(a) != 4]
    rep["crossings_without_four_arms"] = len(bad)

    # 4-5. pairing and over/under
    pairing: Dict[Tuple[int, int], Tuple[int, int]] = {}
    over_pair: Dict[int, Tuple[Tuple[int, int], Tuple[int, int]]] = {}
    conf: Dict[int, float] = {}
    for ci, a in arms.items():
        if len(a) != 4:
            continue
        opts = [((0, 1), (2, 3)), ((0, 2), (1, 3)), ((0, 3), (1, 2))]
        best = min(opts, key=lambda o: float(a[o[0][0]][2] @ a[o[0][1]][2] + a[o[1][0]][2] @ a[o[1][1]][2]))
        s = []
        for (i1, i2) in best:
            u = a[i1][2] - a[i2][2]
            s.append(float(_axial(u) @ cr[ci]["over_axial"]))
        top = 0 if s[0] >= s[1] else 1
        # sigmoid(kappa * margin), shrunk towards 1/2 by the test-time-view consistency c
        # (mean resultant length of the over-axis across views): c^gamma -> 0 when the
        # views disagree, i.e. when the image does not show which strand is on top.
        c_ = min(1.0, cr[ci].get("over_consistency", 1.0))
        conf[ci] = 0.5 + (1 / (1 + math.exp(-kappa * abs(s[0] - s[1]))) - 0.5) * c_ ** gamma
        P = [((a[i1][0], a[i1][1]), (a[i2][0], a[i2][1])) for i1, i2 in best]
        for x, y in P:
            pairing[x], pairing[y] = y, x
        over_pair[ci] = (P[top], P[1 - top])

    # 6. trace components
    diagram = None
    if not bad and arcs:
        comps, passages = [], {}
        used = set()
        for ai in range(len(arcs)):
            if ai in used:
                continue
            pts: List[Tuple[float, float]] = []
            events = []
            cur_arc, cur_end = ai, 0  # traverse arc from end 0 to end 1
            start = (ai, 0)
            closed = False
            guard = 0
            while True:
                used.add(cur_arc)
                path = arcs[cur_arc] if cur_end == 0 else arcs[cur_arc][::-1]
                pts.extend((float(x), float(y)) for y, x in path)
                exit_end = 1 - cur_end
                ci = arc_end_cross.get((cur_arc, exit_end))
                if ci is None:
                    break
                partner = pairing.get((cur_arc, exit_end))
                if partner is None:
                    break
                cx, cy = cr[ci]["pos"]
                pts.append((cx, cy))
                is_over = (cur_arc, exit_end) in over_pair[ci][0]
                events.append((ci, is_over, len(pts) - 1))
                cur_arc, cur_end = partner
                if (cur_arc, cur_end) == start:
                    closed = True
                    break
                guard += 1
                if guard > 4 * len(arcs) + 4:
                    break
            if len(pts) < 3:
                continue
            xy = np.array([[x, H - y] for x, y in pts])
            if closed:
                xy = xy[:-0 or None]
            comp = Component(xy, closed=closed)
            cidx = len(comps)
            comps.append(comp)
            ring = np.vstack([xy, xy[:1]]) if closed else xy
            cum = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(ring, axis=0), axis=1))])
            for ci, is_over, idx in events:
                passages.setdefault(ci, {})["over" if is_over else "under"] = Passage(cidx, float(cum[idx]))
        crossings = []
        for ci, pz in passages.items():
            if "over" in pz and "under" in pz:
                cx, cy = cr[ci]["pos"]
                crossings.append(Crossing(ci + 1, (cx, H - cy), pz["over"], pz["under"], conf.get(ci, 0.5),
                                          {"heat": cr[ci]["score"]}))
        diagram = Diagram(comps, crossings, {"image_height": H, "source": "knot_aware_nanotransformer",
                                             "tangent_window": max(3.0, 0.6 * w), "confidence_calibrated": True})
        rep["components"] = len(comps)
        rep["open_components"] = sum(not c.closed for c in comps)

    # 7. gates
    eu = euler_check(np.asarray(img), len(cr), w, open_ends=free_ends, mask=mask,
                     one_sided=euler_one_sided, min_area_frac=euler_min_area_frac)
    rep["euler"] = eu
    if bad:
        rep["abstain_reasons"].append(f"{len(bad)} crossing(s) without exactly four arms")
    if rep["unexplained_branch_points"]:
        rep["abstain_reasons"].append("skeleton branches away from any detected crossing (possible missed crossing)")
    if free_ends:
        rep["abstain_reasons"].append("free cable ends: record a closure before naming a type")
    if eu.get("applicable") and not eu["consistent"]:
        rep["abstain_reasons"].append("Euler check: " + eu.get("note", "inconsistent"))
    if diagram is not None and not diagram.is_closed and not free_ends:
        rep["abstain_reasons"].append("traversal did not close")
    rep["crossings"] = [{"pos_px": c["pos"], "score": c["score"], "p_over_correct": conf.get(i)} for i, c in enumerate(cr)]
    return diagram, rep
