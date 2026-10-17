"""
neural_parser.py - Photo -> geometric knot diagram with the knot-aware NanoTransformer.

Steps
  1. Network maps: cable probability, crossing heatmap, over-strand axis, strand axis.
  2. Mask = most persistent superlevel set of the cable probability (cable.denoise,
     measure-theoretic: parts of measure below (0.6 w)^2 are null).
     The skeleton comes from the centre-line ridge head (thresholded, specks below
     the same measure removed, short facing gaps bridged), which keeps strands that
     touch apart; the mask skeleton is the fallback for models without that head.
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
     Euler check (holes of the ridge band, or of the stable mask = crossings + pieces).
"""

from __future__ import annotations

import math
import os
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


def _bridge_ends(sk: np.ndarray, ridge: np.ndarray, max_len: float):
    """Join pairs of skeleton end points that face each other across a short ridge gap."""
    import cv2

    ys, xs = np.nonzero(sk)
    ends = []
    for y, x in zip(ys, xs):
        nb = _neighbors(sk, y, x)
        if len(nb) == 1:
            # outward direction from the last ~6 skeleton pixels
            path, prev, cur = [(y, x)], None, (y, x)
            for _ in range(6):
                nxt = [p for p in _neighbors(sk, *cur) if p != prev and p not in path]
                if len(nxt) != 1:
                    break
                prev, cur = cur, nxt[0]
                path.append(cur)
            d = np.array([x - path[-1][1], y - path[-1][0]], float)
            ends.append(((y, x), d / (np.linalg.norm(d) + 1e-9)))
    used, n = set(), 0
    out = sk.copy().astype(np.uint8)
    cand = []
    for i in range(len(ends)):
        for j in range(i + 1, len(ends)):
            (p, dp), (q, dq) = ends[i], ends[j]
            v = np.array([q[1] - p[1], q[0] - p[0]], float)
            L = np.linalg.norm(v)
            if L == 0 or L > max_len:
                continue
            u = v / L
            if dp @ u < 0.5 or dq @ -u < 0.5:
                continue
            t = np.linspace(0, 1, max(2, int(L)))
            ry = np.clip(np.round(p[0] + t * (q[0] - p[0])).astype(int), 0, sk.shape[0] - 1)
            rx = np.clip(np.round(p[1] + t * (q[1] - p[1])).astype(int), 0, sk.shape[1] - 1)
            if ridge[ry, rx].mean() < 0.15:
                continue
            cand.append((L, i, j, p, q))
    for L, i, j, p, q in sorted(cand):
        if i in used or j in used:
            continue
        used |= {i, j}
        cv2.line(out, (int(p[1]), int(p[0])), (int(q[1]), int(q[0])), 1, 1)
        n += 1
    from skimage.morphology import skeletonize

    return skeletonize(out.astype(bool)), n


def _drop_open_pieces(band: np.ndarray, cr, w: float, k: int):
    """Remove centre-line pieces that cannot belong to a closed knot diagram: separate pieces
    that carry no detected crossing and are open (their pruned skeleton has an end). On
    cluttered backgrounds, these are background texture read as cable. A closed loop without
    crossings (an unknot, an unlinked component) is kept. With a single piece nothing is removed,
    so a lone open cable still abstains for its free ends."""
    import cv2
    from skimage.morphology import skeletonize

    n_c, lab = cv2.connectedComponents(band.astype(np.uint8), connectivity=8)
    if n_c <= 2:
        return band, 0
    r = int(math.ceil(0.9 * w + 3))
    H, W = band.shape
    out, removed = band.copy(), 0
    for ci in range(1, n_c):
        comp = lab == ci
        has_crossing = False
        for c in cr:
            x, y = int(round(c["pos"][0])), int(round(c["pos"][1]))
            if comp[max(0, y - r):min(H, y + r + 1), max(0, x - r):min(W, x + r + 1)].any():
                has_crossing = True
                break
        if has_crossing:
            continue
        sk = _prune(_prune(skeletonize(comp), k), k)
        ys, xs = np.nonzero(sk)
        if len(ys) == 0 or any(len(_neighbors(sk, y, x)) == 1 for y, x in zip(ys, xs)):
            out[comp] = False
            removed += 1
    return out, removed


def parse(img: np.ndarray, model, heat_thr: float = 0.35, kappa: Optional[float] = None,
          pred: Optional[Dict[str, np.ndarray]] = None, gamma: Optional[float] = None,
          prune_frac: float = 0.8, euler_one_sided: bool = False,
          euler_min_area_frac: float = 0.6, use_center: bool = True, center_thr: float = 0.5,
          bridge_frac: float = 1.5, ridge: str = "hi", euler_gate: bool = True,
          gap_check: Optional[Any] = None, gap_tau: Optional[float] = None,
          gap_delta: Optional[float] = None, gap_lambda: Optional[float] = None,
          clutter_filter: Optional[bool] = None) -> Tuple[Optional[Diagram], Dict[str, Any]]:
    """``use_center``: build the skeleton from the centre-line ridge head when the model has one.
    Where two strands pass closer than a cable width the mask merges them into one blob
    (spurious branches, lost holes); their ridges stay separate. Ridge gaps shorter than
    ``bridge_frac`` cable widths between two facing free ends are bridged.

    ``gap_check`` ("fuse", "veto" or off; default: environment variable KA_GAP, where 1 means
    "fuse"): network-independent evidence for each over/under reading (cable.gap_measure).
    D is the interrupted length (Lebesgue measure) of the strand read as under, minus that of
    the strand read as over, measured on the full-resolution picture.
      fuse: logit p' = logit p + lambda * D, where lambda is a maximum-likelihood fit on
            development crossings. If p' < 1/2, the reading flips.
      veto: p -> 1/2 where D < delta.
    The evidence is always measured and reported (``gap_margin`` per crossing)."""
    import cv2
    from skimage.morphology import skeletonize

    from ..knot_aware import predict
    from .consistency import euler_check

    kappa = kappa or getattr(model, "extra", {}).get("kappa", KAPPA_DEFAULT)
    gamma = gamma if gamma is not None else getattr(model, "extra", {}).get("gamma", 4.0)
    pred = pred or predict(model, img)
    H, W = np.asarray(img).shape[:2]
    cell = H / pred["mask"].shape[0]
    rep: Dict[str, Any] = {"count_head": pred.get("count", float("nan")), "abstain_reasons": []}

    # 2. measure-stable mask from the probability map
    prob = np.clip(cv2.resize(pred["mask"].astype(np.float32), (W, H), interpolation=cv2.INTER_CUBIC), 0, 1)
    m0 = prob > 0.5
    dt = cv2.distanceTransform(m0.astype(np.uint8), cv2.DIST_L2, 5)
    sk0 = skeletonize(m0)
    w = float(2 * np.median(dt[sk0])) if sk0.any() else 8.0
    mask, info = stable_segmentation(prob, w, levels=np.linspace(0.15, 0.85, 29))
    rep["mask"] = {k: info[k] for k in ("stable", "threshold", "area_fraction") if k in info}
    rep["width_px"] = w

    # 3. crossings from heatmap peaks (sub-cell refinement by local centroid). A model with no
    # crossing head (a segmentation baseline) gets its crossings from skeleton junctions below.
    cr: List[Dict[str, Any]] = []
    heat = pred.get("heat")
    peaks = ((heat == cv2.dilate(heat, np.ones((3, 3), np.uint8))) & (heat > heat_thr)) if heat is not None \
        else np.zeros((1, 1), bool)
    for i, j in zip(*np.nonzero(peaks)):
        i0, i1, j0, j1 = max(i - 1, 0), min(i + 2, heat.shape[0]), max(j - 1, 0), min(j + 2, heat.shape[1])
        wgt = heat[i0:i1, j0:j1]
        yy, xx = np.mgrid[i0:i1, j0:j1]
        cy = float((wgt * (yy + 0.5)).sum() / wgt.sum()) * cell
        cx = float((wgt * (xx + 0.5)).sum() / wgt.sum()) * cell
        # weighted mean of the over-axis; NOT renormalised, so disagreement between
        # test-time views (a shorter mean vector) lowers the over/under margin
        if "over" in pred:
            ov = (pred["over"][:, i0:i1, j0:j1] * wgt).sum((1, 2)) / wgt.sum()
            cr.append({"pos": (cx, cy), "score": float(heat[i, j]), "over_axial": ov,
                       "over_consistency": float(np.linalg.norm(ov))})
        else:
            cr.append({"pos": (cx, cy), "score": float(heat[i, j]), "over_axial": None})
    rep["crossings_detected"] = len(cr)

    # skeleton, pruned, cut around crossings
    k = max(2, int(prune_frac * w))
    use_hi = ridge == "hi" and "center_hi" in pred
    if use_center and (use_hi or "center" in pred):
        src = pred["center_hi"] if use_hi else pred["center"]
        ridge = np.clip(cv2.resize(src.astype(np.float32), (W, H), interpolation=cv2.INTER_CUBIC), 0, 1)
        band = ridge > center_thr
        # drop specks below the measure threshold (as in the stable segmentation)
        n_b, lab_b, st_b, _ = cv2.connectedComponentsWithStats(band.astype(np.uint8), connectivity=8)
        keep = np.zeros(n_b, bool)
        keep[1:] = st_b[1:, cv2.CC_STAT_AREA] >= (0.6 * w) ** 2
        band = keep[lab_b]
        if clutter_filter is None:
            clutter_filter = os.environ.get("KA_CLUTTER", "0") == "1"
        if clutter_filter:
            band, rep["clutter_pieces_removed"] = _drop_open_pieces(band, cr, w, k)
        sk = skeletonize(band)
        sk = _prune(_prune(sk, k), k)
        sk, nb = _bridge_ends(sk, ridge, bridge_frac * w)
        rep["skeleton_source"] = "centre-line ridge" + (" (256 grid)" if use_hi else " (128 grid)")
        rep["bridged_gaps"] = nb
        # Euler count on the ridge band: holes between close strands survive here
        topo_mask = cv2.dilate(band.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool) | sk
    else:
        sk = skeletonize(mask)
        # corner spurs run into the mask boundary, where the distance to the edge is small;
        # the true centreline sits about w/2 from both edges
        dtm = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
        sk &= dtm > 0.25 * w
        sk = _prune(_prune(sk, k), k)
        rep["skeleton_source"] = "mask"
        topo_mask = mask
    if heat is None:
        # crossings = clusters of skeleton junctions (an X gives one 4-way or two 3-way points)
        ys0, xs0 = np.nonzero(sk)
        js = [(y, x) for y, x in zip(ys0, xs0) if _is_junction(sk, y, x)]
        groups: List[List[Tuple[int, int]]] = []
        for p in js:
            for g in groups:
                if any(math.hypot(p[0] - q[0], p[1] - q[1]) <= 1.2 * w for q in g):
                    g.append(p)
                    break
            else:
                groups.append([p])
        for g in groups:
            gy, gx = np.mean([q[0] for q in g]), np.mean([q[1] for q in g])
            cr.append({"pos": (float(gx), float(gy)), "score": 1.0, "over_axial": None})
        rep["crossings_detected"] = len(cr)
        rep["crossings_from"] = "skeleton junctions"
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
    free_pts: List[Tuple[float, float]] = []
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
                free_pts.append((float(p[1]), float(p[0])))
            else:
                c = cr[best]
                v = np.array([q[1] - c["pos"][0], q[0] - c["pos"][1]], float)
                arms[best].append((ai, end, v / (np.linalg.norm(v) + 1e-9)))
    rep["free_ends"] = free_ends
    rep["free_end_px"] = free_pts
    bad = [ci for ci, a in arms.items() if len(a) != 4]
    rep["crossings_without_four_arms"] = len(bad)

    # 4-5. pairing and over/under
    from . import gap_measure as gm
    if gap_check is None:
        gap_check = os.environ.get("KA_GAP", "0")
    gap_mode = {True: "fuse", "1": "fuse", "fuse": "fuse", "veto": "veto"}.get(gap_check, "off")
    gap_tau = (gm.GAP_VETO_TAU if gap_mode == "veto" else gm.GAP_TAU) if gap_tau is None else gap_tau
    gap_delta = gm.GAP_DELTA if gap_delta is None else gap_delta
    gap_lambda = gm.GAP_LAMBDA if gap_lambda is None else gap_lambda
    img_f = np.asarray(img)[..., :3].astype(np.float32)
    gap_ev: Dict[int, Tuple[Any, Any]] = {}
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
            s.append(float(_axial(u) @ cr[ci]["over_axial"]) if cr[ci]["over_axial"] is not None else 0.0)
        top = 0 if s[0] >= s[1] else 1
        d1, d2 = a[best[0][0]][2] - a[best[0][1]][2], a[best[1][0]][2] - a[best[1][1]][2]
        cosang = abs(float(d1 @ d2)) / (np.linalg.norm(d1) * np.linalg.norm(d2) + 1e-9)
        cr[ci]["cross_angle_deg"] = float(np.degrees(np.arccos(min(1.0, cosang))))
        # sigmoid(kappa * margin), shrunk towards 1/2 by the test-time-view consistency c
        # (mean resultant length of the over-axis across views): c^gamma -> 0 when the
        # views disagree, i.e. when the image does not show which strand is on top.
        c_ = min(1.0, cr[ci].get("over_consistency", 1.0)) if cr[ci]["over_axial"] is not None else 0.0
        conf[ci] = 0.5 + (1 / (1 + math.exp(-kappa * abs(s[0] - s[1]))) - 0.5) * c_ ** gamma
        P = [((a[i1][0], a[i1][1]), (a[i2][0], a[i2][1])) for i1, i2 in best]
        # image evidence for the reading: interrupted length of the under strand vs the over strand
        ev = []
        for pair in (P[top], P[1 - top]):
            try:
                path = gm.strand_path(arcs, pair[0], pair[1], max_len=6 * w)
                ev.append(gm.strand_evidence(img_f, path, cr[ci]["pos"], w, cr[ci]["cross_angle_deg"]))
            except Exception:  # noqa  (degenerate arm; treated as unmeasurable)
                ev.append(None)
        gap_ev[ci] = (ev[0], ev[1])
        cr[ci]["gap_margin"] = gm.evidence_margin(ev[0], ev[1], gap_tau)
        if gap_mode == "veto" and conf[ci] > 0.5 and (cr[ci]["gap_margin"] is None or cr[ci]["gap_margin"] < gap_delta):
            cr[ci]["gap_veto"] = True
            conf[ci] = 0.5
        elif gap_mode == "fuse" and cr[ci]["gap_margin"] is not None:
            pc = min(max(conf[ci], 1e-9), 1 - 1e-9)
            pf = 1 / (1 + math.exp(-(math.log(pc / (1 - pc)) + gap_lambda * cr[ci]["gap_margin"])))
            if pf < 0.5:                      # the picture outweighs the network: the other strand is on top
                top, pf = 1 - top, 1 - pf
                cr[ci]["gap_flipped"] = True
            cr[ci]["p_network"] = conf[ci]
            conf[ci] = pf
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
    eu = euler_check(np.asarray(img), len(cr), w, open_ends=free_ends, mask=topo_mask,
                     one_sided=euler_one_sided, min_area_frac=euler_min_area_frac)
    rep["euler"] = eu
    if bad:
        rep["abstain_reasons"].append(f"{len(bad)} crossing(s) without exactly four arms")
    if rep["unexplained_branch_points"]:
        rep["abstain_reasons"].append("skeleton branches away from any detected crossing (possible missed crossing)")
    if free_ends:
        rep["abstain_reasons"].append("free cable ends: record a closure before naming a type")
    if euler_gate and eu.get("applicable") and not eu["consistent"]:
        rep["abstain_reasons"].append("Euler check: " + eu.get("note", "inconsistent"))
    if diagram is not None and not diagram.is_closed and not free_ends:
        rep["abstain_reasons"].append("traversal did not close")
    rep["crossings"] = [{"pos_px": c["pos"], "score": c["score"], "p_over_correct": conf.get(i),
                         "cross_angle_deg": c.get("cross_angle_deg"), "gap_margin": c.get("gap_margin"),
                         "gap_veto": c.get("gap_veto", False), "gap_flipped": c.get("gap_flipped", False),
                         "p_network": c.get("p_network")} for i, c in enumerate(cr)]
    rep["gap_check"] = {"mode": gap_mode, "kind": gm.GAP_KIND, "tau": gap_tau, "delta": gap_delta, "lambda": gap_lambda,
                        "vetoed": sum(bool(c.get("gap_veto")) for c in cr),
                        "flipped": sum(bool(c.get("gap_flipped")) for c in cr)}
    rep["_gap_evidence"] = gap_ev            # per-crossing correlation curves (numpy; not for JSON)
    return diagram, rep


def parse_best(img: np.ndarray, model, pred: Optional[Dict[str, np.ndarray]] = None,
               ridges: Optional[Tuple[str, ...]] = None, **kw):
    """Parse with each available centre-line source (256 grid, then 128 grid) and combine.

    Each candidate must pass every gate on its own. If one or more pass and all
    passing candidates give the same knot type (most likely over/under reading), the
    first passing parse is returned; if they disagree, the parse abstains. With no
    passing candidate the first parse is returned with its reasons. Trying two
    skeletons raises coverage; requiring agreement keeps the gates from being shopped.
    """
    from ..knot_aware import predict
    from .identify import identify_diagram, type_key

    pred = pred or predict(model, img)
    sources = [s for s, k in (("hi", "center_hi"), ("lo", "center")) if k in pred and (ridges is None or s in ridges)] or ["lo"]
    results = [(s,) + tuple(parse(img, model, pred=pred, ridge=s, **kw)) for s in sources]
    passing = [(s, D, rep) for s, D, rep in results if D is not None and D.is_closed and not rep["abstain_reasons"]]
    if not passing:
        s, D, rep = results[0]
        rep["candidates"] = {s_: r_["abstain_reasons"][:2] for s_, _, r_ in results}
        return D, rep
    types = set()
    for s, D, rep in passing:
        try:
            types.add(type_key(identify_diagram(D)))
        except Exception as e:  # noqa
            types.add(f"error {str(e)[:30]}")
    s, D, rep = passing[0]
    rep["candidates"] = {s_: r_["abstain_reasons"][:2] for s_, _, r_ in results}
    rep["ridge_used"] = s
    if len(types) > 1:
        rep["abstain_reasons"].append(f"skeleton sources disagree: {sorted(types)}")
    return D, rep
