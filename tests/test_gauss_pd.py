import math
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from knot_recognition import gauss_pd


def _draw_line(img, r0, c0, r1, c1):
    dr = abs(r1 - r0)
    dc = abs(c1 - c0)
    sr = 1 if r0 < r1 else -1
    sc = 1 if c0 < c1 else -1
    err = dr - dc
    r, c = r0, c0
    while True:
        img[r, c] = True
        if r == r1 and c == c1:
            break
        e2 = 2 * err
        if e2 > -dc:
            err -= dc
            r += sr
        if e2 < dr:
            err += dr
            c += sc


def _make_plus(size=21, margin=2):
    img = np.zeros((size, size), dtype=bool)
    mid = size // 2
    _draw_line(img, margin, mid, size - margin - 1, mid)
    _draw_line(img, mid, margin, mid, size - margin - 1)
    return img


def _make_rectangle_loop(size=21, margin=4):
    img = np.zeros((size, size), dtype=bool)
    top = margin
    bottom = size - margin - 1
    left = margin
    right = size - margin - 1
    _draw_line(img, top, left, top, right)
    _draw_line(img, bottom, left, bottom, right)
    _draw_line(img, top, left, bottom, left)
    _draw_line(img, top, right, bottom, right)
    return img


def _make_lissajous(size=101, n_points=800):
    img = np.zeros((size, size), dtype=bool)
    ts = np.linspace(0.0, 2.0 * math.pi, n_points, endpoint=False)
    xs = np.sin(2 * ts)
    ys = np.sin(3 * ts + 0.3)
    xs = (xs + 1.0) * 0.5 * (size - 1)
    ys = (ys + 1.0) * 0.5 * (size - 1)
    pts = [(int(round(y)), int(round(x))) for x, y in zip(xs, ys)]
    for (r0, c0), (r1, c1) in zip(pts, pts[1:] + pts[:1]):
        _draw_line(img, r0, c0, r1, c1)
    return img


def test_cluster_junctions_plus_shape():
    skel = _make_plus()
    G = gauss_pd.skeleton_to_graph(skel, connectivity=4)
    junctions, junction_map = gauss_pd.cluster_junctions(G, deg_thresh=3)
    assert len(junctions) == 1
    assert len(junction_map) > 0


def test_simplify_graph_plus_shape_edges():
    skel = _make_plus()
    G = gauss_pd.skeleton_to_graph(skel, connectivity=4)
    junctions, junction_map = gauss_pd.cluster_junctions(G, deg_thresh=3)
    SG = gauss_pd.simplify_graph(G, junctions, junction_map)
    junction_nodes = [n for n, d in SG.nodes(data=True) if d.get("type") == "junction"]
    assert len(junction_nodes) == 1
    jn = junction_nodes[0]
    assert SG.degree(jn) == 4


def test_extract_gauss_no_crossings_rectangle():
    skel = _make_rectangle_loop()
    gauss, pd = gauss_pd.extract_gauss_code(skel, cfg={"spur_length": 0, "connectivity": 4})
    assert gauss == []
    assert pd["crossings"] == []
    assert pd["edge_labels"] == {}


def test_extract_gauss_crossing_pd_plus_shape():
    skel = _make_plus()
    gauss, pd = gauss_pd.extract_gauss_code(skel, cfg={"spur_length": 0, "connectivity": 4})
    assert pd["crossings"]
    crossing_id, arcs = pd["crossings"][0]
    assert crossing_id == 1
    assert len(arcs) == 4


def test_extract_gauss_lissajous_structure():
    skel = _make_lissajous()
    gauss, pd = gauss_pd.extract_gauss_code(skel, cfg={"spur_length": 0, "connectivity": 8})
    assert len(pd["crossings"]) > 0
    if gauss:
        max_id = max(cid for cid, _ in pd["crossings"])
        assert all(1 <= g <= max_id for g in gauss)
        assert len(gauss) <= 2 * max_id
