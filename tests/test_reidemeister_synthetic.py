import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from knot_recognition.reidemeister import ReidemeisterConfig, ReidemeisterDetector


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


def _make_canvas(size=128):
    return np.zeros((size, size), dtype=bool)


def _make_r1_loop(size=128, margin=24):
    img = _make_canvas(size)
    top = margin
    bottom = size - margin - 1
    left = margin
    right = size - margin - 1
    _draw_line(img, top, left, top, right)
    _draw_line(img, bottom, left, bottom, right)
    _draw_line(img, top, left, bottom, left)
    _draw_line(img, top, right, bottom, right)
    return img


def _make_r2_parallel(size=128):
    img = _make_canvas(size)
    mid = size // 2
    left = 24
    right = size - 24
    offset = 20
    # two distinct paths between shared junctions (theta-like)
    _draw_line(img, mid, left, mid - offset, left)
    _draw_line(img, mid - offset, left, mid - offset, right)
    _draw_line(img, mid - offset, right, mid, right)

    _draw_line(img, mid, left, mid + offset, left)
    _draw_line(img, mid + offset, left, mid + offset, right)
    _draw_line(img, mid + offset, right, mid, right)

    # add spurs at junctions to reach degree >= 3
    _draw_line(img, mid, left, mid - 10, left - 6)
    _draw_line(img, mid, right, mid + 10, right + 6)
    return img


def _make_r3_triangle(size=128):
    img = _make_canvas(size)
    a = (20, size // 2)
    b = (size - 20, 30)
    c = (size - 20, size - 30)
    _draw_line(img, a[0], a[1], b[0], b[1])
    _draw_line(img, b[0], b[1], c[0], c[1])
    _draw_line(img, c[0], c[1], a[0], a[1])
    # add spurs at vertices for junction degree
    _draw_line(img, a[0], a[1], a[0] - 10, a[1])
    _draw_line(img, a[0], a[1], a[0], a[1] - 10)
    _draw_line(img, b[0], b[1], b[0], b[1] + 10)
    _draw_line(img, b[0], b[1], b[0] - 10, b[1])
    _draw_line(img, c[0], c[1], c[0], c[1] - 10)
    _draw_line(img, c[0], c[1], c[0] + 10, c[1])
    return img


def _count(moves, move_type):
    return sum(1 for m in moves if m.get("move") == move_type)


def test_reidemeister_synthetic_r1():
    cfg = ReidemeisterConfig.synthetic_defaults()
    detector = ReidemeisterDetector(cfg)
    skel = _make_r1_loop()
    moves = detector.detect_skeleton(skel)
    assert _count(moves, "R1") >= 1


def test_reidemeister_synthetic_r2():
    cfg = ReidemeisterConfig.synthetic_defaults()
    detector = ReidemeisterDetector(cfg)
    skel = _make_r2_parallel()
    moves = detector.detect_skeleton(skel)
    assert _count(moves, "R2") >= 1


def test_reidemeister_synthetic_r3():
    cfg = ReidemeisterConfig.synthetic_defaults()
    detector = ReidemeisterDetector(cfg)
    skel = _make_r3_triangle()
    moves = detector.detect_skeleton(skel)
    assert _count(moves, "R3") >= 1
