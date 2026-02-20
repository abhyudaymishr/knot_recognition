import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from knot_recognition.solver import SolverConfig, reduce_skeleton


def _make_loop_with_tail() -> np.ndarray:
    skel = np.zeros((7, 7), dtype=bool)
    # square loop
    skel[2, 2:5] = True
    skel[4, 2:5] = True
    skel[2:5, 2] = True
    skel[2:5, 4] = True
    # tail attached to bottom center
    skel[4:7, 3] = True
    return skel


def test_reduce_r1_loop_removes_nodes():
    skel = _make_loop_with_tail()
    cfg = SolverConfig(max_iters=2, enable_r2=False, r1_max_cycle_len=50, r1_min_cycle_len=4)
    reduced, stats = reduce_skeleton(skel, cfg)
    assert stats["r1_removed"] >= 1
    assert reduced.sum() < skel.sum()


def test_reduce_no_change_on_line():
    skel = np.zeros((5, 5), dtype=bool)
    skel[2, 1:4] = True
    cfg = SolverConfig(max_iters=2, enable_r1=True, enable_r2=True)
    reduced, stats = reduce_skeleton(skel, cfg)
    assert reduced.sum() == skel.sum()
    assert stats["r1_removed"] == 0
