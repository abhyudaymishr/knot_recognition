import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from knot_recognition.protein import detect_crossings, gauss_code_from_crossings, project_polyline


def test_crossing_over_under():
    # Two segments crossing in projection:
    # segment 0-1 at z=1, segment 2-3 at z=0
    coords = np.array(
        [
            [-1.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [0.0, -1.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=float,
    )
    view = np.array([0.0, 0.0, 1.0])
    xy, depth = project_polyline(coords, view)
    crossings = detect_crossings(xy, depth)
    assert len(crossings) == 1
    c = crossings[0]
    # segment 0 should be over segment 2
    assert c.over == 0
    assert c.under == 2
    gauss = gauss_code_from_crossings(crossings, coords.shape[0])
    assert gauss == [1, -1]
