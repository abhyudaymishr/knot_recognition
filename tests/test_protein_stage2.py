import sys
from pathlib import Path
import tempfile

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from knot_recognition.protein import (
    detect_crossings,
    extract_ca_polyline,
    gauss_code_from_crossings,
    project_polyline,
)


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


def test_extract_ca_prefers_altloc_and_records_residue_gaps():
    def atom(serial, alt, residue, x, occupancy=1.0):
        return (
            f"ATOM  {serial:5d}  CA {alt:1s}ALA A{residue:4d} "
            f"  {x:8.3f}{0.0:8.3f}{0.0:8.3f}{occupancy:6.2f}{20.0:6.2f}          C  \n"
        )

    pdb = (
        "MODEL        1\n"
        + atom(1, "A", 1, 0.0, 0.4)
        + atom(2, "B", 1, 1.0, 0.8)
        + atom(3, " ", 3, 3.0)
        + "ENDMDL\nMODEL        2\n"
        + atom(4, " ", 4, 40.0)
        + "ENDMDL\n"
    )
    with tempfile.NamedTemporaryFile("w", suffix=".pdb") as f:
        f.write(pdb)
        f.flush()
        backbone = extract_ca_polyline(f.name, chain_id="A")
    assert backbone.ca_coords[:, 0].tolist() == [1.0, 3.0]
    assert backbone.residue_ids == ("1", "3")
    assert backbone.gap_after.tolist() == [True]


def test_crossing_detection_skips_segments_across_residue_gaps():
    xy = np.array([[-1.0, 0.0], [1.0, 0.0], [0.0, -1.0], [0.0, 1.0]])
    depth = np.array([1.0, 1.0, 0.0, 0.0])
    assert detect_crossings(xy, depth, gap_after=np.array([True, False, False])) == []
