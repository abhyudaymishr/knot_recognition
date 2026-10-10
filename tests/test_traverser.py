import numpy as np
import pytest
from knot_recognition.traverser import KnotTraverser, StrandIntersection


def _make_crossing_image(size=101):
    img = np.ones((size, size, 3), dtype=np.uint8) * 255
    # Draw dark line for overstrand: vertical (col 50, rows 10 to 90)
    img[10:91, 48:53] = 0
    # Draw dark line for understrand with break: horizontal (row 50, cols 10..40 and 60..90)
    img[48:53, 10:41] = 0
    img[48:53, 60:91] = 0
    return img


def test_knot_traverser_crossing_detection():
    img = _make_crossing_image()
    traverser = KnotTraverser()
    result = traverser.process_image(img)
    # Checks that it completes and returns structural results
    assert isinstance(result.gauss_code, list)
    assert isinstance(result.pd_string, str)
    assert result.pd_string.startswith("PD[")
