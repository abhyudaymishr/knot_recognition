import numpy as np
import pytest
from knot_recognition.gauss_pd import _canon_edge
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


def test_knot_traverser_user_api_and_opencv_compatibility(tmp_path):
    import cv2

    img = _make_crossing_image()
    traverser = KnotTraverser(cluster_radius=12, sample_radius=18)
    result = traverser.process_image(img)
    assert isinstance(result.intersections, list)
    assert isinstance(result.gauss_code, list)
    assert isinstance(result.pd_string, str)
    assert isinstance(result.annotated_image, np.ndarray)

    # Verify cv2.imwrite works directly on result.annotated_image
    out_path = str(tmp_path / "annotated.png")
    ok = cv2.imwrite(out_path, result.annotated_image)
    assert ok
    read_back = cv2.imread(out_path)
    assert read_back is not None
    assert read_back.shape == result.annotated_image.shape

    # Verify PIL-style save also succeeds
    out_pil_path = str(tmp_path / "annotated_pil.png")
    result.annotated_image.save(out_pil_path)
    assert cv2.imread(out_pil_path) is not None


def _four_arm_junction(gray):
    import networkx as nx
    from knot_recognition.gauss_pd import _edge_direction_at_node

    graph = nx.MultiGraph()
    center = 0
    graph.add_node(center, type="junction", pos=(50.0, 50.0))
    branches = []
    specs = [
        (1, [(50, 50), (42, 50), (34, 50), (26, 50)]),
        (2, [(50, 50), (58, 50), (66, 50), (74, 50)]),
        (3, [(50, 50), (50, 42), (50, 34), (50, 26)]),
        (4, [(50, 50), (50, 58), (50, 66), (50, 74)]),
    ]
    arms = []
    for node, path in specs:
        graph.add_node(node, type="endpoint", pos=path[-1])
        graph.add_edge(center, node, path=path)
        edge = (center, node, 0)
        arms.append((edge, _edge_direction_at_node(path, at_start=True), path))
    # Pair opposite directions into two strands.
    branches = [(arms[0], arms[1]), (arms[2], arms[3])]
    return graph, branches


def test_crossing_classifier_resolves_clear_gap_without_forcing_ambiguous():
    # Horizontal strand has a clear gap; vertical strand remains continuous.
    gray = np.full((101, 101), 255, dtype=np.uint8)
    gray[48:53, 10:41] = 0
    gray[48:53, 60:91] = 0
    gray[10:91, 48:53] = 0
    graph, branches = _four_arm_junction(gray)

    result = KnotTraverser(sample_radius=18)._resolve_over_under_crossings(
        graph, {0: branches}, gray
    )[0]
    assert result.status == "resolved"
    assert result.confidence >= 0.28
    assert result.over_edges == ((0, 1, 0), (0, 2, 0))
    assert result.under_edges == ((0, 3, 0), (0, 4, 0))

    # With both strands continuous the grayscale evidence cannot determine
    # depth; the classifier must abstain rather than invent a crossing label.
    ambiguous = np.full((101, 101), 255, dtype=np.uint8)
    ambiguous[48:53, 10:91] = 0
    ambiguous[10:91, 48:53] = 0
    graph2, branches2 = _four_arm_junction(ambiguous)
    uncertain = KnotTraverser(sample_radius=18)._resolve_over_under_crossings(
        graph2, {0: branches2}, ambiguous
    )[0]
    assert uncertain.status == "ambiguous"
    assert uncertain.over_direction == (0.0, 0.0)

    pairing = {}
    for branch in branches:
        a, b = (_canon_edge(*arm[0]) for arm in branch)
        pairing[(0, a)] = b
        pairing[(0, b)] = a
    traverser = KnotTraverser(sample_radius=18)
    events, crossings, _ = traverser._traverse_curve(
        graph, pairing, [result]
    )
    assert sorted(events, key=lambda event: event[1]) == [(1, False), (1, True)]
    assert len(crossings) == 1

    uncertain_events, uncertain_pd, _ = traverser._traverse_curve(
        graph, pairing, [uncertain]
    )
    assert uncertain_events == []
    assert uncertain_pd == []
