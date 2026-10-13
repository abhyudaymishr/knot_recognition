"""Cable tools: diagram geometry, identification, move logging, closure, uncertainty, traversal adapter."""

import json
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from knot_recognition.cable.closure import ClosureRecord, closure_sensitivity
from knot_recognition.cable.diagram import braid_closure_curves, from_curves_3d
from knot_recognition.cable.identify import identify_diagram, type_key
from knot_recognition.cable.moves import reduce_diagram
from knot_recognition.cable.uncertainty import propagate

DB = ROOT / "data" / "knotinfo" / "knotinfo_full_dataset.json"
needs_db = pytest.mark.skipif(not DB.exists(), reason="KnotInfo dataset not extracted")


def D_of(word):
    return from_curves_3d(braid_closure_curves(word))


# --------------------------------------------------------------- geometry & chirality
def test_right_handed_trefoil_has_positive_crossings_and_tau_plus_one():
    D = D_of([1, 1, 1])  # rising-left strand passes over: right-handed by the right-hand rule
    assert [D.sign(c) for c in D.crossings] == [1, 1, 1]
    assert D.to_link().knot_floer_homology()["tau"] == 1
    assert D_of([-1, -1, -1]).to_link().knot_floer_homology()["tau"] == -1


def test_flip_changes_crossing_and_trefoil_unknots():
    D = D_of([1, 1, 1])
    assert D.flip([D.crossings[0].cid]).to_link().knot_floer_homology()["seifert_genus"] == 0


# --------------------------------------------------------------- identification
@needs_db
@pytest.mark.parametrize("word,expected", [
    ([1, 1, 1], "3_1"), ([1, -2, 1, -2], "4_1"),
    ([1, 1, 1, 2, 2, 2], "3_1 # 3_1"), ([1, 1, 1, -2, -2, -2], "3_1 # 3_1*"),
])
def test_identifies_practical_knots(word, expected):
    r = identify_diagram(D_of(word))
    assert r["type"] == expected


@needs_db
def test_square_and_granny_names():
    assert "granny" in identify_diagram(D_of([1, 1, 1, 2, 2, 2]))["practical_name"]
    assert "square" in identify_diagram(D_of([1, 1, 1, -2, -2, -2]))["practical_name"]


def test_links_hopf_and_unlink():
    hopf = identify_diagram(D_of([1, 1]))
    assert hopf["split"] == "interlinked" and hopf["linking_matrix"][0][1] in (1, -1)
    unlink = identify_diagram(D_of([1, -1]))
    assert unlink["split"] == "separable"


# --------------------------------------------------------------- moves
def test_move_log_untangles_unknot_and_stops_at_real_knot():
    r = reduce_diagram(D_of([1, 2, -1, -2]))
    assert r["crossings_after_steps"] == 0 and r["knotted"] is False
    assert {s["move"] for s in r["steps"]} <= {"R1", "R2"}
    k = reduce_diagram(D_of([1, 1, 1, 2, -2]))
    assert k["knotted"] is True and any(s["move"] == "R2" for s in k["steps"])


# --------------------------------------------------------------- closure
def _open_at(c, idx, gap=3):
    n = len(c)
    return c[[(idx + gap + k) % n for k in range(n - 2 * gap)]]


@needs_db
def test_closure_sensitivity_detects_end_inside_tangle():
    c = braid_closure_curves([1, 1, 1])[0]
    outer = from_curves_3d([_open_at(c, int(np.argmax(c[:, 0])))], closed=[False])
    assert closure_sensitivity(outer)["closure_independent"]
    inner_idx = int(np.argmin(np.linalg.norm(c[:, :2] - np.array([0.5, 1.5]), axis=1)))
    inner = from_curves_3d([_open_at(c, inner_idx)], closed=[False])
    assert not closure_sensitivity(inner)["closure_independent"]


def test_closure_record_requires_evidence_for_physical_methods():
    assert ClosureRecord("fixture", "behind").validate()
    rec = ClosureRecord("fixture", "behind", fixture_id="panel-01")
    assert rec.validate() == [] and rec.is_measurement
    assert ClosureRecord.from_json(rec.to_json()).fixture_id == "panel-01"


# --------------------------------------------------------------- uncertainty
@needs_db
def test_uncertainty_probabilities_match_hand_calculation():
    D = D_of([1, 1, 1])
    D.crossings[0].confidence, D.crossings[1].confidence = 0.6, 0.8
    r = propagate(D)
    dist = {d["type"]: d["probability"] for d in r["distribution"]}
    assert dist["3_1"] == pytest.approx(0.48) and dist["0_1"] == pytest.approx(0.52)
    assert r["decision"] == "abstain" and r["next_best_views"][0]["cid"] == D.crossings[0].cid


# --------------------------------------------------------------- traversal adapter
def _traversal_from_diagram(D, H=800, px=80.0):
    """Ground-truth TraversalResult in the traverser's own graph format (pixel scale ``px`` per unit)."""
    from knot_recognition.gauss_pd import _canon_edge
    from knot_recognition.traverser import StrandIntersection, TraversalResult

    SG = nx.MultiGraph()
    for c in D.crossings:
        SG.add_node(c.cid, type="junction", pos=(H - px * c.pos[1], px * c.pos[0]))
    pairing, arms = {}, {}
    for ci, comp in enumerate(D.components):
        ev = sorted([(c.over.t, c.cid, True) for c in D.crossings if c.over.component == ci] +
                    [(c.under.t, c.cid, False) for c in D.crossings if c.under.component == ci])
        edges = []
        for k in range(len(ev)):
            (t0, a, _), (t1, b, _) = ev[k], ev[(k + 1) % len(ev)]
            if t1 <= t0:
                t1 += comp.length
            pts = [comp.point_at(t) for t in np.linspace(t0, t1, 40)]
            path = [(H - px * p[1], px * p[0]) for p in pts]
            if a > b:  # traverser paths start at the edge's first node
                a, b, path = b, a, path[::-1]
            key = SG.add_edge(a, b, path=path)
            edges.append(_canon_edge(a, b, key))
        for k, (_, cid, ov) in enumerate(ev):
            e_in, e_out = edges[k - 1], edges[k]
            pairing[(cid, e_in)] = e_out
            pairing[(cid, e_out)] = e_in
            arms[(cid, ov)] = (e_in, e_out)
    inters = [StrandIntersection(crossing_id=c.cid, pos=(H - px * c.pos[1], px * c.pos[0]), status="resolved", confidence=1.0,
                                 junction_node=c.cid, over_edges=arms[(c.cid, True)], under_edges=arms[(c.cid, False)])
              for c in D.crossings]
    return TraversalResult([], "", [], "PD[]", inters, skeleton_graph=SG, pairing=pairing, image_shape=(H, H))


@needs_db
@pytest.mark.parametrize("word", [[1, 1, 1], [-1, -1, -1], [1, -2, 1, -2], [1, 1, 1, -2, -2, -2], [1, 1]])
def test_adapter_recovers_type_from_traverser_graph(word):
    from knot_recognition.cable.from_traversal import diagram_from_traversal

    D = D_of(word)
    Dp, rep = diagram_from_traversal(_traversal_from_diagram(D))
    assert Dp.is_closed and not rep["dropped_crossings"]
    assert type_key(identify_diagram(Dp)) == type_key(identify_diagram(D))


def test_near_tangent_geometry_still_gives_planar_pd():
    """Regression: a deformed curve with a 0.008 depth gap once produced a non-planar PD."""
    import snappy
    from knot_recognition.cable.synth import make_instance

    D = from_curves_3d(make_instance([1, 1, 2, -1, -3, 2, -3], 175080940, wobble=0.15))
    assert len(snappy.Link(D.to_pd()).crossings) == len(D.crossings)


# --------------------------------------------------------------- KnotBench-Photo
def test_euler_check_counts_crossings_from_the_image():
    from knot_recognition.cable.consistency import euler_check
    from knot_recognition.cable.synth import make_instance, render

    rng = np.random.default_rng(0)
    img, ann, D = render(make_instance([1, 1, 1], 0, wobble=0.0), "S0_silhouette", "black_coax", 10, rng)
    img = np.asarray(img)
    n = len(ann["crossings"])
    assert euler_check(img, n, ann["cable_width_px"])["consistent"]
    assert not euler_check(img, n - 1, ann["cable_width_px"])["consistent"]  # a missed crossing is caught
    assert euler_check(img, n, ann["cable_width_px"], open_ends=2)["applicable"] is False


def test_group_split_keeps_cables_and_test_sessions_disjoint():
    from knot_recognition.cable import photo_bench as pb

    recs = [{"image_id": f"c{c}_s{s}", "cable_id": f"c{c}", "session_id": f"s{s}"} for c in range(10) for s in range(4)]
    split = pb.group_split(recs, test_sessions=["s3"], seed=1)
    assert pb.check_split(recs, split) == []
    assert split["test"] and split["train"] and split["val"]


def test_crossing_matching_and_over_under_scoring():
    from knot_recognition.cable import photo_bench as pb

    m = pb.match_crossings(np.array([[0, 0], [100, 100]]), np.array([[98, 103], [500, 500], [2, 1]]), radius=10)
    assert sorted(m) == [(0, 2), (1, 0)]
    assert pb.over_correct([1, 0], [1, 0], [0, 1]) is True
    assert pb.over_correct([0, 1], [1, 0], [0, 1]) is False
