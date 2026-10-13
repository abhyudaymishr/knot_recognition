"""Guards for the KnotBench v2 protocol: split disjointness, no label columns, backbone invariance."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "scripts")]
DB = ROOT / "data" / "knotinfo" / "knotinfo_full_dataset.json"

pytestmark = pytest.mark.skipif(not DB.exists(), reason="KnotInfo dataset not extracted")


@pytest.fixture(scope="module")
def v2():
    spec = importlib.util.spec_from_file_location("run_benchmark_v2", ROOT / "scripts" / "run_benchmark_v2.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def db():
    return json.load(open(DB))


def test_task2_splits_are_knot_disjoint(v2, db):
    tr, va, te = v2.split_by_crossings(db, 0, 10), v2.split_by_crossings(db, 11, 11), v2.split_by_crossings(db, 12, 12)
    assert not set(tr) & set(va) and not set(tr) & set(te) and not set(va) & set(te)
    assert (len(tr), len(va), len(te)) == (250, 552, 2176)


def test_oracle_features_never_contain_the_signature_target(v2, db):
    ks = v2.split_by_crossings(db, 3, 6)
    feats = {k: {"diagram": [0.0] * 13} for k in ks}
    X, cols = v2.oracle_matrix(db, feats, ks, "signature")
    assert "signature" not in cols and "abs_sig_half" not in cols
    assert X.shape[1] == len(cols)
    _, cols_u = v2.oracle_matrix(db, feats, ks, "unknotting_number")
    assert "signature" in cols_u  # kept, but reported only as an oracle row


def test_diagram_features_have_no_invariant_columns(v2):
    assert v2.DIAGRAM_COLS[:4] == ["c", "n_pos", "n_neg", "writhe"]
    assert all(c.startswith("faces_deg") for c in v2.DIAGRAM_COLS[4:])


@pytest.mark.parametrize("backbone", ["mpnn", "gin", "gps", "subgraph"])
def test_backbones_are_invariant_to_relabelling(backbone, db):
    from run_rigorous_tasks import build_pure_graph_tensors, canonicalize_pd_snappy, safe_randomize_pd
    from knot_recognition.bench_backbones import RegressionModel

    pd = db["8_17"]["pd"]
    g1 = build_pure_graph_tensors(canonicalize_pd_snappy(pd))
    g2 = build_pure_graph_tensors(canonicalize_pd_snappy(safe_randomize_pd(pd, seed=11)[0]))
    torch.manual_seed(0)
    m = RegressionModel(backbone, 4).eval()
    with torch.no_grad():
        assert torch.allclose(m([g1]), m([g2]), atol=1e-5)


def test_parameter_budgets_are_matched(v2):
    from knot_recognition.bench_backbones import RegressionModel, count_params

    counts = [count_params(RegressionModel(b, 4, hidden=v2.matched_hidden(b))) for b in ["mpnn", "gin", "gps", "subgraph"]]
    assert max(counts) / min(counts) < 1.1
