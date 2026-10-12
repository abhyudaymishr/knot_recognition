"""
test_honest_protocol.py - Unit Tests for KnotInfo Benchmark, Diagram Generator, and Canonical Models.
"""

import pytest
import torch
import numpy as np

from Knotgen.flype_generator import generate_isotopic_diagrams, braid_tangle_flype
from knot_recognition.knotinfo_bench import KnotInfoBenchmark, parse_unknotting_number
from knot_recognition.canonical_models import (
    BaseKnotGraphModel,
    AlexanderFeatureBaseline,
    CanonicalHashLookupTable,
    build_graph_tensors_from_pd,
)


def test_knotinfo_benchmark_parsing():
    bench = KnotInfoBenchmark()
    train_k, test_9, test_10 = bench.get_splits()

    assert len(train_k) == 36 or len(train_k) == 37  # c <= 8 prime knots + unknot
    assert len(test_9) == 49   # 9-crossing knots
    assert len(test_10) == 165 # 10-crossing knots

    entry_4_1 = bench.get_knot("4_1")
    assert entry_4_1.crossing_number == 4
    assert entry_4_1.signature == 0
    assert entry_4_1.three_genus == 1
    assert entry_4_1.braid_index == 3
    assert entry_4_1.unknotting_number == 1.0


def test_parse_unknotting_number_intervals():
    val_num, raw = parse_unknotting_number("1")
    assert val_num == 1.0

    val_interval, raw_interval = parse_unknotting_number("[2,3]")
    assert val_interval == 2.5
    assert raw_interval == "[2,3]"


def test_hard_negatives_mining():
    bench = KnotInfoBenchmark()
    train_k, _, _ = bench.get_splits()
    negs = bench.mine_hard_negatives("8_9", train_k, num_negatives=5)
    assert len(negs) == 5
    assert "8_9" not in negs


def test_trivial_predictors():
    bench = KnotInfoBenchmark()
    train_k, test_9, _ = bench.get_splits()
    triv = bench.evaluate_trivial_predictors(train_k, test_9)
    assert "signature" in triv
    assert "three_genus" in triv
    assert "braid_index" in triv
    assert "unknotting_number" in triv
    assert triv["signature"]["mean_mae"] > 0.0


def test_flype_diagram_generator():
    braid_4_1 = [1, -2, 1, -2]
    diags = generate_isotopic_diagrams("4_1", braid_4_1, base_nstrands=3, num_diagrams=4, max_extra_crossings=6, seed=42)
    assert len(diags) == 4
    # Diagram 0 is canonical
    assert diags[0].is_canonical
    assert diags[0].crossing_count == 4
    # Perturbed diagrams have >= 4 crossings
    for d in diags[1:]:
        assert d.crossing_count >= 4
        assert len(d.pd_code) == d.crossing_count


def test_canonical_models_forward():
    pd = [(1, 4, 2, 5), (5, 0, 6, 1), (7, 3, 0, 2), (3, 7, 4, 6)]
    signs = [1, 1, 1, 1]

    for mode in ["raw", "r1_r2", "r1_only", "r2_only", "snappy_simplify"]:
        m = BaseKnotGraphModel(mode=mode)
        z = m(pd, signs)
        assert z.shape == torch.Size([128])
        assert torch.isclose(torch.norm(z), torch.tensor(1.0), atol=1e-4)


def test_alexander_feature_baseline():
    bench = KnotInfoBenchmark()
    alex = AlexanderFeatureBaseline(out_dim=128, bench=bench)
    pd = [(1, 4, 2, 5), (5, 0, 6, 1), (7, 3, 0, 2), (3, 7, 4, 6)]
    z = alex.extract_features(pd, [1, 1, 1, 1], knot_name="4_1")
    assert z.shape == torch.Size([128])
    assert torch.isclose(torch.norm(z), torch.tensor(1.0), atol=1e-4)


def test_canonical_hash_lookup_table():
    lut = CanonicalHashLookupTable(out_dim=128)
    pd = [(1, 4, 2, 5), (5, 0, 6, 1), (7, 3, 0, 2), (3, 7, 4, 6)]
    lut.register("4_1", pd, [1, 1, 1, 1])

    # Querying registered diagram matches
    z_seen = lut.predict(pd, [1, 1, 1, 1])
    assert torch.isclose(torch.norm(z_seen), torch.tensor(1.0), atol=1e-4)

    # Querying unseen diagram gives non-matching random unit vector
    pd_unseen = [(1, 2, 3, 4), (5, 6, 7, 8), (9, 10, 11, 12)]
    z_unseen1 = lut.predict(pd_unseen)
    z_unseen2 = lut.predict(pd_unseen)
    sim = torch.dot(z_unseen1, z_unseen2).item()
    # Independent random vectors have dot product near 0 (rarely > 0.5)
    assert abs(sim) < 0.6
