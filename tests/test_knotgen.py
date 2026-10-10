import pytest
import sympy as sp
from Knotgen.knotgen import (
    unknot,
    torus,
    rational,
    pretzel,
    is_rational_knot,
    alexander_burau,
    from_snappy,
    twist_family,
)

t = sp.symbols("t")


def test_knotgen_braid_and_torus():
    u = unknot()
    assert u.determinant() == 1
    assert sp.expand(u.alexander() - 1) == 0

    t23 = torus(2, 3)
    assert sp.expand(t23.alexander() - (t**2 - t + 1)) == 0
    assert sp.expand(t23.jones() - (t + t**3 - t**4)) == 0
    assert t23.determinant() == 3

    t25 = torus(2, 5)
    assert sp.expand(t25.alexander() - (t**4 - t**3 + t**2 - t + 1)) == 0


def test_knotgen_rational():
    k41 = rational([2, 2])
    assert k41.determinant() == 5
    assert sp.expand(k41.alexander() - (-t**2 + 3 * t - 1)) == 0
    assert sp.expand(k41.jones() - (t**2 - t + 1 - 1 / t + t**-2)) == 0

    k52 = rational([2, 3])
    assert k52.determinant() == 7
    assert sp.expand(k52.alexander() - (2 * t**2 - 3 * t + 2)) == 0

    # 10/3 numerator is even -> link
    assert not is_rational_knot([3, 3])
    with pytest.raises(ValueError, match="closes to a 2-component link"):
        rational([3, 3])


def test_knotgen_pretzel():
    p = pretzel(3, 3, -2)  # 8_19 = T(3,4)
    assert sp.expand(p.alexander() - (t**6 - t**5 + t**3 - t + 1)) == 0


def test_knotgen_burau():
    # 3_1 trefoil
    delta = alexander_burau([1, 1, 1], 2)
    assert sp.expand(delta - (t**2 - t + 1)) == 0


def test_knotgen_snappy_integration():
    k = from_snappy("10_165")
    assert k.name == "10_165"
    assert len(k.crossings) == 10
    ident = k.snappy_identify()
    assert "10_165(0,0)" in ident.get("identify", [])
    assert ident.get("volume", 0.0) > 12.0


def test_census_braids_table():
    from Knotgen.build_census import load_census
    from Knotgen.knotgen import braid

    table = load_census()
    assert len(table) == 250, "Expected 250 entries in census_braids.json"

    # Test rebuilding 10_165 strictly from braid word without SnapPy
    entry = table["10_165"]
    assert entry["verified"] is True
    k10 = braid(entry["braid"], entry["nstrands"], name="10_165")
    assert k10.determinant() == 45
    assert len(k10.crossings) == entry["braid_length"]


def test_knots_through_9_json():
    import json
    from pathlib import Path

    p = Path("Knotgen/knots_through_9.json")
    if p.exists():
        with p.open() as f:
            data = json.load(f)
        total = sum(len(v) for v in data.values())
        assert total == 84, "Expected 84 knots through 9 crossings"
