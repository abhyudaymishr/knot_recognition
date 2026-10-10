"""
build_census.py - build and VERIFY a braid-word table for the unknot + 249 prime knots (3_1 .. 10_165).

Run once (needs `pip install snappy`):   python build_census.py
Writes census_braids.json, then knotgen can regenerate any knot from its braid word:

    from build_census import load_census
    from knotgen import braid
    word, n = load_census()["10_165"]["braid"], load_census()["10_165"]["nstrands"]
    K = braid(word, n)

Verification (mirror images are treated as equal, as in the Rolfsen table):
  * hyperbolic knots: exterior of braid closure is isometric to the census exterior
  * non-hyperbolic (torus) knots: Alexander polynomial from knotgen matches torus(p,q)
NOTE: untested in the sandbox where this was written (no snappy there) - run it and read the report.
"""
import json
import sys
from pathlib import Path
import sympy as sp
import snappy

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

try:
    from Knotgen.knotgen import braid, torus, t
except ImportError:
    from knotgen import braid, torus, t

SIZES = {3: 1, 4: 1, 5: 2, 6: 3, 7: 7, 8: 21, 9: 49, 10: 165}
# Torus knots in the table (p,q): the only non-hyperbolic ones in range
TORUS = {"3_1": (2, 3), "5_1": (2, 5), "7_1": (2, 7), "9_1": (2, 9),
         "8_19": (3, 4), "10_124": (3, 5)}


def all_names():
    for c, m in SIZES.items():
        for i in range(1, m + 1):
            yield f"{c}_{i}"


def verify(name, word, nstrands):
    L = snappy.Link(name)
    M_ref = L.exterior()
    if name in TORUS:
        p, q = TORUS[name]
        try:
            a = braid(word, nstrands).alexander()
            b = torus(p, q).alexander()
            return sp.expand(a - b) == 0, "alexander vs torus(p,q)"
        except Exception as e:
            return False, f"alexander check failed: {e}"
    try:
        M = snappy.Link(braid_closure=word).exterior()
        return bool(M.is_isometric_to(M_ref)), "isometry"
    except Exception as e:                     # e.g. if SnapPy cannot decide
        return False, f"isometry check failed: {e}"


def build(path="Knotgen/census_braids.json"):
    table, bad = {}, []
    # unknot
    table["unknot"] = dict(braid=[], nstrands=1, crossings=0, verified=True, check="trivial")
    for name in all_names():
        L = snappy.Link(name)
        word = [int(g) for g in L.braid_word()]
        n = max(abs(g) for g in word) + 1 if word else 1
        ok, how = verify(name, word, n)
        table[name] = dict(braid=word, nstrands=n, braid_length=len(word),
                           crossings=len(L.PD_code()), verified=ok, check=how)
        if not ok:
            bad.append((name, how))
    with open(path, "w") as f:
        json.dump(table, f, indent=2)
    print(f"entries: {len(table)}  (expected 250)")
    print(f"unverified: {len(bad)}")
    for name, how in bad[:20]:
        print("  ", name, how)
    return table


def load_census(path="Knotgen/census_braids.json"):
    with open(path) as f:
        return json.load(f)


if __name__ == "__main__":
    build()
