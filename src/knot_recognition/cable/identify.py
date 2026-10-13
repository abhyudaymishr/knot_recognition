"""
identify.py - Name what a closed cable diagram is, and say how sure we are.

Knots
  1. Reduce the diagram (SnapPy global simplification). Zero crossings left is a
     certificate of the unknot.
  2. Split diagrammatic connected sums into prime summands (spherogram
     ``deconnect_sum``), so composites such as the square and granny knots are named.
  3. Each prime summand is fingerprinted with knot Floer homology (Alexander
     polynomial, Seifert genus, tau) and matched against KnotInfo (prime knots
     up to 12 crossings). Candidates are confirmed with an exterior-isometry
     certificate when the complement is hyperbolic.
  4. Chirality comes from the sign of tau relative to the tabulated knot.

Links (several cables)
  Per-component knot types, the linking matrix, and a split / non-split verdict:
  a split diagram after simplification certifies "separable"; a non-zero linking
  number or a hyperbolic complement certifies "interlinked"; otherwise undetermined.

Every answer carries ``certainty``: "certified", "invariant_match" or "undetermined".
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import snappy
from spherogram.links.simplify import deconnect_sum

ROOT = Path(__file__).resolve().parents[3]
KNOTINFO = ROOT / "data" / "knotinfo" / "knotinfo_full_dataset.json"

PRACTICAL_NAMES = {
    "0_1": "no knot (unknot)",
    "3_1": "overhand knot (closed: trefoil)",
    "4_1": "figure-eight knot",
    "5_1": "cinquefoil knot",
    "5_2": "three-twist knot",
    "6_1": "stevedore knot",
}


# ----------------------------------------------------------------------------- helpers

def _alexander_from_hfk(hfk: Dict[str, Any]) -> Tuple[int, ...]:
    coeff: Dict[int, int] = {}
    for (a, m), r in hfk["ranks"].items():
        coeff[a] = coeff.get(a, 0) + (-1) ** (m % 2) * r
    lo, hi = min(coeff), max(coeff)
    poly = [coeff.get(k, 0) for k in range(lo, hi + 1)]
    return _normalise(poly)


def _normalise(poly: Sequence[int]) -> Tuple[int, ...]:
    p = list(poly)
    while p and p[0] == 0:
        p.pop(0)
    while p and p[-1] == 0:
        p.pop()
    if sum(p) < 0:
        p = [-c for c in p]
    return tuple(p)


@lru_cache(maxsize=1)
def _knotinfo_index(path: str = str(KNOTINFO)) -> Dict[Tuple[int, ...], List[Dict[str, Any]]]:
    db = json.load(open(path))
    index: Dict[Tuple[int, ...], List[Dict[str, Any]]] = {}
    for name, v in db.items():
        av = v.get("alexander_vector") or []
        if not (isinstance(av, list) and len(av) > 2) or not v.get("pd"):
            continue
        index.setdefault(_normalise(av[2:]), []).append({
            "name": name, "crossings": v["crossings"], "pd": v["pd"],
            "tau": v.get("ozsvath_szabo_tau_invariant"), "genus": v.get("three_genus"),
            "signature": v.get("signature"),
        })
    return index


def _hyperbolic(M) -> bool:
    try:
        return M.solution_type() == "all tetrahedra positively oriented" and float(M.volume()) > 0.1
    except Exception:
        return False


# ----------------------------------------------------------------------------- knots

def identify_prime(K: "snappy.Link") -> Dict[str, Any]:
    K = K.copy()
    K.simplify("global")
    n = len(K.crossings)
    if n == 0:
        return {"name": "0_1", "certainty": "certified", "method": "diagram reduces to 0 crossings", "chirality": "amphichiral"}
    hfk = K.knot_floer_homology()
    alex = _alexander_from_hfk(hfk)
    out: Dict[str, Any] = {
        "fingerprint": {"alexander": list(alex), "tau": hfk["tau"], "genus": hfk["seifert_genus"],
                        "fibered": hfk["fibered"], "hfk_total_rank": hfk["total_rank"], "diagram_crossings": n},
    }
    cands = [c for c in _knotinfo_index().get(alex, [])
             if c["crossings"] <= n and (c["genus"] is None or c["genus"] == hfk["seifert_genus"])
             and (c["tau"] is None or abs(c["tau"]) == abs(hfk["tau"]))]
    if not cands:
        out.update(name=None, certainty="undetermined",
                   method="no KnotInfo prime knot (c <= 12) matches the invariants")
        return out
    M = K.exterior()
    hyper = _hyperbolic(M)
    confirmed = []
    if hyper:
        for c in cands:
            try:
                if M.is_isometric_to(snappy.Link(c["pd"]).exterior()):
                    confirmed.append(c)
            except Exception:
                pass
    if len(confirmed) == 1:
        best, certainty, method = confirmed[0], "certified", "exterior isometry (Mostow rigidity) with the KnotInfo diagram"
    elif len(cands) == 1:
        best, certainty, method = cands[0], "invariant_match", "unique KnotInfo match on Alexander polynomial, genus and |tau|"
    else:
        out.update(name=None, candidates=[c["name"] for c in cands], certainty="undetermined",
                   method="several KnotInfo knots share these invariants")
        return out
    tau, ref = hfk["tau"], best["tau"]
    if tau == 0 or ref in (None, 0):
        chir = "not determined by tau" if tau == 0 else "unknown"
    else:
        chir = "as tabulated" if (tau > 0) == (ref > 0) else "mirror of tabulated"
    out.update(name=best["name"], certainty=certainty, method=method, chirality=chir)
    return out


def identify_knot(K: "snappy.Link") -> Dict[str, Any]:
    """Name a one-component diagram, splitting connected sums into prime summands."""
    K = K.copy()
    K.simplify("global")
    if len(K.crossings) == 0:
        return {"type": "0_1", "practical_name": PRACTICAL_NAMES["0_1"], "certainty": "certified",
                "summands": [], "method": "diagram reduces to 0 crossings"}
    pieces = [P for P in deconnect_sum(K) if len(P.crossings) > 0]
    summands = [identify_prime(P) for P in pieces]
    summands = [s for s in summands if s.get("name") != "0_1"] or summands
    names = [s.get("name") or "?" for s in summands]
    certainty = min((s["certainty"] for s in summands),
                    key=["undetermined", "invariant_match", "certified"].index)
    typ = " # ".join(sorted(names))
    practical = PRACTICAL_NAMES.get(typ)
    if len(summands) == 2 and all(s.get("name") == "3_1" for s in summands):
        taus = [s["fingerprint"]["tau"] for s in summands]
        practical = "granny knot (closed)" if taus[0] == taus[1] else "square / reef knot (closed)"
        typ = "3_1 # 3_1" if taus[0] == taus[1] else "3_1 # 3_1*"
    if len(summands) > 1:
        method = "diagrammatic connected-sum decomposition; each summand identified separately"
        certainty = certainty if certainty != "certified" else "certified_summands"
    else:
        method = summands[0].get("method")
    return {"type": typ, "practical_name": practical, "certainty": certainty, "summands": summands, "method": method}


# ----------------------------------------------------------------------------- links

def identify_link(L: "snappy.Link") -> Dict[str, Any]:
    n = len(L.link_components)
    if n == 1:
        r = identify_knot(L)
        r["components"] = 1
        return r
    lk = [list(map(int, row)) for row in L.linking_matrix()]
    comps = [identify_knot(L.sublink([i])) for i in range(n)]
    S = L.copy()
    S.simplify("global")
    pieces = S.split_link_diagram() if len(S.crossings) else []
    if len(S.crossings) == 0:
        verdict, how, cert = "separable", "simplification removes every crossing: all cables come apart", "certified"
    elif getattr(S, "unlinked_unknot_components", 0) > 0:
        verdict, cert = "partly separable", "certified"
        how = (f"{S.unlinked_unknot_components} cable(s) come free after simplification; "
               "see the linking matrix for the pairs that remain interlinked")
    elif len(pieces) > 1:
        verdict, how = "separable", f"diagram splits into {len(pieces)} unlinked parts after simplification"
        cert = "certified"
    elif any(lk[i][j] for i in range(n) for j in range(n) if i != j):
        verdict, how, cert = "interlinked", "non-zero linking number", "certified"
    elif _hyperbolic(L.exterior()):
        verdict, how, cert = "interlinked", "hyperbolic complement (irreducible, hence non-split)", "certified"
    else:
        verdict, how, cert = "undetermined", "linking numbers are zero and no split was found", "undetermined"
    census = None
    try:
        if verdict in ("separable", "partly separable"):
            raise ValueError("split link: census lookup not applicable")
        census = [str(x) for x in L.exterior().identify()]
    except Exception:
        pass
    return {"components": n, "component_types": comps, "linking_matrix": lk,
            "split": verdict, "split_certainty": cert, "split_method": how, "census_names": census}


def identify_diagram(diagram) -> Dict[str, Any]:
    """Entry point for a closed ``cable.diagram.Diagram``."""
    L = diagram.to_link()
    free = diagram.meta.get("free_loops", 0)
    res = identify_link(L) if len(L.crossings) else {"type": "0_1", "practical_name": PRACTICAL_NAMES["0_1"],
                                                     "certainty": "certified", "components": 0}
    if free:
        res["free_unknotted_loops"] = free
    return res


def type_key(res: Dict[str, Any]) -> str:
    """Short hashable label used when aggregating over uncertain readings."""
    if res.get("components", 1) > 1:
        parts = ",".join(sorted(c["type"] for c in res["component_types"]))
        return f"link[{parts}|{res['split']}|lk={res['linking_matrix']}]"
    if res.get("type") and "?" not in res["type"]:
        chir = [s.get("chirality") for s in res.get("summands", []) if s.get("chirality") in ("mirror of tabulated",)]
        return res["type"] + ("*" * len(chir) if res["type"] not in ("3_1 # 3_1", "3_1 # 3_1*") else "")
    fp = [s.get("fingerprint", {}) for s in res.get("summands", [])]
    return "unknown:" + json.dumps(fp, sort_keys=True)
