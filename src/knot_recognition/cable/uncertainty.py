"""
uncertainty.py - Report every knot type a photo is consistent with.

Each crossing read from a photo has a confidence that its over/under reading is
right. A wrong reading is a crossing change, which can change the knot type, so
a single label hides real risk. ``propagate`` enumerates the readings of the
least-confident crossings, identifies each resulting diagram, and returns a
probability for every possible type. ``next_best_views`` ranks the uncertain
crossings by how much resolving each one (a close-up photo, a second angle)
would reduce the remaining uncertainty about the type.

Crossing confidences are treated as independent probabilities. The traverser's
current scores are evidence margins, not calibrated probabilities; calibrate
them on KnotBench-Photo before reading the output as frequencies.
"""

from __future__ import annotations

import itertools
import math
from typing import Any, Dict, List, Optional

from .diagram import Diagram
from .identify import identify_diagram, type_key


def _entropy(ps) -> float:
    return -sum(p * math.log2(p) for p in ps if p > 0)


def propagate(diagram: Diagram, threshold: float = 0.95, max_uncertain: int = 10,
              decide_at: float = 0.95) -> Dict[str, Any]:
    if not diagram.is_closed:
        raise ValueError("close open components first (cable.closure) - the type depends on the closure")
    unc = sorted([c for c in diagram.crossings if c.confidence < threshold], key=lambda c: c.confidence)
    used, held = unc[:max_uncertain], unc[max_uncertain:]
    ids = [c.cid for c in used]
    conf = {c.cid: min(max(c.confidence, 1e-6), 1 - 1e-6) for c in used}
    cache: Dict[str, Dict[str, Any]] = {}
    readings = []
    for bits in itertools.product((0, 1), repeat=len(ids)):
        flips = [cid for cid, b in zip(ids, bits) if b]
        p = 1.0
        for cid, b in zip(ids, bits):
            p *= (1 - conf[cid]) if b else conf[cid]
        d = diagram.flip(flips)
        pd_key = str(sorted(d.to_pd()))
        if pd_key not in cache:
            r = identify_diagram(d)
            cache[pd_key] = {"key": type_key(r), "result": r}
        readings.append({"flipped": flips, "bits": bits, "p": p, "type": cache[pd_key]["key"]})
    dist: Dict[str, float] = {}
    for r in readings:
        dist[r["type"]] = dist.get(r["type"], 0.0) + r["p"]
    ranked = sorted(dist.items(), key=lambda kv: -kv[1])
    top, ptop = ranked[0]
    out = {
        "uncertain_crossings": [{"cid": c.cid, "confidence": c.confidence} for c in used],
        "held_fixed_crossings": [{"cid": c.cid, "confidence": c.confidence} for c in held],
        "readings_evaluated": len(readings),
        "distribution": [{"type": k, "probability": v} for k, v in ranked],
        "entropy_bits": _entropy(dist.values()),
        "most_likely": top,
        "most_likely_probability": ptop,
        "decision": "report" if ptop >= decide_at else "abstain",
        "next_best_views": _next_best(readings, ids, conf),
    }
    if held:
        out["warning"] = (f"{len(held)} further uncertain crossing(s) were held at their current reading "
                          f"(max_uncertain={max_uncertain}); the distribution is conditional on them.")
    return out


def _next_best(readings: List[Dict[str, Any]], ids: List[int], conf: Dict[int, float]) -> List[Dict[str, Any]]:
    total: Dict[str, float] = {}
    for r in readings:
        total[r["type"]] = total.get(r["type"], 0.0) + r["p"]
    H = _entropy(total.values())
    ranking = []
    for k, cid in enumerate(ids):
        exp_h = 0.0
        for b in (0, 1):
            sub: Dict[str, float] = {}
            pb = 0.0
            for r in readings:
                if r["bits"][k] == b:
                    sub[r["type"]] = sub.get(r["type"], 0.0) + r["p"]
                    pb += r["p"]
            if pb > 0:
                exp_h += pb * _entropy([v / pb for v in sub.values()])
        ranking.append({"cid": cid, "confidence": conf[cid], "expected_information_gain_bits": H - exp_h})
    ranking.sort(key=lambda x: -x["expected_information_gain_bits"])
    return ranking
