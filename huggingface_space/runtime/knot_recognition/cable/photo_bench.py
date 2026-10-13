"""
photo_bench.py - KnotBench-Photo: annotation schema, group splits and metrics.

One record per image (JSON):

{
  "image_id": "c017_s3_v2",            # unique
  "file": "images/c017_s3_v2.jpg",
  "cable_id": "c017",                  # physical cable instance  -> split unit
  "session_id": "s3",                  # capture session (place, light, camera)  -> split unit
  "view": 2,                           # view index within the session
  "source": "real" | "synthetic",
  "cable": {"type": "coax RG6", "diameter_mm": 6.9, "colour": "black"},
  "capture": {"camera": "...", "background": "...", "lighting": "...", "distance_cm": 60},
  "ground_truth": {
     "type": "3_1",                    # knot/link type of the closed cable (cable.identify.type_key)
     "how_known": "tied from a diagram and verified" | "rendered",
     "closure": {...ClosureRecord...}  # required when the physical cable has free ends
  },
  "annotation": {
     "cable_width_px": 14,
     "components": [[[x, y], ...], ...],                 # centrelines, pixels
     "crossings": [{"cid": 1, "pos_px": [x, y],
                    "over_tangent_px": [dx, dy], "under_tangent_px": [dx, dy],
                    "visibility": "clear" | "partly_occluded" | "occluded",
                    "annotator_confidence": 0.0-1.0}],
     "endpoints_px": [[x, y], ...]
  }
}

Metrics (per image, then aggregated with cable-level bootstrap CIs):
  crossing precision / recall / F1   Hungarian matching within one cable width
  over/under accuracy                on matched crossings
  topology accuracy                  answered images whose type equals the ground truth
  abstention rate, coverage          the pipeline may decline to answer
  selective accuracy, risk-coverage  accuracy as a function of confidence threshold
  ECE                                calibration of the reported type probability
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

REQUIRED = ["image_id", "cable_id", "session_id", "source", "ground_truth", "annotation"]


def validate_record(r: Dict[str, Any]) -> List[str]:
    errs = [f"missing {k}" for k in REQUIRED if k not in r]
    gt = r.get("ground_truth", {})
    if "type" not in gt:
        errs.append("ground_truth.type missing")
    ann = r.get("annotation", {})
    for c in ann.get("crossings", []):
        for k in ("pos_px", "over_tangent_px", "under_tangent_px"):
            if k not in c:
                errs.append(f"crossing {c.get('cid')} lacks {k}")
    if ann.get("endpoints_px") and not gt.get("closure"):
        errs.append("open cable (endpoints annotated) without a recorded closure")
    return errs


def group_split(records: Sequence[Dict[str, Any]], test_sessions: Sequence[str], val_fraction: float = 0.2,
                test_fraction: float = 0.2, seed: int = 0) -> Dict[str, List[str]]:
    """Split by physical cable AND capture session: no cable or session appears in two splits' test sides.

    test  = held-out cables, imaged in held-out sessions
    val   = held-out cables, imaged in the remaining sessions
    train = remaining cables in the remaining sessions
    Images of test cables in non-test sessions (and train cables in test sessions) are excluded.
    """
    rng = np.random.default_rng(seed)
    cables = sorted({r["cable_id"] for r in records})
    perm = list(rng.permutation(cables))
    n_test = max(1, int(round(test_fraction * len(cables))))
    n_val = max(1, int(round(val_fraction * len(cables))))
    test_c, val_c = set(perm[:n_test]), set(perm[n_test:n_test + n_val])
    ts = set(test_sessions)
    out = {"train": [], "val": [], "test": [], "excluded": []}
    for r in records:
        c, s = r["cable_id"], r["session_id"]
        if c in test_c:
            out["test" if s in ts else "excluded"].append(r["image_id"])
        elif c in val_c:
            out["val" if s not in ts else "excluded"].append(r["image_id"])
        else:
            out["train" if s not in ts else "excluded"].append(r["image_id"])
    return out


def check_split(records: Sequence[Dict[str, Any]], split: Dict[str, List[str]]) -> List[str]:
    by_id = {r["image_id"]: r for r in records}
    problems = []
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        ca = {by_id[i]["cable_id"] for i in split[a]}
        cb = {by_id[i]["cable_id"] for i in split[b]}
        if ca & cb:
            problems.append(f"cable overlap {a}/{b}: {sorted(ca & cb)[:5]}")
    st = {by_id[i]["session_id"] for i in split["test"]}
    sn = {by_id[i]["session_id"] for i in split["train"] + split["val"]}
    if st & sn:
        problems.append(f"session overlap test/non-test: {sorted(st & sn)}")
    return problems


# ----------------------------------------------------------------------------- per-image metrics

def match_crossings(gt_pos: np.ndarray, pred_pos: np.ndarray, radius: float) -> List[Tuple[int, int]]:
    if len(gt_pos) == 0 or len(pred_pos) == 0:
        return []
    D = np.linalg.norm(gt_pos[:, None, :] - pred_pos[None, :, :], axis=2)
    cost = np.where(D <= radius, D, 1e6)
    gi, pj = linear_sum_assignment(cost)
    return [(int(g), int(p)) for g, p in zip(gi, pj) if D[g, p] <= radius]


def over_correct(pred_over: Sequence[float], gt_over: Sequence[float], gt_under: Sequence[float]) -> Optional[bool]:
    v = np.asarray(pred_over, float)
    if np.linalg.norm(v) == 0:
        return None
    v /= np.linalg.norm(v)
    co = abs(float(v @ np.asarray(gt_over, float)))
    cu = abs(float(v @ np.asarray(gt_under, float)))
    return co > cu


# ----------------------------------------------------------------------------- aggregation

def cable_bootstrap(values: Sequence[float], groups: Sequence[str], reps: int = 2000, seed: int = 0) -> List[float]:
    vals, groups = np.asarray(values, float), np.asarray(groups)
    ok = ~np.isnan(vals)
    vals, groups = vals[ok], groups[ok]
    if len(vals) == 0:
        return [float("nan"), float("nan")]
    uniq = np.unique(groups)
    idx = {g: np.where(groups == g)[0] for g in uniq}
    rng = np.random.default_rng(seed)
    bs = [vals[np.concatenate([idx[g] for g in rng.choice(uniq, len(uniq))])].mean() for _ in range(reps)]
    return [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def risk_coverage(conf: Sequence[float], correct: Sequence[bool]) -> List[Dict[str, float]]:
    conf, correct = np.asarray(conf, float), np.asarray(correct, bool)
    order = np.argsort(-conf)
    out = []
    for k in range(1, len(order) + 1):
        sel = order[:k]
        out.append({"coverage": k / len(order), "threshold": float(conf[order[k - 1]]),
                    "accuracy": float(correct[sel].mean())})
    return out


def ece(conf: Sequence[float], correct: Sequence[bool], bins: int = 10) -> float:
    conf, correct = np.asarray(conf, float), np.asarray(correct, float)
    if len(conf) == 0:
        return float("nan")
    e = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        m = (conf > lo) & (conf <= hi) if b else (conf >= lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs(conf[m].mean() - correct[m].mean())
    return float(e)
