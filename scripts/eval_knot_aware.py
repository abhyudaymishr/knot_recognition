"""
eval_knot_aware.py - Score the knot-aware NanoTransformer parser on KnotBench-Photo (synthetic_v1).

Same images, splits and matching rules as scripts/run_knotbench_photo.py, so the
numbers compare directly with the old skeleton traverser
(results/knotbench_photo/synthetic_v1.json).

Over/under confidence: P(correct) = sigmoid(kappa * margin). kappa is fitted by
maximum likelihood on the benchmark's *train* split only, then frozen.

Output: results/knotbench_photo/knot_aware_v1.json
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

import numpy as np
from PIL import Image

from knot_recognition.cable import photo_bench as pb
from knot_recognition.cable.identify import identify_diagram, type_key
from knot_recognition.cable.neural_parser import parse
from knot_recognition.cable.uncertainty import propagate
from knot_recognition.knot_aware import load, predict

DATA = ROOT / "data" / "knotbench_photo" / "synthetic_v1"
CKPT = ROOT / "checkpoints" / "knot_aware_nanotransformer.pth"


def pair_dirs(D, crossing_id):
    """Pixel-space directions of the over and under strands at a parsed crossing."""
    c = next(x for x in D.crossings if x.cid == crossing_id)
    h = D.meta.get("tangent_window")
    to = D.components[c.over.component].tangent_at(c.over.t, h)
    tu = D.components[c.under.component].tangent_at(c.under.t, h)
    return [to[0], -to[1]], [tu[0], -tu[1]], c


def evaluate_image(model, r, kappa):
    img = np.asarray(Image.open(DATA / r["file"]).convert("RGB"))
    ann, gt = r["annotation"], r["ground_truth"]["type"]
    t0 = time.perf_counter()
    pred = predict(model, img)
    D, rep = parse(img, model, kappa=kappa, pred=pred)
    sec = time.perf_counter() - t0
    gpos = np.array([c["pos_px"] for c in ann["crossings"]]).reshape(-1, 2)
    ppos = np.array([c["pos_px"] for c in rep["crossings"]]).reshape(-1, 2)
    m = pb.match_crossings(gpos, ppos, ann["cable_width_px"] + 4)
    row = {"image_id": r["image_id"], "cable_id": r["cable_id"], "session": r["session_id"], "gt": gt,
           "n_gt": len(gpos), "n_pred": len(ppos), "tp": len(m), "sec": sec,
           "count_head": rep["count_head"], "abstain_reasons": rep["abstain_reasons"], "margins": []}
    ou, pairing = [], []
    if D is not None:
        ids = {c.cid for c in D.crossings}
        for g, p in m:
            if (p + 1) not in ids:
                continue
            vo, vu, c = pair_dirs(D, p + 1)
            gc = ann["crossings"][g]
            ok = pb.over_correct(vo, gc["over_tangent_px"], gc["under_tangent_px"])
            ou.append(bool(ok))
            go, gu = np.asarray(gc["over_tangent_px"]), np.asarray(gc["under_tangent_px"])
            cos30 = math.cos(math.radians(30))
            a = abs(np.dot(vo, go)) > cos30 and abs(np.dot(vu, gu)) > cos30
            b = abs(np.dot(vo, gu)) > cos30 and abs(np.dot(vu, go)) > cos30
            pairing.append(bool(a or b))
            p_conf = c.confidence
            margin = math.log(p_conf / (1 - p_conf)) / kappa if 0 < p_conf < 1 else 0.0
            row["margins"].append((margin, bool(ok)))
    row["over_under_correct"], row["over_under_n"] = int(sum(ou)), len(ou)
    row["pairing_correct"], row["pairing_n"] = int(sum(pairing)), len(pairing)
    answer, p_type = None, None
    if D is not None and D.is_closed and not rep["abstain_reasons"]:
        try:
            u = propagate(D, threshold=0.95, max_uncertain=8)
            p_type = u["most_likely_probability"]
            answer = u["most_likely"] if u["decision"] == "report" else None
            row["most_likely"] = u["most_likely"]
            if answer is None:
                row["abstain_reasons"] = row["abstain_reasons"] + ["type probability below 0.95"]
        except Exception as e:  # noqa
            row["abstain_reasons"] = row["abstain_reasons"] + [f"error: {str(e)[:60]}"]
    row["answer"], row["p"] = answer, p_type
    return row


def fit_kappa(margins):
    best, best_ll = 4.0, -1e18
    for k in np.linspace(0.5, 30, 60):
        ll = sum(math.log(1 / (1 + math.exp(-k * m))) if ok else math.log(1 - 1 / (1 + math.exp(-k * m)) + 1e-12)
                 for m, ok in margins)
        if ll > best_ll:
            best, best_ll = float(k), ll
    return best


def summarise(rows):
    groups = [r["cable_id"] for r in rows]
    tp = sum(r["tp"] for r in rows); npred = sum(r["n_pred"] for r in rows); ngt = sum(r["n_gt"] for r in rows)
    ans = [r["answer"] for r in rows]
    correct = [a == r["gt"] for a, r in zip(ans, rows)]
    n_ans = sum(a is not None for a in ans)
    reasons = {}
    for r in rows:
        for x in r["abstain_reasons"]:
            k = x.split(":")[0][:48]
            reasons[k] = reasons.get(k, 0) + 1
    return {
        "images": len(rows), "cables": len(set(groups)),
        "crossing_precision": tp / npred if npred else float("nan"),
        "crossing_recall": tp / ngt if ngt else float("nan"),
        "recall_ci95_cable_bootstrap": pb.cable_bootstrap([r["tp"] / r["n_gt"] if r["n_gt"] else np.nan for r in rows], groups),
        "over_under_accuracy": sum(r["over_under_correct"] for r in rows) / max(1, sum(r["over_under_n"] for r in rows)),
        "over_under_n": sum(r["over_under_n"] for r in rows),
        "pairing_accuracy": sum(r["pairing_correct"] for r in rows) / max(1, sum(r["pairing_n"] for r in rows)),
        "count_head_mae": float(np.mean([abs(r["count_head"] - r["n_gt"]) for r in rows])),
        "coverage": n_ans / len(rows),
        "topology_accuracy_all": float(np.mean(correct)),
        "topology_accuracy_ci95_cable_bootstrap": pb.cable_bootstrap([float(c) for c in correct], groups),
        "selective_accuracy": (sum(c for c, a in zip(correct, ans) if a is not None) / n_ans) if n_ans else None,
        "wrong_confident_answers": sum(1 for a, r in zip(ans, rows) if a is not None and a != r["gt"]),
        "abstain_reasons": reasons,
        "sec_per_image_median": float(np.median([r["sec"] for r in rows])),
    }


def main():
    model = load(CKPT)
    model.extra["gamma"] = 2.0  # chosen on data/knot_aware/parser_dev (scripts/make_parser_dev_set.py), not here
    recs = [json.loads(l) for l in open(DATA / "records.jsonl")]
    split = json.load(open(DATA / "splits.json"))
    by_id = {r["image_id"]: r for r in recs}
    # 1. calibrate kappa on the benchmark train split
    cal = [evaluate_image(model, by_id[i], kappa=1.0) for i in split["train"]]
    margins = [mg for r in cal for mg in r["margins"]]
    kappa = fit_kappa(margins) if margins else 4.0
    model.extra["kappa"] = kappa
    print(f"kappa fitted on {len(margins)} train-split crossings: {kappa:.2f}")
    # 2. evaluate everything with the frozen kappa
    t0 = time.time()
    rows = [evaluate_image(model, r, kappa) for r in recs]
    print(f"evaluated {len(rows)} images in {time.time()-t0:.0f}s")
    sess = sorted({r["session"] for r in rows})
    photo = [r for r in rows if r["session"] != "S0_silhouette"]
    res = {
        "kappa": kappa, "kappa_fit_crossings": len(margins), "gamma": model.extra["gamma"], "tta_views": 8,
        "overall_photo_sessions": summarise(photo),
        "test_split": summarise([r for r in rows if r["image_id"] in set(split["test"])]),
        "unseen_sessions_S3_S4": summarise([r for r in rows if r["session"] in ("S3_dim", "S4_clutter")]),
        "by_session": {s: summarise([r for r in rows if r["session"] == s]) for s in sess},
        "per_image": [{k: v for k, v in r.items() if k != "margins"} for r in rows],
    }
    old_p = ROOT / "results" / "knotbench_photo" / "synthetic_v1.json"
    if old_p.exists():
        old = json.load(open(old_p))
        res["old_traverser"] = {
            "overall_photo_sessions": {k: old["overall"][k] for k in ("crossing_precision", "crossing_recall", "over_under_accuracy")}
            | {"coverage": old["overall"]["adapter"]["coverage"], "wrong_confident_answers_without_gate": old["overall"]["adapter_no_gate"].get("wrong_confident_answers")},
            "silhouette": {k: old["diagnostic_silhouette"][k] for k in ("crossing_precision", "crossing_recall", "over_under_accuracy")},
        }
    out = ROOT / "results" / "knotbench_photo" / "knot_aware_v1.json"
    json.dump(res, open(out, "w"), indent=1, default=float)
    for name in ("overall_photo_sessions", "test_split", "unseen_sessions_S3_S4"):
        o = res[name]
        print(name, json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in o.items()
                                if k not in ("recall_ci95_cable_bootstrap", "topology_accuracy_ci95_cable_bootstrap")}))
    for s, o in res["by_session"].items():
        print(f"  {s:<14} P {o['crossing_precision']:.2f} R {o['crossing_recall']:.2f} OU {o['over_under_accuracy']:.2f} "
              f"pair {o['pairing_accuracy']:.2f} cov {o['coverage']:.2f} sel {o['selective_accuracy']} wrong {o['wrong_confident_answers']}")


if __name__ == "__main__":
    main()
