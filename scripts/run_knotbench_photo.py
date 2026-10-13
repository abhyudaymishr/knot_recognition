"""
run_knotbench_photo.py - Build the synthetic KnotBench-Photo split and score photo pipelines on it.

Pipelines scored (all on the same images):
  native        current app path: traverser PD code (resolved crossings only) -> identify;
                abstains when any crossing is ambiguous, as the Space does.
  adapter       traverser -> cable.from_traversal Diagram (geometry-derived PD signs)
                -> cable.uncertainty.propagate; abstains below 0.95 type probability.
  adapter_gt_ou adapter diagram with ground-truth over/under injected at matched crossings:
                isolates crossing detection + traversal from over/under reading.

Real photographs use the same records and metrics; see docs/knotbench_photo_protocol.md.
Outputs: data/knotbench_photo/synthetic_v1/{images/, records.jsonl, splits.json}
         results/knotbench_photo/synthetic_v1.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

import numpy as np
import snappy

from knot_recognition.cable import photo_bench as pb
from knot_recognition.cable.from_traversal import diagram_from_traversal
from knot_recognition.cable.identify import identify_diagram, identify_link, type_key
from knot_recognition.cable.synth import CABLE_COLOURS, DIAGRAM_SESSION, SESSIONS as PHOTO_SESSIONS, make_instance, render

# four photo-like sessions + one diagnostic session (solid silhouette: crossings visible, over/under not)
SESSIONS = {**PHOTO_SESSIONS, "S0_silhouette": DIAGRAM_SESSION["S0_silhouette"]}
from knot_recognition.cable.uncertainty import propagate
from knot_recognition.cable.consistency import euler_check
from knot_recognition.cable.diagram import from_curves_3d
from knot_recognition.traverser import KnotTraverser

DATA = ROOT / "data" / "knotbench_photo" / "synthetic_v1"
OUT = ROOT / "results" / "knotbench_photo"

TYPES = {  # intended type -> braid word (rendering is checked against the intended type)
    "0_1 (twisted)": [1, 2, -1, -2],
    "3_1": [1, 1, 1],
    "3_1*": [-1, -1, -1],
    "4_1": [1, -2, 1, -2],
    "5_1": [1, 1, 1, 1, 1],
    "5_2": None,
    "6_1": None,
    "granny": [1, 1, 1, 2, 2, 2],
    "square": [1, 1, 1, -2, -2, -2],
    "hopf link": [1, 1],
    "unlink (2 cables)": [1, -1],
}


def well_posed(D, width_px: float) -> bool:
    """Reject renders a careful human could not annotate: crossings too close or too shallow."""
    P = np.array([c.pos for c in D.crossings]) if D.crossings else np.zeros((0, 2))
    if len(P) > 1:
        dmin = np.min(np.linalg.norm(P[:, None] - P[None], axis=2) + np.eye(len(P)) * 1e9)
        if dmin < 2.0 * width_px:
            return False
    for c in D.crossings:
        to = D.components[c.over.component].tangent_at(c.over.t)
        tu = D.components[c.under.component].tangent_at(c.under.t)
        if abs(to[0] * tu[1] - to[1] * tu[0]) < 0.35:  # < ~20 degrees
            return False
    return not touching(D, width_px)


def touching(D, width_px: float) -> bool:
    """Two strand pieces closer than 1.3 cable widths away from any crossing (they would merge in a photo)."""
    from scipy.spatial import cKDTree

    pts, meta = [], []
    for ci, comp in enumerate(D.components):
        n = max(8, int(comp.length / (0.5 * width_px)))
        for t in np.linspace(0, comp.length, n, endpoint=False):
            pts.append(comp.point_at(t))
            meta.append((ci, t, comp.length))
    pts = np.array(pts)
    cross = np.array([c.pos for c in D.crossings]).reshape(-1, 2)
    for i, j in cKDTree(pts).query_pairs(1.3 * width_px):
        (ci, ti, L), (cj, tj, _) = meta[i], meta[j]
        if ci == cj and min(abs(ti - tj), L - abs(ti - tj)) < 3 * width_px:
            continue
        mid = (pts[i] + pts[j]) / 2
        if len(cross) and np.min(np.linalg.norm(cross - mid, axis=1)) < 2.5 * width_px:
            continue
        return True
    return False


def build(n_cables: int, seed: int):
    db = json.load(open(ROOT / "data" / "knotinfo" / "knotinfo_full_dataset.json"))
    (DATA / "images").mkdir(parents=True, exist_ok=True)
    records = []
    rng_master = np.random.default_rng(seed)
    cable_no = 0
    rejected = {"type_changed": 0, "ill_posed": 0}
    for tname, word in TYPES.items():
        word = word or db[tname]["braid"]
        intended = None
        made = 0
        attempt = 0
        while made < n_cables and attempt < 60 * n_cables:
            attempt += 1
            s = int(rng_master.integers(1 << 30))
            curves = make_instance(word, s, wobble=0.15)
            if intended is None:
                intended = type_key(identify_diagram(from_curves_3d(make_instance(word, s, wobble=0.0))))
            try:
                gt_key = type_key(identify_diagram(from_curves_3d(curves)))
            except Exception:
                rejected["invalid_geometry"] = rejected.get("invalid_geometry", 0) + 1
                continue
            if gt_key != intended:
                rejected["type_changed"] += 1
                continue
            cable_no += 1
            cable_id = f"c{cable_no:03d}"
            colour = list(CABLE_COLOURS)[int(rng_master.integers(len(CABLE_COLOURS)))]
            width = int(rng_master.integers(7, 13))
            ok_all = []
            for sess in SESSIONS:
                rng = np.random.default_rng(int(rng_master.integers(1 << 30)))
                img, ann, Dpx = render(curves, sess, colour, width, rng)
                if not well_posed(Dpx, ann["cable_width_px"]):
                    ok_all = None
                    break
                ok_all.append((sess, img, ann, Dpx))
            if ok_all is None:
                rejected["ill_posed"] += 1
                cable_no -= 1
                continue
            made += 1
            for sess, img, ann, Dpx in ok_all:
                iid = f"{cable_id}_{sess}"
                img.save(DATA / "images" / f"{iid}.jpg", quality=95)
                records.append({
                    "image_id": iid, "file": f"images/{iid}.jpg", "cable_id": cable_id, "session_id": sess,
                    "view": 0, "source": "synthetic",
                    "cable": {"type": colour, "width_px": width},
                    "capture": SESSIONS[sess],
                    "ground_truth": {"type": gt_key, "intended": tname, "how_known": "rendered"},
                    "annotation": ann,
                })
    with open(DATA / "records.jsonl", "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    photo = [r for r in records if r["session_id"] != "S0_silhouette"]
    split = pb.group_split(photo, test_sessions=["S3_dim", "S4_clutter"], seed=seed)
    problems = pb.check_split(photo, split)
    assert not problems, problems
    json.dump(split, open(DATA / "splits.json", "w"), indent=1)
    return records, split, rejected


def score(records, split, eng):
    rows = []
    t0 = time.time()
    for k, r in enumerate(records):
        img = np.asarray(__import__("PIL.Image", fromlist=["Image"]).open(DATA / r["file"]).convert("RGB"))
        ann, gt = r["annotation"], r["ground_truth"]["type"]
        t1 = time.perf_counter()
        res = eng.process_image(img)
        trav_sec = time.perf_counter() - t1
        H = res.image_shape[0]
        scale = H / ann["image_size"][1]
        cand = [i for i in res.intersections if i.status in ("resolved", "ambiguous")]
        pred_pos = np.array([[i.pos[1] / scale, i.pos[0] / scale] for i in cand]).reshape(-1, 2)
        gt_pos = np.array([c["pos_px"] for c in ann["crossings"]]).reshape(-1, 2)
        radius = ann["cable_width_px"] + 4
        m = pb.match_crossings(gt_pos, pred_pos, radius)
        tp = len(m)
        row = {"image_id": r["image_id"], "cable_id": r["cable_id"], "session": r["session_id"], "gt": gt,
               "n_gt": len(gt_pos), "n_pred": len(pred_pos), "tp": tp,
               "n_rejected": sum(i.status == "rejected" for i in res.intersections),
               "n_ambiguous": sum(i.status == "ambiguous" for i in res.intersections),
               "traverser_sec": trav_sec}
        # over/under on matched crossings, via the adapter's geometry (covers ambiguous ones too)
        ou = []
        D = rep = None
        try:
            D, rep = diagram_from_traversal(res)
        except Exception as e:  # noqa
            row["adapter_error"] = str(e)
        if D is not None:
            byid = {c.cid: c for c in D.crossings}
            for g, p in m:
                c = byid.get(cand[p].crossing_id)
                if c is None:
                    continue
                to = D.components[c.over.component].tangent_at(c.over.t)
                v = [to[0], -to[1]]  # math -> pixel (y down)
                gc = ann["crossings"][g]
                ok = pb.over_correct(v, gc["over_tangent_px"], gc["under_tangent_px"])
                if ok is not None:
                    ou.append(ok)
        row["over_under_correct"] = int(sum(ou))
        row["over_under_n"] = len(ou)
        # native pipeline (what the Space does today)
        if not res.intersections or row["n_ambiguous"] > 0 or not res.pd_code:
            row["native"] = {"answer": None}
        else:
            try:
                r_nat = identify_link(snappy.Link([tuple(x) for _, x in res.pd_code]))
                row["native"] = {"answer": type_key(r_nat), "p": 1.0}
            except Exception as e:
                row["native"] = {"answer": "invalid_pd", "p": 1.0, "error": str(e)[:80]}
        # adapter + uncertainty pipeline without the Euler gate (to measure what the gate prevents)
        row["adapter_no_gate"] = {"answer": None}
        if D is not None and D.is_closed and (D.crossings or D.components):
            try:
                u0 = propagate(D, max_uncertain=8)
                row["adapter_no_gate"] = {"answer": u0["most_likely"] if u0["decision"] == "report" else None,
                                          "p": u0["most_likely_probability"], "most_likely": u0["most_likely"]}
            except Exception as e:
                row["adapter_no_gate"] = {"answer": None, "reason": f"error: {str(e)[:80]}"}
        # adapter + uncertainty pipeline, gated by the Euler crossing-count check
        if D is None or not D.crossings and not D.components:
            row["adapter"] = {"answer": None, "reason": "no diagram"}
        elif not (eu := euler_check(img, len(D.crossings), ann["cable_width_px"]))["consistent"]:
            row["adapter"] = {"answer": None, "reason": "euler check: crossing count inconsistent with image",
                              "euler": eu}
        elif not D.is_closed:
            row["adapter"] = {"answer": None, "reason": f"{rep['open_components']} open component(s): traversal broke"}
        else:
            try:
                u = propagate(D, max_uncertain=8)
                row["adapter"] = {"answer": u["most_likely"] if u["decision"] == "report" else None,
                                  "most_likely": u["most_likely"], "p": u["most_likely_probability"],
                                  "next_view": (u["next_best_views"] or [{}])[0].get("cid")}
            except Exception as e:
                row["adapter"] = {"answer": None, "reason": f"error: {str(e)[:80]}"}
        # perception-oracle row: detection + traversal only
        if D is not None and D.is_closed and len(m) == len(gt_pos) == len(D.crossings):
            byid = {c.cid: c for c in D.crossings}
            flips = []
            for g, p in m:
                c = byid.get(cand[p].crossing_id)
                to = D.components[c.over.component].tangent_at(c.over.t)
                gc = ann["crossings"][g]
                if not pb.over_correct([to[0], -to[1]], gc["over_tangent_px"], gc["under_tangent_px"]):
                    flips.append(c.cid)
            try:
                row["adapter_gt_ou"] = {"answer": type_key(identify_diagram(D.flip(flips))), "p": 1.0}
            except Exception as e:  # e.g. traversal produced a non-planar PD
                row["adapter_gt_ou"] = {"answer": None, "reason": f"error: {str(e)[:80]}"}
        else:
            row["adapter_gt_ou"] = {"answer": None}
        rows.append(row)
        if (k + 1) % 50 == 0:
            print(f"  scored {k+1}/{len(records)} ({time.time()-t0:.0f}s)")
    return rows


def summarise(rows, subset_ids=None):
    R = [r for r in rows if subset_ids is None or r["image_id"] in subset_ids]
    groups = [r["cable_id"] for r in R]
    tp = sum(r["tp"] for r in R); npred = sum(r["n_pred"] for r in R); ngt = sum(r["n_gt"] for r in R)
    out = {"images": len(R), "cables": len(set(groups)),
           "crossing_precision": tp / npred if npred else float("nan"),
           "crossing_recall": tp / ngt if ngt else float("nan"),
           "recall_ci95_cable_bootstrap": pb.cable_bootstrap([r["tp"] / r["n_gt"] if r["n_gt"] else np.nan for r in R], groups),
           "over_under_accuracy": (sum(r["over_under_correct"] for r in R) / max(1, sum(r["over_under_n"] for r in R))),
           "over_under_n": sum(r["over_under_n"] for r in R),
           "rejected_junctions_per_image": float(np.mean([r["n_rejected"] for r in R])) if R else float("nan"),
           "traverser_sec_median": float(np.median([r["traverser_sec"] for r in R])) if R else float("nan")}
    out["crossing_f1"] = (2 * out["crossing_precision"] * out["crossing_recall"] /
                          (out["crossing_precision"] + out["crossing_recall"])) if tp else 0.0
    for pipe in ("native", "adapter_no_gate", "adapter", "adapter_gt_ou"):
        ans = [r[pipe]["answer"] for r in R]
        answered = [a is not None for a in ans]
        correct = [a == r["gt"] for a, r in zip(ans, R)]
        n_ans = sum(answered)
        out[pipe] = {
            "coverage": n_ans / len(R) if R else float("nan"),
            "abstention_rate": 1 - n_ans / len(R) if R else float("nan"),
            "topology_accuracy_all": float(np.mean(correct)) if R else float("nan"),
            "topology_accuracy_ci95_cable_bootstrap": pb.cable_bootstrap([float(c) for c in correct], groups),
            "selective_accuracy": (sum(c for c, a in zip(correct, answered) if a) / n_ans) if n_ans else None,
            "invalid_pd": sum(a == "invalid_pd" for a in ans),
        }
        if pipe == "adapter":
            ml = [(r["adapter"].get("p"), r["adapter"].get("most_likely") == r["gt"]) for r in R if r["adapter"].get("p") is not None]
            if ml:
                out[pipe]["ece_type_probability"] = pb.ece([p for p, _ in ml], [c for _, c in ml])
                rc = pb.risk_coverage([p for p, _ in ml], [c for _, c in ml])
                out[pipe]["risk_coverage"] = rc[:: max(1, len(rc) // 10)]
            out[pipe]["failure_reasons"] = {}
            out[pipe]["wrong_confident_answers"] = sum(1 for r in R if r["adapter"]["answer"] not in (None, r["gt"]))
            out["adapter_no_gate"]["wrong_confident_answers"] = sum(1 for r in R if r["adapter_no_gate"]["answer"] not in (None, r["gt"]))
            for r in R:
                reason = r["adapter"].get("reason")
                if reason:
                    key = reason.split(":")[0] if "open component" not in reason else "open components (traversal broke)"
                    key = "euler check failed" if reason.startswith("euler") else key
                    out[pipe]["failure_reasons"][key] = out[pipe]["failure_reasons"].get(key, 0) + 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cables-per-type", type=int, default=6)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    t0 = time.time()
    records, split, rejected = build(a.cables_per_type, a.seed)
    print(f"built {len(records)} images ({len(records)//len(SESSIONS)} cables), rejected {rejected}, {time.time()-t0:.0f}s")
    # oracle sanity check: ground-truth geometry must identify to the label
    rows = score(records, split, KnotTraverser())
    res = {
        "dataset": {"images": len(records), "cables": len({r['cable_id'] for r in records}),
                    "types": sorted({r['ground_truth']['type'] for r in records}),
                    "sessions": list(SESSIONS), "rejected_while_building": rejected,
                    "split_sizes": {k: len(v) for k, v in split.items()},
                    "note": "Synthetic renders with exact ground truth. Not a substitute for real photographs."},
        "overall": summarise(rows, {r["image_id"] for r in rows if r["session"] != "S0_silhouette"}),
        "diagnostic_silhouette": summarise(rows, {r["image_id"] for r in rows if r["session"] == "S0_silhouette"}),
        "by_session": {s: summarise(rows, {r["image_id"] for r in rows if r["session"] == s}) for s in SESSIONS},
        "test_split": summarise(rows, set(split["test"])),
        "note_overall": "overall/test_split cover the four photo-like sessions; S0_silhouette is reported separately as a diagnostic.",
        "by_type": {t: summarise(rows, {r["image_id"] for r in rows if r["gt"] == t})
                    for t in sorted({r["gt"] for r in rows})},
        "per_image": rows,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(OUT / "synthetic_v1.json", "w"), indent=1, default=float)
    o = res["overall"]
    print(json.dumps({k: v for k, v in o.items() if k not in ("native", "adapter")}, indent=1, default=float))
    for p in ("native", "adapter_no_gate", "adapter", "adapter_gt_ou"):
        print(p, json.dumps({k: v for k, v in o[p].items() if k != "risk_coverage"}, default=float))
    print(f"done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
