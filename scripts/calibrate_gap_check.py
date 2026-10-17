"""
calibrate_gap_check.py - Fix the gap-measure check's settings (tau, delta) on development data only.

The development images are below. None of them belongs to a held-out set (KnotBench-Photo
val/test, blind sets, real pictures):
  knotbench_train  KnotBench-Photo synthetic_v1, train split (72 tube renders)
  parser_dev       data/knot_aware/parser_dev (40 photo-style, 40 silhouettes)
  gap_dev          new styled renders, built here with fresh seeds: 24 per style (tube, ribbon,
                   diagram, rope, shaded), plus 40 line drawings with nearly invisible gaps
                   (erased gap 1.05-1.4 line widths, while training used 1.6-4)

Steps
  1. Run the model once per image (8 views) and parse with both ridges, with the check off.
     Keep the diagram, the per-crossing evidence curves, and the ground-truth matching.
  2. Fuse mode (used): logit p' = logit p + lambda * D. For each tau, lambda is fitted by
     maximum likelihood on the matched development crossings (as kappa is). The (tau, lambda)
     pair with the highest likelihood is kept. As a check, each image is then decided again
     (type probability >= 0.95).
  3. Veto mode (ablation): for every (tau, delta) on a grid, apply p -> 1/2 where D < delta and
     decide each image again. The utility is U = correct - 19 * wrong (a wrong answer costs
     19 correct ones, the odds of the 0.95 report threshold). The veto is adopted only if it
     beats running no check.

Usage: python scripts/calibrate_gap_check.py [checkpoint]
Output: results/gap_check/calibration.json (+ features cache)
"""
from __future__ import annotations

import copy
import json
import math
import os
import pickle
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "src"), str(ROOT), str(ROOT / "scripts")]
import numpy as np
from PIL import Image

from knot_recognition.cable import gap_measure as gm

OUT = ROOT / "results" / "gap_check"
KIND = ["edge"]
GAP_DEV = ROOT / "data" / "knot_aware" / "gap_dev"
TAUS = (0.4, 0.5, 0.6, 0.7, 0.8)
KINDS = ("edge", "xsec")   # boundary continuity, cross-section correlation (cable.gap_measure)
DELTAS = (-1.0, -0.5, 0.0, 0.5, 1.0)


def build_gap_dev():
    from knot_recognition.cable import styles
    from knot_recognition.cable.identify import identify_diagram, type_key

    (GAP_DEV / "images").mkdir(parents=True, exist_ok=True)
    master = np.random.default_rng(31337)
    plan = [(st, None) for st in styles.STYLES for _ in range(24)] + [("diagram", (1.05, 1.4))] * 40
    recs = []
    for i, (style, gr) in enumerate(plan):
        s = int(master.integers(1 << 30)) + (1 << 31)
        curves, kind = styles.parametric_instance(np.random.default_rng(s))
        img, ann, D, meta = styles.render_styled(curves, np.random.default_rng(s), style=style, gap_mult_range=gr)
        try:
            label = type_key(identify_diagram(D))
        except Exception as e:  # noqa
            label = f"error {e}"
        iid = f"gapdev{i:03d}"
        img.save(GAP_DEV / "images" / f"{iid}.jpg", quality=95)
        recs.append({"image_id": iid, "file": f"images/{iid}.jpg", "source": "gap_dev",
                     "style": style + ("_thin_gap" if gr else ""), "seed": s, "curve": kind,
                     "ground_truth": {"type": label}, "annotation": ann,
                     "render": {k: meta[k] for k in ("background", "width", "gap_mult", "line_w")}})
    with open(GAP_DEV / "records.jsonl", "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    print("gap_dev:", len(recs), "images")


def dev_records():
    kb = ROOT / "data" / "knotbench_photo" / "synthetic_v1"
    split = json.load(open(kb / "splits.json"))
    by = {json.loads(l)["image_id"]: json.loads(l) for l in open(kb / "records.jsonl")}
    out = [(kb, by[i], "knotbench_train") for i in split["train"]]
    pdv = ROOT / "data" / "knot_aware" / "parser_dev"
    out += [(pdv, json.loads(l), "parser_dev_" + json.loads(l)["style"]) for l in open(pdv / "records.jsonl")]
    out += [(GAP_DEV, json.loads(l), "gap_dev_" + json.loads(l)["style"]) for l in open(GAP_DEV / "records.jsonl")]
    return out


def collect(spec: str, partial: Path = None, deadline: float = None):
    import eval_knot_aware as ek
    from knot_recognition.cable import photo_bench as pb
    from knot_recognition.cable.neural_parser import parse_best
    from knot_recognition.knot_aware import predict

    model = ek.load_any(spec)
    model.extra.setdefault("gamma", 2.0)
    feats = pickle.load(open(partial, "rb")) if partial is not None and partial.exists() else []
    done = {f["image_id"] for f in feats}
    for k, (folder, r, group) in enumerate(dev_records()):
        if r["image_id"] in done:
            continue
        if deadline and time.time() > deadline:
            pickle.dump(feats, open(partial, "wb"))
            print(f"PAUSED after {len(feats)} images", flush=True)
            sys.exit(75)
        img = np.asarray(Image.open(folder / r["file"]).convert("RGB"))
        pred = model.predict_maps(img) if hasattr(model, "predict_maps") else predict(model, img)
        D, rep = parse_best(img, model, pred=pred, gap_check=False)
        ann = r["annotation"]
        crossings = []
        if D is not None:
            ids = {c.cid for c in D.crossings}
            gpos = np.array([c["pos_px"] for c in ann["crossings"]]).reshape(-1, 2)
            ppos = np.array([c["pos_px"] for c in rep["crossings"]]).reshape(-1, 2)
            match = dict((p, g) for g, p in pb.match_crossings(gpos, ppos, ann["cable_width_px"] + 4))
            for ci, (eo, eu) in rep["_gap_evidence"].items():
                if ci + 1 not in ids:
                    continue
                vo, vu, c = ek.pair_dirs(D, ci + 1)
                ok = None
                if ci in match:
                    gc = ann["crossings"][match[ci]]
                    ok = pb.over_correct(vo, gc["over_tangent_px"], gc["under_tangent_px"])
                crossings.append({"cid": ci + 1, "p": c.confidence, "over_ok": ok, "ev_over": eo, "ev_under": eu})
        feats.append({"image_id": r["image_id"], "group": group, "label": r["ground_truth"]["type"],
                      "passes_gates": bool(D is not None and D.is_closed and not rep["abstain_reasons"]),
                      "diagram": D if D is not None else None, "crossings": crossings})
        if k % 25 == 0:
            print(k, group, flush=True)
    return feats


def _logit(p):
    p = min(max(p, 1e-9), 1 - 1e-9)
    return math.log(p / (1 - p))


def decide(f, tau, delta, lam=None):
    """Image answer after the veto (delta) or the fusion (lam); None = abstain."""
    from knot_recognition.cable.uncertainty import propagate

    if not f["passes_gates"]:
        return None, 0
    D = copy.deepcopy(f["diagram"])
    vetoed = 0
    by = {c["cid"]: c for c in f["crossings"]}
    flip = []
    for c in D.crossings:
        e = by.get(c.cid)
        if e is None:
            continue
        m = gm.evidence_margin(e["ev_over"], e["ev_under"], tau, KIND[0])
        if lam is not None:
            if m is not None:
                pf = 1 / (1 + math.exp(-(_logit(c.confidence) + lam * m)))
                c.confidence = pf
                if pf < 0.5:
                    flip.append(c.cid)
                    vetoed += 1
        elif delta is not None and c.confidence > 0.5 and (m is None or m < delta):
            c.confidence = 0.5
            vetoed += 1
    if flip:
        D = D.flip(flip)
    try:
        u = propagate(D, threshold=0.95, max_uncertain=8)
    except Exception:  # noqa
        return None, vetoed
    return (u["most_likely"] if u["decision"] == "report" else None), vetoed


def fit_lambda(feats, tau):
    L, Dv, y = [], [], []
    for f in feats:
        for c in f["crossings"]:
            if c["over_ok"] is None:
                continue
            m = gm.evidence_margin(c["ev_over"], c["ev_under"], tau, KIND[0])
            L.append(_logit(c["p"]))
            Dv.append(0.0 if m is None else m)
            y.append(bool(c["over_ok"]))
    L, Dv, y = np.array(L), np.array(Dv), np.array(y)

    def ll(lam):
        p = 1 / (1 + np.exp(-(L + lam * Dv)))
        return float(np.sum(np.where(y, np.log(p + 1e-12), np.log(1 - p + 1e-12))))
    lams = np.round(np.linspace(0, 6, 61), 2)
    vals = [ll(v) for v in lams]
    i = int(np.argmax(vals))
    return float(lams[i]), vals[i], ll(0.0), len(y), int((~y).sum())


def score(feats, tau, delta, lam=None):
    from knot_recognition.cable.compare import same_answer

    by_group = {}
    for f in feats:
        a, v = decide(f, tau, delta, lam)
        g = by_group.setdefault(f["group"], {"images": 0, "answered": 0, "correct": 0, "wrong": 0, "vetoed_crossings": 0})
        g["images"] += 1
        g["vetoed_crossings"] += v
        if a is not None:
            g["answered"] += 1
            g["correct" if same_answer(a, f["label"]) else "wrong"] += 1
    tot = {k: sum(g[k] for g in by_group.values()) for k in ("images", "answered", "correct", "wrong", "vetoed_crossings")}
    tot["utility"] = tot["correct"] - 19 * tot["wrong"]
    return tot, by_group


def crossing_stats(feats, tau):
    """Separation of correct and wrong network readings by the evidence margin D."""
    rows = []
    for f in feats:
        for c in f["crossings"]:
            if c["over_ok"] is None:
                continue
            rows.append((f["group"], bool(c["over_ok"]), gm.evidence_margin(c["ev_over"], c["ev_under"], tau, KIND[0])))
    out = {}
    for grp in sorted({r[0] for r in rows}) + ["all"]:
        R = [r for r in rows if grp == "all" or r[0] == grp]
        ok = [m for _, o, m in R if o and m is not None]
        bad = [m for _, o, m in R if not o and m is not None]
        auc = None
        if ok and bad:
            auc = float(np.mean([(a > b) + 0.5 * (a == b) for a in ok for b in bad]))
        out[grp] = {"crossings": len(R), "unmeasured": sum(m is None for _, _, m in R),
                    "net_correct": len([1 for _, o, _ in R if o]), "net_wrong": len([1 for _, o, _ in R if not o]),
                    "median_D_when_net_correct": float(np.median(ok)) if ok else None,
                    "median_D_when_net_wrong": float(np.median(bad)) if bad else None,
                    "auc_D_separates_correct_from_wrong": auc}
    return out


def main():
    spec = sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "checkpoints" / "knot_aware_nanotransformer.pth")
    OUT.mkdir(parents=True, exist_ok=True)
    if not (GAP_DEV / "records.jsonl").exists():
        build_gap_dev()
    import eval_knot_aware as ek
    cache = OUT / f"features2_{ek.spec_name(spec)}.pkl"
    if cache.exists():
        feats = pickle.load(open(cache, "rb"))
    else:
        budget = float(os.environ.get("CAL_BUDGET", 0))
        feats = collect(spec, cache.with_suffix(".partial"), time.time() + budget if budget else None)
        pickle.dump(feats, open(cache, "wb"))
        cache.with_suffix(".partial").unlink(missing_ok=True)
    base, base_g = score(feats, 0.5, None)
    fusion = []
    for kind in KINDS:
        KIND[0] = kind
        for tau in TAUS:
            lam, llv, ll0, n, nwrong = fit_lambda(feats, tau)
            fusion.append({"kind": kind, "tau": tau, "lambda": lam, "loglik": llv, "loglik_without_D": ll0,
                           "crossings": n, "network_wrong": nwrong})
            print(f"fuse {kind} tau {tau:.2f}: lambda {lam:.2f}  log-likelihood {llv:.1f} (without D {ll0:.1f})", flush=True)
    fz_dev = max(fusion, key=lambda r: r["loglik"])
    fz = fz_dev
    adopt = os.environ.get("ADOPT_KIND")       # a measure chosen for a stated reason other than dev likelihood
    if adopt:
        fz = max([r for r in fusion if r["kind"] == adopt], key=lambda r: r["loglik"])
    KIND[0] = fz["kind"]
    fz_tot, fz_g = score(feats, fz["tau"], None, fz["lambda"])
    print("fuse chosen", fz, {k: fz_tot[k] for k in ("answered", "correct", "wrong", "utility")}, flush=True)
    grid = []                               # veto ablation, with the measure chosen for fusion
    for tau in TAUS:
        for delta in DELTAS:
            tot, g = score(feats, tau, delta)
            grid.append({"kind": KIND[0], "tau": tau, "delta": delta, **tot, "by_group": g})
            print(f"tau {tau:.2f} delta {delta:+.2f}  answered {tot['answered']:3d} correct {tot['correct']:3d} "
                  f"wrong {tot['wrong']:2d}  U {tot['utility']}", flush=True)
    best = max(grid, key=lambda r: (r["utility"], r["correct"], -abs(r["delta"])))
    res = {"model": ek.spec_name(spec), "D": "G(strand read as under) - G(strand read as over), cable widths",
           "profile_half_width_w": gm.PROFILE_HALF_WIDTH, "profile_points": gm.PROFILE_POINTS,
           "no_check": {**base, "by_group": base_g},
           "fuse": {"rule": "logit p' = logit p + lambda * D; reading flips if p' < 1/2",
                    "selection": "maximum likelihood of the network-reading correctness on matched development crossings",
                    "best_by_dev_likelihood": {k: fz_dev[k] for k in ("kind", "tau", "lambda")},
                    "chosen": {"kind": fz["kind"], "tau": fz["tau"], "lambda": fz["lambda"]},
                    "adopted_kind_reason": os.environ.get("ADOPT_REASON") if adopt else None, "fits": fusion,
                    "image_level": {**fz_tot, "by_group": fz_g},
                    "crossing_separation": crossing_stats(feats, fz["tau"])},
           "veto": {"rule": "p -> 1/2 where D < delta (or D unmeasurable)",
                    "selection": "max utility U = correct - 19 * wrong on development images; adopted only if > no check",
                    "best": {k: best[k] for k in ("tau", "delta")}, "best_result": best,
                    "adopted": best["utility"] > base["utility"], "grid": grid}}
    json.dump(res, open(OUT / f"calibration_{ek.spec_name(spec)}.json", "w"), indent=1, default=float)
    print("no check:", {k: base[k] for k in ("answered", "correct", "wrong", "utility")})
    print("fuse:", res["fuse"]["chosen"], {k: fz_tot[k] for k in ("answered", "correct", "wrong", "utility")})
    print("veto best:", res["veto"]["best"], {k: best[k] for k in ("answered", "correct", "wrong", "utility")},
          "adopted" if res["veto"]["adopted"] else "not adopted")
    for grp, v in fz_g.items():
        print(f"  {grp:<26} fuse {v}  (no check: {base_g[grp]})")
    for grp, v in res["fuse"]["crossing_separation"].items():
        print(f"  D separation {grp:<26} {v}")


if __name__ == "__main__":
    main()
