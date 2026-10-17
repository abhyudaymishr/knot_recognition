"""
eval_suite.py - Every evaluation of one model configuration, resumable image by image.

Runs, for one model spec (and optionally the gap-measure evidence, KA_GAP fuse mode), the
same per-image code as the single-set scripts, so results compare directly with earlier files:
  kb        KnotBench-Photo synthetic_v1 (scripts/eval_knot_aware.py; kappa fitted on its train split)
  close     close-strands set              (scripts/eval_close_strands.py)
  real      raw_knot real pictures         (scripts/eval_real_pictures.py, knot-aware pipeline)
  blind_v1  frozen blind set v1            (scripts/eval_blind_set.py; ledger entry, role != main unless frozen)
  blind_v2  frozen blind set v2            (idem)
Each image's row is appended to results/suite/<name>/<set>.jsonl as soon as it is done, so the
run can be stopped (--budget seconds, exit code 75) and resumed on any machine. When a set is
complete, its summary is written where the single-set script would write it.

Usage: python scripts/eval_suite.py MODEL_SPEC --name NAME [--gap] [--sets kb,close,real,blind_v1,blind_v2]
                                   [--role ablation] [--budget 0] [--kappa K]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "src"), str(ROOT), str(ROOT / "scripts")]
import numpy as np
from PIL import Image

SETS = ("kb", "close", "real", "blind_v1", "blind_v2")


class Budget(Exception):
    pass


def jsonl(path):
    return [json.loads(l) for l in open(path)] if path.exists() else []


def run_rows(path, items, fn, deadline):
    """items: list of (key, payload); fn(payload) -> row dict. Appends rows not yet in path."""
    done = {r["_key"] for r in jsonl(path)}
    with open(path, "a") as f:
        for key, payload in items:
            if key in done:
                continue
            if deadline and time.time() > deadline:
                raise Budget()
            row = fn(payload)
            row["_key"] = key
            f.write(json.dumps(row, default=float) + "\n")
            f.flush()
    rows = {r["_key"]: r for r in jsonl(path)}
    return [rows[k] for k, _ in items]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--name", required=True)
    ap.add_argument("--gap", action="store_true", help="fuse the gap-measure evidence (KA_GAP=fuse)")
    ap.add_argument("--clutter", action="store_true", help="parser drops open centre-line pieces without crossings (KA_CLUTTER=1)")
    ap.add_argument("--sets", default=",".join(SETS))
    ap.add_argument("--role", default="ablation", choices=["main", "baseline", "ablation"])
    ap.add_argument("--budget", type=float, default=0, help="seconds; 0 = no limit")
    ap.add_argument("--kappa", type=float, default=None, help="skip the kappa fit and use this value")
    ap.add_argument("--threads", type=int, default=0)
    a = ap.parse_args()
    t0 = time.time()
    deadline = t0 + a.budget if a.budget else None
    os.environ["KA_GAP"] = "fuse" if a.gap else "0"
    os.environ["KA_CLUTTER"] = "1" if a.clutter else "0"
    if a.threads:
        import torch
        torch.set_num_threads(a.threads)

    import eval_blind_set as eb
    import eval_knot_aware as ek
    import eval_real_pictures as er
    from knot_recognition.cable.ledger import record_run, sha256_file
    from knot_recognition.knot_aware import letterbox

    work = ROOT / "results" / "suite" / a.name
    work.mkdir(parents=True, exist_ok=True)
    meta_p = work / "meta.json"
    meta = json.load(open(meta_p)) if meta_p.exists() else {"model": a.model, "gap": a.gap, "clutter": a.clutter,
                                                             "role": a.role, "ledger": {}}
    if meta["model"] != a.model or meta["gap"] != a.gap or meta.get("clutter", False) != a.clutter:
        raise SystemExit(f"refused: {work} belongs to {meta['model']} gap={meta['gap']} clutter={meta.get('clutter', False)}")
    model = ek.load_any(a.model)
    model.extra.setdefault("gamma", 2.0)
    kb_dir = ROOT / "data" / "knotbench_photo" / "synthetic_v1"
    kb_recs = [json.loads(l) for l in open(kb_dir / "records.jsonl")]
    split = json.load(open(kb_dir / "splits.json"))
    by_id = {r["image_id"]: r for r in kb_recs}

    try:
        # kappa: the frozen value, a given value, or a fit on the KnotBench train split (gap evidence off)
        if "kappa" not in meta:
            if a.kappa is not None:
                meta["kappa"], meta["kappa_from"] = a.kappa, "given"
            elif "kappa" in getattr(model, "extra", {}) and "frozen" in a.name:
                meta["kappa"], meta["kappa_from"] = model.extra["kappa"], "checkpoint (frozen)"
            else:
                cal = run_rows(work / "kappa_fit.jsonl", [(i, by_id[i]) for i in split["train"]],
                               lambda r: ek.evaluate_image(model, r, kappa=1.0, gap_check=False), deadline)
                margins = [tuple(m) for r in cal for m in r["margins"]]
                meta["kappa"], meta["kappa_from"] = (ek.fit_kappa(margins) if margins else 4.0), \
                    f"fit on {len(margins)} KnotBench train-split crossings"
            json.dump(meta, open(meta_p, "w"), indent=1)
        kappa = meta["kappa"]
        model.extra["kappa"] = kappa
        sets = [s for s in a.sets.split(",") if s]

        if "kb" in sets:
            rows = run_rows(work / "kb.jsonl", [(r["image_id"], r) for r in kb_recs],
                            lambda r: ek.evaluate_image(model, r, kappa), deadline)
            photo = [r for r in rows if r["session"] != "S0_silhouette"]
            res = {"kappa": kappa, "gamma": model.extra["gamma"], "tta_views": 8, "gap_check": a.gap, "clutter_filter": a.clutter,
                   "overall_photo_sessions": ek.summarise(photo),
                   "test_split": ek.summarise([r for r in rows if r["image_id"] in set(split["test"])]),
                   "unseen_sessions_S3_S4": ek.summarise([r for r in rows if r["session"] in ("S3_dim", "S4_clutter")]),
                   "by_session": {s: ek.summarise([r for r in rows if r["session"] == s]) for s in sorted({r["session"] for r in rows})},
                   "per_image": [{k: v for k, v in r.items() if k not in ("margins", "_key")} for r in rows],
                   "checkpoint": ek.spec_name(a.model), "suite": a.name}
            json.dump(res, open(ROOT / "results" / "knotbench_photo" / f"{a.name}.json", "w"), indent=1, default=float)

        if "close" in sets:
            cdir = ROOT / "data" / "knot_aware" / "close_strands"
            recs = [json.loads(l) for l in open(cdir / "records.jsonl")]

            def close_row(r):
                r2 = dict(r, cable_id=r["image_id"], session_id="close")
                row = ek.evaluate_image(model, r2, kappa, img=np.asarray(Image.open(cdir / r["file"]).convert("RGB")))
                row["bin"], row["sep"] = r["bin"], r["min_separation_widths"]
                return row
            rows = run_rows(work / "close.jsonl", [(r["image_id"], r) for r in recs], close_row, deadline)
            keys = ("images", "crossing_precision", "crossing_recall", "over_under_accuracy", "coverage",
                    "selective_accuracy", "wrong_confident_answers", "abstain_reasons")
            res = {"checkpoint": ek.spec_name(a.model), "kappa": kappa, "gap_check": a.gap, "clutter_filter": a.clutter, "by_bin": {}}
            for b in sorted({r["bin"] for r in rows}, key=lambda s: float(s.split("-")[0])):
                s = ek.summarise([r for r in rows if r["bin"] == b])
                res["by_bin"][b] = {k: s[k] for k in keys}
            s = ek.summarise(rows)
            res["all"] = {k: s[k] for k in keys}
            res["per_image"] = [{k: v for k, v in r.items() if k not in ("margins", "_key")} for r in rows]
            json.dump(res, open(ROOT / "results" / "knotbench_photo" / f"close_strands_{a.name}.json", "w"), indent=1, default=float)

        if "real" in sets:
            import csv

            from knot_recognition.knot_aware import predict
            cats = {r["file"]: r["category"] for r in csv.DictReader(open(ROOT / "data" / "real_knot_pictures" / "categories.csv"))}

            def real_row(f):
                row = {"file": f, "label": f.split("/")[0].replace("0_unknot", "0_1"), "category": cats[f]}
                img = er.letterbox(Image.open(er.RAW / f))
                try:
                    pred = model.predict_maps(img) if hasattr(model, "predict_maps") else predict(model, img)
                    D, rep = er.parse(img, model, pred=pred)
                    row["knot_aware"], row["knot_aware_reason"] = er.run_one(D, rep)
                except Exception as e:  # noqa
                    row["knot_aware"], row["knot_aware_reason"] = None, f"error {str(e)[:30]}"
                return row
            rows = run_rows(work / "real.jsonl", [(f, f) for f in sorted(cats)], real_row, deadline)
            res = {}
            for cat in ["all"] + sorted({r["category"] for r in rows}):
                R = [r for r in rows if cat == "all" or r["category"] == cat]
                ans = [r for r in R if r["knot_aware"] is not None]
                ok = [r for r in ans if er.base(r["knot_aware"]) == er.base(r["label"])]
                res[cat] = {"images": len(R), "answered": len(ans), "correct": len(ok), "wrong": len(ans) - len(ok),
                            "abstain_reasons": dict(Counter(r["knot_aware_reason"] for r in R if r["knot_aware"] is None).most_common(5)),
                            "wrong_examples": [(r["file"], r["label"], r["knot_aware"]) for r in ans if r not in ok][:8]}
            out = {"images": len(rows), "by_pipeline": {"knot_aware": res}, "checkpoint": ek.spec_name(a.model),
                   "kappa": kappa, "gap_check": a.gap, "clutter_filter": a.clutter, "suite": a.name,
                   "per_image": [{k: v for k, v in r.items() if k != "_key"} for r in rows]}
            json.dump(out, open(ROOT / "results" / f"real_knot_pictures_{a.name}.json", "w"), indent=1)

        for s in ("blind_v1", "blind_v2"):
            if s not in sets:
                continue
            folder = ROOT / "data" / f"blind_real_{s.split('_')[1]}"
            msha = sha256_file(folder / "manifest.jsonl")
            if msha != (folder / "MANIFEST_SHA256").read_text().strip():
                raise SystemExit(f"refused: {folder.name} manifest changed after freezing")
            if (a.gap or a.clutter) and a.role == "main":
                raise SystemExit("refused: the gap evidence / clutter filter change the frozen parser; score as an ablation")
            if s not in meta["ledger"]:                  # one ledger entry per (set, configuration), at first scoring
                ck = a.model.split(":", 1)[1].split(",")[0] if ":" in a.model else a.model
                meta["ledger"][s] = record_run(f"blind:{folder.name}", msha, ck, role=a.role,
                                               note=f"{a.model} suite={a.name} gap={a.gap} clutter={a.clutter} kappa={kappa}")
                json.dump(meta, open(meta_p, "w"), indent=1)
            man = [json.loads(l) for l in open(folder / "manifest.jsonl")]
            man = [r for r in man if not r.get("excluded")]

            def blind_row(r):
                if sha256_file(folder / r["file"]) != r["sha256"]:
                    raise SystemExit(f"refused: {r['file']} changed after freezing")
                img = letterbox(Image.open(folder / r["file"]), 512)[0]
                try:
                    ans, why, ncr = eb.answer_one(model, img)
                except Exception as e:  # noqa
                    ans, why, ncr = None, f"error {str(e)[:30]}", 0
                return {"image_id": r["image_id"], "category": r["category"], "label": r["label"], "answer": ans,
                        "correct": eb.same_type(ans, r["label"]) if ans else None, "reason": why, "crossings_detected": ncr,
                        **({"halo_over_line_width": r["halo_over_line_width"]} if "halo_over_line_width" in r else {})}
            rows = run_rows(work / f"{s}.jsonl", [(r["image_id"], r) for r in man], blind_row, deadline)
            rows = [{k: v for k, v in r.items() if k != "_key"} for r in rows]
            res = {"set": folder.name, "manifest_sha256": msha, "model": a.name, "role": a.role, "ledger": meta["ledger"][s],
                   "gap_check": a.gap, "clutter_filter": a.clutter, "kappa": kappa, "by_category": eb.summarise_rows(rows), "per_image": rows}
            out = ROOT / "results" / "blind" / f"{folder.name}_{a.name}.json"
            out.parent.mkdir(parents=True, exist_ok=True)
            json.dump(res, open(out, "w"), indent=1)
    except Budget:
        print(f"PAUSED {a.name} after {time.time() - t0:.0f}s", flush=True)
        sys.exit(75)
    (work / f"DONE_{'_'.join(sets)}").write_text(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()) + "\n")
    print(f"SUITE DONE {a.name} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
