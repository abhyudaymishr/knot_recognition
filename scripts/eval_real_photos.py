"""
eval_real_photos.py - Score the knot-aware parser on REAL annotated cable photographs.

Input: a folder with records.jsonl (made with tools/photo_annotator.html, schema in
cable/photo_bench.py) and the images it names. Each photo is letterboxed to 512 x 512
(the parser's working size) and its annotation is mapped with the same transform.

Reports the KnotBench-Photo metrics (crossing P/R, over/under by visibility, coverage,
selective accuracy, wrong answers) with cable-level bootstrap intervals, plus a
per-cable multi-view result: a cable counts as answered when any view answers and
no view gives a different type.

Score once: the set is frozen by the hash of records.jsonl and every image; each run is
logged in results/heldout_ledger.jsonl (cable/ledger.py). Only the frozen main model
(results/frozen_model.json) may be scored as role "main"; baselines use --role baseline.

Usage: python scripts/eval_real_photos.py data/knotbench_photo/real_v1 [--ckpt PATH] [--role main|baseline|ablation]
Output: results/knotbench_photo/real_<folder>[_<checkpoint>].json
"""
import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "src"), str(ROOT), str(ROOT / "scripts")]
import numpy as np
from PIL import Image

import eval_knot_aware as ek
from knot_recognition.cable import photo_bench as pb
from knot_recognition.knot_aware import letterbox, load


def letterbox_with_transform(img: Image.Image, size=512):
    return letterbox(img, size)


def map_annotation(ann, s, off):
    f = lambda p: [p[0] * s + off[0], p[1] * s + off[1]]
    out = dict(ann)
    out["cable_width_px"] = ann["cable_width_px"] * s
    out["crossings"] = [dict(c, pos_px=f(c["pos_px"])) for c in ann["crossings"]]
    out["components"] = [[f(p) for p in comp] for comp in ann.get("components", [])]
    out["endpoints_px"] = [f(p) for p in ann.get("endpoints_px", [])]
    return out


def set_hash(folder: Path, recs) -> str:
    import hashlib

    from knot_recognition.cable.ledger import sha256_file

    h = hashlib.sha256(open(folder / "records.jsonl", "rb").read())
    for r in sorted(recs, key=lambda r: r["image_id"]):
        h.update(sha256_file(folder / r["file"]).encode())
    return h.hexdigest()


def main():
    import argparse

    from knot_recognition.cable.ledger import record_run

    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--ckpt", default=str(ek.CKPT))
    ap.add_argument("--role", default="main", choices=["main", "baseline", "ablation"])
    ap.add_argument("--new-main-reason", default=None)
    a = ap.parse_args()
    folder, ckpt = Path(a.folder), Path(a.ckpt)
    model = load(ckpt)
    model.extra.setdefault("gamma", 2.0)
    kappa = model.extra.get("kappa", 26.5)
    recs = [json.loads(l) for l in open(folder / "records.jsonl") if l.strip()]
    sh = set_hash(folder, recs)
    entry = record_run(f"real_cables:{folder.name}", sh, ckpt, role=a.role, allow_new_main=a.new_main_reason)
    problems = {r["image_id"]: pb.validate_record(r) for r in recs}
    rows = []
    for r in recs:
        if problems[r["image_id"]]:
            print("skip", r["image_id"], problems[r["image_id"]])
            continue
        img, s, off = letterbox_with_transform(Image.open(folder / r["file"]))
        rr = dict(r, annotation=map_annotation(r["annotation"], s, off))
        row = ek.evaluate_image(model, rr, kappa, img=img)
        vis = [c.get("visibility", "clear") for c in r["annotation"]["crossings"]]
        row["visibility"] = vis
        rows.append(row)
    res = {"checkpoint": ckpt.name, "set_sha256": sh, "ledger": entry, "kappa": kappa, "folder": str(folder), "skipped": {k: v for k, v in problems.items() if v},
           "all": ek.summarise(rows) if rows else None}
    cables = {}
    for r in rows:
        cables.setdefault(r["cable_id"], []).append(r)
    mv = []
    for c, rs in cables.items():
        answers = {r["answer"] for r in rs if r["answer"] is not None}
        mv.append({"cable_id": c, "gt": rs[0]["gt"], "views": len(rs),
                   "answer": answers.pop() if len(answers) == 1 else None, "conflict": len(answers) > 1})
    res["multi_view"] = {"cables": len(mv), "answered": sum(m["answer"] is not None for m in mv),
                         "correct": sum(m["answer"] == m["gt"] for m in mv),
                         "conflicting_views": sum(m["conflict"] for m in mv), "per_cable": mv}
    res["per_image"] = [{k: v for k, v in r.items() if k != "margins"} for r in rows]
    tag = "" if a.role == "main" else "_" + ckpt.stem
    out = ROOT / "results" / "knotbench_photo" / f"real_{folder.name}{tag}.json"
    json.dump(res, open(out, "w"), indent=1, default=float)
    if rows:
        print(json.dumps({k: v for k, v in res["all"].items() if "ci95" not in k}, default=float, indent=1))
    print("multi-view:", {k: v for k, v in res["multi_view"].items() if k != "per_cable"})


if __name__ == "__main__":
    main()
