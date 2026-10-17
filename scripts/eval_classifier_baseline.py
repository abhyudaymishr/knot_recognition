"""
eval_classifier_baseline.py - The image-to-label baseline: the original NanoTransformer that maps
an image to one of 250 classes (unknot + 249 prime knots up to 10 crossings).

It was trained on raw_knot/ (scripts/train_nanotransformer.py), so the 166 real pictures are part
of its training data and are NOT a test for it; they are reported only as a training-set check.

Measures on each set:
  forced      top-1 class, always answers (its native mode)
  selective   answers only when the top softmax probability >= 0.95 (the same threshold our system
              applies to its type probability)
  in_vocab    fraction of images whose true type is one of its 250 classes (links, composites and
              > 10-crossing knots are not)
Correct = same type up to mirror image.

Usage: python scripts/eval_classifier_baseline.py
Output: results/baselines/classifier_250.json
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "src"), str(ROOT), str(ROOT / "scripts")]
import numpy as np
from PIL import Image

from knot_recognition.cable.ledger import record_run, sha256_file
from knot_recognition.knot_aware import letterbox
from knot_recognition.nanotransformer import load_nanotransformer

CKPT = ROOT / "checkpoints" / "nanotransformer_knot250.pth"
THR = 0.95


def norm(label: str) -> str:
    return "0_1" if label == "0_unknot" else label


def summarise(rows, classes):
    n = len(rows)
    inv = [r for r in rows if r["gt"].rstrip("*") in classes]
    forced_ok = sum(r["top"] == r["gt"].rstrip("*") for r in rows)
    sel = [r for r in rows if r["p"] >= THR]
    sel_ok = sum(r["top"] == r["gt"].rstrip("*") for r in sel)
    return {"images": n, "in_vocabulary": len(inv), "forced_accuracy": forced_ok / n if n else None,
            "forced_wrong": n - forced_ok, "selective_answered": len(sel), "selective_correct": sel_ok,
            "selective_wrong": len(sel) - sel_ok, "median_top_probability": float(np.median([r["p"] for r in rows])) if rows else None}


def run(model, items):
    rows = []
    for img, gt, meta in items:
        out = model.predict_image(Image.fromarray(img), top_k=1)
        rows.append({"gt": gt, "top": norm(out["top_label"]), "p": out["top_probability"], **meta})
    return rows


def main():
    model = load_nanotransformer(CKPT)
    classes = {norm(c) for c in model.classes}
    res = {"checkpoint": CKPT.name, "classes": len(classes), "threshold": THR,
           "note": "trained on raw_knot/; real_pictures are its training data"}

    D = ROOT / "data" / "knotbench_photo" / "synthetic_v1"
    recs = [json.loads(l) for l in open(D / "records.jsonl")]
    items = [(np.asarray(Image.open(D / r["file"]).convert("RGB")), r["ground_truth"]["type"],
              {"session": r["session_id"]}) for r in recs]
    rows = run(model, items)
    res["knotbench_photo"] = {"photo_sessions": summarise([r for r in rows if r["session"] != "S0_silhouette"], classes),
                              "silhouettes": summarise([r for r in rows if r["session"] == "S0_silhouette"], classes)}

    C = ROOT / "data" / "knot_aware" / "close_strands"
    recs = [json.loads(l) for l in open(C / "records.jsonl")]
    rows = run(model, [(np.asarray(Image.open(C / r["file"]).convert("RGB")), r["ground_truth"]["type"], {}) for r in recs])
    res["close_strands"] = summarise(rows, classes)

    B = ROOT / "data" / "blind_real_v1"
    if (B / "manifest.jsonl").exists():
        msha = sha256_file(B / "manifest.jsonl")
        assert msha == (B / "MANIFEST_SHA256").read_text().strip(), "blind manifest changed after freezing"
        res["blind_ledger"] = record_run("blind:blind_real_v1", msha, CKPT, role="baseline", note="250-class classifier")
        man = [json.loads(l) for l in open(B / "manifest.jsonl")]
        man = [m for m in man if not m.get("excluded")]
        rows = run(model, [(letterbox(Image.open(B / m["file"]), 512)[0], m["label"], {"category": m["category"]}) for m in man])
        res["blind_real_v1"] = {c: summarise([r for r in rows if c == "all" or r["category"] == c], classes)
                                for c in ["all"] + sorted({r["category"] for r in rows})}

    import csv
    raw = Path("/mnt/user-data/uploads/knot_recognition/raw_knot")
    cat = list(csv.DictReader(open(ROOT / "data" / "real_knot_pictures" / "categories.csv")))
    if raw.exists():
        rows = run(model, [(letterbox(Image.open(raw / c["file"]), 512)[0], c["file"].split("/")[0].replace("0_unknot", "0_1"), {})
                           for c in cat])
        res["real_pictures_TRAINING_DATA"] = summarise(rows, classes)

    out = ROOT / "results" / "baselines" / "classifier_250.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(out, "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k not in ("blind_ledger",)}, indent=1))


if __name__ == "__main__":
    main()
