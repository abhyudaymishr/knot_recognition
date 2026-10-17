"""
eval_blind_set.py - Score a frozen blind set once (data/blind_real_v1, scripts/make_blind_set.py).

The manifest's SHA-256 must match MANIFEST_SHA256 (written when the set was frozen), and
every run is logged in results/heldout_ledger.jsonl. Role "main" is only accepted for the
frozen model (results/frozen_model.json).

Measures (labels are knot or link types only, no crossing annotation):
  answered, correct (same type up to mirror image of the whole diagram), wrong.

Usage: python scripts/eval_blind_set.py [MODEL_SPEC] [--role main|baseline|ablation] [--set data/blind_real_v1]
  MODEL_SPEC: knot-aware checkpoint path, 'unet_seg:<ckpt>' or 'unet_seg+kp:<seg>,<kp>'
Output: results/blind/<set>_<model>.json
"""
import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "src"), str(ROOT), str(ROOT / "scripts")]
from PIL import Image

import eval_knot_aware as ek
from knot_recognition.cable.compare import same_link
from knot_recognition.cable.ledger import record_run, sha256_file
from knot_recognition.cable.neural_parser import parse_best
from knot_recognition.cable.uncertainty import propagate
from knot_recognition.knot_aware import letterbox, predict


def same_type(answer: str, label: str) -> bool:
    """Equal as types, up to mirroring the whole diagram (a picture's chirality convention
    is not known for drawings and renders taken from tables).

    type_key appends one '*' per mirrored summand at the END of the name, except for the
    granny ('3_1 # 3_1') and square ('3_1 # 3_1*') knots, which are distinct types and are
    compared exactly. Links are compared exactly (linking numbers already up to sign)."""
    if answer is None:
        return False
    if answer.startswith("link") or label.startswith("link"):
        return same_link(answer, label)
    special = ("3_1 # 3_1", "3_1 # 3_1*")
    if answer in special or label in special:
        return answer == label
    parts = lambda k: sorted(x.strip() for x in k.rstrip("*").split("#"))
    return parts(answer) == parts(label)


def answer_one(model, img):
    pred = model.predict_maps(img) if hasattr(model, "predict_maps") else predict(model, img)
    D, rep = parse_best(img, model, pred=pred)
    if D is None or not D.is_closed or rep.get("abstain_reasons"):
        return None, (rep.get("abstain_reasons") or ["no closed diagram"])[0].split(":")[0][:48], len(rep.get("crossings", []))
    try:
        u = propagate(D, threshold=0.95, max_uncertain=8)
    except Exception as e:  # noqa
        return None, f"error {str(e)[:30]}", len(rep.get("crossings", []))
    if u["decision"] != "report":
        return None, "type probability below 0.95", len(rep.get("crossings", []))
    return u["most_likely"], None, len(rep.get("crossings", []))


def summarise_rows(rows):
    out = {}
    for cat in ["all"] + sorted({r["category"] for r in rows}):
        R = [r for r in rows if cat == "all" or r["category"] == cat]
        ans = [r for r in R if r["answer"] is not None]
        ok = [r for r in ans if r["correct"]]
        out[cat] = {"images": len(R), "answered": len(ans), "correct": len(ok), "wrong": len(ans) - len(ok),
                    "abstain_reasons": dict(Counter(r["reason"] for r in R if r["answer"] is None).most_common(6)),
                    "wrong_examples": [(r["image_id"], r["label"], r["answer"]) for r in ans if not r["correct"]][:10]}
    return out


def rescore(path: Path, note: str):
    """Recompute correctness from stored answers (no model is run): used when the comparison
    itself had a bug. The note is kept in the file."""
    res = json.load(open(path))
    for r in res["per_image"]:
        r["correct"] = same_type(r["answer"], r["label"]) if r["answer"] else None
    res["by_category"] = summarise_rows(res["per_image"])
    res.setdefault("scoring_fixes", []).append(note)
    json.dump(res, open(path, "w"), indent=1)
    return res


def main():
    if len(sys.argv) > 2 and sys.argv[1] == "--rescore":
        res = rescore(Path(sys.argv[2]), sys.argv[3] if len(sys.argv) > 3 else "rescored")
        for cat, v in res["by_category"].items():
            print(f"{cat:<9} n={v['images']:<4} answered {v['answered']:<3} correct {v['correct']:<3} wrong {v['wrong']:<3}")
        return
    ap = argparse.ArgumentParser()
    ap.add_argument("model", nargs="?", default=str(ek.CKPT))
    ap.add_argument("--role", default="main", choices=["main", "baseline", "ablation"])
    ap.add_argument("--set", default=str(ROOT / "data" / "blind_real_v1"))
    ap.add_argument("--new-main-reason", default=None)
    ap.add_argument("--tag", default="", help="suffix for the output file (e.g. _gap for KA_GAP=1 runs)")
    a = ap.parse_args()
    folder = Path(a.set)
    msha = sha256_file(folder / "manifest.jsonl")
    frozen = (folder / "MANIFEST_SHA256").read_text().strip()
    if msha != frozen:
        raise SystemExit("refused: manifest.jsonl changed after the set was frozen")
    ckpt_file = a.model.split(":", 1)[1].split(",")[0] if ":" in a.model else a.model
    gap = os.environ.get("KA_GAP", "0") == "1"
    if gap and a.role == "main":
        raise SystemExit("refused: the gap-measure check changes the frozen parser; score it as an ablation")
    entry = record_run(f"blind:{folder.name}", msha, ckpt_file, role=a.role, allow_new_main=a.new_main_reason,
                       note=a.model + (" +gap_check" if gap else "") + (f" kappa={os.environ['KA_KAPPA']}" if os.environ.get("KA_KAPPA") else ""))
    model = ek.load_any(a.model)
    if os.environ.get("KA_KAPPA"):
        model.extra["kappa"] = float(os.environ["KA_KAPPA"])
    rows = []
    for line in open(folder / "manifest.jsonl"):
        r = json.loads(line)
        if r.get("excluded"):
            continue
        if sha256_file(folder / r["file"]) != r["sha256"]:
            raise SystemExit(f"refused: {r['file']} changed after freezing")
        img = letterbox(Image.open(folder / r["file"]), 512)[0]
        try:
            ans, why, ncr = answer_one(model, img)
        except Exception as e:  # noqa
            ans, why, ncr = None, f"error {str(e)[:30]}", 0
        rows.append({"image_id": r["image_id"], "category": r["category"], "label": r["label"], "answer": ans,
                     "correct": same_type(ans, r["label"]) if ans else None, "reason": why, "crossings_detected": ncr})
    res = {"set": folder.name, "manifest_sha256": msha, "model": ek.spec_name(a.model) + a.tag, "role": a.role, "ledger": entry,
           "gap_check": gap, "kappa": model.extra.get("kappa"),
           "by_category": summarise_rows(rows)}
    res["per_image"] = rows
    out = ROOT / "results" / "blind" / f"{folder.name}_{ek.spec_name(a.model)}{a.tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(out, "w"), indent=1)
    for cat, v in res["by_category"].items():
        print(f"{cat:<9} n={v['images']:<4} answered {v['answered']:<3} correct {v['correct']:<3} wrong {v['wrong']:<3} {v['abstain_reasons']}")


if __name__ == "__main__":
    main()
