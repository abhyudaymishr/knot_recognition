"""
eval_clutter_filter.py - Effect of the parser's clutter filter (KA_CLUTTER) on development images only:
data/knot_aware/clutter_dev (S3/S4-style renders), parser_dev, gap_dev. Every image is predicted
once and parsed with the filter off and on. Output: results/clutter_filter/<model>.json
Usage: python scripts/eval_clutter_filter.py MODEL_SPEC NAME KAPPA
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "src"), str(ROOT), str(ROOT / "scripts")]
import numpy as np
from PIL import Image

import eval_knot_aware as ek
from knot_recognition.cable.compare import same_answer
from knot_recognition.cable.neural_parser import parse_best
from knot_recognition.cable.uncertainty import propagate
from knot_recognition.knot_aware import predict


def decide(D, rep):
    if D is None or not D.is_closed or rep["abstain_reasons"]:
        return None, rep["abstain_reasons"][0].split(":")[0][:40] if rep["abstain_reasons"] else "no diagram"
    try:
        u = propagate(D, threshold=0.95, max_uncertain=8)
    except Exception as e:  # noqa
        return None, "error"
    return (u["most_likely"], None) if u["decision"] == "report" else (None, "type probability below 0.95")


def main():
    spec, name, kappa = sys.argv[1], sys.argv[2], float(sys.argv[3])
    model = ek.load_any(spec)
    model.extra.update(kappa=kappa, gamma=2.0)
    sets = [("clutter_dev", ROOT / "data/knot_aware/clutter_dev"), ("parser_dev", ROOT / "data/knot_aware/parser_dev"),
            ("gap_dev", ROOT / "data/knot_aware/gap_dev")]
    rows = []
    for sname, folder in sets:
        for l in open(folder / "records.jsonl"):
            r = json.loads(l)
            img = np.asarray(Image.open(folder / r["file"]).convert("RGB"))
            pred = model.predict_maps(img) if hasattr(model, "predict_maps") else predict(model, img)
            row = {"set": sname, "group": f"{sname}:{r.get('style', '')}", "image_id": r["image_id"], "gt": r["ground_truth"]["type"]}
            for flag in (False, True):
                D, rep = parse_best(img, model, pred=pred, clutter_filter=flag, gap_check=False)
                a, why = decide(D, rep)
                row["on" if flag else "off"] = {"answer": a, "correct": same_answer(a, row["gt"]) if a else None, "reason": why,
                                                "removed": rep.get("clutter_pieces_removed", 0)}
            rows.append(row)
    out = {"model": spec, "kappa": kappa, "by_group": {}}
    for g in sorted({r["group"] for r in rows}) + ["all"]:
        R = [r for r in rows if g == "all" or r["group"] == g]
        out["by_group"][g] = {f: {"images": len(R), "answered": sum(r[f]["answer"] is not None for r in R),
                                  "wrong": sum(r[f]["answer"] is not None and not r[f]["correct"] for r in R)} for f in ("off", "on")}
    out["changed"] = [{k: r[k] for k in ("image_id", "group", "gt", "off", "on")} for r in rows
                      if (r["off"]["answer"], r["off"]["reason"]) != (r["on"]["answer"], r["on"]["reason"])]
    Path(ROOT / "results" / "clutter_filter").mkdir(parents=True, exist_ok=True)
    json.dump(out, open(ROOT / "results" / "clutter_filter" / f"{name}.json", "w"), indent=1)
    for g, v in out["by_group"].items():
        print(f"{g:<28} off {v['off']['answered']:3d} ans / {v['off']['wrong']} wrong   on {v['on']['answered']:3d} ans / {v['on']['wrong']} wrong")


if __name__ == "__main__":
    main()
