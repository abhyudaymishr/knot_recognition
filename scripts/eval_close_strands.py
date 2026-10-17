"""
eval_close_strands.py - Parser coverage and errors as strands get closer (dev set only).

Usage: python scripts/eval_close_strands.py [checkpoint] [out_json]
"""
import json, sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "src"), str(ROOT), str(ROOT / "scripts")]
import eval_knot_aware as ek
from knot_recognition.knot_aware import load

DATA = ROOT / "data" / "knot_aware" / "close_strands"


def main():
    ckpt = sys.argv[1] if len(sys.argv) > 1 else str(ek.CKPT)
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "results" / "knotbench_photo" / "close_strands.json"
    model = ek.load_any(ckpt)
    import os
    kappa = float(os.environ.get("KA_KAPPA", model.extra.get("kappa", 26.5)))
    ek.DATA = DATA
    recs = [json.loads(l) for l in open(DATA / "records.jsonl")]
    rows = []
    for r in recs:
        r = dict(r, cable_id=r["image_id"], session_id="close")
        row = ek.evaluate_image(model, r, kappa)
        row["bin"] = r["bin"]; row["sep"] = r["min_separation_widths"]
        rows.append(row)
    res = {"checkpoint": ek.spec_name(ckpt), "kappa": kappa, "by_bin": {}}
    for b in sorted({r["bin"] for r in rows}, key=lambda s: float(s.split("-")[0])):
        rs = [r for r in rows if r["bin"] == b]
        s = ek.summarise(rs)
        res["by_bin"][b] = {k: s[k] for k in ("images", "crossing_precision", "crossing_recall", "over_under_accuracy",
                                              "coverage", "selective_accuracy", "wrong_confident_answers", "abstain_reasons")}
    s = ek.summarise(rows)
    res["all"] = {k: s[k] for k in ("images", "crossing_precision", "crossing_recall", "over_under_accuracy",
                                    "coverage", "selective_accuracy", "wrong_confident_answers", "abstain_reasons")}
    res["per_image"] = [{k: v for k, v in r.items() if k != "margins"} for r in rows]
    json.dump(res, open(out, "w"), indent=1, default=float)
    for b, o in list(res["by_bin"].items()) + [("all", res["all"])]:
        print(f"{b:<9} P {o['crossing_precision']:.2f} R {o['crossing_recall']:.2f} OU {o['over_under_accuracy']:.2f} "
              f"cov {o['coverage']:.2f} sel {o['selective_accuracy']} wrong {o['wrong_confident_answers']} {o['abstain_reasons']}")


if __name__ == "__main__":
    main()
