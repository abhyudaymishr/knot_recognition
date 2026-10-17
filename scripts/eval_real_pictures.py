"""
eval_real_pictures.py - First real-image test: the labelled knot pictures in raw_knot/.

166 real-world images (drawings, logos, 3D renders and photographs of physical
objects), labelled only by knot type (folder name). No crossing annotations exist,
so the measures are topological: answered / correct up to mirror image / wrong.
Categories were assigned by eye from a contact sheet
(data/real_knot_pictures/categories.csv); labels inherit any errors in the folders.

Pipelines: the skeleton traverser + geometric adapter + uncertainty (old) and the
knot-aware NanoTransformer parser (new), both with their abstention gates.
Output: results/real_knot_pictures.json
"""
import csv, glob, json, os, sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
import numpy as np
from PIL import Image

from knot_recognition.cable.from_traversal import diagram_from_traversal
from knot_recognition.cable.identify import identify_diagram, type_key
from knot_recognition.cable.neural_parser import parse_best as parse
from knot_recognition.cable.uncertainty import propagate
from knot_recognition.knot_aware import load
from knot_recognition.traverser import KnotTraverser

RAW = Path(os.environ.get("RAW_KNOT", ROOT / "raw_knot"))


def letterbox(img: Image.Image, size=512) -> np.ndarray:
    img = img.convert("RGB")
    a = np.asarray(img)
    bg = tuple(int(v) for v in np.median(np.concatenate([a[0], a[-1], a[:, 0], a[:, -1]]), axis=0))
    img.thumbnail((int(size * 0.9), int(size * 0.9)), Image.LANCZOS)
    canvas = Image.new("RGB", (size, size), bg)
    canvas.paste(img, ((size - img.width) // 2, (size - img.height) // 2))
    return np.asarray(canvas)


def base(t):
    return None if t is None else t.replace("*", "").replace(" # ", "#")


def run_one(D, rep):
    if D is None or not D.is_closed or rep.get("abstain_reasons"):
        return None, (rep.get("abstain_reasons") or ["no closed diagram"])[0].split(":")[0][:40]
    try:
        u = propagate(D, threshold=0.95, max_uncertain=8)
    except Exception as e:
        return None, f"error {str(e)[:30]}"
    return (u["most_likely"], None) if u["decision"] == "report" else (None, "type probability below 0.95")


def main():
    cats = {r["file"]: r["category"] for r in csv.DictReader(open(ROOT / "data" / "real_knot_pictures" / "categories.csv"))}
    sys.path.insert(0, str(ROOT / "scripts"))
    import eval_knot_aware as ek
    from knot_recognition.knot_aware import predict
    ckpt = os.environ.get("KNOT_AWARE_CKPT", str(ROOT / "checkpoints" / "knot_aware_nanotransformer.pth"))
    model = ek.load_any(ckpt)
    if os.environ.get("KA_KAPPA"):
        model.extra["kappa"] = float(os.environ["KA_KAPPA"])
    trav = KnotTraverser()
    rows = []
    for f in sorted(cats):
        label = f.split("/")[0].replace("0_unknot", "0_1")
        img = letterbox(Image.open(RAW / f))
        row = {"file": f, "label": label, "category": cats[f]}
        try:
            pred = model.predict_maps(img) if hasattr(model, "predict_maps") else predict(model, img)
            D, rep = parse(img, model, pred=pred)
            row["knot_aware"], row["knot_aware_reason"] = run_one(D, rep)
        except Exception as e:
            row["knot_aware"], row["knot_aware_reason"] = None, f"error {str(e)[:30]}"
        if os.environ.get("SKIP_TRAVERSER"):
            row["traverser"], row["traverser_reason"] = None, "skipped"
            rows.append(row)
            continue
        try:
            res = trav.process_image(img)
            D, rep = diagram_from_traversal(res)
            rep["abstain_reasons"] = (["open components"] if not D.is_closed else []) + \
                (["ambiguous crossings"] if any(i.status == "ambiguous" for i in res.intersections) else [])
            row["traverser"], row["traverser_reason"] = run_one(D, rep)
        except Exception as e:
            row["traverser"], row["traverser_reason"] = None, f"error {str(e)[:30]}"
        rows.append(row)
    out = {"images": len(rows), "note": __doc__.strip().splitlines()[0], "by_pipeline": {}}
    for pipe in ("traverser", "knot_aware"):
        res = {}
        for cat in ["all"] + sorted({r["category"] for r in rows}):
            R = [r for r in rows if cat == "all" or r["category"] == cat]
            ans = [r for r in R if r[pipe] is not None]
            ok = [r for r in ans if base(r[pipe]) == base(r["label"])]
            res[cat] = {"images": len(R), "answered": len(ans), "correct": len(ok), "wrong": len(ans) - len(ok),
                        "abstain_reasons": dict(Counter(r[pipe + "_reason"] for r in R if r[pipe] is None).most_common(5)),
                        "wrong_examples": [(r["file"], r["label"], r[pipe]) for r in ans if r not in ok][:8]}
        out["by_pipeline"][pipe] = res
    out["per_image"] = rows
    out["checkpoint"] = ek.spec_name(ckpt)
    json.dump(out, open(os.environ.get("OUT_JSON", ROOT / "results" / "real_knot_pictures.json"), "w"), indent=1)
    for pipe, res in out["by_pipeline"].items():
        print(pipe)
        for cat, v in res.items():
            print(f"   {cat:<10} n={v['images']:<4} answered {v['answered']:<3} correct {v['correct']:<3} wrong {v['wrong']:<3} {v['abstain_reasons']}")
            for w in v["wrong_examples"][:4]:
                print("        WRONG", w)


if __name__ == "__main__":
    main()
