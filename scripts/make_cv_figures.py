"""
make_cv_figures.py - Figures for the computer-vision paper (paper/cv/figures/).

  teaser.png         4 cases with the v3 model: a real render reported, a benchmark
                     cable reported, a silhouette abstained, touching strands abstained;
                     each with predicted crossings and centre-line ridge
  close_strands.pdf  answered fraction by strand gap, v1 / v2 / v3 (results JSONs)
  real_answered.png  all real pictures answered by v3

Usage: RAW_KNOT=<path to raw_knot> python scripts/make_cv_figures.py
"""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
import numpy as np
from PIL import Image, ImageDraw

OUT = ROOT / "paper" / "cv" / "figures"
KB = ROOT / "results" / "knotbench_photo"
RAW = Path(os.environ.get("RAW_KNOT", ROOT / "raw_knot"))
# ordinal blue ramp (light -> dark = v1 -> v3), validated palette steps
RAMP = {"v1": "#6da7ec", "v2": "#2a78d6", "v3": "#104281"}
INK, MUTED, GRID = "#1f1f1e", "#5f5e5a", "#e4e3df"


def close_strands_chart():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    res = {k: json.load(open(KB / f"close_strands_{k}.json")) for k in ("v1", "v2", "v3")}
    bins = list(res["v1"]["by_bin"])
    labels = ["0.6–1.0", "1.0–1.3", "1.3–1.7", "≥ 1.7"]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8})
    fig, ax = plt.subplots(figsize=(3.3, 1.9), dpi=300)
    w = 0.26
    names = {"v1": "v1 mask skeleton", "v2": "v2 ridge (128 grid)", "v3": "v3 two ridges (256 + 128)"}
    for i, k in enumerate(("v1", "v2", "v3")):
        ys = [100 * res[k]["by_bin"][b]["coverage"] for b in bins]
        xs = np.arange(len(bins)) + (i - 1) * (w + 0.02)
        ax.bar(xs, ys, w, color=RAMP[k], label=names[k], linewidth=0)
        for x, y, b in zip(xs, ys, bins):
            if res[k]["by_bin"][b]["wrong_confident_answers"]:
                ax.text(x, y + 2, "1 wrong", ha="center", va="bottom", fontsize=5.5, color=MUTED, rotation=90)
    ax.set_xticks(np.arange(len(bins)), labels)
    ax.set_xlabel("smallest gap between strands (cable widths)", color=INK)
    ax.set_ylabel("images answered (%)", color=INK)
    ax.set_ylim(0, 100)
    ax.yaxis.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, length=2)
    ax.legend(frameon=False, fontsize=6.5, loc="upper left")
    fig.tight_layout(pad=0.3)
    fig.savefig(OUT / "close_strands.pdf")
    plt.close(fig)


def overlay(img, pred, rep):
    import cv2

    a = np.asarray(img).copy()
    if "center_hi" in pred:
        r = cv2.resize(pred["center_hi"].astype(np.float32), a.shape[1::-1], interpolation=cv2.INTER_CUBIC)
        m = np.clip(r, 0, 1)[..., None]
        m = m * (m > 0.5)
        a = (a * (1 - 0.4 * m) + np.array([255, 196, 0]) * 0.4 * m).astype(np.uint8)
    for c in rep["crossings"]:
        x, y = (int(round(v)) for v in c["pos_px"])
        p = c.get("p_over_correct")
        col = (21, 150, 60) if p is not None and p > 0.9 else (230, 100, 20)
        cv2.circle(a, (x, y), 11, col, 3)
    return Image.fromarray(a)


def teaser():
    from knot_recognition.cable.neural_parser import parse_best
    from knot_recognition.cable.uncertainty import propagate
    from knot_recognition.knot_aware import letterbox, load, predict

    m = load(ROOT / "checkpoints" / "knot_aware_nanotransformer.pth")
    recs = {json.loads(l)["image_id"]: json.loads(l) for l in open(ROOT / "data/knotbench_photo/synthetic_v1/records.jsonl")}
    bench = json.load(open(KB / "knot_aware_v3.json"))["per_image"]
    pick_b = next(r for r in bench if r["session"] == "S2_desk" and r["answer"] and r["answer"] != "0_1")
    sil = next(r for r in bench if r["session"] == "S0_silhouette" and r["gt"] == pick_b["gt"])
    close = json.load(open(KB / "close_strands_v3.json"))["per_image"]
    touch = next(r for r in close if r["bin"] == "0.6-1.0" and r["answer"] is None)
    cases = [
        ("Real render", letterbox(Image.open(RAW / "6_1/Blue_Stevedore_Knot.png"), 512)[0], "6_1"),
        ("Benchmark, wood desk", np.asarray(Image.open(ROOT / "data/knotbench_photo/synthetic_v1" / recs[pick_b["image_id"]]["file"]).convert("RGB")), pick_b["gt"]),
        ("Silhouette", np.asarray(Image.open(ROOT / "data/knotbench_photo/synthetic_v1" / recs[sil["image_id"]]["file"]).convert("RGB")), sil["gt"]),
        ("Strands touching", np.asarray(Image.open(ROOT / "data/knot_aware/close_strands/images" / f"{touch['image_id']}.jpg").convert("RGB")), touch["gt"]),
    ]
    tiles = []
    for title, img, gt in cases:
        pred = predict(m, img)
        D, rep = parse_best(img, m, pred=pred)
        if D is not None and D.is_closed and not rep["abstain_reasons"]:
            u = propagate(D, threshold=0.95, max_uncertain=8)
            verdict = f"reports {u['most_likely']}  (p = {u['most_likely_probability']:.2f})" if u["decision"] == "report" \
                else f"abstains: type p = {u['most_likely_probability']:.2f} < 0.95"
        else:
            reason = rep["abstain_reasons"][0].split(":")[0]
            reason = {"skeleton branches away from any detected crossing (possible missed crossing)": "stray skeleton branch"}.get(reason, reason)
            verdict = "abstains: " + reason
        t = overlay(img, pred, rep).resize((384, 384))
        canvas = Image.new("RGB", (384, 384 + 44), "white")
        canvas.paste(t, (0, 44))
        d = ImageDraw.Draw(canvas)
        gt_s = f"{gt.split('|')[0].count(',') + 1}-component link" if gt.startswith("link") else gt
        d.text((6, 4), f"{title}   (truth {gt_s})", fill=(30, 30, 30))
        d.text((6, 22), verdict, fill=(21, 120, 50) if verdict.startswith("reports") else (180, 80, 10))
        tiles.append(np.asarray(canvas))
    Image.fromarray(np.hstack(tiles)).save(OUT / "teaser.png")


def real_grid():
    from knot_recognition.knot_aware import letterbox

    R = json.load(open(ROOT / "results" / "real_knot_pictures_v3.json"))
    tiles = []
    for r in [r for r in R["per_image"] if r["knot_aware"]]:
        a, _, _ = letterbox(Image.open(RAW / r["file"]), 256)
        im = Image.fromarray(a)
        d = ImageDraw.Draw(im)
        d.rectangle([0, 0, 256, 16], fill=(255, 255, 255))
        d.text((4, 2), f"label {r['label']}  read {r['knot_aware']}  ({r['category']})", fill=(0, 0, 0))
        tiles.append(np.asarray(im))
    while len(tiles) % 6:
        tiles.append(np.full_like(tiles[0], 255))
    Image.fromarray(np.vstack([np.hstack(tiles[i:i + 6]) for i in range(0, len(tiles), 6)])).save(OUT / "real_answered.png")


def blind_failures():
    """Post-scoring error analysis: every wrong answer of the frozen model on the blind set,
    with its detected crossings and over/under probabilities (the set was scored before this)."""
    import cv2

    from knot_recognition.cable.neural_parser import parse_best
    from knot_recognition.knot_aware import letterbox, load, predict

    res = json.load(open(ROOT / "results" / "blind" / "blind_real_v1_knot_aware_nanotransformer.json"))
    wrong = [r for r in res["per_image"] if r["answer"] is not None and not r["correct"]]
    m = load(ROOT / "checkpoints" / "knot_aware_nanotransformer.pth")
    tiles = []
    for r in wrong:
        img = letterbox(Image.open(ROOT / "data" / "blind_real_v1" / "images" / f"{r['image_id']}.png"), 512)[0]
        pred = predict(m, img)
        D, rep = parse_best(img, m, pred=pred)
        a = img.copy()
        for c in rep["crossings"]:
            x, y = (int(round(v)) for v in c["pos_px"])
            cv2.circle(a, (x, y), 12, (230, 100, 20), 3)
        t = Image.fromarray(a).resize((384, 384))
        canvas = Image.new("RGB", (384, 430), "white")
        canvas.paste(t, (0, 46))
        d = ImageDraw.Draw(canvas)
        ps = [c["p_over_correct"] for c in rep["crossings"] if c["p_over_correct"]]
        d.text((6, 4), f"truth {r['label']}   read {r['answer']}", fill=(30, 30, 30))
        d.text((6, 24), f"{len(ps)} crossings, over/under p = {min(ps):.2f}-{max(ps):.2f}", fill=(180, 80, 10))
        tiles.append(np.asarray(canvas))
    Image.fromarray(np.hstack(tiles)).save(OUT / "blind_failures.png")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    close_strands_chart()
    real_grid()
    teaser()
    blind_failures()
    print("wrote", sorted(p.name for p in OUT.iterdir()))
