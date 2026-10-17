"""
show_strand_order.py - Draw the strand order the pointer head reads from one image.

Left: the image with the points on the predicted centre-line, each strand drawn in its own
colour in the order the head links them; dashed links jump across a crossing disc (grey circle);
arrows give the reading direction (an image does not fix it; the head picks one).
Right: one crossing, close up. Points are numbered in reading order (row by row), as in
"1 2 3 / 4 5 6", and the text lists the order in which each strand visits them, which jumps
across the crossing.

The image is fixed by rule, not chosen by result: the first validation image at or after --from
(default 150: the half of val_v3 that no design decision looked at) with at least four crossings,
or --index. Its crossing for the close-up is the one nearest the image centre.

  python scripts/show_strand_order.py --head checkpoints/strand_pointer/smoke_s0.pth
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from knot_recognition.knot_aware import load
from knot_recognition.strand_pointer import (cut_radius, decode, edge_report, heat_peaks, label_points, load_head,
                                             load_records, mask_width, ridge_points, targets, walks)

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]      # categorical slots 1-4 (validated, light)
INK, MUTED = "#1f1f1e", "#6b6a63"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--head", default=str(ROOT / "checkpoints/strand_pointer/smoke_s0.pth"))
    ap.add_argument("--split", default="val_v3")
    ap.add_argument("--index", type=int, default=-1)
    ap.add_argument("--from", dest="start", type=int, default=150)
    ap.add_argument("--out", default=str(ROOT / "results/strand_pointer/strand_order.png"))
    a = ap.parse_args()

    head = load_head(a.head)
    trunk = load(ROOT / head.extra["trunk"]).eval()
    recs = load_records(ROOT / "data/knot_aware" / f"{a.split}_strands.pkl")
    npy = ROOT / "data/knot_aware" / a.split / "X.npy"
    X = np.load(npy, mmap_mode="r") if npy.exists() else np.load(ROOT / "data/knot_aware" / f"{a.split}.npz")["X"]
    k = a.index if a.index >= 0 else next(i for i, r in enumerate(recs) if i >= a.start and len(r["crossings"]) >= 4)
    rec = recs[k]

    with torch.no_grad():
        out = trunk(torch.from_numpy(np.array(X[k : k + 1])).permute(0, 3, 1, 2).float() / 255.0, return_feats=True)
        prob = torch.sigmoid(out["center_hi"][0]).numpy()
        C = heat_peaks(torch.sigmoid(out["heat"][0]).numpy())
        w = mask_width(torch.sigmoid(out["mask"][0]).numpy())
        pts = label_points(ridge_points(prob, C, w, head.cfg.spacing), rec, head.cfg.spacing)
        xy, valid, _, _ = targets([pts])
        p = torch.softmax(head(out["feats"], xy, valid, torch.tensor([cut_radius(w)])), -1)[0].numpy()
    edges = decode(p)
    strands = sorted(walks(edges, pts.n), key=lambda s: -len(s["order"]))
    rep = edge_report(edges, pts)
    R = cut_radius(w)

    grey = np.asarray(X[k])[..., :3].astype(float).mean(-1) / 255.0
    img = np.repeat((0.40 * grey + 0.58)[..., None], 3, -1)      # grey and light: the photo recedes, colour marks strands
    fig, (ax, az) = plt.subplots(1, 2, figsize=(12.5, 7.2))
    for axis in (ax, az):
        axis.imshow(img, extent=(0, 256, 256, 0), interpolation="bilinear")
        axis.set_xticks([]); axis.set_yticks([])
        for sp in axis.spines.values():
            sp.set_color("#d9d8d2")
        for c in C:
            axis.add_patch(plt.Circle(c, R, fill=False, lw=1, ec=MUTED, ls="-", alpha=0.7))

    def draw(axis, lw, arrows):
        for si, st in enumerate(strands):
            col = SERIES[si % len(SERIES)] if si < len(SERIES) else MUTED
            o = st["order"] + ([st["order"][0]] if st["closed"] else [])
            for a_, b_ in zip(o[:-1], o[1:]):
                P, Q = pts.xy[a_], pts.xy[b_]
                jump = np.linalg.norm(P - Q) > 2.5 * head.cfg.spacing
                axis.plot([P[0], Q[0]], [P[1], Q[1]], color=col, lw=lw, ls=(0, (3, 2)) if jump else "-",
                          solid_capstyle="round", zorder=3)
            if arrows:
                for t in range(6, len(o) - 1, 14):
                    P, Q = pts.xy[o[t]], pts.xy[o[t + 1]]
                    axis.annotate("", xy=Q, xytext=P, arrowprops=dict(arrowstyle="-|>", color=col, lw=0, mutation_scale=11),
                                  zorder=4)

    draw(ax, 2.0, True)
    ax.scatter(pts.xy[:, 0], pts.xy[:, 1], s=4, c=INK, zorder=5, linewidths=0)
    n_cl = sum(1 for s in strands if s["closed"])
    ax.set_title(f"Validation image {k}: {rep['hit']} of {rep['true']} true links, "
                 f"{rep['jump_hit']} of {rep['jump_true']} crossing jumps;\n{len(strands)} strand pieces, {n_cl} closed",
                 fontsize=10.5, color=INK, loc="left")
    handles = [plt.Line2D([], [], color=SERIES[i], lw=2, label=f"strand {i + 1}") for i in range(min(len(strands), 4))]
    if len(strands) > 4:
        handles.append(plt.Line2D([], [], color=MUTED, lw=2, label="other strand pieces"))
    handles.append(plt.Line2D([], [], color=INK, lw=1.6, ls=(0, (3, 2)), label="jump across a crossing"))
    ax.legend(handles=handles, loc="lower left", fontsize=9, frameon=True, framealpha=0.9, edgecolor="#d9d8d2")

    # close-up: the crossing nearest the centre
    c = C[np.argmin(np.linalg.norm(C - 128, axis=1))] if len(C) else np.array([128.0, 128.0])
    half = R + 3.2 * head.cfg.spacing
    draw(az, 2.6, False)
    inside = np.nonzero((np.abs(pts.xy - c) <= half - 1).all(1))[0]
    rows = np.round((pts.xy[inside, 1] - (c[1] - half)) / head.cfg.spacing)
    order = inside[np.lexsort((pts.xy[inside, 0], rows))]          # reading order: row by row
    label = {int(i): n + 1 for n, i in enumerate(order)}
    for i, n in label.items():
        az.scatter(*pts.xy[i], s=170, c="white", edgecolors=INK, linewidths=1, zorder=6)
        az.text(*pts.xy[i], str(n), ha="center", va="center", fontsize=7.5, color=INK, zorder=7)
    az.set_xlim(c[0] - half, c[0] + half); az.set_ylim(c[1] + half, c[1] - half)
    lines = []
    for si, st in enumerate(strands):
        seq = [label[i] for i in st["order"] if i in label]
        if len(seq) >= 2:
            o = st["order"]
            runs, cur = [], []
            for i in o + (o[:1] if st["closed"] else []):
                if i in label:
                    cur.append(i)
                elif cur:
                    runs.append(cur); cur = []
            if cur:
                runs.append(cur)
            seg = max(runs, key=len)
            txt = " → ".join(str(label[i]) for i in seg)
            lines.append((si, f"strand {si + 1}: {txt}"))
    az.set_title("Close-up of one crossing: points numbered in reading order", fontsize=10.5, color=INK, loc="left")
    az.text(0.0, -0.035, "Order in which each strand visits them:", transform=az.transAxes, fontsize=9.5, color=MUTED, va="top")
    for r_, (si, t) in enumerate(lines[:3]):
        az.text(0.0, -0.085 - 0.06 * r_, t, transform=az.transAxes, fontsize=10, color=INK, va="top",
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=SERIES[si % 4] if si < 4 else MUTED, lw=1.4))
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=150, bbox_inches="tight", pad_inches=0.15)
    print(f"[saved] {a.out}  image {k}: {rep}")


if __name__ == "__main__":
    main()
