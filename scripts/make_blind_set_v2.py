"""
make_blind_set_v2.py - Build and freeze blind set v2 WITHOUT viewing it, before any Minkowski /
Lebesgue model or the gap-measure check is scored on held-out data.

Why a second set: blind v1 was scored once, and its five wrong answers (drawings with very
small gaps) were then inspected. That inspection motivated the gap-measure check, so v1 can
no longer test the check without bias. v2 contains only new pictures that nobody has seen:

  renders   the knotess curves (KnotPlot-derived, MIT) of v1 seen from NEW directions
            (seed 20261015): one view within 25 degrees of the principal-plane normal and one
            random view per curve. Ray-traced with POV-Ray exactly as in v1. A new direction
            gives a new diagram, often with different crossings.
  drawings  line drawings of the same curves from a new front view, drawn with matplotlib
            (a renderer not used anywhere in training). Each crossing has a white halo around
            the over strand. The halo width per side is log-uniform in [0.1, 0.8] line widths,
            so some gaps are clear and some are nearly invisible. That is the case the gap
            check is for, and the right behaviour there is to abstain.

Labels are exact: the type of the projected diagram computed from the 3D curve (renders) or
of the drawn diagram (drawings). Near-duplicates of raw_knot/ and of blind v1 are dropped
(difference hash, Hamming distance <= 8). The manifest is frozen with SHA-256 hashes.

Usage: python scripts/make_blind_set_v2.py --knotess <clone> --raw <raw_knot>
Output: data/blind_real_v2/{images/, manifest.jsonl, MANIFEST_SHA256}
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "src"), str(ROOT), str(ROOT / "scripts")]
import numpy as np
from PIL import Image

import make_blind_set as v1

OUT = ROOT / "data" / "blind_real_v2"


def front_direction(allp: np.ndarray, rng) -> np.ndarray:
    ctr = allp.mean(0)
    nrm = np.linalg.svd(allp - ctr, full_matrices=False)[2][-1]
    while True:
        d = nrm + np.tan(np.radians(25)) * rng.uniform(0, 1) * rng.normal(size=3)
        d /= np.linalg.norm(d)
        if abs(d @ nrm) > np.cos(np.radians(25)):
            return d * rng.choice([-1, 1])


def view_matrix(d: np.ndarray, rng) -> np.ndarray:
    """Proper rotation whose third row is d (viewer at +d) with a random in-plane angle."""
    a = np.cross(d, rng.normal(size=3))
    a /= np.linalg.norm(a)
    b = np.cross(d, a)                       # a x b = d  ->  det = +1
    th = rng.uniform(0, 2 * math.pi)
    e1 = math.cos(th) * a + math.sin(th) * b
    e2 = np.cross(d, e1)
    return np.stack([e1, e2, d])


def draw(comps, rng, out: Path, size: int = 512):
    """Line drawing with a halo gap at every crossing; returns (label, metadata)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from knot_recognition.cable.diagram import from_curves_3d
    from knot_recognition.cable.identify import identify_diagram, type_key

    allp = np.vstack(comps)
    R = view_matrix(front_direction(allp, rng), rng)
    D = from_curves_3d(comps, view=R)
    label = type_key(identify_diagram(D))
    lw = float(rng.uniform(2.0, 8.0))                                  # line width, px
    halo = lw * float(np.exp(rng.uniform(np.log(0.1), np.log(0.8))))  # gap per side, px
    coloured = rng.random() < 0.3
    paper = rng.random() < 0.4
    bg = tuple(rng.uniform(0.9, 0.99, 3) * np.array([1.0, 0.98, 0.93])) if paper else (1.0, 1.0, 1.0)
    cols = [tuple(rng.uniform(0.05, 0.7, 3)) for _ in D.components] if coloured else [(0.05, 0.05, 0.05)] * len(D.components)

    fig = plt.figure(figsize=(size / 100, size / 100), dpi=100, facecolor=bg)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor(bg)
    pts = np.vstack([c.points for c in D.components])
    lo, hi = pts.min(0), pts.max(0)
    span = float(np.max(hi - lo)) / rng.uniform(0.7, 0.86)
    ctr = (lo + hi) / 2 + rng.uniform(-0.04, 0.04, 2) * span
    ax.set_xlim(ctr[0] - span / 2, ctr[0] + span / 2)
    ax.set_ylim(ctr[1] - span / 2, ctr[1] + span / 2)
    ax.axis("off")
    px_per_unit = size / span
    pt = 72.0 / 100.0                                                  # points per pixel at dpi 100
    for c, col in zip(D.components, cols):
        P = np.vstack([c.points, c.points[:1]]) if c.closed else c.points
        ax.plot(P[:, 0], P[:, 1], color=col, lw=lw * pt, solid_capstyle="round", solid_joinstyle="round")
    half = 2.5 * (lw + 2 * halo) / px_per_unit
    for cr in D.crossings:
        comp = D.components[cr.over.component]
        ts = np.linspace(cr.over.t - half, cr.over.t + half, 25)
        P = np.array([comp.point_at(t) for t in ts])
        ax.plot(P[:, 0], P[:, 1], color=bg, lw=(lw + 2 * halo) * pt, solid_capstyle="butt")
        ax.plot(P[:, 0], P[:, 1], color=cols[cr.over.component], lw=lw * pt, solid_capstyle="round")
    fig.savefig(out, dpi=100, facecolor=bg)
    plt.close(fig)
    return label, {"line_width_px": round(lw, 2), "halo_per_side_px": round(halo, 2),
                   "halo_over_line_width": round(halo / lw, 3), "crossings": len(D.crossings)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--knotess", required=True)
    ap.add_argument("--raw", required=True)
    ap.add_argument("--seed", type=int, default=20261015)
    ap.add_argument("--only", default=None, help="comma list of knotess ids (for a smoke test)")
    a = ap.parse_args()
    img_dir = OUT / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    links = v1.knotess_links(Path(a.knotess))
    if a.only:
        links = {k: v for k, v in links.items() if k in a.only.split(",")}
    rows = []
    for lid, comps in links.items():
        tag = lid.replace(".", "_")
        try:
            base_label = v1.exact_label(comps)
        except Exception as e:  # noqa
            print("skip", lid, str(e)[:60])
            continue
        if base_label.startswith("unknown"):
            continue
        for v, view in enumerate(("front", "random")):
            iid = f"v2_render_{tag}_{view}"
            out = img_dir / f"{iid}.png"
            v1.render_pov(v1.pov_scene(comps, rng, view=view), out)
            rows.append({"image_id": iid, "file": f"images/{out.name}", "category": "render", "label": base_label,
                         "source": "knotess centrelines (KnotPlot-derived), POV-Ray", "license": "MIT (geometry)",
                         "label_from": "exact, from the 3D curve", "knotess_id": lid, "view": view})
        iid = f"v2_drawing_{tag}"
        out = img_dir / f"{iid}.png"
        try:
            label, meta = draw(comps, rng, out)
        except Exception as e:  # noqa
            print("skip drawing", lid, str(e)[:60])
            continue
        if label.startswith("unknown"):
            out.unlink(missing_ok=True)
            continue
        rows.append({"image_id": iid, "file": f"images/{out.name}", "category": "drawing", "label": label,
                     "source": "knotess centrelines, drawn with matplotlib (halo gaps)", "license": "MIT (geometry)",
                     "label_from": "exact, from the drawn diagram", "knotess_id": lid, **meta})
        print(lid, len(rows), flush=True)

    ref = [v1.dhash(Image.open(p)) for p in Path(a.raw).rglob("*") if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".gif")]
    ref += [v1.dhash(Image.open(p)) for p in (ROOT / "data" / "blind_real_v1" / "images").glob("*.png")]
    for r in rows:
        h = v1.dhash(Image.open(OUT / r["file"]))
        r["dhash_min_distance_to_raw_knot_and_v1"] = dmin = min(bin(h ^ x).count("1") for x in ref)
        if dmin <= 8:
            r["excluded"] = "near-duplicate of a development or blind-v1 picture"
        r["sha256"] = v1.sha(OUT / r["file"])
    with open(OUT / "manifest.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r, sort_keys=True) + "\n")
    msha = v1.sha(OUT / "manifest.jsonl")
    (OUT / "MANIFEST_SHA256").write_text(msha + "\n")
    n_ok = sum("excluded" not in r for r in rows)
    print(f"{len(rows)} images ({n_ok} kept): {sum(r['category'] == 'drawing' for r in rows)} drawings, "
          f"{sum(r['category'] == 'render' for r in rows)} renders; manifest {msha[:12]}")


if __name__ == "__main__":
    main()
