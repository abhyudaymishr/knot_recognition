"""
make_close_strands_set.py - Stress set for strands that pass close together (dev only).

Random braids rendered in training-style sessions; each image records the minimum
centre-line separation between strand pieces that are not at a crossing, in cable
widths. Images are binned by that separation so the parser's coverage and errors can
be measured as strands get closer. Seeds are disjoint from training, the parser dev
set and KnotBench-Photo.
Output: data/knot_aware/close_strands/records.jsonl + images
"""
import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "src"), str(ROOT), str(ROOT / "scripts")]
import numpy as np
from scipy.spatial import cKDTree

from knot_recognition.cable import synth
from knot_recognition.cable.diagram import from_curves_3d
from knot_recognition.cable.identify import identify_diagram, type_key
from make_knot_aware_data import random_session, random_word

OUT = ROOT / "data" / "knot_aware" / "close_strands"
BINS = [(0.6, 1.0), (1.0, 1.3), (1.3, 1.7), (1.7, 99.0)]


def min_separation(D, w):
    """Smallest distance between strand points that are not adjacent along a strand and not at a crossing."""
    pts, meta = [], []
    for ci, comp in enumerate(D.components):
        n = max(8, int(comp.length / (0.4 * w)))
        for t in np.linspace(0, comp.length, n, endpoint=False):
            pts.append(comp.point_at(t)); meta.append((ci, t, comp.length))
    pts = np.array(pts)
    cross = np.array([c.pos for c in D.crossings]).reshape(-1, 2)
    best = 1e9
    for i, j in cKDTree(pts).query_pairs(3.0 * w):
        (ci, ti, L), (cj, tj, _) = meta[i], meta[j]
        if ci == cj and min(abs(ti - tj), L - abs(ti - tj)) < 3 * w:
            continue
        mid = (pts[i] + pts[j]) / 2
        if len(cross) and np.min(np.linalg.norm(cross - mid, axis=1)) < 2.5 * w:
            continue
        best = min(best, float(np.linalg.norm(pts[i] - pts[j])))
    return best / w


def main(per_bin=25, seed=5005):
    (OUT / "images").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    recs, counts = [], {b: 0 for b in BINS}
    tries = 0
    while min(counts.values()) < per_bin and tries < 20000:
        tries += 1
        word = random_word(rng)
        s = int(rng.integers(1 << 30)) + (5 << 31)
        curves = synth.make_instance(word, s, wobble=float(rng.uniform(0.15, 0.35)))
        try:
            gt = type_key(identify_diagram(from_curves_3d(curves)))
        except Exception:
            continue
        synth.SESSIONS["_close"] = random_session(rng)
        colour = list(synth.CABLE_COLOURS)[int(rng.integers(len(synth.CABLE_COLOURS)))]
        img, ann, D = synth.render(curves, "_close", colour, int(rng.integers(8, 13)), np.random.default_rng(s))
        if not ann["crossings"] or len(ann["crossings"]) > 16:
            continue
        sep = min_separation(D, ann["cable_width_px"])
        b = next((b for b in BINS if b[0] <= sep < b[1]), None)
        if b is None or counts[b] >= per_bin:
            continue
        iid = f"close{len(recs):03d}"
        img.save(OUT / "images" / f"{iid}.jpg", quality=95)
        recs.append({"image_id": iid, "file": f"images/{iid}.jpg", "min_separation_widths": sep,
                     "bin": f"{b[0]}-{b[1] if b[1] < 99 else 'inf'}", "ground_truth": {"type": gt}, "annotation": ann})
        counts[b] += 1
    with open(OUT / "records.jsonl", "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    print({f"{b[0]}-{b[1]}": c for b, c in counts.items()}, "tries", tries)


if __name__ == "__main__":
    main()
