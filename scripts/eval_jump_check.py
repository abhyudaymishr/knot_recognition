"""
eval_jump_check.py - Re-verify the pointer head's jumps with a separate, non-jump linear test.

After the head links an arm end across a crossing (a jump), strand_pointer.check_jumps walks the
straight segment through the crossing: both arms must arrive pointing along it, and the trunk's
centre-line ridge must stay high along it. A jump that fails is dropped, so its ends stay open
(the parser abstains there instead of reading a wrong strand).

  --sweep      dev images (val 0-149): every combination of the thresholds below
  default      held-out images (val 150-299): the two settings fixed on dev (strand_pointer.JUMP_CHECKS)

Reports jumps found (recall), long links that are right (precision of links longer than two
spacings), wrong long links, images with no wrong long link, and links overall, for true strand
points and for points on the predicted centre-line.

  python scripts/eval_jump_check.py --sweep --start 0 --n 150
  python scripts/eval_jump_check.py --start 150 --n 150
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src"), str(ROOT)]

import numpy as np
import torch

from knot_recognition.knot_aware import load
from knot_recognition.strand_pointer import JUMP_CHECKS, check_jumps, cut_radius, decode, load_head, load_records, targets
from train_strand_pointer import images, make_points


def stats(edges, pts, spacing):
    T = {frozenset(e[:2]) for e in pts.edges}
    J = {frozenset(e[:2]) for e in pts.edges if e[2]}
    E = {frozenset(e) for e in edges}
    long_ = [frozenset(e) for e in edges if np.linalg.norm(pts.xy[e[0]] - pts.xy[e[1]]) > 2 * spacing]
    right = sum(1 for e in long_ if e in T)
    return np.array([len(J & E), len(J), right, len(long_), len(T & E), len(T), int(right == len(long_))], float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--head", default=str(ROOT / "checkpoints/strand_pointer/smoke3_s0.pth"))
    ap.add_argument("--split", default="val_v3")
    ap.add_argument("--start", type=int, default=150)
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    head = load_head(a.head)
    trunk = load(ROOT / head.extra["trunk"]).eval()
    recs = load_records(ROOT / "data/knot_aware" / f"{a.split}_strands.pkl")
    X = images(a.split)
    sp = head.cfg.spacing
    settings = {"head alone": None}
    if a.sweep:
        for ang, rm, rmin in itertools.product((40, 55, 70, 90), (0.3, 0.4, 0.5), (0.0, 0.1)):
            settings[f"{ang} deg, ridge mean {rm}, min {rmin}"] = (ang, rm, rmin)
    else:
        settings.update({f"{k} {v}": v for k, v in JUMP_CHECKS.items()})
    agg = {m: {k: np.zeros(7) for k in settings} for m in ("true", "ridge")}
    for i in range(a.start, min(a.start + a.n, len(recs))):
        with torch.no_grad():
            out = trunk(torch.from_numpy(np.array(X[i : i + 1])).permute(0, 3, 1, 2).float() / 255.0, return_feats=True)
        ridge = torch.sigmoid(out["center_hi"][0]).numpy()
        for mode in ("true", "ridge"):
            got = make_points(out, 0, recs[i], mode, None, 0.0, 10 ** 6, sp)
            if got is None:
                continue
            pts, C, w = got
            xy, valid, _, _ = targets([pts])
            with torch.no_grad():
                p = torch.softmax(head(out["feats"], xy, valid, torch.tensor([cut_radius(w)])), -1)[0].numpy()
            E = decode(p)
            for k, th in settings.items():
                keep = E if th is None else check_jumps(E, pts.xy, ridge, sp, *th)[0]
                agg[mode][k] += stats(keep, pts, sp)
    res = {"head": a.head, "images": [a.start, a.start + a.n], "results": {}}
    for m, d in agg.items():
        print(f"--- {m} points")
        for k, v in d.items():
            r = {"jumps_found": v[0] / v[1], "long_links_right": v[2] / max(v[3], 1), "wrong_long_links": int(v[3] - v[2]),
                 "images_without_wrong_long_link": int(v[6]), "link_recall": v[4] / v[5], "jumps": int(v[1])}
            res["results"].setdefault(m, {})[k] = r
            print(f"  {k:34s} jumps found {r['jumps_found']:.3f}  long links right {r['long_links_right']:.3f} "
                  f"({r['wrong_long_links']} wrong)  images with no wrong long link {r['images_without_wrong_long_link']}  "
                  f"links {r['link_recall']:.3f}")
    out = Path(a.out or ROOT / "results/strand_pointer" / ("jump_check_dev_sweep.json" if a.sweep else "jump_check_heldout.json"))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
