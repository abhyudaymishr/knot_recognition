"""
eval_strand_pointer.py - Score a pointer head against the parser's rules on held-out images,
image by image, with paired bootstrap intervals.

The smoke-test heads were designed by looking at failures on validation images 0-149, so those
are not a clean test. Images 150-299 of val_v3 were not used for training, for any design
decision, or viewed: --start 150 scores those.

For each image and each point source (true strand points; points on the trunk's predicted
centre-line), counts the links between consecutive points and the crossing jumps recovered by the
head, by the rules (strand_pointer.straightest, given the same points, crossings, width and
ridge band), and by the head with every jump re-checked along the ridge and free ends re-linked
(strand_pointer.verify_relink). Reports totals and 95% bootstrap intervals (2000 resamples of
images) of head minus rules, and of re-checked head minus head.

  python scripts/eval_strand_pointer.py --head checkpoints/strand_pointer/smoke3_s0.pth --start 150 --n 150
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src"), str(ROOT)]

import cv2
import numpy as np
import torch

from knot_recognition.knot_aware import load
from knot_recognition.strand_pointer import (cut_radius, decode, edge_report, load_head, load_records, straightest, targets,
                                             verify_relink)
from train_strand_pointer import images, make_points


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--head", required=True)
    ap.add_argument("--split", default="val_v3")
    ap.add_argument("--start", type=int, default=150)
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    head = load_head(a.head)
    trunk = load(ROOT / head.extra["trunk"]).eval()
    recs = load_records(ROOT / "data/knot_aware" / f"{a.split}_strands.pkl")
    X = images(a.split)
    rows = []
    for i in range(a.start, min(a.start + a.n, len(recs))):
        with torch.no_grad():
            out = trunk(torch.from_numpy(np.array(X[i : i + 1])).permute(0, 3, 1, 2).float() / 255.0, return_feats=True)
        ridge = torch.sigmoid(out["center_hi"][0]).numpy()
        band = cv2.dilate((ridge > 0.5).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        for mode in ("true", "ridge"):
            got = make_points(out, 0, recs[i], mode, None, 0.0, 10 ** 6, head.cfg.spacing)
            if got is None:
                continue
            pts, C, w = got
            xy, valid, _, _ = targets([pts])
            with torch.no_grad():
                rad = torch.tensor([cut_radius(w)]) if head.cfg.radius else None
                p = torch.softmax(head(out["feats"], xy, valid, rad), -1)[0].numpy()
            E = decode(p)
            r_head = edge_report(E, pts)
            r_check = edge_report(verify_relink(E, p, pts.xy, ridge, cut_radius(w), head.cfg.spacing), pts)
            r_rule = edge_report(straightest(pts.xy, w, head.cfg.spacing, crossings=C, band=band), pts)
            rows.append({"image": i, "mode": mode, "head": r_head, "head_check": r_check, "rules": r_rule})
    rng = np.random.default_rng(0)
    res = {"head": a.head, "split": a.split, "images": [a.start, a.start + a.n], "by_points": {}}
    for mode in ("true", "ridge"):
        R = [r for r in rows if r["mode"] == mode]
        tot = lambda who, k: np.array([r[who][k] for r in R], float)
        d = {}
        for name, num, den in (("link_recall", "hit", "true"), ("jump_recall", "jump_hit", "jump_true")):
            hn, rn, dd = tot("head", num), tot("rules", num), tot("head", den)
            boots = []
            for _ in range(2000):
                s = rng.integers(len(R), size=len(R))
                boots.append((hn[s].sum() - rn[s].sum()) / max(dd[s].sum(), 1))
            d[name] = {"head": hn.sum() / dd.sum(), "rules": rn.sum() / dd.sum(),
                       "diff": (hn.sum() - rn.sum()) / dd.sum(), "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]}
        for name, k in (("link_precision", "pred"),):
            d[name] = {"head": tot("head", "hit").sum() / tot("head", k).sum(), "rules": tot("rules", "hit").sum() / tot("rules", k).sum()}
        ex_h = np.array([r["head"]["exact"] for r in R]); ex_r = np.array([r["rules"]["exact"] for r in R])
        d["exact_images"] = {"head": int(ex_h.sum()), "rules": int(ex_r.sum()), "images": len(R),
                             "only_head": int((ex_h & ~ex_r).sum()), "only_rules": int((ex_r & ~ex_h).sum())}
        d["jumps"] = int(tot("head", "jump_true").sum())
        d["links"] = int(tot("head", "true").sum())
        d["links_to_clutter"] = {"head": int(tot("head", "to_clutter").sum()), "rules": int(tot("rules", "to_clutter").sum())}
        # the head with every jump re-checked along the ridge and free ends re-linked, against the head
        ck = {}
        for name, num, den in (("link_recall", "hit", "true"), ("jump_recall", "jump_hit", "jump_true")):
            cn, hn, dd = tot("head_check", num), tot("head", num), tot("head", den)
            boots = []
            for _ in range(2000):
                s = rng.integers(len(R), size=len(R))
                boots.append((cn[s].sum() - hn[s].sum()) / max(dd[s].sum(), 1))
            ck[name] = {"head_check": cn.sum() / dd.sum(), "head": hn.sum() / dd.sum(), "diff": (cn.sum() - hn.sum()) / dd.sum(),
                        "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]}
        ck["link_precision"] = {"head_check": tot("head_check", "hit").sum() / tot("head_check", "pred").sum(),
                                "head": tot("head", "hit").sum() / tot("head", "pred").sum()}
        ex_c = np.array([r["head_check"]["exact"] for r in R])
        ck["exact_images"] = {"head_check": int(ex_c.sum()), "head": int(ex_h.sum()), "images": len(R),
                              "only_check": int((ex_c & ~ex_h).sum()), "only_head": int((ex_h & ~ex_c).sum())}
        d["recheck"] = ck
        res["by_points"][mode] = d
    out = Path(a.out or ROOT / "results/strand_pointer" / f"heldout_{Path(a.head).stem}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
