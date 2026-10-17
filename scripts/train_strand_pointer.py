"""
train_strand_pointer.py - Train the next-point pointer head (knot_recognition.strand_pointer).

The head learns, for every point on a strand, which points are next to it along the strand,
including the jump across a crossing (the under- or over-pass leaves the disc on the far side,
not on the other strand). The UNetFormer trunk is frozen and only the head trains. (Training
both needs the dense losses alongside, or the trunk's other heads drift; not done here.)

Each training image gets its points one of two ways:
  ridge  (probability --ridge-frac): points on the trunk's own predicted centre-line ridge, cut
         around its own predicted crossings, labelled by the nearest true strand. This is what
         the head sees at test time, including clutter points (target: none).
  true   points every 4 px along the true strands (jittered), plus the off-cable points of the
         predicted ridge as distractors.
Augmentation: the 8 flips/quarter turns, applied to the image and to the strand paths.

Evaluation (validation split, no augmentation) for both point sources: links between
consecutive points recovered (recall, precision), crossing jumps recovered, links drawn to
clutter points, and images whose every link is right; the same for the parser's rules on the
same points (strand_pointer.straightest) as the baseline.

  python scripts/train_strand_pointer.py --trunk checkpoints/beat_unet/mps_uf_s0.pth --steps 1500
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

import cv2
import numpy as np
import torch

from knot_recognition.knot_aware import load
from knot_recognition.strand_pointer import (PointerConfig, StrandPointerHead, cut_radius, decode, edge_report, heat_peaks,
                                             label_points, load_records, mask_width, pointer_loss, ridge_points,
                                             save_head, straightest, targets, transform_record, true_points,
                                             with_distractors)

DATA = ROOT / "data" / "knot_aware"


def sha(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:12]


def images(split: str) -> np.ndarray:
    """Images of a split: the memory-mapped .npy copy when present (scripts/npz_to_npy.py), else the .npz."""
    npy = DATA / split / "X.npy"
    return np.load(npy, mmap_mode="r") if npy.exists() else np.load(DATA / f"{split}.npz")["X"]


def views(x: torch.Tensor, recs, rng):
    xs, rs = [], []
    for b in range(x.shape[0]):
        hf, rot = bool(rng.random() < 0.5), int(rng.integers(4))
        xi = x[b].flip(-1) if hf else x[b]
        xs.append(torch.rot90(xi, rot, (-2, -1)))
        rs.append(transform_record(recs[b], hf, rot))
    return torch.stack(xs), rs


def make_points(out, b, rec, mode, rng, jitter, max_points, spacing):
    """-> (points, crossings used for the cut, cable width), or None. Ridge points use the trunk's
    own crossings and width (as at test time); true points use the true ones."""
    prob = torch.sigmoid(out["center_hi"][b]).float().cpu().numpy()
    if mode == "ridge":
        heat = torch.sigmoid(out["heat"][b]).float().cpu().numpy()
        w = mask_width(torch.sigmoid(out["mask"][b]).float().cpu().numpy())
        C = heat_peaks(heat)
        pts = label_points(ridge_points(prob, C, w, spacing), rec, spacing)
    else:
        w = rec["width"]
        C = np.array([c["pos"] for c in rec["crossings"]], np.float32).reshape(-1, 2)
        pts = true_points(rec, spacing, jitter=jitter, rng=rng)
        if rng is not None:                       # distractors: the predicted ridge's off-cable points
            lab = label_points(ridge_points(prob, C, w, spacing), rec, spacing)
            pts = with_distractors(pts, lab.xy[lab.comp < 0])
    return (pts, C, w) if 2 <= pts.n <= max_points else None


@torch.no_grad()
def evaluate(trunk, head, X, recs, dev, n, max_points, spacing, thr=0.15):
    head.eval()
    agg = {m: {k: {"true": 0, "pred": 0, "hit": 0, "jump_true": 0, "jump_hit": 0, "to_clutter": 0, "exact": 0, "images": 0}
               for k in ("pointer", "straightest")} for m in ("true", "ridge")}
    for i in range(n):
        x = torch.from_numpy(np.asarray(X[i : i + 1])).permute(0, 3, 1, 2).float().to(dev) / 255.0
        out = trunk(x, return_feats=True)
        for mode in ("true", "ridge"):
            got = make_points(out, 0, recs[i], mode, None, 0.0, 10 ** 6, spacing)
            if got is None:
                continue
            pts, C, w = got
            xy, valid, _, _ = targets([pts])
            if pts.n <= max_points:
                rad = torch.tensor([cut_radius(w)], device=dev)
                p = torch.softmax(head(out["feats"], xy.to(dev), valid.to(dev), rad), -1)[0].float().cpu().numpy()
                pred = decode(p, thr)
            else:
                pred = []
            band = cv2.dilate((torch.sigmoid(out["center_hi"][0]).cpu().numpy() > 0.5).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
            for name, edges in (("pointer", pred), ("straightest", straightest(pts.xy, w, spacing, crossings=C, band=band))):
                r = edge_report(edges, pts)
                a = agg[mode][name]
                for k in ("true", "pred", "hit", "jump_true", "jump_hit", "to_clutter"):
                    a[k] += r[k]
                a["exact"] += int(r["exact"])
                a["images"] += 1
    res = {}
    for mode, d in agg.items():
        res[mode] = {}
        for name, a in d.items():
            res[mode][name] = {"link_recall": a["hit"] / max(a["true"], 1), "link_precision": a["hit"] / max(a["pred"], 1),
                               "jump_recall": a["jump_hit"] / max(a["jump_true"], 1), "links_to_clutter": a["to_clutter"],
                               "exact_images": a["exact"], "images": a["images"], "jumps": a["jump_true"], "links": a["true"]}
    head.train()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trunk", default=str(ROOT / "checkpoints/beat_unet/mps_uf_s0.pth"))
    ap.add_argument("--train-split", default="train_v3")
    ap.add_argument("--val-split", default="val_v3")
    ap.add_argument("--n-train", type=int, default=0, help="first n training images (0: all with strand records)")
    ap.add_argument("--n-val", type=int, default=100)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--ridge-frac", type=float, default=0.5)
    ap.add_argument("--jitter", type=float, default=0.75)
    ap.add_argument("--max-points", type=int, default=640)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-every", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "checkpoints/strand_pointer/smoke_s0.pth"))
    ap.add_argument("--log", default=str(ROOT / "results/strand_pointer/smoke_s0.json"))
    a = ap.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    dev = torch.device(a.device)

    trunk = load(a.trunk).to(dev).eval()
    for p in trunk.parameters():
        p.requires_grad_(False)
    cfg = PointerConfig(feat_dims=tuple([trunk.cfg.embed_dim] + list(trunk.cfg.widths[::-1][1:])))
    head = StrandPointerHead(cfg).to(dev)
    opt = torch.optim.AdamW(head.parameters(), a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.steps, pct_start=0.1)

    recs = load_records(DATA / f"{a.train_split}_strands.pkl")
    n = min(len(recs), a.n_train) if a.n_train else len(recs)
    X = images(a.train_split)[:n]
    vrecs = load_records(DATA / f"{a.val_split}_strands.pkl")
    VX = images(a.val_split)[: a.n_val]
    print(f"trunk {a.trunk} ({sum(p.numel() for p in trunk.parameters())/1e6:.2f}M, frozen), "
          f"head {sum(p.numel() for p in head.parameters())/1e6:.2f}M; {n} training images, {len(VX)} validation", flush=True)

    hist, t0, run = [], time.time(), []
    for step in range(1, a.steps + 1):
        idx = rng.choice(n, a.bs, replace=False)
        x = torch.from_numpy(np.asarray(X[np.sort(idx)])).permute(0, 3, 1, 2).float() / 255.0
        idx = np.sort(idx)
        x, rs = views(x, [recs[i] for i in idx], rng)
        x = x.to(dev)
        with torch.no_grad():
            out = trunk(x, return_feats=True)
        batch, keep, rad = [], [], []
        for b in range(len(idx)):
            mode = "ridge" if rng.random() < a.ridge_frac else "true"
            got = make_points(out, b, rs[b], mode, rng, a.jitter, a.max_points, cfg.spacing)
            if got is not None:
                batch.append(got[0])
                rad.append(cut_radius(got[2]))
                keep.append(b)
        if not batch:
            continue
        xy, valid, tgt, wgt = (t.to(dev) for t in targets(batch))
        feats = {k: v[keep] for k, v in out["feats"].items()}
        loss = pointer_loss(head(feats, xy, valid, torch.tensor(rad, device=dev)), tgt, valid, wgt)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        opt.step()
        sched.step()
        run.append(loss.item())
        if step % 50 == 0 or step == 1:
            print(f"step {step}/{a.steps} loss {np.mean(run[-50:]):.4f} ({time.time()-t0:.0f}s)", flush=True)
            hist.append({"step": step, "loss": float(np.mean(run[-50:])), "sec": time.time() - t0})
        if a.eval_every and step % a.eval_every == 0 and step < a.steps:
            ev = evaluate(trunk, head, VX, vrecs, dev, min(30, len(VX)), a.max_points, cfg.spacing)
            print("  val(30):", json.dumps({m: {k: round(v["link_recall"], 4) for k, v in d.items()} for m, d in ev.items()}), flush=True)

    ev = evaluate(trunk, head, VX, vrecs, dev, len(VX), a.max_points, cfg.spacing)
    extra = {"trunk": str(Path(a.trunk).relative_to(ROOT)) if str(a.trunk).startswith(str(ROOT)) else a.trunk,
             "trunk_sha": sha(a.trunk), "steps": a.steps, "train_images": n, "args": vars(a), "val": ev}
    save_head(head, a.out, extra)
    Path(a.log).parent.mkdir(parents=True, exist_ok=True)
    Path(a.log).write_text(json.dumps({"history": hist, "val": ev, "extra": {k: v for k, v in extra.items() if k != "val"}},
                                      indent=1, default=str))
    print(json.dumps(ev, indent=1))
    print(f"[saved] {a.out}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
