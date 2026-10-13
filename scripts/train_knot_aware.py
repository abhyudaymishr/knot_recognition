"""
train_knot_aware.py - Train the knot-aware NanoTransformer on data/knot_aware/{train,val}.npz.

Augmentation uses the dihedral group (flips and 90-degree rotations); axial
targets are transformed consistently: with the angle measured in pixel
coordinates (x right, y down), a horizontal flip maps a -> pi - a, a vertical
flip maps a -> -a, and a 90-degree rotation maps a -> a + pi/2 (sign chosen to
match torch.rot90). Checked by tests/test_knot_aware.py.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

import numpy as np
import torch

from knot_recognition.knot_aware import KnotAwareNanoTransformer, save, total_loss

KEYS = ("mask", "heat", "over", "over_w", "orient", "orient_w", "count")


def load_split(name):
    z = np.load(ROOT / "data" / "knot_aware" / f"{name}.npz")
    return z["X"], {k: z[k] for k in KEYS}


def to_tensors(X, T, idx):
    x = torch.from_numpy(X[idx]).permute(0, 3, 1, 2).float() / 255.0
    t = {k: torch.from_numpy(np.asarray(T[k][idx], dtype=np.float32)) for k in KEYS}
    # the crossing's own cell (each local maximum of the target heatmap) is the positive: set it to 1
    h = t["heat"]
    peak = (h == torch.nn.functional.max_pool2d(h[:, None], 3, 1, 1)[:, 0]) & (h > 0.5)
    t["heat"] = torch.where(peak, torch.ones_like(h), h)
    return x, t


def axial_hflip(v):   # a -> pi - a : (cos2a, sin2a) -> (cos2a, -sin2a)
    return torch.stack([v[:, 0], -v[:, 1]], 1)


def axial_vflip(v):   # a -> -a : same doubled-angle effect
    return torch.stack([v[:, 0], -v[:, 1]], 1)


def axial_rot90(v):   # a -> a - pi/2 for torch.rot90(k=1, dims=(-2,-1)) in (x right, y down): 2a -> 2a - pi
    return -v


def augment(x, t, rng):
    if rng.random() < 0.5:
        x = x.flip(-1)
        for k in ("mask", "heat", "over_w", "orient_w"):
            t[k] = t[k].flip(-1)
        for k in ("over", "orient"):
            t[k] = axial_hflip(t[k].flip(-1))
    if rng.random() < 0.5:
        x = x.flip(-2)
        for k in ("mask", "heat", "over_w", "orient_w"):
            t[k] = t[k].flip(-2)
        for k in ("over", "orient"):
            t[k] = axial_vflip(t[k].flip(-2))
    if rng.random() < 0.5:
        x = torch.rot90(x, 1, (-2, -1))
        for k in ("mask", "heat", "over_w", "orient_w"):
            t[k] = torch.rot90(t[k], 1, (-2, -1))
        for k in ("over", "orient"):
            t[k] = axial_rot90(torch.rot90(t[k], 1, (-2, -1)))
    return x, t


@torch.no_grad()
def evaluate(model, X, T, bs=32):
    model.eval()
    tot, parts, n = 0.0, {}, 0
    tp = fp = fn = 0
    ou_ok = ou_n = 0
    for b in range(0, len(X), bs):
        idx = np.arange(b, min(b + bs, len(X)))
        x, t = to_tensors(X, T, idx)
        out = model(x)
        loss, p = total_loss(out, t)
        tot += float(loss) * len(idx)
        for k, v in p.items():
            parts[k] = parts.get(k, 0) + v * len(idx)
        n += len(idx)
        heat = torch.sigmoid(out["heat"])
        peaks = (heat == torch.nn.functional.max_pool2d(heat[:, None], 3, 1, 1)[:, 0]) & (heat > 0.3)
        gt = t["heat"] > 0.99
        gt_d = torch.nn.functional.max_pool2d(gt[:, None].float(), 7, 1, 3)[:, 0] > 0
        pr_d = torch.nn.functional.max_pool2d(peaks[:, None].float(), 7, 1, 3)[:, 0] > 0
        tp += int((peaks & gt_d).sum()); fp += int((peaks & ~gt_d).sum()); fn += int((gt & ~pr_d).sum())
        w = t["over_w"] > 0
        po = out["over"] / out["over"].norm(dim=1, keepdim=True).clamp(min=1e-6)
        cos = (po * t["over"]).sum(1)  # cos(2 * angle error)
        ou_ok += int((cos[w] > 0).sum()); ou_n += int(w.sum())  # error < 45 degrees
    model.train()
    return {"loss": tot / n, **{k: v / n for k, v in parts.items()},
            "crossing_precision": tp / max(tp + fp, 1), "crossing_recall": (tp) / max(tp + fn, 1),
            "over_axis_within_45deg": ou_ok / max(ou_n, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=14)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1.5e-3)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--out", default=str(ROOT / "checkpoints" / "knot_aware_nanotransformer.pth"))
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    Xtr, Ttr = load_split("train")
    Xva, Tva = load_split("val")
    model = KnotAwareNanoTransformer()
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    steps = a.epochs * int(np.ceil(len(Xtr) / a.bs))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps, pct_start=0.1)
    best, history = float("inf"), []
    t0 = time.time()
    for ep in range(a.epochs):
        perm = rng.permutation(len(Xtr))
        for b in range(0, len(perm), a.bs):
            x, t = to_tensors(Xtr, Ttr, np.sort(perm[b:b + a.bs]))
            x, t = augment(x, t, rng)
            out = model(x)
            loss, _ = total_loss(out, t)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
        ev = evaluate(model, Xva, Tva)
        ev["epoch"], ev["minutes"] = ep + 1, (time.time() - t0) / 60
        history.append(ev)
        print(json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in ev.items()}), flush=True)
        if ev["loss"] < best:
            best = ev["loss"]
            Path(a.out).parent.mkdir(parents=True, exist_ok=True)
            save(model, a.out, {"val": ev, "train_images": len(Xtr)})
    json.dump(history, open(ROOT / "results" / "knot_aware_training.json", "w"), indent=1)


if __name__ == "__main__":
    main()
