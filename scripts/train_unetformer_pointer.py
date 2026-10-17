"""
train_unetformer_pointer.py - Train the UNetFormer and the next-point pointer head together.

The UNetFormer learns its usual maps (mask, crossings, over/under, strand axis, centre-line ridges,
count) and, through the pointer head on its feature maps, the order in which each strand visits
its points, including the jump across every crossing (protocol 13). The loss is the UNetFormer's
own loss plus --pointer-weight times the pointer loss, so the strand order also shapes the
features the other maps are read from. --pointer-weight 0 trains the same pipeline without the
head (a matched control).

Recipe as in the matched comparison of protocol 12 (train_v3, 12 epochs, batch 16, lr 1.5e-3,
AdamW, one-cycle, the epoch with the best validation loss is kept). Augmentation: one of the 8
flips / quarter turns per image, applied to the image, the dense targets (axial maps transformed
consistently) and the strand paths.

Points for the head: from the true strand paths (every 4 px, 0.75 px jitter, plus off-cable
distractors), and, from epoch --ridge-from on, half of the images use points on the model's own
predicted centre-line ridge, cut around its own predicted crossings and labelled by the nearest
true strand (refreshed every --ridge-every epochs). That is what the head sees at test time.

The jump re-checks (strand_pointer.recheck: curved ridge check with re-linking, then the straight
check) have no weights; they are applied when the result is scored (scripts/eval_recheck.py).

Speed: images, targets and points are prepared by --workers CPU processes while the GPU trains;
the loss is read back only every 50 steps.

  python scripts/train_unetformer_pointer.py --device mps --seed 0 --out checkpoints/uf_pointer/mps_ufptr_s0.pth

Writes <out> (the UNetFormer, loadable by knot_aware.load and the parser), <out stem>_head.pth
(the pointer head, whose extra["trunk"] names <out>), and a history JSON.
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import pickle
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src"), str(ROOT)]

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from knot_recognition.knot_aware import save, total_loss
from knot_recognition.strand_pointer import (PointerConfig, Points, StrandPointerHead, cut_radius, decode, edge_report,
                                             heat_peaks, label_points, load_records, mask_width, neighbour_index,
                                             pointer_loss, ridge_points, save_head, target_from_index, targets,
                                             transform_record, transform_xy, true_points, with_distractors)
from knot_recognition.unetformer import UNetFormer, UNetFormerConfig

DATA = ROOT / "data" / "knot_aware"
DENSE = ("mask", "heat", "over", "over_w", "orient", "orient_w", "count", "center", "center_hi")
AXIAL = ("over", "orient")


# --------------------------------------------------------------------------- views
def view_dense(t, hflip: bool, rot: int):
    """Dense targets of one image under torch.rot90(x.flip(-1) if hflip else x, rot): maps are
    flipped and turned; the axial maps (cos 2a, sin 2a) also change sign as in train_knot_aware
    (a flip: sin 2a -> -sin 2a; a quarter turn: both components change sign)."""
    out = {}
    for k, v in t.items():
        if k == "count":
            out[k] = v
            continue
        a = v[..., ::-1] if hflip else v
        a = np.rot90(a, rot, axes=(-2, -1))
        if k in AXIAL:
            a = a.copy()
            if hflip:
                a[1] = -a[1]
            if rot % 2:
                a = -a
        out[k] = np.ascontiguousarray(a, dtype=np.float32)
    return out


def view_image(x, hflip: bool, rot: int):
    """(C, H, W) image under the same transform."""
    a = x[..., ::-1] if hflip else x
    return np.ascontiguousarray(np.rot90(a, rot, axes=(-2, -1)))


def peak_ones(h):
    """The crossing's own cell is the positive (as train_knot_aware.to_tensors)."""
    from scipy.ndimage import maximum_filter

    return np.where((h == maximum_filter(h, 3, mode="nearest")) & (h > 0.5), 1.0, h).astype(np.float32)


def offcable_points(mask128, rng, n_max):
    """A few random points well away from the cable (distractors before ridge points exist)."""
    n = int(rng.integers(0, n_max + 1))
    free = np.argwhere(mask128 < 0.02)
    if n == 0 or not len(free):
        return np.zeros((0, 2), np.float32)
    pick = free[rng.integers(len(free), size=n)]
    return ((pick[:, ::-1] + rng.random((n, 2))) * 2.0).astype(np.float32)      # (x, y) on the 256 grid


def _empty(spacing):
    return Points(np.zeros((0, 2), np.float32), np.zeros(0, int), np.zeros(0), [], spacing)


# --------------------------------------------------------------------------- data (CPU workers)
class JointData(Dataset):
    """One training image: augmented image, dense targets and strand points. Arrays are memory-mapped
    and opened lazily in each worker process."""

    def __init__(self, split, seed, use_points=True, ridge_cache=None, ridge_frac=0.5, jitter=0.75,
                 max_points=640, spacing=4.0, n_distract=12):
        self.split, self.seed, self.use_points = split, seed, use_points
        self.ridge_cache, self.ridge_frac, self.jitter = ridge_cache, ridge_frac, jitter
        self.max_points, self.spacing, self.n_distract = max_points, spacing, n_distract
        self.epoch = 0
        self._A = None
        self.n = len(np.load(DATA / split / "X.npy", mmap_mode="r"))

    def _open(self):
        if self._A is None:
            self._A = {k: np.load(DATA / self.split / f"{k}.npy", mmap_mode="r") for k in ("X",) + DENSE}
            self._recs = load_records(DATA / f"{self.split}_strands.pkl") if self.use_points else None
            self._cache = pickle.loads(Path(self.ridge_cache).read_bytes()) if (self.use_points and self.ridge_cache) else None
        return self._A

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        A = self._open()
        rng = np.random.default_rng([self.seed, self.epoch, int(i)])
        hf, rot = bool(rng.random() < 0.5), int(rng.integers(4))
        x = view_image(np.asarray(A["X"][i]).transpose(2, 0, 1), hf, rot)
        t = {k: np.asarray(A[k][i], np.float32) for k in DENSE}
        t["heat"] = peak_ones(t["heat"])
        t = view_dense(t, hf, rot)
        pts, rad = _empty(self.spacing), 1.0
        if self.use_points:
            rec = transform_record(self._recs[i], hf, rot)
            c = self._cache[i] if self._cache is not None else None
            if c is not None and rng.random() < self.ridge_frac:
                pts = Points(transform_xy(c["xy"], hf, rot).astype(np.float32), c["comp"], c["s"], c["edges"],
                             self.spacing, "ridge")
                rad = cut_radius(c["width"])
            else:
                pts = true_points(rec, self.spacing, jitter=self.jitter, rng=rng)
                off = (transform_xy(c["xy"][c["comp"] < 0], hf, rot) if c is not None
                       else offcable_points(t["mask"], rng, self.n_distract))
                pts = with_distractors(pts, off)
                rad = cut_radius(rec["width"])
            if not 2 <= pts.n <= self.max_points:
                pts = _empty(self.spacing)
        return torch.from_numpy(x), {k: torch.from_numpy(np.asarray(v, np.float32)) for k, v in t.items()}, pts, float(rad)


def collate(batch):
    x = torch.stack([b[0] for b in batch])
    t = {k: torch.stack([b[1][k] for b in batch]) for k in batch[0][1]}
    ptr = neighbour_index([b[2] for b in batch], pad_to=64) if any(b[2].n for b in batch) else None
    return x, t, ptr, torch.tensor([b[3] for b in batch], dtype=torch.float32)


# --------------------------------------------------------------------------- ridge points from the model
_RECS = None


def _init_ridge_worker(split):
    global _RECS
    _RECS = load_records(DATA / f"{split}_strands.pkl")


def _ridge_job(job):
    prob, heat, mask, i, spacing = job
    w = mask_width(mask.astype(np.float32))
    xy = ridge_points(prob.astype(np.float32), heat_peaks(heat.astype(np.float32)), w, spacing)
    p = label_points(xy, _RECS[i], spacing)
    return {"xy": p.xy, "comp": p.comp, "s": p.s, "edges": p.edges, "width": w}


@torch.no_grad()
def build_ridge_cache(model, split, dev, path, pool, spacing, n=None, bs=32, chunk=256):
    """Points on the model's own centre-line ridge for the first n images of the split (base view)."""
    model.eval()
    X = np.load(DATA / split / "X.npy", mmap_mode="r")
    n = n or len(X)
    out = []
    for c0 in range(0, n, chunk):
        jobs = []
        for b in range(c0, min(c0 + chunk, n), bs):
            x = torch.from_numpy(np.asarray(X[b:min(b + bs, c0 + chunk, n)])).permute(0, 3, 1, 2).float().div(255).to(dev)
            o = model(x)
            P = torch.sigmoid(o["center_hi"]).half().cpu().numpy()
            H = torch.sigmoid(o["heat"]).half().cpu().numpy()
            M = torch.sigmoid(o["mask"]).half().cpu().numpy()
            jobs += [(P[k], H[k], M[k], b + k, spacing) for k in range(len(P))]
        out += pool.map(_ridge_job, jobs, chunksize=4)
    model.train()
    tmp = Path(path).with_suffix(".tmp")
    tmp.write_bytes(pickle.dumps(out))
    tmp.replace(path)
    return out


# --------------------------------------------------------------------------- validation
@torch.no_grad()
def pointer_val(model, head, dev, n=60, spacing=4.0):
    """Validation images 0..n-1 (never the held-out 150-299): pointer loss on true points, and
    links / jumps recovered from true points and from points on the model's own ridge."""
    model.eval(); head.eval()
    X = np.load(DATA / "val_v3" / "X.npy", mmap_mode="r")
    recs = load_records(DATA / "val_v3_strands.pkl")
    agg = {m: np.zeros(4) for m in ("true", "ridge")}
    losses = []
    for i in range(n):
        x = torch.from_numpy(np.asarray(X[i:i + 1])).permute(0, 3, 1, 2).float().div(255).to(dev)
        o = model(x, return_feats=True)
        for mode in ("true", "ridge"):
            if mode == "true":
                pts, w = true_points(recs[i], spacing), recs[i]["width"]
            else:
                w = mask_width(torch.sigmoid(o["mask"][0]).cpu().numpy())
                xy = ridge_points(torch.sigmoid(o["center_hi"][0]).cpu().numpy(),
                                  heat_peaks(torch.sigmoid(o["heat"][0]).cpu().numpy()), w, spacing)
                pts = label_points(xy, recs[i], spacing)
            if pts.n < 2:
                continue
            xy, valid, tgt, wgt = (v.to(dev) for v in targets([pts]))
            lg = head(o["feats"], xy, valid, torch.tensor([cut_radius(w)], device=dev))
            if mode == "true":
                losses.append(float(pointer_loss(lg, tgt, valid, wgt)))
            r = edge_report(decode(torch.softmax(lg, -1)[0].float().cpu().numpy()), pts)
            agg[mode] += [r["hit"], r["true"], r["jump_hit"], r["jump_true"]]
    model.train(); head.train()
    res = {"pointer_loss": float(np.mean(losses)) if losses else float("nan")}
    for m, v in agg.items():
        res[f"{m}_links"] = v[0] / max(v[1], 1)
        res[f"{m}_jumps"] = v[2] / max(v[3], 1)
    return res


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="auto", help="mps (Apple GPU) | cuda | cpu | auto")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1.5e-3)
    ap.add_argument("--pointer-weight", type=float, default=1.0, help="0 = the same pipeline without the head (control)")
    ap.add_argument("--ridge-from", type=int, default=4, help="epoch (1-based) from which half of the images use ridge points")
    ap.add_argument("--ridge-every", type=int, default=3)
    ap.add_argument("--max-points", type=int, default=640)
    ap.add_argument("--workers", type=int, default=0, help="CPU processes preparing batches (0: 4, or fewer on small machines)")
    ap.add_argument("--threads", type=int, default=0, help="torch CPU threads of the main process (0: leave)")
    ap.add_argument("--init", default=None, help="start the UNetFormer from this checkpoint instead of from scratch")
    ap.add_argument("--max-steps", type=int, default=0, help="stop each epoch after this many steps (smoke test)")
    ap.add_argument("--n-train", type=int, default=0, help="first n training images only (smoke test)")
    ap.add_argument("--val-n", type=int, default=60)
    ap.add_argument("--out", required=True)
    ap.add_argument("--history", default=None, help="default: results/uf_pointer/<out stem>_history.json")
    a = ap.parse_args()

    if a.device == "auto":
        a.device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    dev = torch.device(a.device)
    if a.threads:
        torch.set_num_threads(a.threads)
    # each worker is a full Python process with torch loaded (~0.4 GB): 4 keep the GPU fed without
    # pushing a Mac's shared memory into swap, which slows every step many times over
    workers = a.workers or max(1, min(4, (os.cpu_count() or 2) - 2))
    torch.manual_seed(a.seed)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    head_out = out.with_name(out.stem + "_head.pth")
    cache_path = out.with_name(out.stem + "_ridge_cache.pkl")
    use_head = a.pointer_weight > 0

    import train_knot_aware as tka                 # same validation as every other trunk
    Xva, Tva = tka.load_split("val_v3")

    model = UNetFormer(UNetFormerConfig()).to(dev)
    if a.init:
        sd = torch.load(a.init, map_location="cpu", weights_only=False)["state_dict"]
        print("init from", a.init, model.load_state_dict(sd, strict=False), flush=True)
    head = StrandPointerHead(PointerConfig(feat_dims=tuple([model.cfg.embed_dim] + list(model.cfg.widths[::-1][1:])))).to(dev)
    params = list(model.parameters()) + (list(head.parameters()) if use_head else [])
    ds = JointData("train_v3", a.seed, use_points=use_head, max_points=a.max_points, spacing=head.cfg.spacing)
    n_train = min(ds.n, a.n_train) if a.n_train else ds.n
    per_epoch = math.ceil(n_train / a.bs) if not a.max_steps else min(a.max_steps, math.ceil(n_train / a.bs))
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.epochs * per_epoch, pct_start=0.1)
    print(f"device {dev}; UNetFormer {sum(p.numel() for p in model.parameters())/1e6:.2f}M"
          + (f" + pointer head {sum(p.numel() for p in head.parameters())/1e6:.2f}M (weight {a.pointer_weight})" if use_head else " (no head: control)")
          + f"; batch {a.bs}, {per_epoch} steps per epoch, {a.epochs} epochs, {workers} data workers", flush=True)

    ctx = mp.get_context("spawn")
    best, hist, t0 = float("inf"), [], time.time()
    subset = torch.utils.data.Subset(ds, range(n_train))
    mem = (lambda: torch.mps.driver_allocated_memory() / 1e9) if dev.type == "mps" else (lambda: float("nan"))
    for ep in range(1, a.epochs + 1):
        if use_head and ep >= a.ridge_from and (ep - a.ridge_from) % a.ridge_every == 0:
            tr = time.time()
            # helper processes only while refreshing: they would otherwise hold memory all run
            with ctx.Pool(workers, initializer=_init_ridge_worker, initargs=("train_v3",)) as pool:
                build_ridge_cache(model, "train_v3", dev, cache_path, pool, head.cfg.spacing, n=n_train)
            ds.ridge_cache = str(cache_path)
            print(f"  ridge points refreshed from the model ({time.time()-tr:.0f}s)", flush=True)
        ds.epoch = ep
        g = torch.Generator().manual_seed(a.seed * 1000 + ep)
        dl = DataLoader(subset, batch_size=a.bs, shuffle=True, generator=g, num_workers=workers, collate_fn=collate,
                        prefetch_factor=3, multiprocessing_context=ctx, drop_last=False)
        run = {"dense": torch.zeros((), device=dev), "pointer": torch.zeros((), device=dev)}
        k, t_last = 0, time.time()
        for step, (x, t, ptr, rad) in enumerate(dl):
            if a.max_steps and step >= a.max_steps:
                break
            x = x.to(dev, non_blocking=True).float().div_(255.0)
            t = {kk: v.to(dev, non_blocking=True) for kk, v in t.items()}
            o = model(x, return_feats=use_head)
            ld, _ = total_loss(o, t, as_float=False)
            loss = ld
            run["dense"] += ld.detach()
            if use_head and ptr is not None:
                xy, valid, nbr, wgt = (v.to(dev, non_blocking=True) for v in ptr)
                lp = pointer_loss(head(o["feats"], xy, valid, rad.to(dev)), target_from_index(nbr), valid, wgt)
                loss = loss + a.pointer_weight * lp
                run["pointer"] += lp.detach()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            k += 1
            if k % 50 == 0:
                sps = (time.time() - t_last) / 50
                t_last = time.time()
                print(f"  epoch {ep} step {k}/{per_epoch}: dense {float(run['dense'])/k:.4f}"
                      + (f", pointer {float(run['pointer'])/k:.4f}" if use_head else "")
                      + f" ({(time.time()-t0)/60:.1f} min, {sps:.2f} s/step"
                      + (f", GPU memory {mem():.1f} GB" if dev.type == "mps" else "") + ")", flush=True)
        ev = tka.evaluate(model, Xva, Tva, dev)
        rec = {"epoch": ep, "minutes": (time.time() - t0) / 60, "train_dense": float(run["dense"]) / max(k, 1),
               "val": ev}
        score = ev["loss"]
        if use_head:
            pv = pointer_val(model, head, dev, a.val_n, head.cfg.spacing)
            rec["train_pointer"] = float(run["pointer"]) / max(k, 1)
            rec["pointer_val"] = pv
            score = ev["loss"] + a.pointer_weight * pv["pointer_loss"]
        hist.append(rec)
        if dev.type == "mps":
            torch.mps.empty_cache()
        print(json.dumps({"epoch": ep, "min": round(rec["minutes"], 1), "val_loss": round(ev["loss"], 4),
                          "crossing_P/R": [round(ev["crossing_precision"], 3), round(ev["crossing_recall"], 3)],
                          "over_axis": round(ev["over_axis_within_45deg"], 4),
                          **({k_: round(v_, 4) for k_, v_ in rec.get("pointer_val", {}).items()})}), flush=True)
        if score < best:
            best = score
            extra = {"val": ev, "device": str(dev), "seed": a.seed, "arch": "unetformer", "data": "train_v3",
                     "epoch": ep, "pointer_weight": a.pointer_weight, "joint_pointer": use_head,
                     "recipe": {"epochs": a.epochs, "bs": a.bs, "lr": a.lr}, "init": a.init}
            save(model, out, extra)
            if use_head:
                try:
                    trunk_ref = str(out.resolve().relative_to(ROOT))
                except ValueError:
                    trunk_ref = str(out.resolve())
                save_head(head, head_out, {"trunk": trunk_ref, "joint": True, "epoch": ep, "pointer_val": rec["pointer_val"],
                                           "seed": a.seed, "pointer_weight": a.pointer_weight})
            print(f"  [saved] epoch {ep} -> {out.name}" + (f", {head_out.name}" if use_head else ""), flush=True)
    hist_path = Path(a.history) if a.history else ROOT / "results" / "uf_pointer" / f"{out.stem}_history.json"
    hist_path.parent.mkdir(parents=True, exist_ok=True)
    hist_path.write_text(json.dumps({"args": vars(a), "history": hist}, indent=1, default=float))
    print(f"done in {(time.time()-t0)/60:.1f} min; history {hist_path}", flush=True)


if __name__ == "__main__":
    main()
