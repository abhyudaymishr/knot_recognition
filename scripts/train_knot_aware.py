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

from knot_recognition.knot_aware import KnotAwareConfig, KnotAwareNanoTransformer, save, total_loss

KEYS = ("mask", "heat", "over", "over_w", "orient", "orient_w", "count")
SCALAR_MAPS = ("mask", "heat", "over_w", "orient_w", "center", "center_hi")


def load_split(name):
    """Memory-maps data/knot_aware/<name>/<key>.npy when present (scripts/npz_to_npy.py), so
    concurrent training jobs share one page-cached copy; otherwise loads <name>.npz."""
    global KEYS
    d = ROOT / "data" / "knot_aware" / name
    if d.is_dir():
        files = [p.stem for p in d.glob("*.npy")]
        z = {k: np.load(d / f"{k}.npy", mmap_mode="r") for k in files}
    else:
        z = np.load(ROOT / "data" / "knot_aware" / f"{name}.npz")
        files = z.files
    for k in ("center", "center_hi"):
        if k in files and k not in KEYS:
            KEYS = KEYS + (k,)
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
        for k in [k for k in SCALAR_MAPS if k in t]:
            t[k] = t[k].flip(-1)
        for k in ("over", "orient"):
            t[k] = axial_hflip(t[k].flip(-1))
    if rng.random() < 0.5:
        x = x.flip(-2)
        for k in [k for k in SCALAR_MAPS if k in t]:
            t[k] = t[k].flip(-2)
        for k in ("over", "orient"):
            t[k] = axial_vflip(t[k].flip(-2))
    if rng.random() < 0.5:
        x = torch.rot90(x, 1, (-2, -1))
        for k in [k for k in SCALAR_MAPS if k in t]:
            t[k] = torch.rot90(t[k], 1, (-2, -1))
        for k in ("over", "orient"):
            t[k] = axial_rot90(torch.rot90(t[k], 1, (-2, -1)))
    return x, t


@torch.no_grad()
def evaluate(model, X, T, dev=torch.device("cpu"), bs=32):
    model.eval()
    tot, parts, n = 0.0, {}, 0
    tp = fp = fn = 0
    ou_ok = ou_n = 0
    for b in range(0, len(X), bs):
        idx = np.arange(b, min(b + bs, len(X)))
        x, t = to_tensors(X, T, idx)
        x, t = x.to(dev), {k: v.to(dev) for k, v in t.items()}
        out = model(x)
        loss, p = total_loss(out, t, layer_cake=getattr(getattr(model, "cfg", None), "layer_cake", 0.0),
                             topo=getattr(getattr(model, "cfg", None), "topo_loss", 0.0))
        tot += float(loss) * len(idx)
        for k, v in p.items():
            parts[k] = parts.get(k, 0) + v * len(idx)
        n += len(idx)
        if "heat" not in out:
            continue
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
    ap.add_argument("--data-suffix", default="")
    ap.add_argument("--init", default=None, help="fine-tune from this checkpoint (new heads start fresh)")
    ap.add_argument("--freeze-trunk", action="store_true",
                    help="train only the full-resolution centre-line branch (stem1, up3, fuse3, head_center_hi)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu", help="cpu | mps (Apple GPU) | cuda | auto")
    ap.add_argument("--arch", default="knot_aware", choices=["knot_aware", "unet_seg", "unet_kp", "unetformer"],
                    help="unet_seg / unet_kp: standard CNN baselines (src/knot_recognition/baselines.py); "
                         "unetformer: U-Net with a transformer bottleneck, all heads (src/knot_recognition/unetformer.py)")
    ap.add_argument("--embed-dim", type=int, default=None, help="transformer width (knot_aware: 128, unetformer: 192)")
    ap.add_argument("--depth", type=int, default=None, help="transformer blocks (default 4)")
    ap.add_argument("--heads", type=int, default=None, help="attention heads (knot_aware: 4, unetformer: 6)")
    ap.add_argument("--stem-dim", type=int, default=None, help="knot_aware stem width (default 48)")
    ap.add_argument("--topo-loss", type=float, default=0.0, help="weight of the clDice centre-line topology loss")
    ap.add_argument("--no-fft", action="store_true", help="ablation: RGB input only (drop the FFT-cleaned channel)")
    ap.add_argument("--no-hi", action="store_true", help="ignore the 256-grid ridge target (v2 stage of the recipe)")
    ap.add_argument("--attn-bias", default="none", choices=["none", "minkowski"],
                    help="Minkowski relative-position bias -alpha_h ||r_i - r_j||_p (learnable p, alpha)")
    ap.add_argument("--attn-content", default="dot", choices=["dot", "minkowski"],
                    help="attention logits from -||q - k||_p^p (p = 1 / 2 per head) instead of q.k")
    ap.add_argument("--layer-cake", type=float, default=0.0,
                    help="weight of the Lebesgue (layer-cake) loss on mask and centre-line ridges")
    ap.add_argument("--history", default=str(ROOT / "results" / "knot_aware_training.json"))
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    if a.device == "auto":
        a.device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    dev = torch.device(a.device)
    print("device", dev, flush=True)
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    global KEYS
    Xtr, Ttr = load_split("train" + a.data_suffix)
    Xva, Tva = load_split("val" + a.data_suffix)
    if a.no_hi and "center_hi" in KEYS:
        KEYS = tuple(k for k in KEYS if k != "center_hi")
        Ttr.pop("center_hi", None)
        Tva.pop("center_hi", None)
    size = {k: v for k, v in (("embed_dim", a.embed_dim), ("depth", a.depth), ("num_heads", a.heads)) if v is not None}
    if a.arch == "knot_aware":
        if a.stem_dim is not None:
            size["stem_dim"] = a.stem_dim
        model = KnotAwareNanoTransformer(KnotAwareConfig(hi_res_center="center_hi" in KEYS, in_channels=3 if a.no_fft else 4,
                                                         attn_pos_bias=a.attn_bias, attn_content=a.attn_content,
                                                         layer_cake=a.layer_cake, topo_loss=a.topo_loss, **size))
        print("knot_aware", sum(p_.numel() for p_ in model.parameters()) / 1e6, "M parameters", flush=True)
    elif a.arch == "unetformer":
        from knot_recognition.unetformer import UNetFormer, UNetFormerConfig
        model = UNetFormer(UNetFormerConfig(in_channels=3 if a.no_fft else 4, attn_pos_bias=a.attn_bias,
                                            layer_cake=a.layer_cake, topo_loss=a.topo_loss, **size))
        print("unetformer", sum(p_.numel() for p_ in model.parameters()) / 1e6, "M parameters", flush=True)
    else:
        from knot_recognition.baselines import UNet, UNetConfig, save_unet
        model = UNet(UNetConfig(arch=a.arch, in_channels=3 if a.no_fft else 4))
        print(a.arch, sum(p_.numel() for p_ in model.parameters()) / 1e6, "M parameters", flush=True)
    if a.init:
        sd = torch.load(a.init, map_location="cpu", weights_only=False)["state_dict"]
        print("init from", a.init, model.load_state_dict(sd, strict=False))
    model.to(dev)
    if a.freeze_trunk:
        hi = ("stem1.", "up3.", "fuse3.", "head_center_hi.")
        for n_, p_ in model.named_parameters():
            p_.requires_grad = n_.startswith(hi)
    opt = torch.optim.AdamW([p_ for p_ in model.parameters() if p_.requires_grad], lr=a.lr, weight_decay=1e-4)
    steps = a.epochs * int(np.ceil(len(Xtr) / a.bs))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps, pct_start=0.1)
    best, history = float("inf"), []
    t0 = time.time()
    for ep in range(a.epochs):
        perm = rng.permutation(len(Xtr))
        for b in range(0, len(perm), a.bs):
            x, t = to_tensors(Xtr, Ttr, np.sort(perm[b:b + a.bs]))
            x, t = augment(x, t, rng)
            x, t = x.to(dev), {k: v.to(dev) for k, v in t.items()}
            out = model(x)
            loss, _ = total_loss(out, t, layer_cake=a.layer_cake, topo=a.topo_loss)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
        ev = evaluate(model, Xva, Tva, dev)
        ev["epoch"], ev["minutes"] = ep + 1, (time.time() - t0) / 60
        history.append(ev)
        print(json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in ev.items()}), flush=True)
        if ev["loss"] < best:
            best = ev["loss"]
            Path(a.out).parent.mkdir(parents=True, exist_ok=True)
            (save if a.arch in ("knot_aware", "unetformer") else save_unet)(model, a.out, {"val": ev, "device": str(dev), "train_images": len(Xtr), "data": "train" + a.data_suffix, "init": a.init,
                                "seed": a.seed, "no_fft": a.no_fft, "attn_bias": a.attn_bias,
                                "attn_content": a.attn_content, "layer_cake": a.layer_cake,
                                "topo_loss": a.topo_loss, "arch": a.arch})
    json.dump(history, open(a.history, "w"), indent=1)


if __name__ == "__main__":
    main()
