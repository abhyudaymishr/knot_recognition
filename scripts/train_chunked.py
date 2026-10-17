"""
train_chunked.py - scripts/train_knot_aware.py in resumable time slices.

For machines where one process may only run for a few minutes at a time (for example the
Linux VM of the Claude desktop app, where every command is limited to ~3 minutes). The run
is identical to train_knot_aware.py with the same arguments: same data order, augmentation
stream, optimiser, one-cycle schedule, per-epoch validation and best-checkpoint rule. All state
(model, optimiser, scheduler, numpy random state, position in the epoch, history) is
saved to --state when the time budget runs out, and the next call continues from there.

Exit code 0 when training is finished, 75 when paused (call again with the same arguments).
Usage: python scripts/train_chunked.py <train_knot_aware.py arguments> --state S.pt [--budget 150]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "src"), str(ROOT), str(ROOT / "scripts")]
import numpy as np
import torch

import train_knot_aware as tk
from knot_recognition.knot_aware import KnotAwareConfig, KnotAwareNanoTransformer, save, total_loss


def main():
    t_start = time.time()
    ap = argparse.ArgumentParser()
    for flag, kw in (("--epochs", dict(type=int, default=14)), ("--bs", dict(type=int, default=16)),
                     ("--lr", dict(type=float, default=1.5e-3)), ("--threads", dict(type=int, default=4)),
                     ("--out", dict(required=True)), ("--data-suffix", dict(default="")), ("--init", dict(default=None)),
                     ("--seed", dict(type=int, default=0)), ("--history", dict(required=True)),
                     ("--attn-bias", dict(default="none")), ("--attn-content", dict(default="dot")),
                     ("--layer-cake", dict(type=float, default=0.0)), ("--state", dict(required=True)),
                     ("--budget", dict(type=float, default=150.0))):
        ap.add_argument(flag, **kw)
    ap.add_argument("--freeze-trunk", action="store_true")
    ap.add_argument("--no-fft", action="store_true")
    ap.add_argument("--no-hi", action="store_true")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    Xtr, Ttr = tk.load_split("train" + a.data_suffix)
    Xva, Tva = tk.load_split("val" + a.data_suffix)
    lim = int(os.environ.get("TK_LIMIT", 0))          # tests only: a small subset
    if lim:
        Xtr, Ttr = Xtr[:lim], {k: v[:lim] for k, v in Ttr.items()}
        Xva, Tva = Xva[:lim], {k: v[:lim] for k, v in Tva.items()}
    max_steps = int(os.environ.get("TK_MAX_STEPS_PER_CALL", 0))   # tests only: force pauses
    if a.no_hi and "center_hi" in tk.KEYS:
        tk.KEYS = tuple(k for k in tk.KEYS if k != "center_hi")
        Ttr.pop("center_hi", None)
        Tva.pop("center_hi", None)
    model = KnotAwareNanoTransformer(KnotAwareConfig(hi_res_center="center_hi" in tk.KEYS, in_channels=3 if a.no_fft else 4,
                                                     attn_pos_bias=a.attn_bias, attn_content=a.attn_content,
                                                     layer_cake=a.layer_cake))
    steps_per_epoch = int(np.ceil(len(Xtr) / a.bs))
    state_p = Path(a.state)
    if state_p.exists():
        st = torch.load(state_p, map_location="cpu", weights_only=False)
        model.load_state_dict(st["model"])
    else:
        torch.manual_seed(a.seed)
        model = KnotAwareNanoTransformer(model.cfg)          # same init as train_knot_aware.py
        if a.init:
            sd = torch.load(a.init, map_location="cpu", weights_only=False)["state_dict"]
            print("init from", a.init, model.load_state_dict(sd, strict=False))
        st = None
    if a.freeze_trunk:
        hi = ("stem1.", "up3.", "fuse3.", "head_center_hi.")
        for n_, p_ in model.named_parameters():
            p_.requires_grad = n_.startswith(hi)
    opt = torch.optim.AdamW([p_ for p_ in model.parameters() if p_.requires_grad], lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.epochs * steps_per_epoch, pct_start=0.1)
    rng = np.random.default_rng(a.seed)
    if st is None:
        st = {"epoch": 0, "batch": 0, "perm": None, "best": float("inf"), "history": [], "train_seconds": 0.0,
              "pending_eval": False}
    else:
        opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        rng.bit_generator.state = st["rng"]

    def pause():
        st.update(model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(), rng=rng.bit_generator.state)
        st["train_seconds"] += time.time() - t_start
        tmp = state_p.with_suffix(".tmp")
        torch.save(st, tmp)
        tmp.replace(state_p)
        print(f"PAUSED epoch {st['epoch'] + 1}/{a.epochs} batch {st['batch']}/{steps_per_epoch} "
              f"({st['train_seconds'] / 60:.1f} min so far)", flush=True)
        sys.exit(75)

    left = lambda: a.budget - (time.time() - t_start)
    step_s = None
    n_steps = 0
    while st["epoch"] < a.epochs:
        if not st["pending_eval"]:
            if st["perm"] is None:
                st["perm"] = rng.permutation(len(Xtr))
            perm = st["perm"]
            while st["batch"] < steps_per_epoch:
                if (step_s is not None and left() < 1.5 * step_s + 5) or (max_steps and n_steps >= max_steps):
                    pause()
                t0 = time.time()
                b = st["batch"] * a.bs
                x, t = tk.to_tensors(Xtr, Ttr, np.sort(perm[b:b + a.bs]))
                x, t = tk.augment(x, t, rng)
                out = model(x)
                loss, _ = total_loss(out, t, layer_cake=a.layer_cake)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
                st["batch"] += 1
                n_steps += 1
                step_s = time.time() - t0 if step_s is None else 0.8 * step_s + 0.2 * (time.time() - t0)
            st["pending_eval"] = True
        if left() < 60:
            pause()
        ev = tk.evaluate(model, Xva, Tva)
        ev["epoch"] = st["epoch"] + 1
        ev["minutes"] = (st["train_seconds"] + time.time() - t_start) / 60
        st["history"].append(ev)
        print(json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in ev.items()}), flush=True)
        if ev["loss"] < st["best"]:
            st["best"] = ev["loss"]
            Path(a.out).parent.mkdir(parents=True, exist_ok=True)
            save(model, a.out, {"val": ev, "device": "cpu (chunked)", "train_images": len(Xtr), "data": "train" + a.data_suffix,
                                "init": a.init, "seed": a.seed, "no_fft": a.no_fft, "attn_bias": a.attn_bias,
                                "attn_content": a.attn_content, "layer_cake": a.layer_cake})
        st.update(epoch=st["epoch"] + 1, batch=0, perm=None, pending_eval=False)
    json.dump(st["history"], open(a.history, "w"), indent=1)
    state_p.unlink(missing_ok=True)
    print("FINISHED", a.out, flush=True)


if __name__ == "__main__":
    main()
