"""
make_knot_aware_data.py - Training data for the knot-aware NanoTransformer.

Random braid words (2-4 strands, 2-9 generators) rendered with exact ground truth.
Training backgrounds are restricted to plain-light and wood (sessions S1/S2 styles,
with randomised lighting, blur, noise and JPEG), so the KnotBench-Photo sessions
S3 (dim gradient) and S4 (clutter texture) stay unseen. Cables never reuse the
benchmark's seeds.

Output: data/knot_aware/{train,val}.npz  (images 256x256x4: RGB + FFT-cleaned gray; dense targets on a 128 x 128 grid)
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

import numpy as np

from knot_recognition.cable import synth
from knot_recognition.cable.diagram import from_curves_3d
from knot_recognition.cable.targets import dense_targets
from knot_recognition.knot_aware import preprocess

OUT = ROOT / "data" / "knot_aware"


def random_word(rng):
    n = int(rng.integers(2, 5))
    L = int(rng.integers(2, 10))
    return [int(rng.integers(1, n)) * int(rng.choice([-1, 1])) for _ in range(L)]


def random_session(rng):
    return {"bg": str(rng.choice(["plain_light", "wood"])), "light": float(rng.uniform(0, 0.35)),
            "blur": float(rng.uniform(0, 0.9)), "noise": float(rng.uniform(1, 6)), "jpeg": int(rng.integers(75, 96))}


def make(n: int, seed: int):
    rng = np.random.default_rng(seed)
    X, T = [], {k: [] for k in ("mask", "heat", "over", "over_w", "orient", "orient_w", "count")}
    made = 0
    t0 = time.time()
    while made < n:
        word = random_word(rng)
        s = int(rng.integers(1 << 30)) + (1 << 31)  # disjoint from benchmark seed range
        curves = synth.make_instance(word, s, wobble=0.15)
        try:
            from_curves_3d(curves).to_pd()
        except Exception:
            continue
        sess = random_session(rng)
        synth.SESSIONS["_train"] = sess
        colour = list(synth.CABLE_COLOURS)[int(rng.integers(len(synth.CABLE_COLOURS)))]
        width = int(rng.integers(7, 13))
        img, ann, _ = synth.render(curves, "_train", colour, width, np.random.default_rng(s))
        if len(ann["crossings"]) > 24:
            continue
        x = preprocess(np.asarray(img))
        tg = dense_targets(ann, stride=4)  # stride 4 at 512 == stride 2 at the 256 model input (128 x 128)
        X.append(x)
        for k in T:
            T[k].append(tg[k].astype(np.float16) if k != "count" else tg[k])
        made += 1
        if made % 200 == 0:
            print(f"  {made}/{n} ({time.time()-t0:.0f}s)")
    del synth.SESSIONS["_train"]
    return np.stack(X), {k: np.stack(v) for k, v in T.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=int, default=2400)
    ap.add_argument("--val", type=int, default=240)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for name, n, seed in (("train", a.train, 1001), ("val", a.val, 2002)):
        X, T = make(n, seed)
        np.savez_compressed(OUT / f"{name}.npz", X=X, **T)
        print(f"[saved] {name}: {X.shape}")


if __name__ == "__main__":
    main()
