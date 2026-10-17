"""
make_knot_aware_data.py - Training data for the knot-aware NanoTransformer.

Random braid words (2-4 strands, 2-9 generators) rendered with exact ground truth.
Training backgrounds are restricted to plain-light and wood (sessions S1/S2 styles,
with randomised lighting, blur, noise and JPEG), so the KnotBench-Photo sessions
S3 (dim gradient) and S4 (clutter texture) stay unseen. Cables never reuse the
benchmark's seeds.

v2 (--styled-frac > 0): a fraction of images uses cable.styles.render_styled
(ribbon, textbook gaps, rope, shaded tube, multicoloured links, new backgrounds,
random scale) and the braid wobble is widened so strands pass closer together.
Targets add the centre-line ridge. Seeds stay disjoint from the benchmark, the
parser dev set and the close-strands set.

Output: data/knot_aware/{train,val}{suffix}.npz  (images 256x256x4: RGB + FFT-cleaned gray; dense targets on a 128 x 128 grid)
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

import numpy as np

from knot_recognition.cable import styles, synth
from knot_recognition.cable.diagram import from_curves_3d
from knot_recognition.cable.targets import centerline_target, dense_targets
from knot_recognition.knot_aware import preprocess

OUT = ROOT / "data" / "knot_aware"


def min_separation(D, w):
    sys.path.insert(0, str(ROOT / "scripts"))
    from make_close_strands_set import min_separation as ms
    return ms(D, w)


def random_word(rng):
    n = int(rng.integers(2, 5))
    L = int(rng.integers(2, 10))
    return [int(rng.integers(1, n)) * int(rng.choice([-1, 1])) for _ in range(L)]


def random_session(rng):
    return {"bg": str(rng.choice(["plain_light", "wood"])), "light": float(rng.uniform(0, 0.35)),
            "blur": float(rng.uniform(0, 0.9)), "noise": float(rng.uniform(1, 6)), "jpeg": int(rng.integers(75, 96))}


def strand_record(ann, Dpx, size: int = 256):
    """Ordered strand paths of one image in model-input coordinates (size x size), for the
    next-point pointer targets (knot_recognition.strand_pointer). The renders use OpenCV's
    convention (pixel centres at integers); the record uses torch's grid_sample convention
    (pixel j spans [j, j + 1)), hence (X + 0.5) * f."""
    f = size / float(ann["image_size"][0])
    closed = [bool(c.closed) for c in Dpx.components] if Dpx is not None else [True] * len(ann["components"])
    return {"components": [((np.asarray(c, np.float64) + 0.5) * f).astype(np.float32) for c in ann["components"]],
            "closed": closed,
            "crossings": [{"pos": [(c["pos_px"][0] + 0.5) * f, (c["pos_px"][1] + 0.5) * f], "over": int(c["over_component"]),
                           "under": int(c["under_component"]), "over_tangent": list(c["over_tangent_px"])}
                          for c in ann["crossings"]],
            "width": float(ann["cable_width_px"]) * f, "coords": "edge"}


def make(n: int, seed: int, styled_frac: float = 0.0, keep_strands: bool = False, on_strand=None):
    """``keep_strands``: also return each image's ordered strand paths (strand_record). The random
    stream is unchanged, so the images are identical to a run without it."""
    rng = np.random.default_rng(seed)
    strands = []
    keys = ("mask", "heat", "over", "over_w", "orient", "orient_w", "count") + (("center", "center_hi") if styled_frac > 0 else ())
    X, T = [], {k: [] for k in keys}
    made = 0
    t0 = time.time()
    while made < n:
        word = random_word(rng)
        s = int(rng.integers(1 << 30)) + (1 << 31)  # disjoint from benchmark seed range
        wob = float(rng.uniform(0.15, 0.4)) if styled_frac > 0 else 0.15
        if styled_frac > 0 and rng.random() < 0.3:
            curves, _ = styles.parametric_instance(np.random.default_rng(s))
        else:
            curves = synth.make_instance(word, s, wobble=wob)
        try:
            from_curves_3d(curves).to_pd()
        except Exception:
            continue
        if rng.random() < styled_frac:
            img, ann, Dpx, _ = styles.render_styled(curves, np.random.default_rng(s))
            # strands drawn closer than half a cable width are unreadable even to a person
            if min_separation(Dpx, ann["cable_width_px"]) < 0.5:
                continue
        else:
            sess = random_session(rng)
            synth.SESSIONS["_train"] = sess
            colour = list(synth.CABLE_COLOURS)[int(rng.integers(len(synth.CABLE_COLOURS)))]
            width = int(rng.integers(7, 13))
            img, ann, Dpx = synth.render(curves, "_train", colour, width, np.random.default_rng(s))
        if len(ann["crossings"]) > 24:
            continue
        x = preprocess(np.asarray(img))
        tg = dense_targets(ann, stride=4)  # stride 4 at 512 == stride 2 at the 256 model input (128 x 128)
        if "center_hi" in T:
            tg["center_hi"] = centerline_target(ann, stride=2, sigma_cells=1.0)  # 256 x 256
        X.append(x)
        if keep_strands:
            strands.append(strand_record(ann, Dpx, x.shape[0]))
            if on_strand is not None:
                on_strand(strands)
        for k in T:
            T[k].append(tg[k].astype(np.float16) if k != "count" else tg[k])
        made += 1
        if made % 200 == 0:
            print(f"  {made}/{n} ({time.time()-t0:.0f}s)")
    synth.SESSIONS.pop("_train", None)
    if keep_strands:
        return np.stack(X), {k: np.stack(v) for k, v in T.items()}, strands
    return np.stack(X), {k: np.stack(v) for k, v in T.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=int, default=2400)
    ap.add_argument("--val", type=int, default=240)
    ap.add_argument("--styled-frac", type=float, default=0.0)
    ap.add_argument("--suffix", default="")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    seeds = ((1001, 2002) if not a.suffix else (7007, 8008))
    for name, n, seed in (("train", a.train, seeds[0]), ("val", a.val, seeds[1])):
        X, T = make(n, seed, a.styled_frac)
        np.savez(OUT / f"{name}{a.suffix}.npz", X=X, **T)
        print(f"[saved] {name}: {X.shape}")


if __name__ == "__main__":
    main()
