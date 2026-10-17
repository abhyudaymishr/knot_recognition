"""
make_strand_sidecar.py - Ordered strand paths for an existing knot-aware data split.

The v3 split stores dense maps only. The next-point pointer (knot_recognition.strand_pointer)
needs each strand as an ordered path. The generator is deterministic, so this re-runs it with
the split's seed and keeps the paths; every regenerated image is checked against the saved one.

  python scripts/make_strand_sidecar.py --split val_v3     # 300 images, seed 8008
  python scripts/make_strand_sidecar.py --split train_v3   # 3000 images, seed 7007

Output: data/knot_aware/<split>_strands.pkl (list, one record per image, in split order),
rewritten every 100 images so a partial file can be used while the rest is generated.
The run holds the whole split in memory (about 5 GB for train_v3); do not run it beside a training job
on a small machine. scripts/check_strand_sidecar.py checks a file against the saved centre-line maps.
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src"), str(ROOT)]

import numpy as np

from make_knot_aware_data import OUT, make

SPLITS = {"train_v3": (3000, 7007, 0.6), "val_v3": (300, 8008, 0.6)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=sorted(SPLITS))
    ap.add_argument("--n", type=int, default=None, help="only the first n images")
    a = ap.parse_args()
    n, seed, styled = SPLITS[a.split]
    n = a.n or n
    X_ref = np.load(OUT / f"{a.split}.npz", mmap_mode="r")["X"]
    out = OUT / f"{a.split}_strands.pkl"

    def dump(strands):
        if len(strands) % 100 == 0:
            tmp = out.with_suffix(".tmp")
            tmp.write_bytes(pickle.dumps(strands))
            tmp.replace(out)

    X, _, strands = make(n, seed, styled, keep_strands=True, on_strand=dump)
    bad = [i for i in range(len(X)) if not np.array_equal(X[i], X_ref[i])]
    if bad:
        raise SystemExit(f"regenerated images differ from {a.split} at {bad[:10]} ({len(bad)} total)")
    tmp = out.with_suffix(".tmp")
    tmp.write_bytes(pickle.dumps(strands))
    tmp.replace(out)
    print(f"[saved] {out} ({len(strands)} images, all identical to the saved split)")


if __name__ == "__main__":
    main()
