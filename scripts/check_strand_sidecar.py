"""
check_strand_sidecar.py - Check that each strand record belongs to its image.

Redraws the 256-grid centre-line map from every record (cable.targets.centerline_target, the
function that made the saved training target) and compares it with the saved map of the same
image. A matching record differs only by sub-pixel rounding at a few pixels; a record of another
image differs over the whole map.

  python scripts/check_strand_sidecar.py --split train_v3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

import numpy as np

from knot_recognition.cable.targets import centerline_target
from knot_recognition.strand_pointer import load_records

DATA = ROOT / "data" / "knot_aware"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--max-diff-pixels", type=int, default=60, help="pixels allowed to differ by > 0.05")
    a = ap.parse_args()
    recs = load_records(DATA / f"{a.split}_strands.pkl")
    npy = DATA / a.split / "center_hi.npy"
    ref = np.load(npy, mmap_mode="r") if npy.exists() else np.load(DATA / f"{a.split}.npz")["center_hi"]
    if len(recs) != len(ref):
        raise SystemExit(f"{len(recs)} records for {len(ref)} images")
    worst, bad = 0, []
    for i, r in enumerate(recs):
        ann = {"image_size": [512, 512],          # records: (X + 0.5) / 2 of the 512 render
               "components": [(np.asarray(P, np.float64) * 2 - 0.5).tolist() for P in r["components"]]}
        d = np.abs(centerline_target(ann, stride=2, sigma_cells=1.0) - np.asarray(ref[i], np.float32))
        n = int((d > 0.05).sum())
        worst = max(worst, n)
        if n > a.max_diff_pixels:
            bad.append(i)
    print(f"{a.split}: {len(recs)} records; most differing pixels in one map: {worst} of {ref.shape[1] * ref.shape[2]}; "
          f"mismatched records: {len(bad)} {bad[:10]}")
    if bad:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
