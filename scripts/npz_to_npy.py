"""npz_to_npy.py - Split data/knot_aware/*.npz into one .npy per array so training can memory-map
them (several training jobs then share one page-cached copy instead of each holding ~4 GB)."""
import sys
from pathlib import Path

import numpy as np

D = Path(__file__).resolve().parent.parent / "data" / "knot_aware"
for name in sys.argv[1:] or ["train", "val", "train_v3", "val_v3"]:
    z = np.load(D / f"{name}.npz")
    out = D / name
    out.mkdir(exist_ok=True)
    for k in z.files:
        if not (out / f"{k}.npy").exists():
            np.save(out / f"{k}.npy", z[k])
    print(name, z.files)
