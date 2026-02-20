import argparse
import csv
from pathlib import Path

import numpy as np

from knot_recognition.protein import (
    detect_crossings,
    gauss_code_from_crossings,
    project_polyline,
    sample_viewpoints,
)


def render_projection(xy: np.ndarray, size: int = 128, pad: int = 8) -> np.ndarray:
    mins = xy.min(axis=0)
    maxs = xy.max(axis=0)
    span = np.maximum(maxs - mins, 1e-6)
    norm = (xy - mins) / span
    img = np.zeros((size, size), dtype=np.uint8)
    coords = (norm * (size - 1 - 2 * pad) + pad).astype(int)
    for i in range(len(coords) - 1):
        r0, c0 = coords[i][1], coords[i][0]
        r1, c1 = coords[i + 1][1], coords[i + 1][0]
        rr = np.linspace(r0, r1, num=abs(r1 - r0) + abs(c1 - c0) + 1, dtype=int)
        cc = np.linspace(c0, c1, num=abs(r1 - r0) + abs(c1 - c0) + 1, dtype=int)
        img[rr, cc] = 255
    return img


def gauss_histogram(gauss: list[int], bins: int = 40) -> np.ndarray:
    if not gauss:
        return np.zeros((bins,), dtype=np.float32)
    vals = np.array(gauss, dtype=float)
    maxv = max(1.0, np.max(np.abs(vals)))
    vals = np.clip(vals / maxv, -1.0, 1.0)
    hist, _ = np.histogram(vals, bins=bins, range=(-1.0, 1.0))
    hist = hist.astype(np.float32)
    return hist / max(1.0, hist.sum())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbones", default="data/knotprot/backbones.npz")
    parser.add_argument("--labels", default="data/knotprot/knotprot_labels.csv")
    parser.add_argument("--out", default="data/knotprot/hybrid_dataset.npz")
    parser.add_argument("--manifest", default="data/knotprot/hybrid_manifest.csv")
    parser.add_argument("--viewpoints", type=int, default=32)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-points", type=int, default=None)
    args = parser.parse_args()

    backbones = np.load(args.backbones, allow_pickle=True)
    labels = {}
    with open(args.labels, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = f"{row['pdb_id']}_{row['chain_id']}"
            labels[key] = row["knot_type"]

    keys = [k for k in backbones.keys() if k in labels]
    if args.offset:
        keys = keys[args.offset :]
    if args.limit:
        keys = keys[: args.limit]

    views = sample_viewpoints(args.viewpoints)

    images = []
    gauss_feats = []
    y = []
    manifest_rows = []
    keys_per_sample = []

    for idx, key in enumerate(keys, start=1):
        coords = backbones[key]
        if args.stride and args.stride > 1:
            coords = coords[:: args.stride]
        if args.max_points and coords.shape[0] > args.max_points:
            idxs = np.linspace(0, coords.shape[0] - 1, args.max_points).astype(int)
            coords = coords[idxs]
        for v_id, view in enumerate(views):
            xy, depth = project_polyline(coords, view)
            crossings = detect_crossings(xy, depth)
            gauss = gauss_code_from_crossings(crossings, coords.shape[0])
            img = render_projection(xy)
            images.append(img)
            gauss_feats.append(gauss_histogram(gauss))
            y.append(labels[key])
            manifest_rows.append((key, v_id))
            keys_per_sample.append(key)
        if idx % 50 == 0:
            print(f"processed {idx}/{len(keys)}")

    images = np.stack(images, axis=0)
    gauss_feats = np.stack(gauss_feats, axis=0)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        images=images,
        gauss_feats=gauss_feats,
        labels=np.array(y),
        keys=np.array(keys_per_sample),
    )

    with Path(args.manifest).open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["key", "view_id"])
        w.writerows(manifest_rows)

    print("samples", len(y))
    print("out", out_path)
    print("manifest", args.manifest)


if __name__ == "__main__":
    main()
