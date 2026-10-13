import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from knot_recognition.protein import (
    detect_crossings,
    gauss_code_from_crossings,
    project_polyline,
    sample_viewpoints,
)


def render_projection(xy: np.ndarray, size: int = 128, pad: int = 8,
                     gap_after: np.ndarray | None = None) -> np.ndarray:
    mins = xy.min(axis=0)
    maxs = xy.max(axis=0)
    span = np.maximum(maxs - mins, 1e-6)
    norm = (xy - mins) / span
    img = np.zeros((size, size), dtype=np.uint8)
    coords = (norm * (size - 1 - 2 * pad) + pad).astype(int)
    if gap_after is None:
        gap_after = np.zeros(max(0, len(coords) - 1), dtype=bool)
    for i in range(len(coords) - 1):
        if gap_after[i]:
            continue
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
    parser.add_argument("--provenance", default=None, help="JSON provenance sidecar (defaults to <out>.provenance.json)")
    args = parser.parse_args()

    backbones = np.load(args.backbones, allow_pickle=True)
    labels = {}
    with open(args.labels, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pdb_id = row["pdb_id"].strip().lower()
            chain_id = row["chain_id"].strip() or "_"
            key = f"{pdb_id}_{chain_id}"
            label = row["knot_type"].strip()
            if key in labels and labels[key] != label:
                raise ValueError(f"Conflicting knot labels for chain {key}: {labels[key]!r} and {label!r}")
            labels[key] = label

    all_bb_keys = [k for k in backbones.keys() if not k.startswith("__residue_ids__")
                   and not k.startswith("__gap_after__")]
    matched_keys = [k for k in all_bb_keys if k in labels]
    unmatched_bb = [k for k in all_bb_keys if k not in labels]
    unmatched_lbl = [k for k in labels.keys() if k not in backbones]

    print(f"Total backbones: {len(all_bb_keys)}, Total label records: {len(labels)}")
    print(f"Matched chains: {len(matched_keys)}, Unmatched backbones: {len(unmatched_bb)}, Unmatched labels: {len(unmatched_lbl)}")

    if not matched_keys:
        sample_bb = all_bb_keys[:5]
        sample_lbl = list(labels.keys())[:5]
        raise ValueError(
            f"No backbone keys matched between {args.backbones} and labels in {args.labels}.\n"
            f"Sample backbone keys: {sample_bb}\n"
            f"Sample label keys:    {sample_lbl}\n"
            "Ensure PDB IDs and chain IDs match (e.g. '1j85_A')."
        )

    keys = matched_keys
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
    chain_sampling = {}

    for idx, key in enumerate(keys, start=1):
        coords = backbones[key]
        original_points = int(len(coords))
        gap_key = f"__gap_after__{key}"
        original_gaps = (np.asarray(backbones[gap_key], dtype=bool)
                         if gap_key in backbones else np.zeros(max(0, original_points - 1), dtype=bool))
        selected = np.arange(0, original_points, args.stride if args.stride else 1)
        if args.max_points and len(selected) > args.max_points:
            selected = selected[np.linspace(0, len(selected) - 1, args.max_points).astype(int)]
        coords = coords[selected]
        # Mark a retained segment invalid if it spans any missing-residue gap.
        sampled_gaps = np.array([np.any(original_gaps[a:b])
                                 for a, b in zip(selected[:-1], selected[1:])], dtype=bool)
        chain_sampling[key] = {"original_ca_count": original_points,
                               "processed_point_count": int(len(coords)),
                               "stride": args.stride, "max_points": args.max_points}
        for v_id, view in enumerate(views):
            xy, depth = project_polyline(coords, view)
            crossings = detect_crossings(xy, depth, sampled_gaps)
            gauss = gauss_code_from_crossings(crossings, coords.shape[0])
            img = render_projection(xy, gap_after=sampled_gaps)
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

    def file_hash(path):
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(1024 * 1024), b""):
                h.update(block)
        return h.hexdigest()

    provenance_path = Path(args.provenance) if args.provenance else Path(str(out_path) + ".provenance.json")
    provenance_path.parent.mkdir(parents=True, exist_ok=True)
    provenance = {
        "schema_version": 1,
        "inputs": {"backbones_path": str(Path(args.backbones)),
                   "backbones_sha256": file_hash(args.backbones),
                   "labels_path": str(Path(args.labels)),
                   "labels_sha256": file_hash(args.labels)},
        "parameters": {"viewpoints": args.viewpoints, "stride": args.stride,
                       "max_points": args.max_points, "offset": args.offset,
                       "limit": args.limit},
        "matched_chain_ids": sorted(keys),
        "matched_chain_ids_sha256": hashlib.sha256("\n".join(sorted(keys)).encode()).hexdigest(),
        "chain_sampling": chain_sampling,
        "outputs": {"dataset_path": str(out_path), "dataset_sha256": file_hash(out_path),
                    "manifest_path": str(Path(args.manifest)),
                    "manifest_sha256": file_hash(args.manifest)},
        "warning": "Projection-derived features are heuristic; subsampling is not guaranteed to preserve topology.",
    }
    provenance_path.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    print("provenance", provenance_path)


if __name__ == "__main__":
    main()
