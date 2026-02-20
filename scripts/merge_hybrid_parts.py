import argparse
import csv
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pattern", default="data/knotprot/hybrid_dataset_part*.npz")
    parser.add_argument("--out", default="data/knotprot/hybrid_dataset.npz")
    parser.add_argument("--manifest-pattern", default="data/knotprot/hybrid_manifest_part*.csv")
    parser.add_argument("--manifest-out", default="data/knotprot/hybrid_manifest.csv")
    args = parser.parse_args()

    paths = sorted(Path(".").glob(args.pattern))
    if not paths:
        raise SystemExit("No parts found")

    images_list = []
    feats_list = []
    labels_list = []
    keys_list = []
    saw_keys = False

    for p in paths:
        data = np.load(p, allow_pickle=True)
        images_list.append(data["images"])
        feats_list.append(data["gauss_feats"])
        labels_list.append(data["labels"])
        if "keys" in data:
            keys_list.append(data["keys"])
            saw_keys = True
        elif saw_keys:
            raise SystemExit(f"Missing keys in {p}")

    images = np.concatenate(images_list, axis=0)
    feats = np.concatenate(feats_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)
    if saw_keys:
        keys = np.concatenate(keys_list, axis=0)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if saw_keys:
        np.savez_compressed(out_path, images=images, gauss_feats=feats, labels=labels, keys=keys)
    else:
        np.savez_compressed(out_path, images=images, gauss_feats=feats, labels=labels)

    if args.manifest_pattern:
        manifest_paths = sorted(Path(".").glob(args.manifest_pattern))
        if manifest_paths:
            if len(manifest_paths) != len(paths):
                raise SystemExit("Manifest parts count does not match dataset parts")
            rows = []
            for mp in manifest_paths:
                with mp.open("r", newline="") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        rows.append((row["key"], row["view_id"]))
            with Path(args.manifest_out).open("w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["key", "view_id"])
                writer.writerows(rows)
            print("manifest", args.manifest_out)

    print("parts", len(paths))
    print("samples", len(labels))
    print("out", out_path)


if __name__ == "__main__":
    main()
