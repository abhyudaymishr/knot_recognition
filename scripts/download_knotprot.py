import argparse
import csv
import hashlib
import json
import os
import time
from pathlib import Path
from urllib.request import urlretrieve


def download(pdb_id: str, out_dir: Path) -> Path:
    pdb_id = pdb_id.lower()
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    out_path = out_dir / f"{pdb_id}.pdb"
    if out_path.exists():
        return out_path
    urlretrieve(url, out_path)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="data/knotprot/knotprot_chains.csv")
    parser.add_argument("--out-dir", default="data/knotprot/pdb")
    parser.add_argument("--sleep", type=float, default=0.2)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    csv_path = Path(args.csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    seen = set()
    rows = []
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pdb_id = row["pdb_id"].strip()
            if pdb_id in seen:
                continue
            seen.add(pdb_id)
            rows.append(pdb_id)

    if args.limit:
        rows = rows[: args.limit]

    for i, pdb_id in enumerate(rows, start=1):
        try:
            download(pdb_id, out_dir)
            print(f"[{i}/{len(rows)}] downloaded {pdb_id}")
        except Exception as exc:
            print(f"[{i}/{len(rows)}] failed {pdb_id}: {exc}")
        time.sleep(args.sleep)

    # Freeze the exact downloaded coordinate files by checksum, since an RCSB
    # accession can be revised without changing its identifier.
    records = []
    for pdb_path in sorted(out_dir.glob("*.pdb")):
        h = hashlib.sha256()
        with pdb_path.open("rb") as f:
            for block in iter(lambda: f.read(1024 * 1024), b""):
                h.update(block)
        records.append({"pdb_id": pdb_path.stem.lower(), "url":
                        f"https://files.rcsb.org/download/{pdb_path.name}",
                        "sha256": h.hexdigest(), "bytes": pdb_path.stat().st_size})
    manifest = out_dir / "download_manifest.json"
    manifest.write_text(json.dumps({"schema_version": 1, "files": records},
                                   indent=2, sort_keys=True) + "\n")
    print(f"Wrote checksummed coordinate manifest to {manifest}")


if __name__ == "__main__":
    main()
