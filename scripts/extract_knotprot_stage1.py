#!/usr/bin/env python3
"""extract_knotprot_stage1.py - Extract C-alpha backbones from PDB files for KnotProt pipeline.

Reads PDB files from --pdb-dir, extracts C-alpha coordinates using
knot_recognition.protein.extract_ca_polyline, and outputs:
  - Compressed numpy file (--out): dictionary of key -> (N, 3) float coords
  - CSV manifest (--manifest): summary metadata of extracted chains
"""

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from knot_recognition.protein import extract_ca_polyline


def load_chains_filter(chains_csv: Path) -> Optional[Set[Tuple[str, str]]]:
    if not chains_csv.exists():
        return None
    chains = set()
    with chains_csv.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pdb_id = row.get("pdb_id", "").strip().lower()
            chain_id = row.get("chain_id", "").strip() or "_"
            if pdb_id:
                chains.add((pdb_id, chain_id))
    return chains


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract C-alpha backbones from PDB structures.")
    parser.add_argument("--pdb-dir", default="data/knotprot/pdb", help="Directory containing .pdb files")
    parser.add_argument("--chains-csv", default="data/knotprot/knotprot_chains.csv", help="Optional CSV listing target chains")
    parser.add_argument("--out", default="data/knotprot/backbones.npz", help="Output .npz file path")
    parser.add_argument("--manifest", default="data/knotprot/backbones.csv", help="Output .csv manifest path")
    args = parser.parse_args()

    pdb_dir = Path(args.pdb_dir)
    out_path = Path(args.out)
    manifest_path = Path(args.manifest)
    chains_csv = Path(args.chains_csv)

    if not pdb_dir.exists():
        raise FileNotFoundError(
            f"PDB directory '{pdb_dir}' does not exist. "
            "Please download PDB structures first using `scripts/download_knotprot.py` "
            "or place RCSB PDB files into this directory."
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    target_chains = load_chains_filter(chains_csv)
    pdb_files = sorted(pdb_dir.glob("*.pdb"))
    if not pdb_files:
        raise FileNotFoundError(f"No .pdb files found in '{pdb_dir}'.")

    backbones: Dict[str, np.ndarray] = {}
    manifest_rows: List[Dict[str, str]] = []

    for pdb_path in pdb_files:
        pdb_id = pdb_path.stem.strip().lower()
        # Determine which chain(s) to extract
        if target_chains:
            relevant = [ch for p, ch in target_chains if p == pdb_id]
            if not relevant:
                continue
            chains_to_extract = relevant
        else:
            chains_to_extract = [None]  # Auto-select longest chain

        for ch in chains_to_extract:
            try:
                bb = extract_ca_polyline(str(pdb_path), chain_id=ch)
                chain_norm = bb.chain_id.strip() or "_"
                key = f"{pdb_id}_{chain_norm}"
                backbones[key] = bb.ca_coords
                manifest_rows.append(
                    {
                        "key": key,
                        "pdb_id": pdb_id,
                        "chain_id": chain_norm,
                        "num_ca": str(len(bb.ca_coords)),
                        "path": str(pdb_path),
                    }
                )
                print(f"[EXTRACTED] {key}: {len(bb.ca_coords)} C-alpha atoms")
            except Exception as exc:
                print(f"[SKIP] {pdb_id} chain {ch}: {exc}")

    if target_chains:
        extracted_pairs = {(r["pdb_id"], r["chain_id"]) for r in manifest_rows}
        unmatched_targets = target_chains - extracted_pairs
        print(f"Target chains specified: {len(target_chains)} | Extracted: {len(extracted_pairs)} | Unmatched targets: {len(unmatched_targets)}")

    if not backbones:
        raise RuntimeError("No backbones were successfully extracted.")

    np.savez_compressed(out_path, **backbones)
    print(f"Saved {len(backbones)} backbones to {out_path}")


    with manifest_path.open("w", newline="") as f:
        fieldnames = ["key", "pdb_id", "chain_id", "num_ca", "path"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)
    print(f"Saved manifest to {manifest_path}")


if __name__ == "__main__":
    main()
