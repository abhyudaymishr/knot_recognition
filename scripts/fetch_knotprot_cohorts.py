#!/usr/bin/env python3
"""Fetch versioned KnotProt API records and coordinate cohorts.

Development cohort: the 200 chain IDs in the repository's historical view
manifest, relabeled with KnotProt's explicit K (knot) / S (slipknot) field.
External chain cohort: published (Y), non-redundant API entries whose PDB-chain
IDs do not occur in the development manifest. Multiple knot types for one
chain are permitted only when their K/S status agrees.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
import urllib.request
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

BASE = "https://knotprot.cent.uw.edu.pl"
USER_AGENT = "knot-recognition-research/0.1 (reproducible public-data evaluation)"


def get(url: str, retries: int = 4) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    last = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=45) as response:
                return response.read()
        except Exception as exc:
            last = exc
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"Could not retrieve {url}: {last}")


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_rows(raw: bytes):
    rows = []
    for line in raw.decode("utf-8").splitlines()[1:]:
        fields = line.split(";")
        if len(fields) < 9:
            continue
        pdb, chain, length, status, kind, knot_type, n_cut, c_cut, core = fields[:9]
        key = f"{pdb.strip().lower()}_{chain.strip() or '_'}"
        if kind not in {"K", "S"}:
            continue
        rows.append({"key": key, "pdb_id": pdb.strip().lower(),
                     "chain_id": chain.strip() or "_", "length": int(length),
                     "publication_status": status, "kind": kind,
                     "knot_type": knot_type, "n_cut": n_cut,
                     "c_cut": c_cut, "core_range": core})
    return rows


def resolve_chain_labels(rows, keys, *, published_only=False):
    allowed = set(keys)
    by_key = defaultdict(list)
    for row in rows:
        if row["key"] in allowed and (not published_only or row["publication_status"] == "Y"):
            by_key[row["key"]].append(row)
    chosen = {}
    for key in sorted(allowed):
        matches = by_key.get(key, [])
        statuses = {r["kind"] for r in matches}
        if len(statuses) != 1:
            continue
        # Prefer a published annotation. Preserve all annotation rows in the
        # source snapshot; the selected binary target is unambiguous K or S.
        label = next(iter(statuses))
        chosen[key] = (label, matches)
    return chosen


def write_csv(path: Path, header, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(records)


def fetch_xyz(key: str) -> tuple[np.ndarray, bytes]:
    pdb, chain = key.rsplit("_", 1)
    url = f"{BASE}/chains/{pdb}/{chain}/chain.xyz.txt"
    raw = get(url)
    coords = np.loadtxt(raw.decode("utf-8").splitlines(), dtype=np.float64, usecols=(1, 2, 3))
    if coords.ndim != 2 or coords.shape[1] != 3 or len(coords) < 4:
        raise ValueError(f"Invalid coordinate rows for {key}: {coords.shape}")
    return coords, raw


def save_cohort(name, selection, root, records):
    coord_dir = root / name / "xyz"
    coord_dir.mkdir(parents=True, exist_ok=True)
    backbones = {}
    failures = []
    coordinate_manifest = []
    for n, key in enumerate(sorted(selection), 1):
        try:
            coords, raw = fetch_xyz(key)
            backbones[key] = coords
            (coord_dir / f"{key}.xyz.txt").write_bytes(raw)
            coordinate_manifest.append({"key": key, "xyz_sha256": digest(raw),
                                        "coordinate_count": len(coords)})
            if n % 25 == 0:
                print(f"{name}: retrieved {n}/{len(selection)} chains", flush=True)
            time.sleep(0.05)
        except Exception as exc:
            failures.append({"key": key, "error": str(exc)})
    if backbones:
        np.savez_compressed(root / name / "backbones.npz", **backbones)
    labels = [{"pdb_id": key.rsplit("_", 1)[0], "chain_id": key.rsplit("_", 1)[1],
               "knot_type": selection[key][0]} for key in sorted(backbones)]
    chains = [{"pdb_id": r["pdb_id"], "chain_id": r["chain_id"]} for r in labels]
    write_csv(root / name / "knotprot_labels.csv", ["pdb_id", "chain_id", "knot_type"], labels)
    write_csv(root / name / "knotprot_chains.csv", ["pdb_id", "chain_id"], chains)
    write_csv(root / name / "coordinate_manifest.csv",
              ["key", "xyz_sha256", "coordinate_count"], coordinate_manifest)
    return {"requested_keys": len(selection), "downloaded_keys": len(backbones),
            "failures": failures, "coordinates": coordinate_manifest}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--historical-manifest", type=Path,
                        default=Path("data/knotprot/hybrid_manifest.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/knotprot/research_2026_10"))
    args = parser.parse_args()
    root = args.out_dir
    root.mkdir(parents=True, exist_ok=True)

    raw_all = get(f"{BASE}/_all_chains_knotted_N_C")
    raw_nr = get(f"{BASE}/_nr_chains_knotted_N_C")
    (root / "source_all_chains_knotted_N_C.txt").write_bytes(raw_all)
    (root / "source_nr_chains_knotted_N_C.txt").write_bytes(raw_nr)
    all_rows, nr_rows = read_rows(raw_all), read_rows(raw_nr)
    with args.historical_manifest.open(newline="") as f:
        dev_keys = sorted({row["key"].strip() for row in csv.DictReader(f)})
    dev = resolve_chain_labels(all_rows, dev_keys)
    unresolved = sorted(set(dev_keys) - set(dev))
    if unresolved:
        raise ValueError(f"Historical chains lack a consistent K/S annotation: {unresolved[:12]}")

    ext_rows = [r for r in nr_rows if r["publication_status"] == "Y" and r["key"] not in set(dev_keys)]
    ext_keys = sorted({r["key"] for r in ext_rows})
    ext = resolve_chain_labels(ext_rows, ext_keys, published_only=True)
    if not ext:
        raise ValueError("No published, non-overlapping external chains found")

    print(f"Development: {len(dev)} historical chains; K/S labels are consistent across all annotations.")
    print(f"External candidate: {len(ext)} published, non-redundant chain IDs, disjoint by PDB-chain key.")
    dev_result = save_cohort("development", dev, root, all_rows)
    ext_result = save_cohort("external", ext, root, nr_rows)
    snapshot = {
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_api": BASE,
        "target": "KnotProt full-chain annotation class: K=knot, S=slipknot",
        "development_selection": "chain IDs in historical repository hybrid_manifest.csv; current API K/S labels",
        "external_selection": "published Y records from non-redundant API endpoint, excluding development PDB-chain IDs",
        "historical_manifest_path": str(args.historical_manifest),
        "historical_manifest_sha256": digest(args.historical_manifest.read_bytes()),
        "source_files": {
            "all_chains": {"path": "source_all_chains_knotted_N_C.txt", "sha256": digest(raw_all)},
            "nonredundant": {"path": "source_nr_chains_knotted_N_C.txt", "sha256": digest(raw_nr)},
        },
        "development": dev_result, "external": ext_result,
        "development_label_counts": {label: sum(1 for item in dev.values() if item[0] == label)
                                      for label in ("K", "S")},
        "external_label_counts": {label: sum(1 for item in ext.values() if item[0] == label)
                                   for label in ("K", "S")},
        "pdb_accession_overlap": sorted(
            {key.rsplit("_", 1)[0] for key in dev}
            & {key.rsplit("_", 1)[0] for key in ext}),
        "notes": ["Historical chain IDs are highly redundant; this protocol's external cohort is restricted to the API's published non-redundant list.",
                  "The source is one KnotProt release; this is external-chain validation, not an independent database replication.",
                  "Labels are K/S (knot versus slipknot), not the historical four-class chirality/type labels.",
                  "KnotProt supplies coordinates after its own structure processing; this differs from raw RCSB PDB parsing."],
    }
    (root / "source_manifest.json").write_text(json.dumps(snapshot, indent=2, sort_keys=True) + "\n")
    print(f"Wrote source manifest to {root / 'source_manifest.json'}")
    print(f"Downloaded development/external: {dev_result['downloaded_keys']}/{ext_result['downloaded_keys']} chains")


if __name__ == "__main__":
    main()
