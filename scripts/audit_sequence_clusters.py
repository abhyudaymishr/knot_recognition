#!/usr/bin/env python3
"""Map KnotProt chains onto current RCSB sequence clusters and rerun grouped CV."""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import re
import time
import urllib.request
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from evaluate_knotprot import (load_data, metrics, model_factories,
                               row_features, chain_predictions, add_shape_features)

RCSB = "https://data.rcsb.org/rest/v1/core"
CLUSTERS = "https://cdn.rcsb.org/resources/sequence/clusters/clusters-by-entity-{identity}.txt"


def fetch(url):
    req = urllib.request.Request(url, headers={"User-Agent": "knot-recognition-sequence-audit/1.0"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def json_get(url):
    return json.loads(fetch(url))


def fetch_range_file(url, chunk_size=1_000_000):
    """Download a large public text resource in resumable byte ranges."""
    head = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "knot-recognition-sequence-audit/1.0"})
    with urllib.request.urlopen(head, timeout=60) as r:
        size = int(r.headers["Content-Length"])
    pieces = []
    for start in range(0, size, chunk_size):
        end = min(size - 1, start + chunk_size - 1)
        req = urllib.request.Request(url, headers={"User-Agent": "knot-recognition-sequence-audit/1.0",
                                                   "Range": f"bytes={start}-{end}",
                                                   "Accept-Encoding": "identity"})
        with urllib.request.urlopen(req, timeout=60) as r:
            part = r.read()
            if len(part) != end - start + 1:
                raise IOError(f"RCSB range {start}-{end} returned {len(part)} bytes")
            pieces.append(part)
        if len(pieces) % 5 == 0:
            print(f"RCSB cluster file {end + 1}/{size} bytes", flush=True)
    return b"".join(pieces)


def pdb_entity_maps(pdb):
    entry = json_get(f"{RCSB}/entry/{pdb}")
    ids = entry.get("rcsb_entry_container_identifiers", {}).get("polymer_entity_ids", [])
    mapping = {}
    for entity in ids:
        d = json_get(f"{RCSB}/polymer_entity/{pdb}/{entity}")
        ids_obj = d.get("rcsb_polymer_entity_container_identifiers", {})
        for chain in ids_obj.get("auth_asym_ids", []) + ids_obj.get("asym_ids", []):
            mapping[str(chain)] = str(entity)
    return mapping


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--development", type=Path, required=True)
    p.add_argument("--external", type=Path, required=True)
    p.add_argument("--evaluation", type=Path, required=True)
    p.add_argument("--backbones", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seed", type=int, default=23)
    p.add_argument("--folds", type=int, default=5)
    a = p.parse_args()
    dev, ext = load_data(a.development), load_data(a.external)
    ev = json.loads(a.evaluation.read_text())
    keys = sorted(set(dev["keys"].tolist()) | set(ext["keys"].tolist()))
    pdbs = sorted({k.rsplit("_", 1)[0].upper() for k in keys})
    print(f"Fetching RCSB chain/entity maps for {len(pdbs)} accessions", flush=True)
    cache_path = a.out.with_suffix(".mapping-cache.json")
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {"maps": {}, "errors": {}}
    maps, errors = cache["maps"], cache["errors"]
    pdbs = [pdb for pdb in pdbs if pdb not in maps and pdb not in errors]
    with cf.ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(pdb_entity_maps, pdb): pdb for pdb in pdbs}
        for n, future in enumerate(cf.as_completed(futures), 1):
            pdb = futures[future]
            try: maps[pdb] = future.result()
            except Exception as e: errors[pdb] = str(e)
            cache_path.write_text(json.dumps({"maps": maps, "errors": errors}))
            if n % 25 == 0: print(f"RCSB metadata {n}/{len(pdbs)}", flush=True)
    cluster_by_identity = {}
    raw_hashes = {}
    for ident in (30,):
        cluster_path = a.out.parent / f"rcsb_clusters_by_entity_{ident}.txt"
        if cluster_path.exists():
            raw = cluster_path.read_bytes()
        else:
            raw = fetch_range_file(CLUSTERS.format(identity=ident))
            cluster_path.write_bytes(raw)
        raw_hashes[str(ident)] = hashlib.sha256(raw).hexdigest()
        lookup = {}
        for ix, line in enumerate(raw.decode("utf-8").splitlines()):
            members = re.split(r"[,;\s]+", line.strip())
            for member in filter(None, members): lookup[member.upper()] = f"{ident}:{ix}"
        assignment = {}
        for key in keys:
            pdb, chain = key.rsplit("_", 1); pdb = pdb.upper()
            entity = maps.get(pdb, {}).get(chain)
            cluster = lookup.get(f"{pdb}_{entity}") if entity else None
            assignment[key] = cluster or f"unmapped:{key}"
        cluster_by_identity[str(ident)] = assignment
    result = {
        "source": "RCSB PDB weekly polymer-entity sequence clusters",
        "retrieved_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "cluster_file_sha256": raw_hashes, "identity_thresholds": [30],
        "rcsb_accessions_queried": len(maps) + len(errors), "metadata_failures": errors,
        "chain_entity_mapping_cache": str(cache_path),
        "chain_entity_mapping_cache_sha256": hashlib.sha256(cache_path.read_bytes()).hexdigest(),
        "cluster_file": str(cluster_path),
        "mapping_note": "Chains are mapped through RCSB polymer entities; unmapped chains receive singleton groups, so they cannot cause false leakage but may understate homolog overlap.",
        "chain_cluster_assignments": cluster_by_identity,
        "audit": {},
    }
    train, holdout = set(ev["train_chain_ids"]), set(ev["test_chain_ids"])
    external_keys = set(ext["keys"].tolist())
    for ident, assignment in cluster_by_identity.items():
        dev_keys = np.asarray(sorted(set(dev["keys"].tolist())), dtype=str)
        y = np.asarray([np.unique(dev["labels"][dev["keys"] == k])[0] for k in dev_keys])
        groups = np.asarray([assignment[k] for k in dev_keys])
        holdout_overlap = sorted({assignment[k] for k in train & set(dev_keys)} &
                                 {assignment[k] for k in holdout & set(dev_keys)})
        external_overlap = sorted({assignment[k] for k in train & set(dev_keys)} &
                                  {assignment[k] for k in external_keys})
        splitter = StratifiedGroupKFold(n_splits=a.folds, shuffle=True, random_state=a.seed)
        cv = {}
        for modality, model_name in (("image", "rbf_svm"), ("shape_3d", "extra_trees")):
            if modality == "shape_3d": add_shape_features(dev, a.backbones)
            x = row_features(dev, modality)
            reports, split_ids = [], []
            for fold, (tr, va) in enumerate(splitter.split(np.zeros(len(dev_keys)), y, groups), 1):
                tr_keys, va_keys = dev_keys[tr], dev_keys[va]
                tr_rows = np.flatnonzero(np.isin(dev["keys"], tr_keys))
                model = model_factories(a.seed)[model_name]().fit(x[tr_rows], dev["labels"][tr_rows])
                yt, yp, _ = chain_predictions(model, x, dev, va_keys, np.unique(y), None, "uniform", a.seed)
                reports.append(metrics(yt, yp, np.unique(y), a.seed, bootstraps=500))
                split_ids.append({"fold": fold, "train": sorted(tr_keys.tolist()), "validation": sorted(va_keys.tolist()),
                                  "train_sha256": hashlib.sha256("\n".join(sorted(tr_keys)).encode()).hexdigest(),
                                  "validation_sha256": hashlib.sha256("\n".join(sorted(va_keys)).encode()).hexdigest()})
            cv[modality] = {"model": model_name, "fold_metrics": reports,
                            "mean_balanced_accuracy": float(np.mean([r["balanced_accuracy"] for r in reports])),
                            "fold_chain_ids": split_ids}
        result["audit"][ident] = {
            "n_dev_chains": len(dev_keys), "n_dev_clusters": len(set(groups)),
            "largest_dev_cluster": int(max(np.unique(groups, return_counts=True)[1])),
            "n_unmapped_dev_chains": sum(v.startswith("unmapped:") for k, v in assignment.items() if k in set(dev_keys)),
            "frozen_random_holdout_cluster_overlap_count": len(holdout_overlap),
            "frozen_train_holdout_overlap_clusters": holdout_overlap,
            "development_external_cluster_overlap_count": len(external_overlap),
            "development_external_overlap_clusters": external_overlap,
            "cluster_grouped_cv": cv,
        }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"Wrote RCSB sequence-cluster audit to {a.out}")


if __name__ == "__main__": main()
