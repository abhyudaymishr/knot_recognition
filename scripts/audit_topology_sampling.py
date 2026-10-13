#!/usr/bin/env python3
"""Recompute knot-type distributions after chain downsampling with Topoly.

KnotProt K/S labels are retained only for stratified summaries; they are not
treated as labels produced by this invariant analysis. Open chains are closed
using explicit stochastic two-point closures.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
from topoly import Closure, alexander
import topoly
from evaluate_knotprot import metrics


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def distribution(xyz, tries, closure):
    values = alexander(np.asarray(xyz, dtype=float).tolist(), closure=closure,
                       tries=tries, hide_trivial=False, hide_rare=False,
                       run_parallel=False, translate=True, chiral=False)
    if isinstance(values, str):
        return {values: 1.0}
    return {str(k): float(v) for k, v in values.items()}


def nontrivial(d):
    return float(sum(v for k, v in d.items() if k not in {"0_1", "Unknown", "TMC"}))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backbones", type=Path, required=True)
    p.add_argument("--labels", type=Path, required=True)
    p.add_argument("--reduced-provenance", type=Path, required=True)
    p.add_argument("--tries", type=int, default=30)
    p.add_argument("--seed", type=int, default=23)
    p.add_argument("--evaluation-split", type=Path)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    random.seed(a.seed); np.random.seed(a.seed)
    coords = np.load(a.backbones, allow_pickle=False)
    labels = np.genfromtxt(a.labels, delimiter=",", names=True, dtype=None, encoding="utf-8")
    label_map = {f"{r['pdb_id']}_{r['chain_id']}": str(r["knot_type"]) for r in labels}
    prov = json.loads(a.reduced_provenance.read_text())
    params = prov["parameters"]
    stride, cap = int(params["stride"]), params["max_points"]
    records = []
    for i, key in enumerate(sorted(set(coords.files) & set(label_map)), 1):
        xyz = np.asarray(coords[key], dtype=float)
        reduced = xyz[::stride]
        if cap is not None:
            reduced = reduced[:int(cap)]
        # Match the evaluated capped/strided representation, plus a stride-only
        # representation, while keeping the all-coordinate source as reference.
        stride_only = xyz[::stride]
        representations = (("full", xyz), ("stride_only", stride_only), ("stride_cap", reduced))
        stochastic, deterministic = {}, {}
        for name, points in representations:
            stochastic[name] = distribution(points, a.tries, Closure.TWO_POINTS)
            deterministic[name] = distribution(points, 1, Closure.MASS_CENTER)
        full_p = nontrivial(stochastic["full"])
        row = {"key": key, "knotprot_KS": label_map[key],
               "coordinate_count": len(xyz), "stride_count": len(stride_only),
               "stride_cap_count": len(reduced), "topoly_alexander_stochastic_two_point": stochastic,
               "topoly_alexander_deterministic_mass_center": deterministic,
               "nontrivial_probability": {n: nontrivial(d) for n, d in stochastic.items()},
               "nontrivial_probability_delta_vs_full": {
                   n: nontrivial(d) - full_p for n, d in stochastic.items() if n != "full"},
               "stochastic_30_closure_histogram_not_equal_vs_full": {
                   n: stochastic[n] != stochastic["full"]
                   for n in ("stride_only", "stride_cap")},
               "stochastic_nontrivial_probability_delta_vs_full": {
                   n: nontrivial(stochastic[n]) - full_p
                   for n in ("stride_only", "stride_cap")},
               "deterministic_mass_center_changed_vs_full": {
                   n: deterministic[n] != deterministic["full"]
                   for n in ("stride_only", "stride_cap")}}
        row["deterministic_cap_changed_vs_stride_only"] = (
            deterministic["stride_cap"] != deterministic["stride_only"])
        row["stochastic_cap_nontrivial_probability_delta_vs_stride_only"] = (
            nontrivial(stochastic["stride_cap"]) - nontrivial(stochastic["stride_only"]))
        records.append(row)
        if i % 10 == 0 or i == len(set(coords.files) & set(label_map)):
            print(f"Topoly completed {i}/{len(set(coords.files) & set(label_map))}", flush=True)
    result = {
        "method": "Topoly Alexander polynomial, stochastic TWO_POINTS plus deterministic MASS_CENTER closures",
        "topoly_version": getattr(topoly, "__version__", "1.1.0"),
        "tries_per_representation": a.tries, "seed": a.seed,
        "seed_note": "Python and NumPy RNGs seeded; Topoly's compiled stochastic closure RNG may not honor these seeds. Stochastic histograms are descriptive Monte Carlo estimates; deterministic MASS_CENTER results are the primary paired sampling check.",
        "closed_chain_topology_not_equivalent_to_KnotProt_KS": True,
        "label_provenance_caveat": "KnotProt annotations may themselves use Topoly-like closure analysis; this is a reproducible invariant baseline but not an independent label-generation pipeline.",
        "interpretation": "Recomputed whole-chain Alexander knot distributions test geometric subsampling sensitivity; K/S labels are only stratification covariates because slipknots require subchain fingerprinting.",
        "backbones": str(a.backbones), "backbones_sha256": sha256(a.backbones),
        "labels": str(a.labels), "labels_sha256": sha256(a.labels),
        "reduced_provenance": str(a.reduced_provenance),
        "records": records,
    }
    for label in sorted(set(label_map.values())):
        subset = [r for r in records if r["knotprot_KS"] == label]
        result.setdefault("summary_by_KnotProt_KS", {})[label] = {
            "n_chains": len(subset),
            "mean_nontrivial_probability": {
                n: float(np.mean([r["nontrivial_probability"][n] for r in subset]))
                for n in ("full", "stride_only", "stride_cap")}}
    # A resolved whole-chain Alexander result is a K/S proxy: a nontrivial type
    # maps to K and the unknot to S. Unresolved TMC/Unknown cases are excluded.
    keys_in_order = [r["key"] for r in records]
    y_all = np.asarray([label_map[k] for k in keys_in_order])
    class_names = np.asarray(["K", "S"])
    for rep in ("full", "stride_only", "stride_cap"):
        outputs = [r["topoly_alexander_deterministic_mass_center"][rep] for r in records]
        resolved = np.asarray([not any(k in {"TMC", "Unknown"} for k in output)
                               for output in outputs])
        probs = np.asarray([[0., 1.] if "0_1" in output else [1., 0.]
                            for output in outputs])
        baseline = {
            "interpretation": "MASS_CENTER-closed whole-chain Alexander type: recognized nontrivial knot -> K, unknot -> S; unresolved outcome excluded",
            "n_total_chains": len(records), "n_resolved_chains": int(resolved.sum()),
            "unresolved_chain_ids": [keys_in_order[i] for i in np.flatnonzero(~resolved)],
            "metrics_all_resolved_chains": metrics(y_all[resolved], probs[resolved], class_names, a.seed)}
        if a.evaluation_split:
            split = json.loads(a.evaluation_split.read_text())
            test = set(split["test_chain_ids"])
            ids = np.asarray([i for i, k in enumerate(keys_in_order) if k in test and resolved[i]])
            baseline["n_resolved_untouched_holdout_chains"] = int(len(ids))
            baseline["metrics_resolved_untouched_holdout"] = (
                metrics(y_all[ids], probs[ids], class_names, a.seed) if len(ids) else None)
        result.setdefault("deterministic_mass_center_invariant_baseline", {})[rep] = baseline
    for n in ("stride_only", "stride_cap"):
        result.setdefault("aggregate", {})[n] = {
            "n_chains": len(records),
            "stochastic_30_closure_histogram_not_equal_count": sum(r["stochastic_30_closure_histogram_not_equal_vs_full"][n] for r in records),
            "deterministic_mass_center_changed_count": sum(r["deterministic_mass_center_changed_vs_full"][n] for r in records),
            "cap_vs_stride_only_deterministic_changed_count": sum(r["deterministic_cap_changed_vs_stride_only"] for r in records),
            "mean_abs_nontrivial_probability_delta": float(np.mean([
                abs(r["nontrivial_probability_delta_vs_full"][n]) for r in records]))}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"Wrote topology sensitivity audit to {a.out}")


if __name__ == "__main__":
    main()
