#!/usr/bin/env python3
"""Compare the cap and stride settings on fixed KnotProt chain labels.

This audits representation and prediction sensitivity. Because KnotProt's
full-chain K/S annotation is carried unchanged into each derived dataset, this
does not claim that topology was recomputed after subsampling.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from evaluate_knotprot import (chain_predictions, image_features, load_data,
                               metrics, model_factories, row_features)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--development", type=Path, required=True)
    ap.add_argument("--external-reduced", type=Path, required=True)
    ap.add_argument("--external-full", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=23)
    args = ap.parse_args()

    dev = load_data(args.development)
    reduced = load_data(args.external_reduced)
    full = load_data(args.external_full)
    keys_reduced = np.unique(reduced["keys"])
    keys_full = np.unique(full["keys"])
    if not np.array_equal(keys_reduced, keys_full):
        raise ValueError("Reduced and full cohorts have different chain IDs")
    for key in keys_reduced:
        labels_a = np.unique(reduced["labels"][reduced["keys"] == key])
        labels_b = np.unique(full["labels"][full["keys"] == key])
        if not np.array_equal(labels_a, labels_b):
            raise ValueError(f"Source labels differ for {key}")
    classes = np.unique(dev["labels"])
    if not np.array_equal(classes, np.unique(reduced["labels"])):
        raise ValueError("Reduced external cohort class schema differs")

    reduced_prov = json.loads(Path(str(args.external_reduced) + ".provenance.json").read_text())
    full_prov = json.loads(Path(str(args.external_full) + ".provenance.json").read_text())
    reduced_counts = reduced_prov["chain_sampling"]
    full_counts = full_prov["chain_sampling"]
    affected = sorted(k for k in keys_reduced
                          if reduced_counts[k]["processed_point_count"]
                          < full_counts[k]["processed_point_count"])

    result = {
        "question": "Does the preprocessing variant change the derived input or model predictions?",
        "development_source": str(args.development),
        "external_reduced": str(args.external_reduced),
        "external_full": str(args.external_full),
        "reduced_parameters": reduced_prov["parameters"],
        "full_parameters": full_prov["parameters"],
        "viewpoints": int(len(reduced["labels"]) // len(keys_reduced)),
        "chains_changed_by_preprocessing": affected,
        "n_chains_changed_by_preprocessing": len(affected),
        "source_knotprot_labels_recomputed_after_subsampling": False,
        "label_check": "K/S labels are identical because the original KnotProt chain annotation is carried through; this is not an independent topology check.",
        "metrics": {},
        "per_modality_prediction_flips_on_changed_chains": {},
        "projection_feature_drift_on_changed_chains": None,
    }
    for modality in ("image", "projection_features", "hybrid"):
        train_x = row_features(dev, modality)
        reduced_x = row_features(reduced, modality)
        full_x = row_features(full, modality)
        model = model_factories(args.seed)["logistic"]().fit(train_x, dev["labels"])
        for name, data, features in (("reduced", reduced, reduced_x), ("full", full, full_x)):
            y, p, _ = chain_predictions(model, features, data, keys_reduced, classes,
                                        None, "uniform", args.seed)
            result["metrics"][f"{modality}/{name}"] = metrics(y, p, classes, args.seed)
            result.setdefault("predictions", {}).setdefault(modality, {})[name] = {
                str(k): c for k, c in zip(keys_reduced, classes[p.argmax(axis=1)])
            }
            result.setdefault("probabilities", {}).setdefault(modality, {})[name] = p.tolist()
    changed = np.array([k in affected for k in keys_reduced])
    for modality in ("image", "projection_features", "hybrid"):
        a = np.asarray([result["predictions"][modality]["reduced"][str(k)] for k in keys_reduced])
        b = np.asarray([result["predictions"][modality]["full"][str(k)] for k in keys_reduced])
        result["per_modality_prediction_flips_on_changed_chains"][modality] = {
            "count": int(np.sum((a != b) & changed)),
            "total_changed": int(changed.sum()),
        }
    if affected:
        diffs = []
        for key in affected:
            ia, ib = np.flatnonzero(reduced["keys"] == key), np.flatnonzero(full["keys"] == key)
            a, b = reduced["gauss_feats"][ia], full["gauss_feats"][ib]
            cos = np.sum(a * b, axis=1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12)
            diffs.append({"key": key, "mean_feature_l1": float(np.abs(a - b).mean()),
                          "mean_feature_cosine": float(cos.mean()),
                          "reduced_points": reduced_counts[key]["processed_point_count"],
                          "uncapped_points": full_counts[key]["processed_point_count"]})
        result["projection_feature_drift_on_changed_chains"] = diffs
    result.pop("predictions")
    result.pop("probabilities")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"Wrote sampling sensitivity audit to {args.out}")
    print(f"Preprocessing changed {len(affected)}/{len(keys_reduced)} external chains")


if __name__ == "__main__":
    main()
