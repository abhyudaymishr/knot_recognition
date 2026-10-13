#!/usr/bin/env python3
"""Leak-resistant, chain-level evaluation for KnotProt projection datasets.

The script evaluates image, projection-feature, and hybrid inputs with
non-neural baselines. Views are aggregated into one probability vector per
protein chain before any metric is calculated. A group-stratified development
cross-validation is followed by a single untouched chain holdout. An optional
external NPZ is evaluated only after fitting on the complete development set.

Expected NPZ keys: images, gauss_feats, labels, keys. `keys` must identify the
protein chain for every view. Input NPZ files are treated as immutable sources;
their SHA-256 hashes, split IDs, parameters, and metrics are written to JSON.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, log_loss,
                             recall_score)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_data(path: Path) -> dict[str, Any]:
    raw = np.load(path, allow_pickle=False)
    needed = {"images", "gauss_feats", "labels", "keys"}
    missing = needed.difference(raw.files)
    if missing:
        raise ValueError(f"{path}: missing required arrays {sorted(missing)}")
    data = {k: raw[k] for k in needed}
    n = len(data["labels"])
    if any(len(v) != n for v in data.values()):
        raise ValueError(f"{path}: arrays do not have matching sample counts")
    data["labels"] = data["labels"].astype(str)
    data["keys"] = data["keys"].astype(str)
    if np.any(data["keys"] == ""):
        raise ValueError(f"{path}: empty chain key; refusing sample-level split")
    # A chain must have exactly one class label across all of its views.
    for key in np.unique(data["keys"]):
        labels = np.unique(data["labels"][data["keys"] == key])
        if len(labels) != 1:
            raise ValueError(f"Chain {key!r} has conflicting labels: {labels.tolist()}")
    return data


def add_manifest(data: dict[str, Any], path: Path) -> None:
    if not path.exists():
        return
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != len(data["keys"]):
        raise ValueError(f"Manifest has {len(rows)} rows but NPZ has {len(data['keys'])} samples")
    manifest_keys = np.asarray([r["key"] for r in rows], dtype=str)
    if not np.array_equal(manifest_keys, data["keys"]):
        raise ValueError("Manifest key order does not match NPZ key order; refusing to infer view IDs")
    if "view_id" in rows[0]:
        data["view_id"] = np.asarray([int(r["view_id"]) for r in rows], dtype=int)


def chain_table(data: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    keys = np.unique(data["keys"])
    labels = np.array([np.unique(data["labels"][data["keys"] == k])[0] for k in keys])
    return keys, labels


def group_holdout(keys: np.ndarray, labels: np.ndarray, fraction: float, seed: int):
    """Create one stratified chain holdout; enforce a class in both partitions."""
    splitter = StratifiedGroupKFold(n_splits=max(2, round(1 / fraction)), shuffle=True,
                                    random_state=seed)
    dummy = np.zeros((len(keys), 1))
    candidates = list(splitter.split(dummy, labels, groups=keys))
    # Select a fold with all classes represented on both sides where possible.
    classes = set(labels.tolist())
    valid = [(tr, te) for tr, te in candidates
             if set(labels[tr]) == classes and set(labels[te]) == classes]
    if not valid:
        raise ValueError("Cannot make a chain-level holdout containing every class; "
                         "collect more independent chains per class.")
    return valid[0]


def image_features(images: np.ndarray) -> np.ndarray:
    # Compact, fixed feature map for scalable non-neural image baselines.
    import cv2
    out = [cv2.resize(im.astype(np.float32), (32, 32), interpolation=cv2.INTER_AREA).ravel()
           for im in images]
    return np.asarray(out, dtype=np.float32) / (255.0 if np.max(images) > 1 else 1.0)


def model_factories(seed: int, trees: int = 200):
    return {
        "logistic": lambda: make_pipeline(SimpleImputer(), StandardScaler(),
             LogisticRegression(max_iter=2500, class_weight="balanced", random_state=seed)),
        "rbf_svm": lambda: make_pipeline(SimpleImputer(), StandardScaler(),
             SVC(C=1.0, kernel="rbf", probability=True, class_weight="balanced", random_state=seed)),
        "extra_trees": lambda: make_pipeline(SimpleImputer(),
             ExtraTreesClassifier(n_estimators=trees, class_weight="balanced", n_jobs=-1,
                                  random_state=seed, min_samples_leaf=2)),
    }


def row_features(data: dict[str, Any], modality: str) -> np.ndarray:
    img = image_features(data["images"])
    feat = data["gauss_feats"].astype(np.float32)
    if modality == "image":
        return img
    if modality == "projection_features":
        return feat
    if modality == "hybrid":
        return np.concatenate([img, feat], axis=1)
    if modality == "shape_3d":
        return data["shape_3d_features"]
    raise ValueError(modality)


def shape_features(coords: np.ndarray) -> np.ndarray:
    """Rotation/translation-invariant whole-chain geometry, not a knot invariant."""
    xyz = np.asarray(coords, dtype=np.float64)
    if len(xyz) < 4:
        return np.full(15, np.nan, dtype=np.float32)
    centered = xyz - xyz.mean(axis=0, keepdims=True)
    eig = np.maximum(np.linalg.eigvalsh(centered.T @ centered / len(xyz)), 0.0)
    rg = float(np.sqrt(eig.sum()))
    contour = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
    angles = []
    turns = []
    bonds = np.diff(xyz, axis=0)
    for i in range(len(bonds) - 1):
        denom = np.linalg.norm(bonds[i]) * np.linalg.norm(bonds[i + 1]) + 1e-12
        angles.append(np.arccos(np.clip(np.dot(bonds[i], bonds[i + 1]) / denom, -1, 1)))
    for i in range(len(bonds) - 2):
        a, b, c = bonds[i], bonds[i + 1], bonds[i + 2]
        n1, n2 = np.cross(a, b), np.cross(b, c)
        denom = np.linalg.norm(n1) * np.linalg.norm(n2) + 1e-12
        turns.append(np.arccos(np.clip(np.dot(n1, n2) / denom, -1, 1)))
    shape = eig / (eig.sum() + 1e-12)
    values = [np.log1p(len(xyz)), np.linalg.norm(xyz[-1] - xyz[0]) / (contour.sum() + 1e-12),
              rg / np.sqrt(len(xyz)), rg / (contour.mean() * np.sqrt(len(xyz)) + 1e-12),
              *shape.tolist(), float(contour.mean()), float(contour.std()),
              float(np.mean(angles)), float(np.std(angles)),
              float(np.mean(turns)), float(np.std(turns)),
              float(np.quantile(contour, .1)), float(np.quantile(contour, .9))]
    return np.asarray(values, dtype=np.float32)


def add_shape_features(data: dict[str, Any], backbones_path: Path) -> None:
    backbones = np.load(backbones_path, allow_pickle=False)
    by_key = {key: shape_features(backbones[key]) for key in np.unique(data["keys"])
              if key in backbones.files}
    missing = sorted(set(data["keys"].tolist()) - set(by_key))
    if missing:
        raise ValueError(f"Missing 3D coordinates for {len(missing)} chains: {missing[:5]}")
    data["shape_3d_features"] = np.stack([by_key[key] for key in data["keys"]])


def choose_views(data: dict[str, Any], keys: np.ndarray, count: int | None, mode: str,
                 seed: int) -> np.ndarray:
    keep = []
    rng = np.random.default_rng(seed)
    for key in keys:
        inds = np.flatnonzero(data["keys"] == key)
        if "view_id" in data:
            order = np.argsort(data["view_id"][inds])
            inds = inds[order]
        if count is not None and count < len(inds):
            if mode == "uniform":
                positions = np.linspace(0, len(inds) - 1, count, dtype=int)
            elif mode == "random":
                positions = np.sort(rng.choice(len(inds), count, replace=False))
            else:
                raise ValueError(mode)
            inds = inds[positions]
        keep.extend(inds.tolist())
    return np.asarray(keep, dtype=int)


def chain_predictions(model, x: np.ndarray, data: dict[str, Any], keys: np.ndarray,
                      classes: np.ndarray, count: int | None, view_mode: str, seed: int):
    ids = choose_views(data, keys, count, view_mode, seed)
    probs = model.predict_proba(x[ids])
    model_classes = model.classes_.astype(str)
    # Expand model probabilities to the global class order for fold-safe scoring.
    aligned = np.zeros((len(ids), len(classes)), dtype=float)
    for j, c in enumerate(model_classes):
        aligned[:, np.flatnonzero(classes == c)[0]] = probs[:, j]
    pred, truth, conf = [], [], []
    for key in keys:
        loc = np.flatnonzero(data["keys"][ids] == key)
        p = aligned[loc].mean(axis=0)
        total = p.sum()
        if not np.isfinite(total) or total <= 0:
            raise ValueError(f"Invalid aggregated probabilities for chain {key!r}")
        p = p / total
        pred.append(p)
        truth.append(np.unique(data["labels"][data["keys"] == key])[0])
        conf.append(p)
    return np.asarray(truth), np.asarray(conf), classes


def ece(y: np.ndarray, p: np.ndarray, classes: np.ndarray, bins: int = 10) -> float:
    idx = np.array([int(np.flatnonzero(classes == v)[0]) for v in y])
    confidence = p.max(axis=1)
    correct = (p.argmax(axis=1) == idx).astype(float)
    edges = np.linspace(0, 1, bins + 1)
    return float(sum((m := (confidence > edges[i]) & (confidence <= edges[i + 1])).mean()
                     * abs(correct[m].mean() - confidence[m].mean())
                     for i in range(bins) if np.any((confidence > edges[i]) &
                                                    (confidence <= edges[i + 1]))))


def metrics(y: np.ndarray, p: np.ndarray, classes: np.ndarray, seed: int,
            bootstraps: int = 2000) -> dict[str, Any]:
    yi = np.array([int(np.flatnonzero(classes == v)[0]) for v in y])
    pi = p.argmax(axis=1)
    recalls = recall_score(yi, pi, labels=np.arange(len(classes)), average=None, zero_division=0)
    result = {
        "n_chains": int(len(y)), "accuracy": float(accuracy_score(yi, pi)),
        "macro_f1": float(f1_score(yi, pi, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(yi, pi)),
        "per_class_recall": {str(c): float(r) for c, r in zip(classes, recalls)},
        "confusion_matrix": confusion_matrix(yi, pi, labels=np.arange(len(classes))).tolist(),
        "log_loss": float(log_loss(yi, p, labels=np.arange(len(classes)))),
        "brier_score": float(np.mean(np.sum((p - np.eye(len(classes))[yi]) ** 2, axis=1))),
        "ece_10_bins": ece(y, p, classes),
    }
    rng = np.random.default_rng(seed)
    vals = {k: [] for k in ("accuracy", "macro_f1", "balanced_accuracy")}
    by_class = [np.flatnonzero(yi == c) for c in range(len(classes))]
    for _ in range(bootstraps):
        # Stratified chain bootstrap keeps every class represented in every
        # replicate and holds the observed class counts fixed.
        sample = np.concatenate([rng.choice(ix, size=len(ix), replace=True)
                                 for ix in by_class if len(ix)])
        vals["accuracy"].append(accuracy_score(yi[sample], pi[sample]))
        vals["macro_f1"].append(f1_score(yi[sample], pi[sample], average="macro", zero_division=0))
        vals["balanced_accuracy"].append(balanced_accuracy_score(yi[sample], pi[sample]))
    result["chain_bootstrap_95ci"] = {
        k: [float(np.quantile(v, .025)), float(np.quantile(v, .975))] for k, v in vals.items()
    }
    return result


def run_cv(data, x, dev_keys, dev_labels, classes, folds, seed, view_counts, view_modes, trees):
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    y_by_key = dict(zip(dev_keys, dev_labels))
    rows = []
    for fold, (tr, va) in enumerate(splitter.split(dev_keys, dev_labels, groups=dev_keys), 1):
        train_keys, valid_keys = dev_keys[tr], dev_keys[va]
        tr_rows = np.flatnonzero(np.isin(data["keys"], train_keys))
        for modality in x:
            features = x[modality]
            for model_name, factory in model_factories(seed + fold, trees).items():
                model = factory()
                model.fit(features[tr_rows], data["labels"][tr_rows])
                for mode in view_modes:
                    for nviews in view_counts:
                        truth, prob, _ = chain_predictions(model, features, data, valid_keys,
                            classes, nviews, mode, seed + fold)
                        rows.append({"fold": fold, "modality": modality, "model": model_name,
                                     "view_mode": mode, "views": nviews,
                                     "train_chain_ids": sorted(train_keys.tolist()),
                                     "validation_chain_ids": sorted(valid_keys.tolist()),
                                     "train_chain_ids_sha256": hashlib.sha256(
                                         "\n".join(sorted(train_keys)).encode()).hexdigest(),
                                     "validation_chain_ids_sha256": hashlib.sha256(
                                         "\n".join(sorted(valid_keys)).encode()).hexdigest(),
                                     "metrics": metrics(truth, prob, classes, seed + fold,
                                                        bootstraps=500)})
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=Path("data/knotprot/hybrid_dataset.npz"))
    ap.add_argument("--manifest", type=Path, help="Optional view manifest; defaults to hybrid_manifest.csv beside --data")
    ap.add_argument("--external", type=Path, help="Independent external NPZ, same schema")
    ap.add_argument("--backbones", type=Path, help="Optional chain-coordinate NPZ for a 3D shape baseline")
    ap.add_argument("--external-backbones", type=Path, help="External chain-coordinate NPZ for 3D shape baseline")
    ap.add_argument("--source-manifest", type=Path, help="Acquisition snapshot manifest for external provenance")
    ap.add_argument("--out", type=Path, default=Path("results/knotprot_evaluation.json"))
    ap.add_argument("--seed", type=int, default=23)
    ap.add_argument("--test-fraction", type=float, default=.20)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--trees", type=int, default=200)
    ap.add_argument("--views", default="1,2,4,8,16,all")
    ap.add_argument("--view-modes", default="uniform,random")
    args = ap.parse_args()
    if not args.data.exists():
        raise FileNotFoundError(f"Development dataset missing: {args.data}. Rebuild from versioned chain/label records and PDB files first.")
    data = load_data(args.data)
    if args.backbones:
        add_shape_features(data, args.backbones)
    manifest_path = args.manifest or args.data.parent / "hybrid_manifest.csv"
    add_manifest(data, manifest_path)
    keys, labels = chain_table(data)
    classes = np.unique(labels)
    train_idx, test_idx = group_holdout(keys, labels, args.test_fraction, args.seed)
    train_keys, test_keys = keys[train_idx], keys[test_idx]
    if set(train_keys) & set(test_keys):
        raise AssertionError("Chain leakage between train and test")
    view_counts = [None if v.strip() == "all" else int(v) for v in args.views.split(",")]
    view_modes = [m.strip() for m in args.view_modes.split(",")]
    modalities = ["image", "projection_features", "hybrid"]
    if args.backbones:
        modalities.append("shape_3d")
    all_x = {m: row_features(data, m) for m in modalities}
    dev_labels = labels[train_idx]
    cv = run_cv(data, all_x, train_keys, dev_labels, classes, args.folds, args.seed,
                view_counts, view_modes, args.trees)
    acquisition = (json.loads(args.source_manifest.read_text())
                   if args.source_manifest and args.source_manifest.exists() else {})
    summary = {
        "protocol": "chain-level grouped CV on development chains; untouched chain holdout; optional external set",
        "seed": args.seed, "class_names": classes.tolist(),
        "task_definition": "KnotProt whole-chain status: K=knot, S=slipknot",
        "development_npz": str(args.data), "development_sha256": sha256(args.data),
        "dataset_provenance": (str(Path(str(args.data) + ".provenance.json"))
                               if Path(str(args.data) + ".provenance.json").exists() else None),
        "dataset_provenance_sha256": (sha256(Path(str(args.data) + ".provenance.json"))
                                       if Path(str(args.data) + ".provenance.json").exists() else None),
        "manifest": str(manifest_path) if manifest_path.exists() else None,
        "manifest_sha256": sha256(manifest_path) if manifest_path.exists() else None,
        "train_chain_ids": sorted(train_keys.tolist()), "test_chain_ids": sorted(test_keys.tolist()),
        "train_chain_ids_sha256": hashlib.sha256("\n".join(sorted(train_keys)).encode()).hexdigest(),
        "test_chain_ids_sha256": hashlib.sha256("\n".join(sorted(test_keys)).encode()).hexdigest(),
        "backbones": str(args.backbones) if args.backbones else None,
        "backbones_sha256": sha256(args.backbones) if args.backbones else None,
        "source_manifest": str(args.source_manifest) if args.source_manifest else None,
        "source_manifest_sha256": sha256(args.source_manifest) if args.source_manifest else None,
        "source_pdb_accession_overlap": acquisition.get("pdb_accession_overlap"),
        "source_pdb_accession_overlap_count": len(acquisition.get("pdb_accession_overlap", [])),
        "cv": cv, "holdout": {}, "external": None,
        "limitations": ["All supplied modalities are non-neural baselines; compare against a CNN trained with the same chain partitions before a paper claim.",
                        "The projection-event feature is heuristic and is not a verified knot invariant.",
                        "Bootstrap intervals resample chains; with few chains per class, uncertainty remains large."],
    }
    # One final refit on all development chains, one untouched holdout evaluation.
    for modality, features in all_x.items():
        tr_rows = np.flatnonzero(np.isin(data["keys"], train_keys))
        for model_name, factory in model_factories(args.seed, args.trees).items():
            model = factory().fit(features[tr_rows], data["labels"][tr_rows])
            for mode in view_modes:
                for nviews in view_counts:
                    y, p, _ = chain_predictions(model, features, data, test_keys, classes,
                                                nviews, mode, args.seed)
                    summary["holdout"][f"{modality}/{model_name}/{mode}/{nviews or 'all'}"] = metrics(y, p, classes, args.seed)
    if args.external:
        ext = load_data(args.external)
        if args.backbones:
            if not args.external_backbones:
                raise ValueError("--external-backbones is required when --backbones enables the 3D baseline")
            add_shape_features(ext, args.external_backbones)
        ext_keys, ext_labels = chain_table(ext)
        overlap = set(keys.tolist()) & set(ext_keys.tolist())
        if overlap:
            raise ValueError(f"External set overlaps development chain IDs (examples: {sorted(overlap)[:5]})")
        ext_classes = np.unique(ext_labels)
        if not np.array_equal(classes, ext_classes):
            raise ValueError(f"External class schema {ext_classes.tolist()} differs from development {classes.tolist()}")
        summary["external"] = {"path": str(args.external), "sha256": sha256(args.external), "chain_ids": sorted(ext_keys.tolist()), "metrics": {}}
        summary["external"]["backbones"] = str(args.external_backbones) if args.external_backbones else None
        summary["external"]["backbones_sha256"] = sha256(args.external_backbones) if args.external_backbones else None
        for modality in all_x:
            tr_rows = np.arange(len(data["labels"]))
            ext_x = row_features(ext, modality)
            for model_name, factory in model_factories(args.seed, args.trees).items():
                model = factory().fit(all_x[modality][tr_rows], data["labels"])
                for mode in view_modes:
                    y, p, _ = chain_predictions(model, ext_x, ext, ext_keys, classes,
                                                None, mode, args.seed)
                    summary["external"]["metrics"][f"{modality}/{model_name}/{mode}/all"] = metrics(y, p, classes, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"Wrote chain-level evaluation to {args.out}")
    print(f"Development chains: {len(train_keys)} | untouched test chains: {len(test_keys)} | classes: {classes.tolist()}")


if __name__ == "__main__":
    main()
