#!/usr/bin/env python3
"""Evaluate the projection CNN under frozen RCSB 30%-identity grouped folds."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from evaluate_knotprot import load_data, metrics
from evaluate_knotprot_cnn import ProjectionCNN


def digest(ids):
    return hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--cluster-audit", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--identity", default="30")
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--seed", type=int, default=23)
    a = p.parse_args()
    if not torch.cuda.is_available():
        torch.set_num_threads(min(4, torch.get_num_threads()))
    data = load_data(a.data)
    audit = json.loads(a.cluster_audit.read_text())
    folds = audit["audit"][a.identity]["cluster_grouped_cv"]["image"]["fold_chain_ids"]
    classes = np.unique(data["labels"])
    cidx = {c: i for i, c in enumerate(classes)}
    y = np.asarray([cidx[z] for z in data["labels"]], dtype=np.int64)
    images = data["images"].astype(np.float32)
    if images.max() > 1: images /= 255.0
    x = torch.from_numpy(images[:, None, :, :])
    if x.shape[-2:] != (32, 32):
        x = F.interpolate(x, size=(32, 32), mode="area")
    yt = torch.from_numpy(y)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    reports, split_records = [], []
    for fold in folds:
        seed = a.seed + int(fold["fold"])
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
        train_keys, valid_keys = np.asarray(fold["train"], dtype=str), np.asarray(fold["validation"], dtype=str)
        if set(train_keys) & set(valid_keys): raise ValueError("Sequence-cluster fold has chain overlap")
        tr = np.flatnonzero(np.isin(data["keys"], train_keys))
        va = np.flatnonzero(np.isin(data["keys"], valid_keys))
        counts = np.bincount(y[tr], minlength=len(classes))
        weights = torch.tensor(len(tr) / (len(classes) * counts), dtype=torch.float32, device=device)
        loader = DataLoader(TensorDataset(x[tr], yt[tr]), batch_size=a.batch_size, shuffle=True,
                            generator=torch.Generator().manual_seed(seed))
        model = ProjectionCNN(len(classes)).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        loss_fn = nn.CrossEntropyLoss(weight=weights)
        epoch_losses = []
        for epoch in range(a.epochs):
            model.train(); total_loss = seen = 0
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad(set_to_none=True)
                loss = loss_fn(model(xb), yb)
                loss.backward(); opt.step()
                total_loss += loss.item() * len(yb); seen += len(yb)
            epoch_losses.append(total_loss / seen)
        model.eval()
        with torch.no_grad(): probs = torch.softmax(model(x[va].to(device)), dim=1).cpu().numpy()
        pred_chain, truth_chain = [], []
        for key in valid_keys:
            local = np.flatnonzero(data["keys"][va] == key)
            pred_chain.append(probs[local].mean(axis=0))
            truth_chain.append(np.unique(data["labels"][data["keys"] == key])[0])
        report = metrics(np.asarray(truth_chain), np.asarray(pred_chain), classes, seed)
        reports.append(report)
        split_records.append({"fold": int(fold["fold"]), "train_chain_ids": sorted(train_keys.tolist()),
                              "validation_chain_ids": sorted(valid_keys.tolist()),
                              "train_chain_ids_sha256": digest(train_keys),
                              "validation_chain_ids_sha256": digest(valid_keys),
                              "training_loss_by_epoch": epoch_losses, "metrics": report})
        print(f"fold {fold['fold']}/{len(folds)} balanced_accuracy={report['balanced_accuracy']:.3f}", flush=True)
    result = {
        "model": "ProjectionCNN; image-only; average per-view probabilities to chain",
        "input_resolution": [32, 32],
        "grouping": f"RCSB {a.identity}% sequence-identity groups",
        "seed": a.seed, "epochs_fixed": a.epochs, "device": str(device),
        "development_npz": str(a.data), "development_sha256": hashlib.sha256(a.data.read_bytes()).hexdigest(),
        "cluster_audit": str(a.cluster_audit), "folds": split_records,
        "mean_balanced_accuracy": float(np.mean([r["balanced_accuracy"] for r in reports])),
        "fold_sd_balanced_accuracy": float(np.std([r["balanced_accuracy"] for r in reports], ddof=1)),
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"Wrote grouped CNN evaluation to {a.out}")


if __name__ == "__main__": main()
