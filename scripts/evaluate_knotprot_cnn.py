#!/usr/bin/env python3
"""Train a compact view CNN on the frozen chain split and aggregate by chain."""
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


class ProjectionCNN(nn.Module):
    def __init__(self, nclasses: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 24, 3, padding=1), nn.BatchNorm2d(24), nn.GELU(),
            nn.MaxPool2d(2), nn.Conv2d(24, 48, 3, padding=1),
            nn.BatchNorm2d(48), nn.GELU(), nn.MaxPool2d(2),
            nn.Conv2d(48, 96, 3, padding=1), nn.BatchNorm2d(96), nn.GELU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(.25),
            nn.Linear(96, nclasses))

    def forward(self, x):
        return self.net(x)


def sha_ids(ids):
    return hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--split", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--seed", type=int, default=23)
    a = p.parse_args()
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_data(a.data)
    split = json.loads(a.split.read_text())
    train_keys = np.asarray(split["train_chain_ids"], dtype=str)
    test_keys = np.asarray(split["test_chain_ids"], dtype=str)
    if set(train_keys) & set(test_keys):
        raise ValueError("Frozen CNN train/test chain IDs overlap")
    classes = np.asarray(split["class_names"], dtype=str)
    cidx = {c: i for i, c in enumerate(classes)}
    y = np.asarray([cidx[v] for v in data["labels"]], dtype=np.int64)
    train_rows = np.flatnonzero(np.isin(data["keys"], train_keys))
    test_rows = np.flatnonzero(np.isin(data["keys"], test_keys))
    images = data["images"].astype(np.float32)
    if images.max() > 1:
        images /= 255.0
    x = torch.from_numpy(images[:, None, :, :])
    if x.shape[-2:] != (32, 32):
        x = F.interpolate(x, size=(32, 32), mode="area")
    y_t = torch.from_numpy(y)
    train_dl = DataLoader(TensorDataset(x[train_rows], y_t[train_rows]),
                          batch_size=a.batch_size, shuffle=True,
                          generator=torch.Generator().manual_seed(a.seed))
    model = ProjectionCNN(len(classes)).to(device)
    counts = np.bincount(y[train_rows], minlength=len(classes))
    weights = torch.tensor(len(train_rows) / (len(classes) * counts), dtype=torch.float32, device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(weight=weights)
    history = []
    for epoch in range(a.epochs):
        model.train(); loss_sum = 0.; seen = 0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward(); opt.step()
            loss_sum += loss.item() * len(yb); seen += len(yb)
        history.append(loss_sum / seen)
        print(f"epoch {epoch+1}/{a.epochs} train_loss={history[-1]:.5f}", flush=True)
    model.eval()
    with torch.no_grad():
        probs = torch.softmax(model(x[test_rows].to(device)), dim=1).cpu().numpy()
    y_true, y_chain, p_chain = [], [], []
    for key in test_keys:
        ix = np.flatnonzero(data["keys"][test_rows] == key)
        p_chain.append(probs[ix].mean(axis=0))
        y_chain.append(np.unique(data["labels"][data["keys"] == key])[0])
        y_true.append(key)
    result = {
        "model": "ProjectionCNN (image-only, per-view cross-entropy; mean probability over views)",
        "input_resolution": [32, 32],
        "seed": a.seed, "epochs_fixed_before_holdout_evaluation": a.epochs,
        "development_dataset": str(a.data), "development_sha256": hashlib.sha256(a.data.read_bytes()).hexdigest(),
        "split_source": str(a.split), "train_chain_ids": sorted(train_keys.tolist()),
        "test_chain_ids": sorted(test_keys.tolist()), "train_chain_ids_sha256": sha_ids(train_keys),
        "test_chain_ids_sha256": sha_ids(test_keys), "views_per_chain": int(len(data["labels"]) / len(np.unique(data["keys"]))),
        "device": str(device), "training_loss_by_epoch": history,
        "holdout_metrics": metrics(np.asarray(y_chain), np.asarray(p_chain), classes, a.seed),
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {a.out}; test chains={len(test_keys)}")


if __name__ == "__main__":
    main()
