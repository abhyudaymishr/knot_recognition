"""
train_nanotransformer.py - Local Training Script for Knot NanoTransformer.

Trains the Vision NanoTransformer across all 250 prime knot classes
(from 0_unknot to 10_165) using raw knot diagrams and augmentations.
Computes prototype embeddings for fast topological similarity retrieval
and serializes the checkpoint for local traverser use and Hugging Face deployment.
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
import torchvision.transforms as T
from PIL import Image

# Ensure src is on sys.path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from knot_recognition.nanotransformer import (
    NanoTransformer,
    NanoTransformerConfig,
    load_census_metadata_table,
    save_nanotransformer_checkpoint,
)
from knot_recognition.utils import imread_any


class KnotImageClassificationDataset(Dataset):
    """Loads all image files across 250 knot subdirectories in raw_knot/."""

    def __init__(
        self,
        root_dir: Union[str, Path],
        transform: Optional[Any] = None,
        extensions: Tuple[str, ...] = (".png", ".jpg", ".jpeg", ".gif", ".bmp"),
    ):
        self.root = Path(root_dir)
        self.transform = transform
        self.extensions = tuple(ext.lower() for ext in extensions)

        # Sort classes deterministically
        subdirs = [d.name for d in self.root.iterdir() if d.is_dir() and not d.name.startswith(".")]
        self.classes = sorted(subdirs)
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.idx_to_class = {i: c for i, c in enumerate(self.classes)}

        self.samples: List[Tuple[Path, int, str]] = []
        for c in self.classes:
            folder = self.root / c
            idx = self.class_to_idx[c]
            for f in folder.iterdir():
                if f.is_file() and f.suffix.lower() in self.extensions and not f.name.startswith("."):
                    self.samples.append((f, idx, c))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        path, label, label_name = self.samples[idx]
        img_np = imread_any(str(path))
        pil_img = Image.fromarray(img_np).convert("RGB")
        if self.transform:
            tensor = self.transform(pil_img)
        else:
            tensor = T.functional.to_tensor(pil_img)
        return tensor, label, label_name


def build_transforms(image_size: int = 128) -> Tuple[Any, Any]:
    train_transform = T.Compose([
        T.Resize((image_size, image_size)),
        T.RandomAffine(
            degrees=180,
            translate=(0.06, 0.06),
            scale=(0.92, 1.08),
            interpolation=T.InterpolationMode.BILINEAR,
            fill=255,
        ),
        T.RandomHorizontalFlip(p=0.5),
        T.ColorJitter(brightness=0.15, contrast=0.15),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    eval_transform = T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    return train_transform, eval_transform


def resolve_device(requested: Optional[str] = None) -> torch.device:
    if requested:
        return torch.device(requested)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def compute_class_prototypes(
    model: NanoTransformer,
    dataset: KnotImageClassificationDataset,
    device: torch.device,
    batch_size: int = 32,
) -> torch.Tensor:
    """Computes mean normalized embedding vectors for each of the 250 classes."""
    model.eval()
    num_classes = len(dataset.classes)
    embed_dim = model.config.embed_dim

    class_embeddings: Dict[int, List[torch.Tensor]] = {i: [] for i in range(num_classes)}
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    with torch.inference_mode():
        for tensors, labels, _ in loader:
            tensors = tensors.to(device)
            feats = model.extract_embedding(tensors, normalize=True).cpu()
            for feat, label in zip(feats, labels):
                class_embeddings[int(label)].append(feat)

    prototypes = torch.zeros(num_classes, embed_dim)
    for c_idx in range(num_classes):
        feats_list = class_embeddings[c_idx]
        if feats_list:
            stacked = torch.stack(feats_list, dim=0)
            mean_feat = torch.mean(stacked, dim=0)
            prototypes[c_idx] = F.normalize(mean_feat, p=2, dim=-1)
        else:
            # Fallback random unit vector if empty
            prototypes[c_idx] = F.normalize(torch.randn(embed_dim), p=2, dim=-1)

    return prototypes


def train_model(
    data_dir: str = "raw_knot",
    mapping_csv: str = "mapping_generated.csv",
    census_json: str = "Knotgen/census_braids.json",
    out_checkpoint: str = "checkpoints/nanotransformer_knot250.pth",
    epochs: int = 25,
    batch_size: int = 32,
    lr: float = 1e-3,
    weight_decay: float = 1e-2,
    image_size: int = 128,
    seed: int = 42,
    device_name: Optional[str] = None,
) -> Tuple[NanoTransformer, Dict[str, Any]]:
    # Set seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = resolve_device(device_name)
    print(f"[*] Training NanoTransformer on device: {device}")

    train_tf, eval_tf = build_transforms(image_size=image_size)
    full_ds = KnotImageClassificationDataset(data_dir, transform=train_tf)
    eval_ds = KnotImageClassificationDataset(data_dir, transform=eval_tf)

    num_samples = len(full_ds)
    num_classes = len(full_ds.classes)
    print(f"[*] Loaded dataset: {num_samples} images across {num_classes} knot classes (0_unknot to 10_165).")

    # Stratified or randomized train/val split
    indices = list(range(num_samples))
    random.shuffle(indices)
    split = int(0.85 * num_samples)
    train_idx, val_idx = indices[:split], indices[split:]

    train_loader = DataLoader(
        Subset(full_ds, train_idx),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        Subset(eval_ds, val_idx),
        batch_size=batch_size,
        shuffle=False,
    )

    # Initialize NanoTransformer
    config = NanoTransformerConfig(
        image_size=image_size,
        patch_size=16,
        in_channels=3,
        embed_dim=128,
        depth=4,
        num_heads=4,
        mlp_ratio=2.0,
        num_classes=num_classes,
        dropout=0.1,
        attention_dropout=0.1,
    )
    model = NanoTransformer(config).to(device)
    model.set_classes(full_ds.classes)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    print(f"[*] Starting training for {epochs} epochs...")
    start_time = time.time()
    best_val_acc = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, total_correct, total_samples = 0.0, 0, 0

        for x, y, _ in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item() * x.size(0)
            preds = logits.argmax(dim=-1)
            total_correct += (preds == y).sum().item()
            total_samples += x.size(0)

        scheduler.step()
        train_acc = total_correct / max(1, total_samples)
        avg_train_loss = total_loss / max(1, total_samples)

        # Validation
        model.eval()
        val_correct, val_samples = 0, 0
        with torch.inference_mode():
            for x, y, _ in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                preds = logits.argmax(dim=-1)
                val_correct += (preds == y).sum().item()
                val_samples += x.size(0)

        val_acc = val_correct / max(1, val_samples)
        if val_acc >= best_val_acc:
            best_val_acc = val_acc

        if epoch % 5 == 0 or epoch == epochs or epoch == 1:
            print(f"    Epoch {epoch:2d}/{epochs:2d} | Train Loss: {avg_train_loss:.4f} | "
                  f"Train Acc: {train_acc*100:.1f}% | Val Acc: {val_acc*100:.1f}% | "
                  f"LR: {scheduler.get_last_lr()[0]:.6f}")

    elapsed = time.time() - start_time
    print(f"[*] Training finished in {elapsed:.2f}s! Best Val Accuracy: {best_val_acc*100:.1f}%")

    # Compute prototypes across all images for nearest-neighbor similarity
    print("[*] Computing prototype embeddings for all 250 classes...")
    prototypes = compute_class_prototypes(model, eval_ds, device=device)
    model.set_prototypes(prototypes)

    # Load and attach census metadata
    print("[*] Loading and attaching topological census metadata...")
    census_meta = load_census_metadata_table(
        mapping_csv=REPO_ROOT / mapping_csv,
        census_braids_json=REPO_ROOT / census_json,
    )
    model.census_metadata = census_meta

    # Save checkpoint
    out_path = REPO_ROOT / out_checkpoint
    save_nanotransformer_checkpoint(
        model,
        out_path,
        extra_meta={
            "epochs": epochs,
            "train_acc": float(train_acc),
            "val_acc": float(val_acc),
            "best_val_acc": float(best_val_acc),
            "training_time_seconds": float(elapsed),
        },
    )
    print(f"[+] Saved NanoTransformer checkpoint to {out_path} ({out_path.stat().st_size / 1024:.1f} KB)")

    return model, {
        "train_acc": train_acc,
        "val_acc": val_acc,
        "best_val_acc": best_val_acc,
        "checkpoint_path": str(out_path),
    }


def main():
    parser = argparse.ArgumentParser(description="Train NanoTransformer for 250 knot classes.")
    parser.add_argument("--data-dir", default="raw_knot", help="Path to raw knot dataset")
    parser.add_argument("--mapping-csv", default="mapping_generated.csv", help="Knot mapping CSV")
    parser.add_argument("--census-json", default="Knotgen/census_braids.json", help="Knot census braids JSON")
    parser.add_argument("--out", default="checkpoints/nanotransformer_knot250.pth", help="Output checkpoint path")
    parser.add_argument("--epochs", type=int, default=25, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--image-size", type=int, default=128, help="Input image size")
    parser.add_argument("--device", default=None, help="Device (cpu, mps, cuda)")
    args = parser.parse_args()

    train_model(
        data_dir=args.data_dir,
        mapping_csv=args.mapping_csv,
        census_json=args.census_json,
        out_checkpoint=args.out,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        image_size=args.image_size,
        device_name=args.device,
    )


if __name__ == "__main__":
    main()
