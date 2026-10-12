"""
train_reidemeister_contrastive.py - Train Reidemeister Contrastive Representation Model.

CLI to train a vision model (or hybrid model) under the certified Reidemeister contrastive
loss on census knot diagrams and their ambient isotopy orbits.
"""

from __future__ import annotations

import sys
import argparse
from pathlib import Path
import time
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from torch.utils.data import DataLoader

from src.knot_recognition.reidemeister_contrastive import (
    ReidemeisterAugmentedDataset,
    ReidemeisterContrastiveLoss,
    ReidemeisterVisionModel,
    reidemeister_collate_fn,
)


def train_reidemeister_contrastive(
    census_path: Optional[str] = None,
    max_crossing: Optional[int] = 8,
    epochs: int = 10,
    batch_size: int = 16,
    lr: float = 3e-4,
    margin: float = 1.0,
    lambda_neg: float = 0.5,
    out_checkpoint: str = "checkpoints/reidemeister_contrastive.pth",
    device_str: Optional[str] = None,
) -> None:
    if device_str is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(device_str)

    print(f"Initializing Reidemeister contrastive training on {device}...")
    dataset = ReidemeisterAugmentedDataset(
        census_path=census_path,
        max_crossing_filter=max_crossing,
        img_size=(224, 224),
        num_reidemeister_moves=3,
        samples_per_knot=4,
    )
    print(f"Loaded {len(dataset.knots)} knot classes (total {len(dataset)} augmented diagram pairs).")

    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, drop_last=True, collate_fn=reidemeister_collate_fn
    )

    model = ReidemeisterVisionModel(embed_dim=128, proj_dim=128).to(device)
    loss_fn = ReidemeisterContrastiveLoss(margin=margin, lambda_neg=lambda_neg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    start_time = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, total_pos, total_neg = 0.0, 0.0, 0.0
        batches = 0

        for batch in dataloader:
            x_anchor = batch["image_anchor"].to(device)
            x_pos = batch["image_positive"].to(device)

            optimizer.zero_grad()
            z_anchor, z_pos = model(x_anchor, x_pos)
            loss, stats = loss_fn(z_anchor, z_pos)

            loss.backward()
            optimizer.step()

            total_loss += stats["loss_total"]
            total_pos += stats["loss_pos"]
            total_neg += stats["loss_neg"]
            batches += 1

        avg_loss = total_loss / max(1, batches)
        avg_pos = total_pos / max(1, batches)
        avg_neg = total_neg / max(1, batches)
        print(
            f"Epoch {epoch:02d}/{epochs:02d} | Loss: {avg_loss:.4f} "
            f"(Pos: {avg_pos:.4f}, Neg: {avg_neg:.4f})"
        )

    out_p = Path(out_checkpoint)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "max_crossing": max_crossing,
        "epochs": epochs,
    }, out_p)
    print(f"Training completed in {time.time() - start_time:.1f}s. Saved checkpoint to {out_p}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Reidemeister contrastive model.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--max-crossing", type=int, default=8, help="Train only on <= N crossings (e.g. 8)")
    parser.add_argument("--out", default="checkpoints/reidemeister_contrastive.pth")
    args = parser.parse_args()

    train_reidemeister_contrastive(
        max_crossing=args.max_crossing,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        out_checkpoint=args.out,
    )
