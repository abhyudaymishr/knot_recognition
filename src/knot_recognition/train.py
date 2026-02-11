import argparse
import os
import random
import time
from dataclasses import asdict, dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import DataLoader

from .dataset import KnotImageDataset
from .model import get_resnet
from .utils import ensure_dir


@dataclass
class TrainConfig:
    data_dir: str
    outdir: str = "./checkpoints"
    epochs: int = 20
    batch: int = 32
    lr: float = 1e-3
    image_size: int = 224
    num_workers: int = 4
    seed: Optional[int] = None
    val_split: float = 0.2
    split_strategy: str = "random"  # "random" or "stratified"


class Trainer:
    def __init__(self, config: TrainConfig):
        self.config = config
        if config.seed is not None:
            _seed_everything(config.seed)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.transforms = T.Compose(
            [
                T.Resize((config.image_size, config.image_size)),
                T.RandomHorizontalFlip(),
                T.RandomRotation(10),
                T.ToTensor(),
            ]
        )
        self.dataset = KnotImageDataset(config.data_dir, transform=self.transforms)
        self.dl_train, self.dl_val = self._build_loaders()

        num_classes = len(self.dataset.class_to_idx)
        self.model = get_resnet(num_classes, pretrained=True).to(self.device)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=config.lr)

    def _build_loaders(self):
        idxs = list(range(len(self.dataset)))
        train_idx, val_idx = _split_indices(
            idxs,
            labels=[sample[1] for sample in self.dataset.samples],
            val_split=self.config.val_split,
            strategy=self.config.split_strategy,
            seed=self.config.seed,
        )
        train_ds = torch.utils.data.Subset(self.dataset, train_idx)
        val_ds = torch.utils.data.Subset(self.dataset, val_idx)
        dl_train = DataLoader(
            train_ds,
            batch_size=self.config.batch,
            shuffle=True,
            num_workers=self.config.num_workers,
        )
        dl_val = DataLoader(
            val_ds,
            batch_size=self.config.batch,
            shuffle=False,
            num_workers=self.config.num_workers,
        )
        return dl_train, dl_val

    def train_epoch(self):
        self.model.train()
        total, acc = 0, 0
        loss_sum = 0.0
        for x, y, _, _ in self.dl_train:
            x = x.to(self.device).float()
            y = y.to(self.device)
            out = self.model(x)
            loss = F.cross_entropy(out, y)
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            loss_sum += loss.item() * x.size(0)
            preds = out.argmax(1)
            acc += (preds == y).sum().item()
            total += x.size(0)
        return loss_sum / total, acc / total

    def val_epoch(self):
        self.model.eval()
        total, acc = 0, 0
        loss_sum = 0.0
        with torch.no_grad():
            for x, y, _, _ in self.dl_val:
                x = x.to(self.device).float()
                y = y.to(self.device)
                out = self.model(x)
                loss = F.cross_entropy(out, y)
                loss_sum += loss.item() * x.size(0)
                preds = out.argmax(1)
                acc += (preds == y).sum().item()
                total += x.size(0)
        return loss_sum / total, acc / total

    def fit(self):
        ensure_dir(self.config.outdir)
        best_acc = 0.0
        for epoch in range(self.config.epochs):
            t0 = time.time()
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.val_epoch()
            print(
                f"Epoch {epoch+1}/{self.config.epochs}  "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.4f}  "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}  "
                f"time={time.time()-t0:.1f}s"
            )
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(
                    {
                        "model_state": self.model.state_dict(),
                        "class_to_idx": self.dataset.class_to_idx,
                        "optimizer_state": self.opt.state_dict(),
                        "epoch": epoch + 1,
                        "best_acc": best_acc,
                        "train_loss": train_loss,
                        "train_acc": train_acc,
                        "val_loss": val_loss,
                        "val_acc": val_acc,
                        "train_config": asdict(self.config),
                    },
                    os.path.join(self.config.outdir, "best.pth"),
                )


# Backwards-compatible functional API
def train_epoch(model, dl, opt, device):
    model.train()
    total, acc = 0, 0
    loss_sum = 0.0
    for x, y, _, _ in dl:
        x = x.to(device).float()
        y = y.to(device)
        out = model(x)
        loss = F.cross_entropy(out, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        loss_sum += loss.item() * x.size(0)
        preds = out.argmax(1)
        acc += (preds == y).sum().item()
        total += x.size(0)
    return loss_sum / total, acc / total


def val_epoch(model, dl, device):
    model.eval()
    total, acc = 0, 0
    loss_sum = 0.0
    with torch.no_grad():
        for x, y, _, _ in dl:
            x = x.to(device).float()
            y = y.to(device)
            out = model(x)
            loss = F.cross_entropy(out, y)
            loss_sum += loss.item() * x.size(0)
            preds = out.argmax(1)
            acc += (preds == y).sum().item()
            total += x.size(0)
    return loss_sum / total, acc / total


def train(config: TrainConfig):
    Trainer(config).fit()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--outdir', default='./checkpoints')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--val-split', type=float, default=0.2)
    parser.add_argument('--split-strategy', default="random", choices=["random", "stratified"])
    args = parser.parse_args()

    config = TrainConfig(
        data_dir=args.data_dir,
        outdir=args.outdir,
        epochs=args.epochs,
        batch=args.batch,
        lr=args.lr,
        seed=args.seed,
        val_split=args.val_split,
        split_strategy=args.split_strategy,
    )
    Trainer(config).fit()


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _split_indices(
    idxs: List[int],
    labels: List[int],
    val_split: float,
    strategy: str,
    seed: Optional[int],
) -> Tuple[List[int], List[int]]:
    val_split = min(max(val_split, 0.0), 0.9)
    if len(idxs) == 0:
        return [], []
    rng = np.random.RandomState(seed) if seed is not None else np.random

    if strategy == "stratified":
        by_label = {}
        for idx, label in zip(idxs, labels):
            by_label.setdefault(label, []).append(idx)
        train_idx, val_idx = [], []
        for label, group in by_label.items():
            group = list(group)
            rng.shuffle(group)
            split = int(round((1.0 - val_split) * len(group)))
            split = max(1, split) if len(group) > 1 else len(group)
            train_idx.extend(group[:split])
            val_idx.extend(group[split:])
        rng.shuffle(train_idx)
        rng.shuffle(val_idx)
        return train_idx, val_idx

    idxs = list(idxs)
    rng.shuffle(idxs)
    split_idx = int(round((1.0 - val_split) * len(idxs)))
    split_idx = max(1, split_idx) if len(idxs) > 1 else len(idxs)
    return idxs[:split_idx], idxs[split_idx:]


if __name__ == '__main__':
    main()
