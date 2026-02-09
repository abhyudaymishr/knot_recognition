import argparse
import os
import time
from dataclasses import dataclass
from typing import Optional

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


class Trainer:
    def __init__(self, config: TrainConfig):
        self.config = config
        if config.seed is not None:
            torch.manual_seed(config.seed)
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
        n = len(self.dataset)
        split_idx = int(0.8 * n)
        idxs = list(range(n))
        train_ds = torch.utils.data.Subset(self.dataset, idxs[:split_idx])
        val_ds = torch.utils.data.Subset(self.dataset, idxs[split_idx:])
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
                    {"model_state": self.model.state_dict(), "class_to_idx": self.dataset.class_to_idx},
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
    args = parser.parse_args()

    config = TrainConfig(
        data_dir=args.data_dir,
        outdir=args.outdir,
        epochs=args.epochs,
        batch=args.batch,
        lr=args.lr,
        seed=args.seed,
    )
    Trainer(config).fit()


if __name__ == '__main__':
    main()
