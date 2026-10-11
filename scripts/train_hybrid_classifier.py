import argparse
import csv
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


class HybridDataset(Dataset):
    def __init__(self, npz_path: str, manifest_path: Optional[str] = None):
        data = np.load(npz_path, allow_pickle=True)
        self.images = data["images"].astype(np.float32) / 255.0
        self.gauss_feats = data["gauss_feats"].astype(np.float32)
        self.labels_raw = data["labels"]
        classes = sorted(set(self.labels_raw.tolist()))
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        self.labels = np.array([self.class_to_idx[c] for c in self.labels_raw], dtype=np.int64)
        self.keys = None
        if "keys" in data:
            self.keys = data["keys"].astype(str)
        elif manifest_path:
            self.keys = self._load_manifest_keys(manifest_path)
        if self.keys is not None and len(self.keys) != len(self.labels):
            raise ValueError("Manifest keys length does not match dataset length")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.images[idx][None, :, :]  # 1xHxW
        feat = self.gauss_feats[idx]
        label = self.labels[idx]
        return torch.from_numpy(img), torch.from_numpy(feat), label

    @staticmethod
    def _load_manifest_keys(manifest_path: str) -> np.ndarray:
        with open(manifest_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            keys = [row["key"] for row in reader]
        return np.array(keys, dtype=str)


class HybridModel(nn.Module):
    def __init__(self, num_classes: int, feat_dim: int):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.feat_head = nn.Sequential(
            nn.Linear(feat_dim, 64),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(64 + 64, num_classes)

    def forward(self, img, feat):
        x = self.cnn(img).view(img.size(0), -1)
        f = self.feat_head(feat)
        z = torch.cat([x, f], dim=1)
        return self.classifier(z)


def group_stratified_split(keys: np.ndarray, labels: np.ndarray, train_ratio: float, seed: int):
    rng = np.random.default_rng(seed)
    group_to_indices: dict[str, list[int]] = {}
    group_to_label: dict[str, int] = {}
    for idx, (key, label) in enumerate(zip(keys, labels)):
        group_to_indices.setdefault(str(key), []).append(idx)
        group_to_label.setdefault(str(key), int(label))

    label_to_groups: dict[int, list[str]] = {}
    for group, label in group_to_label.items():
        label_to_groups.setdefault(label, []).append(group)

    train_indices: list[int] = []
    val_indices: list[int] = []
    for label, groups in label_to_groups.items():
        groups = sorted(groups)
        rng.shuffle(groups)
        n_groups = len(groups)
        if n_groups == 1:
            n_train = 1
        else:
            n_train = int(round(train_ratio * n_groups))
            n_train = max(1, min(n_groups - 1, n_train))
        train_groups = set(groups[:n_train])
        for group in groups:
            if group in train_groups:
                train_indices.extend(group_to_indices[group])
            else:
                val_indices.extend(group_to_indices[group])

    train_indices = np.array(sorted(train_indices), dtype=np.int64)
    val_indices = np.array(sorted(val_indices), dtype=np.int64)
    return train_indices, val_indices


def evaluate(model: nn.Module, val_dl: DataLoader, device: torch.device):
    model.eval()
    vtotal, vcorrect, vloss = 0, 0, 0.0
    all_preds = []
    all_true = []
    with torch.no_grad():
        for img, feat, y in val_dl:
            img, feat, y = img.to(device), feat.to(device), y.to(device)
            out = model(img, feat)
            loss = F.cross_entropy(out, y)
            vloss += loss.item() * y.size(0)
            pred = out.argmax(1)
            vcorrect += (pred == y).sum().item()
            vtotal += y.size(0)
            all_preds.extend(pred.cpu().numpy().tolist())
            all_true.extend(y.cpu().numpy().tolist())
    val_acc = vcorrect / max(1, vtotal)
    val_loss = vloss / max(1, vtotal)
    return val_acc, val_loss, all_preds, all_true


def _save_evaluation_plots(all_preds, all_true, class_to_idx, confusion_path_str, per_class_path_str):
    try:
        import matplotlib.pyplot as plt

        n_classes = len(class_to_idx)
        cm = np.zeros((n_classes, n_classes), dtype=np.int64)
        for t, p in zip(all_true, all_preds):
            cm[t, p] += 1

        cm_path = Path(confusion_path_str)
        cm_path.parent.mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(6, 6))
        plt.imshow(cm, cmap="Blues")
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        idx_to_class = {i: c for c, i in class_to_idx.items()}
        classes = [idx_to_class[i] for i in range(n_classes)]
        plt.xticks(range(n_classes), classes, rotation=45)
        plt.yticks(range(n_classes), classes)
        plt.tight_layout()
        plt.savefig(cm_path)
        plt.close()

        per_class = np.diag(cm) / np.maximum(1, cm.sum(axis=1))
        pc_path = Path(per_class_path_str)
        pc_path.parent.mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(8, 4))
        plt.bar(range(len(per_class)), per_class)
        plt.title("Per-Class Accuracy")
        plt.xlabel("Class")
        plt.ylabel("Accuracy")
        plt.xticks(range(n_classes), classes, rotation=45)
        plt.ylim(0.0, 1.05)
        plt.tight_layout()
        plt.savefig(pc_path)
        plt.close()
    except Exception:
        pass


def _save_training_plots(history, plot_path_str, loss_plot_path_str, smooth: int = 3):
    try:
        import matplotlib.pyplot as plt

        epochs = [h[0] for h in history]
        train_accs = [h[1] for h in history]
        val_accs = [h[2] for h in history]
        train_losses = [h[3] for h in history]
        val_losses = [h[4] for h in history]

        plot_path = Path(plot_path_str)
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(6, 4))
        plt.plot(epochs, train_accs, label="train")
        plt.plot(epochs, val_accs, label="val")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Hybrid Classifier Training")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()

        loss_path = Path(loss_plot_path_str)
        loss_path.parent.mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(6, 4))
        plt.plot(epochs, train_losses, label="train")
        plt.plot(epochs, val_losses, label="val")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Hybrid Classifier Loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(loss_path)
        plt.close()

        if smooth and len(epochs) >= smooth:
            w = smooth
            smooth_train = [sum(train_accs[i - w + 1:i + 1]) / w for i in range(w - 1, len(train_accs))]
            smooth_val = [sum(val_accs[i - w + 1:i + 1]) / w for i in range(w - 1, len(val_accs))]
            smooth_epochs = epochs[w - 1:]
            smooth_path = plot_path.with_name("hybrid_training_smooth.png")
            plt.figure(figsize=(6, 4))
            plt.plot(smooth_epochs, smooth_train, label="train")
            plt.plot(smooth_epochs, smooth_val, label="val")
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy")
            plt.title("Hybrid Training (Smoothed)")
            plt.legend()
            plt.tight_layout()
            plt.savefig(smooth_path)
            plt.close()
    except Exception:
        pass


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/knotprot/hybrid_dataset.npz")
    parser.add_argument("--out", default="checkpoints/hybrid_classifier.pth")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--log-csv", default="results/hybrid_training.csv")
    parser.add_argument("--plot", default="results/figures/hybrid_training.png")
    parser.add_argument("--loss-plot", default="results/figures/hybrid_loss.png")
    parser.add_argument("--confusion", default="results/figures/hybrid_confusion.png")
    parser.add_argument("--per-class", default="results/figures/hybrid_per_class.png")
    parser.add_argument("--smooth", type=int, default=3)
    parser.add_argument("--manifest", default="data/knotprot/hybrid_manifest.csv")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--eval-only", action="store_true", help="Skip training; evaluate existing checkpoint from --out")
    args = parser.parse_args()

    dataset = HybridDataset(args.data, args.manifest)
    n = len(dataset)
    if dataset.keys is not None:
        train_idx, val_idx = group_stratified_split(dataset.keys, dataset.labels, 0.8, args.seed)
        train_ds = torch.utils.data.Subset(dataset, train_idx)
        val_ds = torch.utils.data.Subset(dataset, val_idx)
        print(
            f"group split: train_groups={len(set(dataset.keys[train_idx]))} "
            f"val_groups={len(set(dataset.keys[val_idx]))}"
        )
    else:
        split = int(0.8 * n)
        generator = torch.Generator().manual_seed(args.seed)
        indices = torch.randperm(n, generator=generator).tolist()
        train_idx, val_idx = indices[:split], indices[split:]
        train_ds = torch.utils.data.Subset(dataset, train_idx)
        val_ds = torch.utils.data.Subset(dataset, val_idx)
    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HybridModel(num_classes=len(dataset.class_to_idx), feat_dim=dataset.gauss_feats.shape[1]).to(device)

    if args.eval_only:
        ckpt_path = Path(args.out)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint '{ckpt_path}' not found for evaluation.")
        raw = torch.load(ckpt_path, map_location=device)
        if isinstance(raw, dict) and "model_state" in raw:
            state_dict = raw["model_state"]
            saved_class_to_idx = raw.get("class_to_idx")
        else:
            state_dict = raw
            saved_class_to_idx = None

        if saved_class_to_idx is not None and saved_class_to_idx != dataset.class_to_idx:
            raise ValueError(
                f"Checkpoint class mapping {saved_class_to_idx} does not match "
                f"dataset class mapping {dataset.class_to_idx}.\n"
                f"The checkpoint was trained on {len(saved_class_to_idx)} classes: {list(saved_class_to_idx.keys())},\n"
                f"while the dataset provides {len(dataset.class_to_idx)} classes: {list(dataset.class_to_idx.keys())}."
            )

        model.load_state_dict(state_dict)
        print(f"Loaded existing checkpoint from {ckpt_path} (eval-only mode).")
        if isinstance(raw, dict) and "val_indices" in raw and raw["val_indices"] is not None:
            saved_val_idx = raw["val_indices"]
            saved_seed = raw.get("seed")
            print(f"Checkpoint contains recorded validation split metadata (seed={saved_seed}, {len(saved_val_idx)} val samples).")
        else:
            print(
                f"Note: Checkpoint contains model_state and class_to_idx without split indices; "
                f"evaluating on current regenerated holdout split (recomputed with seed={args.seed})."
            )
        val_acc, val_loss, all_preds, all_true = evaluate(model, val_dl, device)
        print(f"[EVALUATION RESULT] Validation Samples: {len(val_ds)} | val_acc: {val_acc:.4f} | val_loss: {val_loss:.4f}")
        _save_evaluation_plots(all_preds, all_true, dataset.class_to_idx, args.confusion, args.per_class)
        return

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    history = []
    all_preds, all_true = [], []
    for epoch in range(args.epochs):
        model.train()
        total, correct, loss_sum = 0, 0, 0.0
        for img, feat, y in train_dl:
            img, feat, y = img.to(device), feat.to(device), y.to(device)
            out = model(img, feat)
            loss = F.cross_entropy(out, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            loss_sum += loss.item() * y.size(0)
            pred = out.argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)
        train_acc = correct / max(1, total)
        train_loss = loss_sum / max(1, total)

        val_acc, val_loss, all_preds, all_true = evaluate(model, val_dl, device)
        print(
            f"Epoch {epoch+1}/{args.epochs} "
            f"train_acc={train_acc:.4f} train_loss={train_loss:.4f} "
            f"val_acc={val_acc:.4f} val_loss={val_loss:.4f}"
        )
        history.append((epoch + 1, train_acc, val_acc, train_loss, val_loss))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_payload = {
        "model_state": model.state_dict(),
        "class_to_idx": dataset.class_to_idx,
        "seed": args.seed,
        "train_indices": train_idx.tolist() if hasattr(train_idx, "tolist") else list(train_idx),
        "val_indices": val_idx.tolist() if hasattr(val_idx, "tolist") else list(val_idx),
        "train_keys": sorted(list(set(dataset.keys[train_idx]))) if dataset.keys is not None else None,
        "val_keys": sorted(list(set(dataset.keys[val_idx]))) if dataset.keys is not None else None,
    }
    torch.save(checkpoint_payload, out_path)
    print("saved checkpoint to", out_path)

    # write log
    log_path = Path(args.log_csv)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as f:
        f.write("epoch,train_acc,val_acc,train_loss,val_loss\n")
        for epoch, tacc, vacc, tloss, vloss in history:
            f.write(f"{epoch},{tacc:.6f},{vacc:.6f},{tloss:.6f},{vloss:.6f}\n")

    _save_training_plots(history, args.plot, args.loss_plot, args.smooth)
    _save_evaluation_plots(all_preds, all_true, dataset.class_to_idx, args.confusion, args.per_class)



if __name__ == "__main__":
    main()
