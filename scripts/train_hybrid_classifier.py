import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


class HybridDataset(Dataset):
    def __init__(self, npz_path: str):
        data = np.load(npz_path, allow_pickle=True)
        self.images = data["images"].astype(np.float32) / 255.0
        self.gauss_feats = data["gauss_feats"].astype(np.float32)
        self.labels_raw = data["labels"]
        classes = sorted(set(self.labels_raw.tolist()))
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        self.labels = np.array([self.class_to_idx[c] for c in self.labels_raw], dtype=np.int64)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.images[idx][None, :, :]  # 1xHxW
        feat = self.gauss_feats[idx]
        label = self.labels[idx]
        return torch.from_numpy(img), torch.from_numpy(feat), label


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
    args = parser.parse_args()

    dataset = HybridDataset(args.data)
    n = len(dataset)
    split = int(0.8 * n)
    train_ds, val_ds = torch.utils.data.random_split(dataset, [split, n - split])
    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HybridModel(num_classes=len(dataset.class_to_idx), feat_dim=dataset.gauss_feats.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    history = []
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

        model.eval()
        vtotal, vcorrect, vloss = 0, 0, 0.0
        with torch.no_grad():
            for img, feat, y in val_dl:
                img, feat, y = img.to(device), feat.to(device), y.to(device)
                out = model(img, feat)
                loss = F.cross_entropy(out, y)
                vloss += loss.item() * y.size(0)
                pred = out.argmax(1)
                vcorrect += (pred == y).sum().item()
                vtotal += y.size(0)
        val_acc = vcorrect / max(1, vtotal)
        val_loss = vloss / max(1, vtotal)
        print(
            f"Epoch {epoch+1}/{args.epochs} "
            f"train_acc={train_acc:.3f} val_acc={val_acc:.3f}"
        )
        history.append((epoch + 1, train_acc, val_acc, train_loss, val_loss))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "class_to_idx": dataset.class_to_idx,
        },
        out_path,
    )
    print("saved", out_path)

    # write log
    log_path = Path(args.log_csv)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as f:
        f.write("epoch,train_acc,val_acc,train_loss,val_loss\n")
        for epoch, tacc, vacc, tloss, vloss in history:
            f.write(f"{epoch},{tacc:.6f},{vacc:.6f},{tloss:.6f},{vloss:.6f}\n")

    # plot
    plot_path = Path(args.plot)
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib.pyplot as plt

        epochs = [h[0] for h in history]
        train_accs = [h[1] for h in history]
        val_accs = [h[2] for h in history]
        train_losses = [h[3] for h in history]
        val_losses = [h[4] for h in history]
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

        loss_path = Path(args.loss_plot)
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

        # smoothed accuracy
        if args.smooth and len(epochs) >= args.smooth:
            w = args.smooth
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

    # per-class accuracy + confusion matrix (no sklearn)
    try:
        import matplotlib.pyplot as plt

        model.eval()
        all_preds = []
        all_true = []
        with torch.no_grad():
            for img, feat, y in val_dl:
                img, feat, y = img.to(device), feat.to(device), y.to(device)
                out = model(img, feat)
                pred = out.argmax(1)
                all_preds.extend(pred.cpu().numpy().tolist())
                all_true.extend(y.cpu().numpy().tolist())

        n_classes = len(dataset.class_to_idx)
        cm = np.zeros((n_classes, n_classes), dtype=np.int64)
        for t, p in zip(all_true, all_preds):
            cm[t, p] += 1

        cm_path = Path(args.confusion)
        plt.figure(figsize=(6, 6))
        plt.imshow(cm, cmap="Blues")
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.tight_layout()
        plt.savefig(cm_path)
        plt.close()

        per_class = np.diag(cm) / np.maximum(1, cm.sum(axis=1))
        pc_path = Path(args.per_class)
        plt.figure(figsize=(8, 4))
        plt.bar(range(len(per_class)), per_class)
        plt.title("Per-Class Accuracy")
        plt.xlabel("Class Index")
        plt.ylabel("Accuracy")
        plt.tight_layout()
        plt.savefig(pc_path)
        plt.close()
    except Exception:
        pass


if __name__ == "__main__":
    main()
