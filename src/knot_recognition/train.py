import argparse
import os
import time
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T
import torch.nn.functional as F
from .dataset import KnotImageDataset
from .model import get_resnet
from .utils import ensure_dir



def train_epoch(model, dl, opt, device):
    model.train()
    total, acc = 0,0
    loss_sum=0.0
    for x,y,_,_ in dl:
        x = x.to(device).float()
        y = y.to(device)
        out = model(x)
        loss = F.cross_entropy(out, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        loss_sum += loss.item()*x.size(0)
        preds = out.argmax(1)
        acc += (preds==y).sum().item()
        total += x.size(0)
    return loss_sum/total, acc/total


def val_epoch(model, dl, device):
    model.eval()
    total, acc=0,0
    loss_sum=0.0
    with torch.no_grad():
        for x,y,_,_ in dl:
            x = x.to(device).float()
            y = y.to(device)
            out = model(x)
            loss = F.cross_entropy(out, y)
            loss_sum += loss.item()*x.size(0)
            preds = out.argmax(1)
            acc += (preds==y).sum().item()
            total += x.size(0)
    return loss_sum/total, acc/total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--outdir', default='./checkpoints')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    args = parser.parse_args()
    ensure_dir(args.outdir)

    # transforms
    ts = T.Compose([
        T.Resize((224,224)),
        T.RandomHorizontalFlip(),
        T.RandomRotation(10),
        T.ToTensor(),
    ])
    dataset = KnotImageDataset(args.data_dir, transform=ts)
    # split
    n = len(dataset)
    si = int(0.8*n)
    idxs = list(range(n))
    train_ds = torch.utils.data.Subset(dataset, idxs[:si])
    val_ds = torch.utils.data.Subset(dataset, idxs[si:])
    dl_train = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=4)
    dl_val = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=4)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_classes = len(dataset.class_to_idx)
    model = get_resnet(num_classes, pretrained=True)
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_acc = 0.0
    for epoch in range(args.epochs):
        t0 = time.time()
        train_loss, train_acc = train_epoch(model, dl_train, opt, device)
        val_loss, val_acc = val_epoch(model, dl_val, device)
        print(f"Epoch {epoch+1}/{args.epochs}  train_loss={train_loss:.4f} train_acc={train_acc:.4f}  val_loss={val_loss:.4f} val_acc={val_acc:.4f}  time={time.time()-t0:.1f}s")
        # save best
        if val_acc>best_acc:
            best_acc=val_acc
            torch.save({'model_state':model.state_dict(), 'class_to_idx':dataset.class_to_idx}, os.path.join(args.outdir,'best.pth'))

if __name__=='__main__':
    main()
