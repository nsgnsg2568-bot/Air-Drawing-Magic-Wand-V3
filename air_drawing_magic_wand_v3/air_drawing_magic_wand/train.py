"""Train the sketch recogniser on Google's QuickDraw dataset.

    python train.py                  # full: ~75 objects, 5000 imgs each, 6 epochs
    python train.py --preset fast    # ~25 common objects, quick to train (good first run)
    python train.py --classes cat dog house sun tree
    python train.py --samples 12000 --epochs 10   # better accuracy, slower

First run needs internet (downloads ~28 MB per class into ./data).
Output: models/quickdraw_cnn.pt  (weights + class names)
Classes that don't exist in QuickDraw are skipped with a warning.
"""
import argparse
import math
import os
import time
import urllib.parse
import urllib.request

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model import QuickDrawCNN

URL = "https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/{}.npy"

FAST = ["apple", "airplane", "bicycle", "cat", "circle", "clock", "cloud", "cup", "dog", "door", "eye",
        "fish", "flower", "house", "key", "moon", "mountain", "smiley face", "square", "star", "sun",
        "tree", "triangle", "umbrella", "car"]

FULL = FAST + [
    "alarm clock", "angel", "ant", "axe", "banana", "baseball bat", "basketball", "bed", "bee", "bird", "book",
    "bread", "bridge", "broom", "bus", "butterfly", "cactus", "cake", "camera", "candle", "carrot", "chair",
    "cow", "crown", "diamond", "dolphin", "donut", "duck", "elephant", "envelope", "guitar", "hammer", "hat",
    "headphones", "ice cream", "knife", "ladder", "leaf", "light bulb", "lightning", "mushroom", "pencil",
    "pizza", "rainbow", "scissors", "snowman", "spider", "sword", "tent", "wine glass",
]


def download(name, data_dir):
    os.makedirs(data_dir, exist_ok=True)
    path = os.path.join(data_dir, name.replace(" ", "_") + ".npy")
    if not os.path.exists(path):
        print(f"  downloading '{name}' ...")
        try:
            urllib.request.urlretrieve(URL.format(urllib.parse.quote(name)), path)
        except Exception as e:
            print(f"  !! skipped '{name}': {e}")
            if os.path.exists(path):
                os.remove(path)
            return None
    return path


def augment(x):
    """x: (B,1,28,28) float in [0,1]. Per-sample random rotation/scale/shift + random thickening."""
    B = x.size(0)
    ang = (torch.rand(B, device=x.device) - 0.5) * 2 * math.radians(12)
    sc = 1 + (torch.rand(B, device=x.device) - 0.5) * 0.24
    tx = (torch.rand(B, device=x.device) - 0.5) * 0.2
    ty = (torch.rand(B, device=x.device) - 0.5) * 0.2
    cos, sin = torch.cos(ang) / sc, torch.sin(ang) / sc
    theta = torch.stack([torch.stack([cos, -sin, tx], 1), torch.stack([sin, cos, ty], 1)], 1)
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    x = F.grid_sample(x, grid, align_corners=False, padding_mode="zeros")
    thick = F.max_pool2d(x, 3, 1, 1)
    m = (torch.rand(B, 1, 1, 1, device=x.device) < 0.35).float()
    return m * thick + (1 - m) * x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", choices=["fast", "full"], default="full")
    ap.add_argument("--classes", nargs="+")
    ap.add_argument("--samples", type=int, default=None, help="images per class")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out", default="models/quickdraw_cnn.pt")
    args = ap.parse_args()
    fast = args.preset == "fast"
    names = args.classes or (FAST if fast else FULL)
    samples = args.samples or (4000 if fast else 5000)
    epochs = args.epochs or (5 if fast else 6)

    device = ("cuda" if torch.cuda.is_available()
              else "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
              else "cpu")
    print("Device:", device)

    xs, ys, classes = [], [], []
    for name in names:
        path = download(name, args.data_dir)
        if path is None:
            continue
        arr = np.load(path, mmap_mode="r")[:samples]
        xs.append(np.array(arr, dtype=np.uint8))
        ys.append(np.full(len(arr), len(classes), dtype=np.int64))
        classes.append(name)
    if len(classes) < 2:
        raise SystemExit("Need at least 2 classes (check your internet connection).")
    print(f"Loaded {len(classes)} classes.")

    X = torch.from_numpy(np.concatenate(xs)).view(-1, 1, 28, 28)   # uint8 keeps RAM small
    Y = torch.from_numpy(np.concatenate(ys))
    perm = torch.randperm(len(X))
    X, Y = X[perm], Y[perm]
    n_val = max(2000, len(X) // 20)
    Xv, Yv, Xt, Yt = X[:n_val], Y[:n_val], X[n_val:], Y[n_val:]
    print(f"Train: {len(Xt)}  Val: {len(Xv)}")

    model = QuickDrawCNN(len(classes)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    steps = epochs * math.ceil(len(Xt) / args.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.05)

    def evaluate():
        model.eval()
        preds = []
        with torch.no_grad():
            for i in range(0, len(Xv), 1024):
                preds.append(model(Xv[i:i + 1024].float().div(255).to(device)).argmax(1).cpu())
        return torch.cat(preds)

    for epoch in range(1, epochs + 1):
        model.train()
        idx = torch.randperm(len(Xt))
        total, t0 = 0.0, time.time()
        for i in range(0, len(Xt), args.batch):
            b = idx[i:i + args.batch]
            xb = augment(Xt[b].float().div(255).to(device))
            yb = Yt[b].to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            sched.step()
            total += loss.item() * len(b)
        acc = (evaluate() == Yv).float().mean().item()
        print(f"epoch {epoch}/{epochs}  loss {total/len(Xt):.3f}  val acc {acc:.1%}  ({time.time()-t0:.0f}s)")

    pred = evaluate()
    print("\nWeakest classes (validation accuracy):")
    per = [((pred[Yv == c] == c).float().mean().item(), classes[c]) for c in range(len(classes)) if (Yv == c).any()]
    for a, n in sorted(per)[:8]:
        print(f"  {n:<14} {a:.0%}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save({"state_dict": model.cpu().state_dict(), "classes": classes}, args.out)
    print("\nSaved", args.out, "- now run: python main.py")


if __name__ == "__main__":
    main()
