#!/usr/bin/env python3
"""
train.py — Train RadarNet-1D on preprocessed RadioML data
==========================================================
Trains the ResNet-1D model, logs metrics, saves the best checkpoint,
and generates evaluation artefacts (confusion matrix, SNR robustness curve).

Usage:
    python train.py --data-dir ../data --epochs 60 --batch-size 256
"""

import argparse
import json
import os
import time

import matplotlib
matplotlib.use("Agg")                    # headless backend for servers
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    confusion_matrix, classification_report,
    ConfusionMatrixDisplay, precision_recall_fscore_support,
)

from model import RadarNet1D, count_parameters

# ── Defaults ─────────────────────────────────────────────────────────────────
DEFENSE_LABELS = ["Normal", "Jammer", "Spoofer", "Interference"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ── Helpers ──────────────────────────────────────────────────────────────────
def load_npy(data_dir: str):
    """Load preprocessed .npy files and convert to PyTorch tensors."""
    splits = {}
    for split in ("train", "val", "test"):
        X = np.load(os.path.join(data_dir, f"X_{split}.npy"))
        y = np.load(os.path.join(data_dir, f"y_{split}.npy"))
        # RadioML shape: (N, 128, 2) → PyTorch conv: (N, 2, 128)
        X = np.transpose(X, (0, 2, 1)).astype(np.float32)
        y = y.astype(np.int64)
        splits[split] = (torch.from_numpy(X), torch.from_numpy(y))
    return splits


def make_loader(X, y, batch_size, shuffle=True):
    return DataLoader(
        TensorDataset(X, y),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=(DEVICE == "cuda"),
    )


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * xb.size(0)
        correct += (logits.argmax(1) == yb).sum().item()
        total += xb.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss = criterion(logits, yb)
        total_loss += loss.item() * xb.size(0)
        preds = logits.argmax(1)
        correct += (preds == yb).sum().item()
        total += xb.size(0)
        all_preds.append(preds.cpu())
        all_labels.append(yb.cpu())
    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    return total_loss / total, correct / total, all_preds, all_labels


def plot_confusion_matrix(y_true, y_pred, save_path):
    """Generate and save a confusion matrix figure."""
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=DEFENSE_LABELS)
    fig, ax = plt.subplots(figsize=(7, 6))
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    ax.set_title("RadarNet-1D Confusion Matrix (Test Set)")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Confusion matrix saved → {save_path}")


def plot_training_curves(history, save_path):
    """Plot train/val loss and accuracy curves."""
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(epochs, history["train_loss"], label="Train")
    ax1.plot(epochs, history["val_loss"], label="Val")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss Curves")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, history["train_acc"], label="Train")
    ax2.plot(epochs, history["val_acc"], label="Val")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Accuracy Curves")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.suptitle("RadarNet-1D Training Progress", fontsize=14)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Training curves saved → {save_path}")


def plot_snr_robustness(model, hdf5_path, device, save_path):
    """
    (Optional) If the raw HDF5 is available, plot accuracy vs SNR.
    Falls back gracefully if file not found.
    """
    try:
        import h5py
        from preprocess import MOD_CLASSES, CLASS_MAP, TIME_STEPS
    except ImportError:
        print("  ⚠ Skipping SNR robustness plot (missing deps)")
        return

    if not os.path.isfile(hdf5_path):
        print(f"  ⚠ Skipping SNR robustness plot ({hdf5_path} not found)")
        return

    print("  Generating SNR robustness curve …")
    with h5py.File(hdf5_path, "r") as f:
        X_all = f["X"][:, :TIME_STEPS, :]     # (N, 128, 2)
        Y_oh  = f["Y"][:]
        Z_snr = f["Z"][:, 0]

    raw_idx = np.argmax(Y_oh, axis=1)
    y_all = np.array([CLASS_MAP[MOD_CLASSES[i]] for i in raw_idx], dtype=np.int64)

    # Normalise
    max_amp = np.max(np.abs(X_all), axis=(1, 2), keepdims=True) + 1e-8
    X_all = X_all / max_amp

    snr_values = sorted(set(Z_snr.tolist()))
    accuracies = []

    model.eval()
    for snr in snr_values:
        mask = Z_snr == snr
        X_s = np.transpose(X_all[mask], (0, 2, 1)).astype(np.float32)
        y_s = y_all[mask]
        if len(y_s) == 0:
            accuracies.append(0.0)
            continue
        with torch.no_grad():
            Xt = torch.from_numpy(X_s).to(device)
            preds = model(Xt).argmax(1).cpu().numpy()
        acc = (preds == y_s).mean()
        accuracies.append(acc)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(snr_values, accuracies, "o-", linewidth=2, markersize=4)
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Accuracy")
    ax.set_title("RadarNet-1D — Accuracy vs. SNR")
    ax.set_ylim([0.0, 1.05])
    ax.axhline(0.92, color="r", linestyle="--", alpha=0.5, label="92% target")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  SNR robustness curve saved → {save_path}")


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Train RadarNet-1D")
    parser.add_argument("--data-dir",   type=str, default=os.path.join("..", "data"))
    parser.add_argument("--out-dir",    type=str, default=os.path.join("..", "checkpoints"))
    parser.add_argument("--epochs",     type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience",   type=int, default=10,
                        help="Early stopping patience (epochs)")
    parser.add_argument("--hdf5",       type=str, default="",
                        help="Path to raw HDF5 for SNR robustness plot")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    fig_dir = os.path.join(args.out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    # ── Data ─────────────────────────────────────────────────────────
    print(f"Device: {DEVICE}")
    splits = load_npy(args.data_dir)
    train_loader = make_loader(*splits["train"], args.batch_size, shuffle=True)
    val_loader   = make_loader(*splits["val"],   args.batch_size, shuffle=False)
    test_loader  = make_loader(*splits["test"],  args.batch_size, shuffle=False)

    # ── Model ────────────────────────────────────────────────────────
    model = RadarNet1D(in_channels=2, num_classes=4).to(DEVICE)
    print(f"RadarNet-1D  |  Parameters: {count_parameters(model):,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    # ── Training loop ────────────────────────────────────────────────
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc = 0.0
    patience_counter = 0
    best_path = os.path.join(args.out_dir, "radarnet_best.pth")

    print(f"\n{'Epoch':>5}  {'TrLoss':>8}  {'TrAcc':>7}  {'VaLoss':>8}  "
          f"{'VaAcc':>7}  {'LR':>10}  {'Time':>6}")
    print("─" * 65)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, DEVICE
        )
        va_loss, va_acc, _, _ = evaluate(model, val_loader, criterion, DEVICE)
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(va_acc)

        elapsed = time.time() - t0
        lr = optimizer.param_groups[0]["lr"]
        print(f"{epoch:5d}  {tr_loss:8.4f}  {tr_acc:6.2%}  {va_loss:8.4f}  "
              f"{va_acc:6.2%}  {lr:10.2e}  {elapsed:5.1f}s")

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            torch.save(model.state_dict(), best_path)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\n⏹ Early stopping at epoch {epoch} "
                      f"(no improvement for {args.patience} epochs)")
                break

    # ── Evaluate best model on test set ──────────────────────────────
    print(f"\n✓ Best validation accuracy: {best_val_acc:.2%}")
    model.load_state_dict(torch.load(best_path, map_location=DEVICE, weights_only=True))
    te_loss, te_acc, te_preds, te_labels = evaluate(
        model, test_loader, criterion, DEVICE
    )
    print(f"  Test accuracy: {te_acc:.2%}  |  Test loss: {te_loss:.4f}\n")

    # Classification report
    report = classification_report(
        te_labels, te_preds, target_names=DEFENSE_LABELS, digits=4
    )
    print(report)

    # ── Save artefacts ───────────────────────────────────────────────
    plot_confusion_matrix(te_labels, te_preds,
                          os.path.join(fig_dir, "confusion_matrix.png"))
    plot_training_curves(history,
                         os.path.join(fig_dir, "training_curves.png"))

    if args.hdf5:
        plot_snr_robustness(model, args.hdf5, DEVICE,
                            os.path.join(fig_dir, "snr_robustness.png"))

    # Save metrics JSON
    p, r, f1, _ = precision_recall_fscore_support(
        te_labels, te_preds, average=None, labels=[0, 1, 2, 3]
    )
    metrics = {
        "best_val_acc": best_val_acc,
        "test_acc": te_acc,
        "test_loss": te_loss,
        "per_class": {
            DEFENSE_LABELS[i]: {"precision": float(p[i]),
                                "recall": float(r[i]),
                                "f1": float(f1[i])}
            for i in range(4)
        },
        "epochs_trained": len(history["train_loss"]),
        "total_params": count_parameters(model),
    }
    metrics_path = os.path.join(args.out_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Metrics saved → {metrics_path}")
    print(f"  Best model   → {best_path}")


if __name__ == "__main__":
    main()
