#!/usr/bin/env python3
"""
preprocess.py — RadioML 2018.01A Preprocessing Pipeline
========================================================
Loads the RadioML 2018.01A HDF5 dataset, slices I/Q samples to 128 time-steps,
maps 24 modulation types to 4 defense anomaly classes, filters by SNR >= 0 dB,
balances classes to 50K samples each, normalises amplitude, and splits into
train / val / test sets (70 / 15 / 15).

Output files (saved to ../data/):
    X_train.npy, y_train.npy   — 140 000 samples
    X_val.npy,   y_val.npy     —  30 000 samples
    X_test.npy,  y_test.npy    —  30 000 samples

Usage:
    python preprocess.py --hdf5 path/to/GOLD_XYZ_OSC.0001_1024.hdf5
"""

import argparse
import os
import sys

import h5py
import numpy as np
from sklearn.model_selection import train_test_split

# ── 24 RadioML modulation names (dataset-canonical order) ────────────────────
MOD_CLASSES = [
    "OOK", "4ASK", "8ASK", "BPSK", "QPSK", "8PSK", "16PSK", "32PSK",
    "16APSK", "32APSK", "64APSK", "128APSK", "16QAM", "32QAM", "64QAM",
    "128QAM", "256QAM", "AM-SSB-WC", "AM-SSB-SC", "AM-DSB-WC", "AM-DSB-SC",
    "FM", "GMSK", "OQPSK",
]

# ── 24-class → 4-class defense anomaly mapping ──────────────────────────────
# 0 = Normal    : legitimate continuous-wave / narrow-band waveforms
# 1 = Jammer    : wideband / high-entropy jamming-like signals
# 2 = Spoofer   : coherent phase-modulated signals mimicking radar returns
# 3 = Interference : multi-level amplitude/phase cross-band interference
CLASS_MAP = {
    "FM": 0, "GMSK": 0, "AM-DSB-WC": 0, "AM-DSB-SC": 0,
    "AM-SSB-WC": 0, "AM-SSB-SC": 0,
    "OOK": 1, "WBFM": 1, "64QAM": 1, "128QAM": 1, "256QAM": 1,
    "BPSK": 2, "QPSK": 2, "8PSK": 2, "16PSK": 2, "32PSK": 2, "OQPSK": 2,
    "16APSK": 3, "32APSK": 3, "64APSK": 3, "128APSK": 3,
    "16QAM": 3, "32QAM": 3, "8ASK": 3, "4ASK": 3,
}

DEFENSE_LABELS = ["Normal", "Jammer", "Spoofer", "Interference"]

# ── Configuration ────────────────────────────────────────────────────────────
TIME_STEPS        = 128        # Slice first N time-steps from 1024
SNR_THRESHOLD     = 0          # Keep only samples with SNR >= this (dB)
SAMPLES_PER_CLASS = 50_000     # Balance to this many per class
NUM_CLASSES       = 4
RANDOM_SEED       = 42


def load_dataset(hdf5_path: str):
    """Load raw arrays from RadioML 2018.01A HDF5 file."""
    print(f"[1/6] Loading dataset from {hdf5_path} …")
    with h5py.File(hdf5_path, "r") as f:
        X = f["X"][:]            # (2 555 904, 1024, 2)  float32 I/Q
        Y_onehot = f["Y"][:]     # (2 555 904, 24)       one-hot labels
        Z_snr = f["Z"][:, 0]    # (2 555 904,)          SNR in dB
    print(f"    Loaded X {X.shape}, Y {Y_onehot.shape}, Z {Z_snr.shape}")
    return X, Y_onehot, Z_snr


def map_labels(Y_onehot: np.ndarray) -> np.ndarray:
    """Convert 24-class one-hot labels to 4-class integer defense labels."""
    print("[2/6] Mapping 24 modulation classes → 4 defense anomaly classes …")
    raw_indices = np.argmax(Y_onehot, axis=1)
    defense_labels = np.array(
        [CLASS_MAP[MOD_CLASSES[idx]] for idx in raw_indices], dtype=np.int64
    )
    for c in range(NUM_CLASSES):
        count = np.sum(defense_labels == c)
        print(f"    Class {c} ({DEFENSE_LABELS[c]}): {count:,} samples")
    return defense_labels


def filter_snr(X, y, snr, threshold: int):
    """Keep only samples with SNR >= threshold."""
    print(f"[3/6] Filtering SNR >= {threshold} dB …")
    mask = snr >= threshold
    X_f, y_f, snr_f = X[mask], y[mask], snr[mask]
    print(f"    Retained {mask.sum():,} / {len(mask):,} samples "
          f"({100 * mask.mean():.1f}%)")
    return X_f, y_f, snr_f


def slice_and_normalise(X: np.ndarray, time_steps: int) -> np.ndarray:
    """Slice to first `time_steps` points and per-sample amplitude normalise."""
    print(f"[4/6] Slicing to {time_steps} time-steps and normalising …")
    X = X[:, :time_steps, :]                                      # (N, 128, 2)
    max_amp = np.max(np.abs(X), axis=(1, 2), keepdims=True) + 1e-8
    X = X / max_amp                                                # [-1, 1]
    return X


def balance_classes(X, y, samples_per_class: int, seed: int):
    """Undersample each class to exactly `samples_per_class` samples."""
    print(f"[5/6] Balancing to {samples_per_class:,} samples per class …")
    rng = np.random.default_rng(seed)
    indices = []
    for c in range(NUM_CLASSES):
        class_idx = np.where(y == c)[0]
        if len(class_idx) < samples_per_class:
            print(f"    ⚠ Class {c} has only {len(class_idx):,} samples — "
                  f"using all (< {samples_per_class:,})")
            chosen = class_idx
        else:
            chosen = rng.choice(class_idx, size=samples_per_class, replace=False)
        indices.append(chosen)
    indices = np.concatenate(indices)
    rng.shuffle(indices)
    print(f"    Total balanced samples: {len(indices):,}")
    return X[indices], y[indices]


def split_and_save(X, y, out_dir: str, seed: int):
    """Stratified split into train/val/test and save as .npy."""
    print("[6/6] Splitting 70/15/15 (stratified) and saving …")
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=seed
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=seed
    )

    os.makedirs(out_dir, exist_ok=True)
    for name, arr in [
        ("X_train", X_train), ("y_train", y_train),
        ("X_val",   X_val),   ("y_val",   y_val),
        ("X_test",  X_test),  ("y_test",  y_test),
    ]:
        path = os.path.join(out_dir, f"{name}.npy")
        np.save(path, arr)
        print(f"    Saved {name:8s}  shape={str(arr.shape):20s}  → {path}")

    print("\n✓ Preprocessing complete.")
    print(f"  Train : {X_train.shape[0]:,} samples")
    print(f"  Val   : {X_val.shape[0]:,} samples")
    print(f"  Test  : {X_test.shape[0]:,} samples")


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess RadioML 2018.01A for RadarNet FPGA inference"
    )
    parser.add_argument(
        "--hdf5", type=str, required=True,
        help="Path to GOLD_XYZ_OSC.0001_1024.hdf5"
    )
    parser.add_argument(
        "--out-dir", type=str, default=os.path.join("..", "data"),
        help="Output directory for .npy files (default: ../data/)"
    )
    parser.add_argument(
        "--seed", type=int, default=RANDOM_SEED,
        help=f"Random seed (default: {RANDOM_SEED})"
    )
    args = parser.parse_args()

    X, Y_onehot, Z_snr = load_dataset(args.hdf5)
    y = map_labels(Y_onehot)
    X, y, _ = filter_snr(X, y, Z_snr, SNR_THRESHOLD)
    X = slice_and_normalise(X, TIME_STEPS)
    X, y = balance_classes(X, y, SAMPLES_PER_CLASS, args.seed)
    split_and_save(X, y, args.out_dir, args.seed)


if __name__ == "__main__":
    main()
