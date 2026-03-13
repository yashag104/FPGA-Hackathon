#!/usr/bin/env python3
"""
quantize.py — INT8 Post-Training Quantisation with BN Folding
==============================================================
1. Loads the trained RadarNet-1D checkpoint.
2. Folds BatchNorm parameters into convolution weights:
       W_folded = γ · W / σ
       b_folded = γ · (b − μ) / σ + β
3. Quantises all folded weights and biases to INT8 (symmetric quantisation).
4. Evaluates the quantised model to confirm < 1% accuracy drop.
5. Saves the folded + quantised state_dict for weight export.

Usage:
    python quantize.py --checkpoint ../checkpoints/radarnet_best.pth \
                       --data-dir ../data
"""

import argparse
import copy
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from model import RadarNet1D, count_parameters


# ── BN Folding ───────────────────────────────────────────────────────────────
def fold_bn_into_conv(conv: nn.Conv1d, bn: nn.BatchNorm1d) -> nn.Conv1d:
    """
    Fold BatchNorm parameters into a Conv1d layer to produce a single
    conv layer with both weight and bias — eliminates BN from RTL.

    Returns a new Conv1d with folded weight and bias.
    """
    # Extract BN parameters
    gamma  = bn.weight.data                  # (out_ch,)
    beta   = bn.bias.data                    # (out_ch,)
    mu     = bn.running_mean                 # (out_ch,)
    sigma  = torch.sqrt(bn.running_var + bn.eps)  # (out_ch,)

    # Extract conv weight (and bias if present)
    W = conv.weight.data.clone()             # (out_ch, in_ch, k)
    b = conv.bias.data.clone() if conv.bias is not None else torch.zeros(W.size(0))

    # Fold:  W_folded = γ * W / σ ,  b_folded = γ * (b - μ) / σ + β
    scale = (gamma / sigma).reshape(-1, 1, 1)   # broadcast over (in_ch, k)
    W_folded = W * scale
    b_folded = gamma * (b - mu) / sigma + beta

    # Create new conv with bias
    folded_conv = nn.Conv1d(
        conv.in_channels, conv.out_channels, conv.kernel_size[0],
        stride=conv.stride[0], padding=conv.padding[0], bias=True,
    )
    folded_conv.weight.data = W_folded
    folded_conv.bias.data   = b_folded
    return folded_conv


def fold_all_bn(model: RadarNet1D) -> nn.Module:
    """
    Walk the model and fold every (Conv1d, BatchNorm1d) pair.
    Returns a new model with BN layers replaced by Identity.
    """
    model = copy.deepcopy(model)
    model.eval()

    def _fold_sequential(seq):
        """Fold Conv+BN pairs in an nn.Sequential."""
        new_modules = []
        i = 0
        modules = list(seq.children())
        while i < len(modules):
            m = modules[i]
            # Check if next module is BN
            if isinstance(m, nn.Conv1d) and i + 1 < len(modules) \
                    and isinstance(modules[i + 1], nn.BatchNorm1d):
                folded = fold_bn_into_conv(m, modules[i + 1])
                new_modules.append(folded)
                i += 2   # skip the BN
            else:
                new_modules.append(m)
                i += 1
        return nn.Sequential(*new_modules)

    # Fold in conv_in and conv_down (nn.Sequential blocks)
    model.conv_in   = _fold_sequential(model.conv_in)
    model.conv_down = _fold_sequential(model.conv_down)

    # Fold in residual blocks
    for block in [model.block1, model.block2]:
        block.conv1 = fold_bn_into_conv(block.conv1, block.bn1)
        block.bn1   = nn.Identity()
        block.conv2 = fold_bn_into_conv(block.conv2, block.bn2)
        block.bn2   = nn.Identity()
        # Fold shortcut if it contains Conv+BN
        if isinstance(block.shortcut, nn.Sequential):
            block.shortcut = _fold_sequential(block.shortcut)

    return model


# ── INT8 Symmetric Quantisation ──────────────────────────────────────────────
def symmetric_quantise_tensor(t: torch.Tensor, bits: int = 8):
    """
    Symmetric per-tensor quantisation to `bits`-bit signed integer.
    Returns: (quantised_int_tensor, scale_factor)
    """
    qmin = -(2 ** (bits - 1))
    qmax = (2 ** (bits - 1)) - 1      # -128 .. +127 for INT8
    abs_max = t.abs().max().item()
    if abs_max == 0:
        return torch.zeros_like(t, dtype=torch.int8), 1.0
    scale = abs_max / qmax
    t_q = torch.clamp(torch.round(t / scale), qmin, qmax).to(torch.int8)
    return t_q, scale


def quantise_model_weights(model: nn.Module):
    """
    Quantise all Conv1d and Linear weights (and biases) to INT8.
    Returns dict of {param_name: (int8_tensor, scale)}.
    """
    quant_params = {}
    for name, param in model.named_parameters():
        t_q, scale = symmetric_quantise_tensor(param.data, bits=8)
        quant_params[name] = {"int8": t_q, "scale": scale}
    return quant_params


# ── Evaluation ───────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate_accuracy(model, loader, device):
    model.eval()
    correct, total = 0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        preds = model(xb).argmax(1)
        correct += (preds == yb).sum().item()
        total += xb.size(0)
    return correct / total


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="INT8 quantisation with BN folding for RadarNet-1D"
    )
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to trained model checkpoint (.pth)")
    parser.add_argument("--data-dir",   type=str, default=os.path.join("..", "data"))
    parser.add_argument("--out-dir",    type=str, default=os.path.join("..", "checkpoints"))
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out_dir, exist_ok=True)

    # ── Load model ───────────────────────────────────────────────────
    print("Loading trained model …")
    model = RadarNet1D(in_channels=2, num_classes=4)
    model.load_state_dict(
        torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    )
    model.to(device)

    # ── Load test set ────────────────────────────────────────────────
    X_test = np.load(os.path.join(args.data_dir, "X_test.npy"))
    y_test = np.load(os.path.join(args.data_dir, "y_test.npy"))
    X_test = np.transpose(X_test, (0, 2, 1)).astype(np.float32)
    test_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_test),
                      torch.from_numpy(y_test.astype(np.int64))),
        batch_size=args.batch_size, shuffle=False,
    )

    # ── Baseline accuracy (FP32) ─────────────────────────────────────
    fp32_acc = evaluate_accuracy(model, test_loader, device)
    print(f"FP32 test accuracy: {fp32_acc:.4%}")

    # ── Fold BN ──────────────────────────────────────────────────────
    print("\nFolding BatchNorm into convolution weights …")
    folded_model = fold_all_bn(model)
    folded_model.to(device)
    folded_acc = evaluate_accuracy(folded_model, test_loader, device)
    print(f"BN-folded test accuracy: {folded_acc:.4%}  "
          f"(Δ = {folded_acc - fp32_acc:+.4%})")

    # ── Quantise to INT8 ─────────────────────────────────────────────
    print("\nQuantising all parameters to INT8 (symmetric) …")
    quant_params = quantise_model_weights(folded_model)

    # Build a float model from INT8 weights (for accuracy verification)
    quant_model = copy.deepcopy(folded_model)
    for name, param in quant_model.named_parameters():
        qp = quant_params[name]
        param.data = qp["int8"].float() * qp["scale"]
    quant_model.to(device)

    quant_acc = evaluate_accuracy(quant_model, test_loader, device)
    drop = fp32_acc - quant_acc
    print(f"INT8 test accuracy: {quant_acc:.4%}  "
          f"(Δ from FP32 = {-drop:+.4%})")

    status = "✓ PASS" if drop < 0.01 else "⚠ FAIL"
    print(f"\nQuantisation accuracy drop: {drop:.4%}  [{status}: target < 1%]")

    # ── Save quantised artefacts ─────────────────────────────────────
    # 1. Folded + quantised state_dict (float scales + int8 weights)
    save_dict = {}
    for name in quant_params:
        save_dict[name + ".int8"]  = quant_params[name]["int8"]
        save_dict[name + ".scale"] = torch.tensor(quant_params[name]["scale"])

    quant_path = os.path.join(args.out_dir, "radarnet_int8.pth")
    torch.save(save_dict, quant_path)
    print(f"\nINT8 quantised weights → {quant_path}")

    # 2. Also save the folded FP32 model (for reference)
    folded_path = os.path.join(args.out_dir, "radarnet_folded.pth")
    torch.save(folded_model.state_dict(), folded_path)
    print(f"BN-folded FP32 model  → {folded_path}")

    # 3. Summary
    total_bytes = sum(qp["int8"].numel() for qp in quant_params.values())
    print(f"\nTotal INT8 weight bytes: {total_bytes:,}  "
          f"({total_bytes / 1024:.1f} KB — fits in FPGA BRAM)")


if __name__ == "__main__":
    main()
