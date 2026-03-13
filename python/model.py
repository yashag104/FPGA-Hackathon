#!/usr/bin/env python3
"""
model.py — ResNet-1D (8-Layer) for RF Anomaly Detection
========================================================
Architecture designed for INT8 FPGA deployment:
    Layer 1 : Conv1D(2→16,  k=7, s=1, pad=3) + BN + ReLU
    Layer 2 : Conv1D(16→32, k=3, s=2, pad=1) + BN + ReLU           # downsample ×2
    Layer 3-4 : Residual Block 1  (32→32, k=3, s=1, pad=1)         # direct skip
    Layer 5-6 : Residual Block 2  (32→64, k=3, s=2, pad=1)         # projection skip
    Layer 7 : Global Average Pooling → 64-dim vector
    Layer 8 : FC(64→4)

Total parameters: ~29 668 (fits in 30 KB INT8 BRAM)
Input:  (batch, 2, 128)   — 128-point I/Q
Output: (batch, 4)        — 4 anomaly class logits
"""

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """
    Two-conv residual block with optional 1×1 projection shortcut.
    When `stride > 1` or `in_ch != out_ch`, the skip path uses a learnable
    1×1 conv + BN to match dimensions — this maps to `proj_conv.v` in RTL.
    """

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3,
                 stride: int = 1):
        super().__init__()
        padding = kernel_size // 2

        # Main path
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size,
                               stride=stride, padding=padding, bias=False)
        self.bn1   = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size,
                               stride=1, padding=padding, bias=False)
        self.bn2   = nn.BatchNorm1d(out_ch)
        self.relu  = nn.ReLU(inplace=True)

        # Shortcut / projection path
        self.shortcut = nn.Identity()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm1d(out_ch),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.relu(out + identity)      # residual addition → INT8 adder in RTL
        return out


class RadarNet1D(nn.Module):
    """
    8-layer ResNet-1D for RadioML anomaly detection.

    Architecture mirrors the RTL module hierarchy:
        conv_in   → conv1d_engine.v (layer 1)
        conv_down → conv1d_engine.v (layer 2, stride=2)
        block1    → residual_block.v (layers 3-4, direct skip)
        block2    → residual_block.v (layers 5-6, proj_conv skip)
        gap       → gap_unit.v (layer 7)
        fc        → fc_layer.v (layer 8)
    """

    def __init__(self, in_channels: int = 2, num_classes: int = 4):
        super().__init__()

        # Layer 1: Input convolution  (2 → 16 ch)
        self.conv_in = nn.Sequential(
            nn.Conv1d(in_channels, 16, kernel_size=7, stride=1,
                      padding=3, bias=False),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
        )

        # Layer 2: Down-sampling convolution  (16 → 32 ch, stride=2)
        self.conv_down = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=3, stride=2,
                      padding=1, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
        )

        # Layers 3-4: Residual Block 1  (32 → 32, no projection)
        self.block1 = ResidualBlock(32, 32, kernel_size=3, stride=1)

        # Layers 5-6: Residual Block 2  (32 → 64, with 1×1 projection)
        self.block2 = ResidualBlock(32, 64, kernel_size=3, stride=2)

        # Layer 7: Global Average Pooling
        self.gap = nn.AdaptiveAvgPool1d(1)

        # Layer 8: Fully Connected  (64 → 4)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, 2, 128) — raw I/Q time-series

        Returns:
            logits: (batch, 4) — anomaly class scores
        """
        x = self.conv_in(x)      # (B, 16, 128)
        x = self.conv_down(x)    # (B, 32,  64)
        x = self.block1(x)       # (B, 32,  64)
        x = self.block2(x)       # (B, 64,  32)
        x = self.gap(x)          # (B, 64,   1)
        x = x.squeeze(-1)        # (B, 64)
        x = self.fc(x)           # (B,  4)
        return x


def count_parameters(model: nn.Module) -> int:
    """Return total trainable parameter count."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ── Quick sanity check ──────────────────────────────────────────────────────
if __name__ == "__main__":
    model = RadarNet1D(in_channels=2, num_classes=4)
    print(model)
    print(f"\nTotal parameters: {count_parameters(model):,}")

    dummy = torch.randn(4, 2, 128)
    out = model(dummy)
    print(f"Input shape : {dummy.shape}")
    print(f"Output shape: {out.shape}")

    # Per-layer parameter breakdown
    print("\n── Parameter breakdown ──")
    for name, param in model.named_parameters():
        print(f"  {name:40s}  {str(list(param.shape)):20s}  = {param.numel():>6,}")
