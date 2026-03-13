#!/usr/bin/env python3
"""
export_weights.py — Export INT8 Weights as .hex Files for Verilog BRAM
======================================================================
Reads the INT8 quantised checkpoint and writes each layer's weights/biases
as .hex files for $readmemh in weight_rom.v.

Usage:
    python export_weights.py --checkpoint ../checkpoints/radarnet_int8.pth \
                             --out-dir ../weights
"""
import argparse, os
import torch, numpy as np


def int8_to_hex(val: int) -> str:
    if val < 0:
        val = val + 256
    return f"{val:02X}"


def export_tensor_hex(tensor, filepath):
    flat = tensor.flatten().numpy().astype(np.int8)
    with open(filepath, "w") as f:
        for val in flat:
            f.write(int8_to_hex(int(val)) + "\n")
    return len(flat)


LAYER_NAMES = {
    "conv_in.0.weight":         "L1_conv_in",
    "conv_in.0.bias":           "L1_conv_in_bias",
    "conv_down.0.weight":       "L2_conv_down",
    "conv_down.0.bias":         "L2_conv_down_bias",
    "block1.conv1.weight":      "L3_res1_conv1",
    "block1.conv1.bias":        "L3_res1_conv1_bias",
    "block1.conv2.weight":      "L4_res1_conv2",
    "block1.conv2.bias":        "L4_res1_conv2_bias",
    "block2.conv1.weight":      "L5_res2_conv1",
    "block2.conv1.bias":        "L5_res2_conv1_bias",
    "block2.conv2.weight":      "L6_res2_conv2",
    "block2.conv2.bias":        "L6_res2_conv2_bias",
    "block2.shortcut.0.weight": "L5_res2_proj",
    "block2.shortcut.0.bias":   "L5_res2_proj_bias",
    "fc.weight":                "L8_fc",
    "fc.bias":                  "L8_fc_bias",
}


def main():
    parser = argparse.ArgumentParser(description="Export INT8 weights to .hex")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--out-dir", type=str, default=os.path.join("..", "weights"))
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    int8_keys = sorted([k for k in ckpt if k.endswith(".int8")])
    scale_keys = sorted([k for k in ckpt if k.endswith(".scale")])

    print(f"Found {len(int8_keys)} INT8 parameter tensors\n")
    total_bytes = 0

    for key in int8_keys:
        param_name = key.replace(".int8", "")
        tensor = ckpt[key]
        scale = ckpt[param_name + ".scale"].item()
        clean = LAYER_NAMES.get(param_name, param_name.replace(".", "_"))
        hex_file = os.path.join(args.out_dir, f"{clean}.hex")
        n = export_tensor_hex(tensor, hex_file)
        total_bytes += n
        shape_s = "x".join(str(s) for s in tensor.shape)
        print(f"  {clean:30s} shape={shape_s:20s} values={n:>6,} scale={scale:.6f}")

    # Scale factors
    scales_path = os.path.join(args.out_dir, "scales.txt")
    with open(scales_path, "w") as f:
        for key in scale_keys:
            pn = key.replace(".scale", "")
            cn = LAYER_NAMES.get(pn, pn.replace(".", "_"))
            f.write(f"{cn} {ckpt[key].item():.10f}\n")

    # Address map
    addr_map = os.path.join(args.out_dir, "address_map.txt")
    addr = 0
    with open(addr_map, "w") as f:
        f.write(f"{'Layer':<30s} {'Start':>8s} {'End':>8s} {'Size':>8s}\n")
        for key in int8_keys:
            pn = key.replace(".int8", "")
            cn = LAYER_NAMES.get(pn, pn.replace(".", "_"))
            sz = ckpt[key].numel()
            f.write(f"{cn:<30s} {addr:>8d} {addr+sz-1:>8d} {sz:>8d}\n")
            addr += sz

    print(f"\n  Total INT8 values: {total_bytes:,} ({total_bytes/1024:.1f} KB)")
    print(f"  Scales  -> {scales_path}")
    print(f"  AddrMap -> {addr_map}")


if __name__ == "__main__":
    main()
