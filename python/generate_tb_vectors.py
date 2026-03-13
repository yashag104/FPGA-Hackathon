#!/usr/bin/env python3
"""
generate_tb_vectors.py — Generate Verilog Testbench Vectors
============================================================
Selects test samples, runs the INT8-quantised model to get expected outputs,
and exports input I/Q data + expected labels as .hex files for tb_radarnet.v.

Usage:
    python generate_tb_vectors.py --checkpoint ../checkpoints/radarnet_int8.pth \
                                  --data-dir ../data --out-dir ../tb_vectors
"""
import argparse, os
import numpy as np, torch
from model import RadarNet1D
from quantize import fold_all_bn, quantise_model_weights
import copy


def float_to_int8_hex(val: float) -> str:
    """Clamp float [-1,1] to INT8, return 2-digit hex."""
    ival = int(round(val * 127.0))
    ival = max(-128, min(127, ival))
    if ival < 0:
        ival += 256
    return f"{ival:02X}"


def main():
    parser = argparse.ArgumentParser(description="Generate Verilog test vectors")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to radarnet_best.pth (FP32 trained)")
    parser.add_argument("--data-dir", type=str, default=os.path.join("..", "data"))
    parser.add_argument("--out-dir", type=str, default=os.path.join("..", "tb_vectors"))
    parser.add_argument("--num-vectors", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # Load test data
    X_test = np.load(os.path.join(args.data_dir, "X_test.npy"))   # (N,128,2)
    y_test = np.load(os.path.join(args.data_dir, "y_test.npy"))

    # Select balanced subset
    rng = np.random.default_rng(args.seed)
    n_per_class = args.num_vectors // 4
    indices = []
    for c in range(4):
        cls_idx = np.where(y_test == c)[0]
        chosen = rng.choice(cls_idx, min(n_per_class, len(cls_idx)), replace=False)
        indices.append(chosen)
    # Fill remainder
    remaining = args.num_vectors - sum(len(x) for x in indices)
    if remaining > 0:
        all_idx = np.concatenate(indices)
        pool = np.setdiff1d(np.arange(len(y_test)), all_idx)
        indices.append(rng.choice(pool, remaining, replace=False))
    indices = np.concatenate(indices)
    rng.shuffle(indices)
    indices = indices[:args.num_vectors]

    X_sel = X_test[indices]  # (50, 128, 2)
    y_sel = y_test[indices]

    # Load model, fold BN, run quantised inference
    model = RadarNet1D(in_channels=2, num_classes=4)
    model.load_state_dict(
        torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    )
    model.eval()
    folded = fold_all_bn(model)
    qp = quantise_model_weights(folded)
    qmodel = copy.deepcopy(folded)
    for name, param in qmodel.named_parameters():
        param.data = qp[name]["int8"].float() * qp[name]["scale"]

    X_torch = torch.from_numpy(
        np.transpose(X_sel, (0, 2, 1)).astype(np.float32)
    )
    with torch.no_grad():
        logits = qmodel(X_torch)
        preds = logits.argmax(1).numpy()

    # Export input vectors — interleaved I/Q per sample
    input_hex = os.path.join(args.out_dir, "input_vectors.hex")
    with open(input_hex, "w") as f:
        for i in range(len(X_sel)):
            for t in range(128):
                i_val = float_to_int8_hex(X_sel[i, t, 0])
                q_val = float_to_int8_hex(X_sel[i, t, 1])
                f.write(f"{i_val}{q_val}\n")  # 16-bit: {I, Q}
    print(f"Input vectors ({len(X_sel)} x 128 x 2) -> {input_hex}")

    # Export expected labels
    labels_hex = os.path.join(args.out_dir, "expected_labels.hex")
    with open(labels_hex, "w") as f:
        for p in preds:
            f.write(f"{int(p):02X}\n")
    print(f"Expected labels -> {labels_hex}")

    # Export true labels for cross-reference
    true_hex = os.path.join(args.out_dir, "true_labels.hex")
    with open(true_hex, "w") as f:
        for l in y_sel:
            f.write(f"{int(l):02X}\n")
    print(f"True labels     -> {true_hex}")

    # Summary
    match = (preds == y_sel).sum()
    print(f"\nQuantised model: {match}/{len(y_sel)} match true labels")
    print(f"  Class distribution: {[int((y_sel==c).sum()) for c in range(4)]}")


if __name__ == "__main__":
    main()
