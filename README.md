# RadarNet: FPGA Hackathon 2026 📡

**Hardware-Accelerated Edge AI for RF Anomaly Detection**

This repository contains the complete ML pipeline and Verilog RTL for **RadarNet**, an INT8-quantized ResNet-1D accelerator targeting AMD/Xilinx FPGA platforms. Designed for the **Defense** application domain, RadarNet classifies raw I/Q radio signals into anomaly categories (Jammer, Spoofer, Interference, or Normal) at real-time speeds (< 1µs latency).

---

## 🚀 Overview

Cloud-based processing of defense radar signals suffers from high latency and bandwidth bottlenecks. **RadarNet** solves this by performing real-time inference at the edge directly on the raw I/Q data stream.

*   **Application Domain:** Defense Systems (Signal Anomaly Detection)
*   **Dataset:** RadioML 2018.01A (24 modulations mapped to 4 defense classes)
*   **AI Model:** ResNet-1D (8 layers, ~29,668 parameters)
*   **Quantization:** Fully INT8 symmetric post-training quantization with Batch-Norm folding.
*   **Hardware Implementation:** Verilog RTL (synthesizable for Xilinx 7-Series).

---

## 🧠 Machine Learning Pipeline (`python/`)

The Python pipeline handles dataset processing, model training, and exporting weights for the FPGA.

### Prerequisites
```bash
pip install torch numpy h5py scikit-learn matplotlib tqdm
```

### Execution Steps

1.  **Download Dataset:** Download the RadioML 2018.01A dataset from Kaggle or DeepSig. Extract the `GOLD_XYZ_OSC.0001_1024.hdf5` file.
2.  **Preprocess:** Maps the 24 classes to 4 anomaly types, filters SNR ≥ 0dB, slices to 128 time-steps, and balances the dataset.
    ```bash
    python python/preprocess.py --hdf5 path/to/dataset.hdf5
    ```
3.  **Train:** Trains the ResNet-1D architecture to >92% test accuracy.
    ```bash
    python python/train.py
    ```
4.  **Quantize:** Folds BatchNorm layers into convolution weights and quantizes the model to INT8. Ensure the accuracy drop is < 1%.
    ```bash
    python python/quantize.py --checkpoint checkpoints/radarnet_best.pth
    ```
5.  **Export Weights:** Converts INT8 weights into `.hex` files for Verilog `$readmemh`.
    ```bash
    python python/export_weights.py --checkpoint checkpoints/radarnet_int8.pth
    ```
6.  **Generate Test Vectors:** Creates sample I/Q inputs and expected labels for RTL simulation.
    ```bash
    python python/generate_tb_vectors.py --checkpoint checkpoints/radarnet_best.pth
    ```

---

## ⚡ Hardware Architecture (`rtl/`)

The RTL design operates on a 100 MHz clock and uses an 8-state Finite State Machine (FSM) to orchestrate data across the network layers.

### Key Features
*   **INT8 MAC Engine:** 24-bit accumulation preventing overflow, saturating down to 8-bit.
*   **Zero-Cost BatchNorm:** Folded entirely during the Python export phase.
*   **Hardware Skip Connections:** Implemented as zero-cost integer additions (Block 1) or via 1x1 projection convolutions (Block 2) to handle stride dimension mismatches.
*   **Efficient BRAM Usage:** All ~29KB of weights fit safely within on-chip block RAM (`weight_rom.v`).

### Module Hierarchy
*   `radarnet_top.v`: Top-level wrapper holding the FSM.
    *   `input_buffer.v`: 128×2 shift register bridging serial input to parallel computation.
    *   `conv1d_engine.v`: Parameterized INT8 MAC engine.
    *   `residual_block.v`: Wraps two conv engines and a skip connection.
        *   `proj_conv.v`: 1x1 projection convolution for downsampling paths.
    *   `gap_unit.v`: Global Average Pooling using arithmetic right shifts (division by 32).
    *   `fc_layer.v`: Fully connected classifier mapping 64 features to 4 logits.
    *   `argmax_out.v`: Combinational logic isolating the maximum logit and flagging anomalies.
    *   `weight_rom.v`: Unified BRAM module holding all parameters.

---

## 🛠️ Verification & Simulation

To verify the hardware matches the software model, run the included testbench which cross-references standard Python outputs against Verilog outputs.

Using **Icarus Verilog**:
```bash
iverilog -o radarnet_sim rtl/*.v
vvp radarnet_sim
```

Using **Vivado (Xilinx)**:
```bash
xvlog rtl/*.v
xelab tb_radarnet -debug all
xsim tb_radarnet -runall
```

The testbench (`tb_radarnet.v`) will feed 50 test vectors into `radarnet_top`, wait for the `done` signal, and compare the RTL `class_id` against the expected Python `expected_labels.hex`.

---

## 📚 Class Mapping Rationale

| Class Index | Defense Type | Modulations | Rationale |
| :--- | :--- | :--- | :--- |
| **0** | **Normal** | FM, AM-*, GMSK | Continuous-wave/narrowband expected emission profiles. |
| **1** | **Jammer** | OOK, WBFM, 64+QAM | Wideband/high-entropy signals (barrage jamming). |
| **2** | **Spoofer** | *PSK, OQPSK | Coherent phase-modulated signals mimicking authentic radar. |
| **3** | **Interference** | *APSK, *ASK, 16/32QAM | Multi-level amplitude/phase cross-band interference. |

This novel mapping aligns the standard RadioML dataset directly with real-world Electronic Warfare (EW) requirements.
