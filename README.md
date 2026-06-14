# Approximate Activation Function Architectures for Hardware-Efficient Neural Network Inference

This repository provides the complete source code and reproducible workflows for our research on piecewise-linear approximation architectures for sigmoid, tanh, and GELU activation functions, targeting FPGA and ASIC deployment.

## Overview

We propose curvature-aware, non-uniform segment allocation strategies for piecewise-linear activation approximation. The repository includes:

- **Software Simulation**: Python-based numerical approximation experiments across FP32, FP8, INT8, and UINT8 data formats
- **RTL Implementations**: Synthesizable Verilog modules for Baseline, Proposed-1 (16 non-uniform segments), Proposed-2 (optimized segment count), and CORDIC-based architectures
- **FPGA Benchmarking**: Automated Vivado flows for Virtex-7 and ZCU106 (UltraScale+) resource/timing/power analysis
- **ML Application Benchmarks**: Inference accuracy validation on LSTM (CIC-IDS2018), HuBERT (PTB-XL ECG), LeNet-5 (MNIST), VGG11 (CIFAR-100), and GPT-2 (SST-2) models

## Directory Structure

```
├── software_simulation/     # Numerical approximation experiments
├── simulation/              # Production simulation, error analysis, visualization
├── rtl/                     # Synthesizable Verilog RTL designs
│   ├── sigmoid/             # Sigmoid: baseline, proposed1, proposed2, proposed2_ext
│   ├── tanh/                # Tanh: baseline, proposed1, proposed2, proposed2_ext
│   └── cordic/              # CORDIC-based implementations (Paper1, Paper2)
├── fpga/                    # FPGA implementation flows
│   ├── virtex7/             # Xilinx Virtex-7 (xc7v585t)
│   └── zcu106/              # Xilinx ZCU106 UltraScale+ (xczu7ev)
├── applications/            # ML model inference benchmarks
│   ├── approx_activation_lib.py  # Shared activation approximation library
│   ├── lstm_cic/            # LSTM on CIC-IDS2018 intrusion detection
│   ├── hubert_ecg/          # HuBERT on PTB-XL ECG classification
│   ├── lenet5_mnist/        # LeNet-5 on MNIST digit recognition
│   ├── vgg11_cifar100/      # VGG11 on CIFAR-100 image classification
│   └── gpt2_sst2/           # GPT-2 on SST-2 sentiment analysis
├── synthesis/               # Vivado synthesis automation scripts
└── results/                 # Pre-computed results, CSV tables, and figures
```

## Prerequisites

### Software
- **Python** ≥ 3.9
- **PyTorch** ≥ 2.0
- **NumPy**, **Matplotlib**, **scikit-learn**, **pandas**
- **Icarus Verilog** (`iverilog`, `vvp`) for RTL simulation
- **Xilinx Vivado** ≥ 2023.1 for FPGA synthesis (optional)

### Installation

```bash
git clone <repository-url>
cd <repository-name>
pip install -r requirements.txt
```

## Quick Start

### 1. Software Simulation (Approximation Error Analysis)

```bash
# Run all activation simulations and generate error plots
cd simulation
python run_activation_simulations.py --activations all --methods all --formats all

# Generate architecture partition summary CSV
python generate_paper_csv.py

# Generate overlay comparison plots
python generate_proposed_images.py
```

### 2. RTL Simulation (Hardware Verification)

```bash
# Requires: iverilog, vvp
cd rtl
python run_rtl_simulations.py --activations all --methods all --formats all
```

### 3. RTL Error Analysis

```bash
cd simulation
python run_error_analysis.py
```

### 4. FPGA Implementation

```bash
# Requires: Vivado
cd fpga/virtex7
./run_vivado.sh                           # Run full 16-design matrix
./run_vivado.sh --activations sigmoid --methods proposed --formats fp32  # Single design
```

### 5. ML Application Benchmarks

```bash
# LeNet-5 on MNIST (smallest, good for quick testing)
cd applications/lenet5_mnist
python infer_activation_accuracy.py --device cpu --allow-cpu

# LSTM on CIC-IDS2018
cd applications/lstm_cic
python train_lstm_cic.py --data-dir ./data   # Train first
python infer_activation_accuracy.py --device cpu --allow-cpu

# GPT-2 on SST-2
cd applications/gpt2_sst2
python infer_activation_accuracy.py --device cpu --allow-cpu
```

## Data and Model Weights

### Datasets
- **MNIST**: Auto-downloaded by `torchvision.datasets.MNIST`
- **CIFAR-100**: Download from [cs.toronto.edu](https://www.cs.toronto.edu/~kriz/cifar.html), extract to `applications/vgg11_cifar100/cifar100/`
- **CIC-IDS2018**: Download from [Kaggle](https://www.kaggle.com/datasets/solarmainframe/ids-intrusion-csv) or the [official source](https://www.unb.ca/cic/datasets/ids-2018.html), place CSV files in `applications/lstm_cic/data/`
- **PTB-XL**: Download from [PhysioNet](https://physionet.org/content/ptb-xl/1.0.3/), extract to `applications/hubert_ecg/ptb-xl-1.0.3/`
- **SST-2**: Auto-downloaded by HuggingFace `datasets`

### Pre-trained Weights
The pre-trained model weights are included directly in their respective application directories.

> **Note**: Due to the size of these files (e.g., HuBERT is 1.1 GB), this repository relies on Git LFS (Large File Storage). Ensure you have Git LFS installed before cloning.


## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.


