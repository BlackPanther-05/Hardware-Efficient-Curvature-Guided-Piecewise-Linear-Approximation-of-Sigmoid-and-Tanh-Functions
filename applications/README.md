# Application Inference Benchmarks

Inference accuracy benchmarks measuring the impact of approximate activation functions on real ML models.

## Shared Library

`approx_activation_lib.py` provides:
- `ApproxActivation` — Piecewise-linear sigmoid/tanh for fp32/fp8/int8/uint8
- `ApproxGELU` — GELU expressed through approximate tanh or sigmoid
- `ApproxReLUReplacement` — Activation substitution for ReLU networks
- `ExactActivation` / `ExactGELU` — FP32 reference implementations

## Models

| Model | Task | Dataset | Activation Sites |
|-------|------|---------|-----------------|
| LSTM | Intrusion Detection | CIC-IDS2018 | sigmoid + tanh gates |
| HuBERT | ECG Classification | PTB-XL | GELU (12 transformer layers) |
| LeNet-5 | Digit Recognition | MNIST | 5 ReLU sites |
| VGG11 | Image Classification | CIFAR-100 | 10 ReLU sites |
| GPT-2 | Sentiment Analysis | SST-2 | GELU (12 transformer layers) |

## LUT Sharing Semantics

- **Baseline scheme**: Each activation site gets its OWN independent LUT copy
- **Proposed scheme**: All sites share ONE LUT instance (reflecting hardware LUT sharing)

## Usage

```bash
# Each model directory has an infer_activation_accuracy.py script
cd lstm_cic
python infer_activation_accuracy.py --device cpu --allow-cpu
```
