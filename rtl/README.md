# RTL Designs

Synthesizable Verilog implementations of activation function approximation architectures.

## Structure

- `sigmoid/` — Sigmoid activation modules
  - `baseline/` — 16 uniform segments (stage-based pipeline)
  - `proposed1/` — 16 curvature-weighted non-uniform segments
  - `proposed2/` — 14 optimized non-uniform segments
  - `proposed2_ext/` — Shared-LUT extension for multi-instance deployment
- `tanh/` — Tanh activation modules (same sub-structure)
- `cordic/` — CORDIC-based implementations
  - `paper1/` — Reference CORDIC architecture 1
  - `paper2/` — Reference CORDIC architecture 2

## Format Variants

Each architecture has per-format implementations under `FP32/`, `FP8/`, `INT8/`, `UINT8/` subdirectories.

## Scripts

- `generate_lut_format_rtl.py` — Generate format-specific RTL from simulation coefficients
- `run_rtl_simulations.py` — Compile and simulate all RTL designs with iverilog/vvp

## Running Simulations

```bash
# Requires: iverilog, vvp (Icarus Verilog)
python run_rtl_simulations.py --activations all --methods all --formats all --keep-going
```
