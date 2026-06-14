# ZCU106 Vivado Implementation Flow

This folder benchmarks the existing sigmoid and tanh RTL on the ZCU106 device:

- Activations: `sigmoid`, `tanh`
- Methods: `baseline`, `proposed`
- Formats: `int8`, `uint8`, `fp8`, `fp32`
- Device: `xczu7ev-ffvc1156-2-e`
- Default timing target: 1.000 ns (1 GHz)

## Important Timing Note

The current format-specific RTL is combinational. Its `clk` input is unused.
The flow therefore adds input and output boundary registers and measures the
DUT as a register-to-register path.

The 1 GHz setting is an aggressive implementation target for comparison. It
does not mean that a valid 1 GHz board clock can be generated or routed on the
ZCU106. A design passes only when post-route WNS and WHS are both non-negative.
External pad paths are excluded because they depend on the final board
interface and package-pin selection.

## Run

Run all 16 implementations:

```bash
./run_vivado.sh
```

Run one design:

```bash
./run_vivado.sh \
  --activations tanh \
  --methods proposed \
  --formats int8 \
  --period-ns 1.0
```

Run synthesis only:

```bash
./run_vivado.sh --action synth
```

Use a more realistic timing target:

```bash
./run_vivado.sh --period-ns 2.5
```

## Outputs

Each design receives an isolated directory under `build/` containing:

- `vivado.log`
- Generated `benchmark.xdc`
- Post-synthesis and post-route checkpoints
- Timing, utilization, power, methodology, and clock reports
- `metrics.csv`

The complete matrix is aggregated into:

```text
results/implementation_summary.csv
```

## Board Testing

This flow performs synthesis, placement, routing, and timing/resource
comparison. It intentionally does not write a bitstream because the benchmark
ports have no ZCU106 package-pin assignments.

A board-test design should connect the activation core through AXI, VIO/ILA,
or a PS-controlled interface and use an achievable PL clock. Package pins and
the actual clock source must then be constrained for that selected interface.
