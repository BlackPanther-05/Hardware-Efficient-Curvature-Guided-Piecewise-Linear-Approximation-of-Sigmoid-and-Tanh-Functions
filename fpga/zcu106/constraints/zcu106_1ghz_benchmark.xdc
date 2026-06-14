# ZCU106 implementation benchmark constraints.
#
# The 1.290 ns clock (775 MHz) is a realistic high-performance timing target used to compare
# designs, reflecting the maximum global clock tree frequency of the -2 speed grade device.
# Package-pin constraints are intentionally omitted because this flow measures
# the registered datapath and does not produce a board-ready bitstream.

create_clock -name benchmark_clk -period 1.290 -waveform {0.000 0.645} [get_ports clk]
set_clock_uncertainty -setup 0.050 [get_clocks benchmark_clk]
set_clock_uncertainty -hold  0.025 [get_clocks benchmark_clk]

# The benchmark compares only the activation core between boundary registers.
# Physical pad timing depends on the board interface and package-pin selection,
# so external input/output paths are excluded from this implementation metric.
set_false_path -from [get_ports b[*]]
set_false_path -to [get_ports z[*]]
