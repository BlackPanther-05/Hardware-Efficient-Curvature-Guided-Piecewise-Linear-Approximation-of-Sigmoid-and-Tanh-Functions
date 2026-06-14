`ifndef DUT_MODULE
`error "DUT_MODULE must name the activation module"
`endif

`ifndef DATA_WIDTH
`error "DATA_WIDTH must be 8 or 32"
`endif

module fpga_benchmark_wrapper (
    input  wire                  clk,
    input  wire [`DATA_WIDTH-1:0] b,
    output reg  [`DATA_WIDTH-1:0] z
);

    reg  [`DATA_WIDTH-1:0] b_reg;
    wire [`DATA_WIDTH-1:0] z_comb;

    // Boundary registers make the combinational DUT measurable as a
    // register-to-register path without changing the activation RTL.
    always @(posedge clk) begin
        b_reg <= b;
        z     <= z_comb;
    end

    `DUT_MODULE dut (
        .clk (clk),
        .b   (b_reg),
        .z   (z_comb)
    );

endmodule
