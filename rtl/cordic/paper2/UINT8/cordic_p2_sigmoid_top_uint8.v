module cordic_p2_sigmoid_top_uint8 (
    input clk,
    input [7:0] b,
    output [7:0] z
);
    wire signed [31:0] b_fp32;
    wire signed [31:0] z_fp32;

    // map b to fp32: x = (b/16)-8.0. b_fp32 = x*64 = b*4 - 512
    assign b_fp32 = ({24'd0, b} * 4) - 512;

    cordic_p2_core core (
        .clk(clk),
        .b(b_fp32),
        .is_tanh(1'b0),
        .z(z_fp32)
    );

    // map z_fp32 to uint8: z_uint8 = (z_fp32 + 2^31) * 255 / 2^32
    // In verilog, we can just flip the sign bit of z_fp32 to map [-2^31, 2^31-1] to [0, 2^32-1]
    wire [31:0] z_unsigned = {~z_fp32[31], z_fp32[30:0]};
    wire [39:0] z_mult = z_unsigned * 40'd255;
    assign z = z_mult[39:32];
endmodule
