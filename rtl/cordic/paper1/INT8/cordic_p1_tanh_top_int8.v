module cordic_p1_tanh_top_int8 (
    input clk,
    input [7:0] b,
    output [7:0] z
);
    wire signed [31:0] b_fp32;
    wire signed [31:0] z_fp32;

    // map b to fp32 (x = b/16 -> x = b_fp32/64 => b_fp32 = b * 4)
    assign b_fp32 = { {24{b[7]}}, b } <<< 2;

    cordic_p1_core core (
        .clk(clk),
        .b(b_fp32),
        .is_tanh(1'b1),
        .z(z_fp32)
    );

    // map z_fp32 to 8-bit output: z_int8 = z_fp32 / 2^24
    wire signed [31:0] z_shifted = (z_fp32 + (1<<23)) >>> 24;
    wire signed [31:0] z_sat = (z_shifted > 127) ? 127 : ((z_shifted < -128) ? -128 : z_shifted);
    assign z = z_sat[7:0];
endmodule
