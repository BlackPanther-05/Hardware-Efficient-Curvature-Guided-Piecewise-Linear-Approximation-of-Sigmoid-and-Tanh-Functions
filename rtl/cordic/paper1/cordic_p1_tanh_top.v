module cordic_p1_tanh_top (
    input clk,
    input [31:0] b,
    output [31:0] z
);
    cordic_p1_core core (
        .clk(clk),
        .b(b),
        .is_tanh(1'b1),
        .z(z)
    );
endmodule
