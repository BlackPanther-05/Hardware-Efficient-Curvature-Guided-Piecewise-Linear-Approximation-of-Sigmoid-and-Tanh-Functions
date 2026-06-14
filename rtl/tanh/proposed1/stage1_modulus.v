// =============================================================
//  stage1_modulus.v
//  Computes |b| for signed Q3.7 input.
// =============================================================
module tanh_stage1_modulus (
    input  wire [10:0] b,
    output wire [10:0] xp,
    output wire        sign_bit
);

    assign sign_bit = b[10];
    assign xp       = b[10] ? (~b + 11'd1) : b;

endmodule
