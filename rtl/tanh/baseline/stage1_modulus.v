// =============================================================
//  stage1_modulus.v
//
//  Computes |b| for the input.
//  Format : b    -> S1.I3.F7  (11 bits: [10]=sign [9:7]=int [6:0]=frac)
//           xp   -> same 11-bit pattern, always positive (sign=0)
//           sign -> captured sign bit of original b, passed to Stage 4
// =============================================================
module tanh_stage1_modulus (
    input  wire [10:0] b,
    output wire [10:0] xp,
    output wire        sign_bit
);

    assign sign_bit = b[10];
    assign xp       = b[10] ? (~b + 11'd1) : b;

endmodule
