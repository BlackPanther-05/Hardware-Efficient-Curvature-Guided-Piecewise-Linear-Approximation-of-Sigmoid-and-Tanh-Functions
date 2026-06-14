// =============================================================
//  stage3_adder.v
//
//  Computes s = p + c
//
//  Both p and c are in S1.F10 (11-bit signed fixed-point).
//  For sigmoid, p and c are both positive and their sum s
//  should be in (0, 1), so no overflow expected.
//
//  Output s is 11 bits S1.F10, clamped to 11'd1023 (=0.999...)
//  on overflow to be safe.
// =============================================================
module stage3_adder (
    input  wire [10:0] p,
    input  wire [10:0] c,
    output wire [10:0] s
);

    wire [11:0] sum_full;
    assign sum_full = {1'b0, p} + {1'b0, c};

    // Clamp: if bit[11] set or sum > 1023 (0.999...) → 1023
    assign s = (sum_full > 12'd1023) ? 11'd1023 : sum_full[10:0];

endmodule
