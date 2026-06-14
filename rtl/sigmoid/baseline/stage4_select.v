// =============================================================
//  stage4_select.v
//
//  Final stage: selects output for positive or negative input.
//
//  For sigmoid:
//    sign_bit == 0  → z = s            (positive half, direct)
//    sign_bit == 1  → z = 1 - s        (negative half)
//                         1.0 in Q1.10 = 1024
//                         so z = 1024 - s
//
//  Format: S1.F10, 11 bits
//    1.0 is represented as 11'd1024 — note this requires
//    bit[10]=1 which is the sign position in S1.F10.
//    However since sigmoid output is in (0,1), output is
//    always < 1, so we represent 1.0 as 12-bit internally
//    and the result fits in 11 bits (result is < 1.0).
//
//    For (1 - s): since s > 0 and s <= 1023/1024,
//    result is in [1/1024, 1.0), fits in 10 bits → output[10]=0
// =============================================================
module stage4_select (
    input  wire [10:0] s,
    input  wire        sign_bit,    // 0 = original input was positive
    output wire [10:0] z
);

    // 1.0 in Q1.10 = 1024 (represented in 11 bits as 11'b10000000000)
    // but since we subtract a positive s < 1.0, result is positive < 1.0
    wire [10:0] one_minus_s;
    assign one_minus_s = 11'd1024 - s;   // 11-bit wraps correctly since s<=1023

    assign z = sign_bit ? one_minus_s : s;

endmodule
