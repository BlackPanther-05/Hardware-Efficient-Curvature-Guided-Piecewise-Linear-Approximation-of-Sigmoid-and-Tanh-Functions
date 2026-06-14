// =============================================================
//  stage2_comparator.v
//
//  Regression-tree comparator block.
//  Compares xp (S1.I3.F7, 11-bit, always positive from Stage 1)
//  against 17 boundary thresholds and outputs a 4-bit xindex
//  that addresses the LUT.
//
//  Boundaries encoded in Q-format matching xp: multiply float by
//  2^7 = 128  and round → 11-bit unsigned integer comparison.
//
//  Boundary table (from Python script, x * 128):
//   Seg  x1_float   x1_int
//    0    0.000        0      → xindex = 0
//    1    0.166       21      → xindex = 1
//    2    0.344       44      → xindex = 2
//    3    0.520       67      → xindex = 3
//    4    0.704       90      → xindex = 4
//    5    0.888      114      → xindex = 5
//    6    1.089      139      → xindex = 6
//    7    1.305      167      → xindex = 7
//    8    1.537      197      → xindex = 8
//    9    1.777      227      → xindex = 9
//   10    2.042      261      → xindex = 10
//   11    2.354      301      → xindex = 11
//   12    2.730      349      → xindex = 12
//   13    3.187      408      → xindex = 13
//   14    3.811      488      → xindex = 14
//   15    4.804      615      → xindex = 15
//   16    6.443      825      → TAIL flag (use constant 0.999)
//
//  xindex  = 4-bit segment address (0..15)
//  is_tail = 1 when xp >= 825 (x >= 6.443), output constant
//
//  Purely combinational (priority encoder style).
// =============================================================
module stage2_comparator (
    input  wire [10:0] xp,
    output reg  [ 3:0] xindex,
    output wire        is_tail
);

    // Tail threshold: 6.443 * 128 = 824.704 → 825
    assign is_tail = (xp >= 11'd825);

    always @(*) begin
        casez (1'b1)
            (xp < 11'd21)  : xindex = 4'd0;
            (xp < 11'd44)  : xindex = 4'd1;
            (xp < 11'd67)  : xindex = 4'd2;
            (xp < 11'd90)  : xindex = 4'd3;
            (xp < 11'd114) : xindex = 4'd4;
            (xp < 11'd139) : xindex = 4'd5;
            (xp < 11'd167) : xindex = 4'd6;
            (xp < 11'd197) : xindex = 4'd7;
            (xp < 11'd227) : xindex = 4'd8;
            (xp < 11'd261) : xindex = 4'd9;
            (xp < 11'd301) : xindex = 4'd10;
            (xp < 11'd349) : xindex = 4'd11;
            (xp < 11'd408) : xindex = 4'd12;
            (xp < 11'd488) : xindex = 4'd13;
            (xp < 11'd615) : xindex = 4'd14;
            default         : xindex = 4'd15;
        endcase
    end

endmodule
