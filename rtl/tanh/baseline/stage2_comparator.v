// =============================================================
//  stage2_comparator.v
//
//  Regression-tree comparator block for tanh positive half.
//  Boundaries are from the paper's tanh table, encoded as Q3.7
//  thresholds: round(x * 128).
// =============================================================
module tanh_stage2_comparator (
    input  wire [10:0] xp,
    output reg  [ 3:0] xindex,
    output wire        is_tail
);

    // Tail threshold: 3.827 * 128 = 489.856 -> 490
    assign is_tail = (xp >= 11'd490);

    always @(*) begin
        casez (1'b1)
            (xp < 11'd11)  : xindex = 4'd0;
            (xp < 11'd24)  : xindex = 4'd1;
            (xp < 11'd35)  : xindex = 4'd2;
            (xp < 11'd47)  : xindex = 4'd3;
            (xp < 11'd60)  : xindex = 4'd4;
            (xp < 11'd74)  : xindex = 4'd5;
            (xp < 11'd88)  : xindex = 4'd6;
            (xp < 11'd104) : xindex = 4'd7;
            (xp < 11'd121) : xindex = 4'd8;
            (xp < 11'd139) : xindex = 4'd9;
            (xp < 11'd161) : xindex = 4'd10;
            (xp < 11'd188) : xindex = 4'd11;
            (xp < 11'd219) : xindex = 4'd12;
            (xp < 11'd265) : xindex = 4'd13;
            (xp < 11'd341) : xindex = 4'd14;
            default         : xindex = 4'd15;
        endcase
    end

endmodule
