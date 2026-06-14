// =============================================================
//  stage2_comparator.v
//  Proposed1 sigmoid sqrt(|f''|) segment comparator.
//  Boundaries are Q3.7 integers: round(x * 128).
// =============================================================
module stage2_comparator (
    input  wire [10:0] xp,
    output reg  [ 3:0] xindex,
    output wire        is_tail
);

    assign is_tail = 1'b0;

    always @(*) begin
        casez (1'b1)
            (xp < 11'd59)  : xindex = 4'd0;
            (xp < 11'd95)  : xindex = 4'd1;
            (xp < 11'd127) : xindex = 4'd2;
            (xp < 11'd157) : xindex = 4'd3;
            (xp < 11'd187) : xindex = 4'd4;
            (xp < 11'd218) : xindex = 4'd5;
            (xp < 11'd249) : xindex = 4'd6;
            (xp < 11'd283) : xindex = 4'd7;
            (xp < 11'd319) : xindex = 4'd8;
            (xp < 11'd359) : xindex = 4'd9;
            (xp < 11'd405) : xindex = 4'd10;
            (xp < 11'd459) : xindex = 4'd11;
            (xp < 11'd525) : xindex = 4'd12;
            (xp < 11'd612) : xindex = 4'd13;
            (xp < 11'd744) : xindex = 4'd14;
            default         : xindex = 4'd15;
        endcase
    end

endmodule
