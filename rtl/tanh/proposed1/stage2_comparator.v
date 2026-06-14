// =============================================================
//  stage2_comparator.v
//  Proposed1 tanh sqrt(|f''|) segment comparator.
// =============================================================
module tanh_stage2_comparator (
    input  wire [10:0] xp,
    output reg  [ 3:0] xindex,
    output wire        is_tail
);

    assign is_tail = 1'b0;

    always @(*) begin
        casez (1'b1)
            (xp < 11'd30)  : xindex = 4'd0;
            (xp < 11'd49)  : xindex = 4'd1;
            (xp < 11'd65)  : xindex = 4'd2;
            (xp < 11'd81)  : xindex = 4'd3;
            (xp < 11'd96)  : xindex = 4'd4;
            (xp < 11'd112) : xindex = 4'd5;
            (xp < 11'd128) : xindex = 4'd6;
            (xp < 11'd146) : xindex = 4'd7;
            (xp < 11'd165) : xindex = 4'd8;
            (xp < 11'd186) : xindex = 4'd9;
            (xp < 11'd211) : xindex = 4'd10;
            (xp < 11'd241) : xindex = 4'd11;
            (xp < 11'd278) : xindex = 4'd12;
            (xp < 11'd331) : xindex = 4'd13;
            (xp < 11'd419) : xindex = 4'd14;
            default         : xindex = 4'd15;
        endcase
    end

endmodule
