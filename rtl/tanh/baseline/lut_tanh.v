// =============================================================
//  lut_tanh.v
//
//  Look-Up Table storing quantized slope (m) and intercept (c)
//  for the 16 tanh segments in the paper image.
//
//  Format: both m and c are positive Q0.10 magnitudes
//    value = integer / 1024.0
// =============================================================
module lut_tanh (
    input  wire [ 3:0] xindex,
    output reg  [10:0] m,
    output reg  [10:0] c
);

    always @(*) begin
        case (xindex)
            4'd0:  begin m = 11'd1022; c = 11'd0;   end
            4'd1:  begin m = 11'd1005; c = 11'd2;   end
            4'd2:  begin m = 11'd972;  c = 11'd8;   end
            4'd3:  begin m = 11'd925;  c = 11'd20;  end
            4'd4:  begin m = 11'd862;  c = 11'd44;  end
            4'd5:  begin m = 11'd787;  c = 11'd80;  end
            4'd6:  begin m = 11'd703;  c = 11'd128; end
            4'd7:  begin m = 11'd609;  c = 11'd193; end
            4'd8:  begin m = 11'd513;  c = 11'd272; end
            4'd9:  begin m = 11'd419;  c = 11'd360; end
            4'd10: begin m = 11'd327;  c = 11'd461; end
            4'd11: begin m = 11'd238;  c = 11'd573; end
            4'd12: begin m = 11'd158;  c = 11'd690; end
            4'd13: begin m = 11'd90;   c = 11'd808; end
            4'd14: begin m = 11'd36;   c = 11'd919; end
            4'd15: begin m = 11'd7;    c = 11'd997; end
            default: begin m = 11'd0;  c = 11'd0;   end
        endcase
    end

endmodule
