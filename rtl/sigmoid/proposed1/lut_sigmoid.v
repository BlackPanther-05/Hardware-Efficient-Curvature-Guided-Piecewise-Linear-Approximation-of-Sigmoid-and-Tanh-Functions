// =============================================================
//  lut_sigmoid.v
//  Proposed1 sigmoid sqrt(|f''|) regression table.
//  m and c are Q0.10 positive values.
// =============================================================
module lut_sigmoid (
    input  wire [ 3:0] xindex,
    output reg  [10:0] m,
    output reg  [10:0] c
);

    always @(*) begin
        case (xindex)
            4'd0:  begin m = 11'd252; c = 11'd513;  end
            4'd1:  begin m = 11'd234; c = 11'd521;  end
            4'd2:  begin m = 11'd215; c = 11'd535;  end
            4'd3:  begin m = 11'd192; c = 11'd557;  end
            4'd4:  begin m = 11'd167; c = 11'd588;  end
            4'd5:  begin m = 11'd142; c = 11'd625;  end
            4'd6:  begin m = 11'd125; c = 11'd654;  end
            4'd7:  begin m = 11'd105; c = 11'd692;  end
            4'd8:  begin m = 11'd80;  c = 11'd747;  end
            4'd9:  begin m = 11'd64;  c = 11'd787;  end
            4'd10: begin m = 11'd49;  c = 11'd829;  end
            4'd11: begin m = 11'd33;  c = 11'd879;  end
            4'd12: begin m = 11'd21;  c = 11'd922;  end
            4'd13: begin m = 11'd12;  c = 11'd959;  end
            4'd14: begin m = 11'd5;   c = 11'd993;  end
            4'd15: begin m = 11'd2;   c = 11'd1010; end
            default: begin m = 11'd0; c = 11'd512;  end
        endcase
    end

endmodule
