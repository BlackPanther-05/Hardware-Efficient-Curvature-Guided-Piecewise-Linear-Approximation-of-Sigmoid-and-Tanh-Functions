// =============================================================
//  lut_tanh.v
//  Proposed1 tanh sqrt(|f''|) regression table.
// =============================================================
module lut_tanh (
    input  wire [ 3:0] xindex,
    output reg  [10:0] m,
    output reg  [10:0] c
);

    always @(*) begin
        case (xindex)
            4'd0:  begin m = 11'd1010; c = 11'd1;    end
            4'd1:  begin m = 11'd938;  c = 11'd17;   end
            4'd2:  begin m = 11'd847;  c = 11'd51;   end
            4'd3:  begin m = 11'd756;  c = 11'd97;   end
            4'd4:  begin m = 11'd660;  c = 11'd157;  end
            4'd5:  begin m = 11'd570;  c = 11'd224;  end
            4'd6:  begin m = 11'd481;  c = 11'd301;  end
            4'd7:  begin m = 11'd384;  c = 11'd397;  end
            4'd8:  begin m = 11'd309;  c = 11'd483;  end
            4'd9:  begin m = 11'd235;  c = 11'd578;  end
            4'd10: begin m = 11'd172;  c = 11'd669;  end
            4'd11: begin m = 11'd116;  c = 11'd761;  end
            4'd12: begin m = 11'd70;   c = 11'd847;  end
            4'd13: begin m = 11'd36;   c = 11'd921;  end
            4'd14: begin m = 11'd11;   c = 11'd986;  end
            4'd15: begin m = 11'd5;    c = 11'd1005; end
            default: begin m = 11'd0;  c = 11'd0;    end
        endcase
    end

endmodule
