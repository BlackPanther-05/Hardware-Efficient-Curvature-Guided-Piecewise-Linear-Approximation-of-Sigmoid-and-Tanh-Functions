module tanh_lut_ext (
    input [3:0] seg,
    output reg [31:0] m_lut,
    output reg [31:0] c_lut
);
    always @(*) begin
        case (seg)
            4'd0: begin m_lut = 32'd1011588; c_lut = 32'd2849; end
            4'd1: begin m_lut = 32'd851885; c_lut = 32'd57550; end
            4'd2: begin m_lut = 32'd679308; c_lut = 32'd156771; end
            4'd3: begin m_lut = 32'd508517; c_lut = 32'd291877; end
            4'd4: begin m_lut = 32'd351730; c_lut = 32'd451451; end
            4'd5: begin m_lut = 32'd217479; c_lut = 32'd622443; end
            4'd6: begin m_lut = 32'd111833; c_lut = 32'd789972; end
            4'd7: begin m_lut = 32'd39050; c_lut = 32'd936374; end
            4'd8: begin m_lut = 32'd817; c_lut = 32'd1043313; end
            default: begin m_lut = 32'd817; c_lut = 32'd1043313; end
        endcase
    end
endmodule
