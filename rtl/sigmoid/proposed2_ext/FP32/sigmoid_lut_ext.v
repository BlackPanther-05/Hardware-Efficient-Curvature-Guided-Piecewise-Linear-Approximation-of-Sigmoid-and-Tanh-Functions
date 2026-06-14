module sigmoid_lut_ext (
    input [3:0] seg,
    output reg [31:0] m_lut,
    output reg [31:0] c_lut
);
    always @(*) begin
        case (seg)
            4'd0: begin m_lut = 32'd257238; c_lut = 32'd524836; end
            4'd1: begin m_lut = 32'd235330; c_lut = 32'd535664; end
            4'd2: begin m_lut = 32'd210299; c_lut = 32'd556060; end
            4'd3: begin m_lut = 32'd183626; c_lut = 32'd585245; end
            4'd4: begin m_lut = 32'd156617; c_lut = 32'd622062; end
            4'd5: begin m_lut = 32'd130204; c_lut = 32'd665203; end
            4'd6: begin m_lut = 32'd105084; c_lut = 32'd713265; end
            4'd7: begin m_lut = 32'd81814; c_lut = 32'd764733; end
            4'd8: begin m_lut = 32'd60859; c_lut = 32'd817932; end
            4'd9: begin m_lut = 32'd42584; c_lut = 32'd871068; end
            4'd10: begin m_lut = 32'd27286; c_lut = 32'd922158; end
            4'd11: begin m_lut = 32'd15200; c_lut = 32'd968942; end
            4'd12: begin m_lut = 32'd6506; c_lut = 32'd1008746; end
            4'd13: begin m_lut = 32'd1317; c_lut = 32'd1038162; end
            default: begin m_lut = 32'd1317; c_lut = 32'd1038162; end
        endcase
    end
endmodule
