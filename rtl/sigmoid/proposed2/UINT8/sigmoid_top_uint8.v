// ============================================================
// UINT8 SIGMOID TOP MODULE (PROPOSED 2 LUT — EXTERNAL)
// ============================================================
// Architecture: segment select -> external LUT -> multiply/add -> output mapping
// ============================================================

module sigmoid_top_uint8 (
    input        clk,
    input  [7:0] b,
    output reg [7:0] z
);

    wire signed [31:0] centered = ($signed({1'b0, b}) * 16) - 32'sd2040;
    wire sign_bit = centered < 0;
    wire [31:0] abs_raw = sign_bit ? -centered : centered;
    wire [31:0] x_abs_scaled = ((abs_raw * 4096) + 127) / 255;

    reg [3:0] seg;
    reg [31:0] m_lut;
    reg [31:0] c_lut;
    reg signed [63:0] y_positive;
    reg signed [63:0] y_final;
    reg signed [63:0] q_out;

    always @(*) begin
        casez (1'b1)
            (x_abs_scaled <= 32'd2066): seg = 4'd0;
            (x_abs_scaled <= 32'd3342): seg = 4'd1;
            (x_abs_scaled <= 32'd4483): seg = 4'd2;
            (x_abs_scaled <= 32'd5583): seg = 4'd3;
            (x_abs_scaled <= 32'd6689): seg = 4'd4;
            (x_abs_scaled <= 32'd7834): seg = 4'd5;
            (x_abs_scaled <= 32'd9056): seg = 4'd6;
            (x_abs_scaled <= 32'd10393): seg = 4'd7;
            (x_abs_scaled <= 32'd11901): seg = 4'd8;
            (x_abs_scaled <= 32'd13665): seg = 4'd9;
            (x_abs_scaled <= 32'd15833): seg = 4'd10;
            (x_abs_scaled <= 32'd18702): seg = 4'd11;
            (x_abs_scaled <= 32'd23056): seg = 4'd12;
            (x_abs_scaled <= 32'd32768): seg = 4'd13;
            default: seg = 4'd13;
        endcase
    end

    always @(*) begin
            case (seg)
                4'd0: begin m_lut = 32'd257238; c_lut = 32'd524836; end
                4'd1: begin m_lut = 32'd235330; c_lut = 32'd535664; end
                4'd2: begin m_lut = 32'd210299; c_lut = 32'd556060; end
                4'd3: begin m_lut = 32'd183626; c_lut = 32'd585245; end
                4'd4: begin m_lut = 32'd156617; c_lut = 32'd622063; end
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

always @(*) begin
        y_positive = ((m_lut * x_abs_scaled) / 4096) + c_lut;
        if (y_positive < 0) y_positive = 0;
        if (y_positive > 1048576) y_positive = 1048576;
        y_final = sign_bit ? (1048576 - y_positive) : y_positive;
        q_out = ((y_final * 255) + (524288)) / 1048576;
        if (q_out < 0) q_out = 0;
        if (q_out > 255) q_out = 255;
        z = q_out[7:0];
    end
endmodule
