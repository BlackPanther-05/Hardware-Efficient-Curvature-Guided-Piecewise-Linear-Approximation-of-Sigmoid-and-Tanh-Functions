// ============================================================
// FP8 TANH TOP MODULE (PROPOSED 2 LUT — EXTERNAL)
// ============================================================
// Architecture: segment select -> external LUT -> multiply/add -> output mapping
// ============================================================

module tanh_top_fp8 (
    input        clk,
    input  [7:0] b,
    output reg [7:0] z
);

    wire signed [7:0] b_signed = b;
    wire sign_bit = b_signed < 0;
    wire [8:0] abs_raw = sign_bit ? -b_signed : b_signed;
    wire [31:0] x_abs_scaled = (abs_raw * 4096) / 16;

    reg [3:0] seg;
    reg [31:0] m_lut;
    reg [31:0] c_lut;
    reg signed [63:0] y_positive;
    reg signed [63:0] y_final;
    reg signed [63:0] q_out;

    always @(*) begin
        casez (1'b1)
            (x_abs_scaled <= 32'd1431): seg = 4'd0;
            (x_abs_scaled <= 32'd2357): seg = 4'd1;
            (x_abs_scaled <= 32'd3239): seg = 4'd2;
            (x_abs_scaled <= 32'd4165): seg = 4'd3;
            (x_abs_scaled <= 32'd5210): seg = 4'd4;
            (x_abs_scaled <= 32'd6480): seg = 4'd5;
            (x_abs_scaled <= 32'd8202): seg = 4'd6;
            (x_abs_scaled <= 32'd11068): seg = 4'd7;
            (x_abs_scaled <= 32'd32768): seg = 4'd8;
            default: seg = 4'd8;
        endcase
    end

    always @(*) begin
            case (seg)
                4'd0: begin m_lut = 32'd983040; c_lut = 32'd0; end
                4'd1: begin m_lut = 32'd851968; c_lut = 32'd65536; end
                4'd2: begin m_lut = 32'd655360; c_lut = 32'd131072; end
                4'd3: begin m_lut = 32'd524288; c_lut = 32'd262144; end
                4'd4: begin m_lut = 32'd327680; c_lut = 32'd458752; end
                4'd5: begin m_lut = 32'd196608; c_lut = 32'd589824; end
                4'd6: begin m_lut = 32'd131072; c_lut = 32'd786432; end
                4'd7: begin m_lut = 32'd65536; c_lut = 32'd917504; end
                4'd8: begin m_lut = 32'd0; c_lut = 32'd1048576; end
                default: begin m_lut = 32'd0; c_lut = 32'd1048576; end
            endcase
        end

always @(*) begin
        y_positive = ((m_lut * x_abs_scaled) / 4096) + c_lut;
        if (y_positive < 0) y_positive = 0;
        if (y_positive > 1048576) y_positive = 1048576;
        y_final = sign_bit ? -y_positive : y_positive;
        q_out = ((y_final * 127) + (524288)) / 1048576;
        if (q_out < -128) q_out = -128;
        if (q_out > 127) q_out = 127;
        z = q_out[7:0];
    end
endmodule
