// ============================================================
// FP32 TANH TOP MODULE (PROPOSED 2 LUT — EXTERNAL)
// ============================================================
// Architecture: segment select -> external LUT -> multiply/add -> output mapping
// ============================================================

module tanh_top_fp32 (
    input        clk,
    input  [31:0] b,
    output reg [31:0] z
);

    wire signed [31:0] b_signed = b;
    wire sign_bit = b_signed < 0;
    wire [31:0] abs_raw = sign_bit ? -b_signed : b_signed;
    wire [31:0] x_abs_scaled = (abs_raw * 4096) / 64;

    reg [3:0] seg;
    reg [31:0] m_lut;
    reg [31:0] c_lut;
    // Constrain multiply operands to fit in one DSP48E1 (25x18 bits)
    wire [17:0] x_mult = (x_abs_scaled > 18'd262143) ? 18'd262143 : x_abs_scaled[17:0];
    reg [20:0] m_val;
    reg signed [63:0] y_positive;
    reg signed [63:0] y_final;
    reg signed [31:0] q_out;

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
            m_val = m_lut[20:0];
        end

    // Output mapping: use shift instead of multiply (y << 11 ≈ y * 2^31 / 2^20)
    always @(*) begin
        y_positive = ((m_val * x_mult) / 4096) + c_lut;
        if (y_positive < 0) y_positive = 0;
        if (y_positive > 1048575) y_positive = 1048575;
        y_final = sign_bit ? -y_positive : y_positive;
        q_out = (y_final <<< 11);
        z = q_out[31:0];
    end
endmodule
