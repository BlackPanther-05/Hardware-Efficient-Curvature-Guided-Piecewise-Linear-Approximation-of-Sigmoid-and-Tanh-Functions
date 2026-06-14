// ============================================================
// UINT8 TANH TOP MODULE (BASELINE LUT)
// ============================================================
// Generated from: Work/simulation_results/activation_segment_coefficients.csv
// Architecture: segment select -> slope/intercept LUT -> multiply/add -> output mapping
// x scale: x_scaled = x * 4096
// coefficient scale: coeff = real * 1048576
// ============================================================

module tanh_top_uint8 (
    input        clk,
    input  [7:0] b,
    output reg [7:0] z
);


    wire signed [31:0] centered = ($signed({1'b0, b}) * 2) - 32'sd255;
    wire sign_bit = centered < 0;
    wire [31:0] abs_raw = sign_bit ? -centered : centered;
    wire [31:0] x_abs_scaled = ((abs_raw * 4096) + 127) / 255;

    reg [3:0] seg;
    reg [31:0] m_lut;
    reg [31:0] c_lut;
    reg signed [63:0] y_positive;
    reg signed [63:0] y_final;
    reg signed [63:0] y_signed;
    reg signed [63:0] q_out;
    reg signed [63:0] q_mag;

    always @(*) begin
        casez (1'b1)
            (x_abs_scaled <= 32'd2048): seg = 4'd0;
            (x_abs_scaled <= 32'd4096): seg = 4'd1;
            (x_abs_scaled <= 32'd6144): seg = 4'd2;
            (x_abs_scaled <= 32'd8192): seg = 4'd3;
            (x_abs_scaled <= 32'd10240): seg = 4'd4;
            (x_abs_scaled <= 32'd12288): seg = 4'd5;
            (x_abs_scaled <= 32'd14336): seg = 4'd6;
            (x_abs_scaled <= 32'd16384): seg = 4'd7;
            (x_abs_scaled <= 32'd18432): seg = 4'd8;
            (x_abs_scaled <= 32'd20480): seg = 4'd9;
            (x_abs_scaled <= 32'd22528): seg = 4'd10;
            (x_abs_scaled <= 32'd24576): seg = 4'd11;
            (x_abs_scaled <= 32'd26624): seg = 4'd12;
            (x_abs_scaled <= 32'd28672): seg = 4'd13;
            (x_abs_scaled <= 32'd30720): seg = 4'd14;
            default: seg = 4'd15;
        endcase
    end

    always @(*) begin
        case (seg)
            4'd0: begin m_lut = 32'd975689; c_lut = 32'd7971; end
            4'd1: begin m_lut = 32'd627100; c_lut = 32'd187473; end
            4'd2: begin m_lut = 32'd298265; c_lut = 32'd511447; end
            4'd3: begin m_lut = 32'd121824; c_lut = 32'd771551; end
            4'd4: begin m_lut = 32'd46649; c_lut = 32'd919634; end
            4'd5: begin m_lut = 32'd17420; c_lut = 32'd991778; end
            4'd6: begin m_lut = 32'd6444; c_lut = 32'd1024352; end
            4'd7: begin m_lut = 32'd2376; c_lut = 32'd1038460; end
            4'd8: begin m_lut = 32'd875; c_lut = 32'd1044414; end
            4'd9: begin m_lut = 32'd322; c_lut = 32'd1046884; end
            4'd10: begin m_lut = 32'd118; c_lut = 32'd1047894; end
            4'd11: begin m_lut = 32'd44; c_lut = 32'd1048303; end
            4'd12: begin m_lut = 32'd16; c_lut = 32'd1048468; end
            4'd13: begin m_lut = 32'd6; c_lut = 32'd1048533; end
            4'd14: begin m_lut = 32'd2; c_lut = 32'd1048559; end
            4'd15: begin m_lut = 32'd1; c_lut = 32'd1048569; end
            default: begin m_lut = 32'd1; c_lut = 32'd1048569; end
        endcase
    end

    always @(*) begin
        y_positive = ((m_lut * x_abs_scaled) / 4096) + c_lut;
        if (y_positive < 0) y_positive = 0;
        if (y_positive > 1048576) y_positive = 1048576;

            y_signed = sign_bit ? -y_positive : y_positive;
            q_out = (((y_signed + 1048576) * 255) + 1048576) / (2097152);
            if (q_out < 0) q_out = 0;
            if (q_out > 255) q_out = 255;
            z = q_out[7:0];

    end

endmodule
