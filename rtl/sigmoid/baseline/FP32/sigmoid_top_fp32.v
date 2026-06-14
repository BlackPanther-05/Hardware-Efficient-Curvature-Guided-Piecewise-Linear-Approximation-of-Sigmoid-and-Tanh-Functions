// ============================================================
// FP32 SIGMOID TOP MODULE (BASELINE LUT)
// ============================================================
// Generated from: Work/simulation_results/activation_segment_coefficients.csv
// Architecture: segment select -> slope/intercept LUT -> multiply/add -> output mapping
// x scale: x_scaled = x * 4096
// coefficient scale: coeff = real * 1048576
// ============================================================

module sigmoid_top_fp32 (
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
            4'd0: begin m_lut = 32'd257324; c_lut = 32'd524821; end
            4'd1: begin m_lut = 32'd228041; c_lut = 32'd540291; end
            4'd2: begin m_lut = 32'd181468; c_lut = 32'd587177; end
            4'd3: begin m_lut = 32'd132456; c_lut = 32'd660571; end
            4'd4: begin m_lut = 32'd90720; c_lut = 32'd743711; end
            4'd5: begin m_lut = 32'd59464; c_lut = 32'd821500; end
            4'd6: begin m_lut = 32'd37864; c_lut = 32'd886014; end
            4'd7: begin m_lut = 32'd23667; c_lut = 32'd935495; end
            4'd8: begin m_lut = 32'd14623; c_lut = 32'd971534; end
            4'd9: begin m_lut = 32'd8970; c_lut = 32'd996883; end
            4'd10: begin m_lut = 32'd5478; c_lut = 32'd1014285; end
            4'd11: begin m_lut = 32'd3337; c_lut = 32'd1026029; end
            4'd12: begin m_lut = 32'd2029; c_lut = 32'd1033854; end
            4'd13: begin m_lut = 32'd1233; c_lut = 32'd1039017; end
            4'd14: begin m_lut = 32'd748; c_lut = 32'd1042399; end
            4'd15: begin m_lut = 32'd454; c_lut = 32'd1044601; end
            default: begin m_lut = 32'd454; c_lut = 32'd1044601; end
        endcase
    end

    always @(*) begin
        y_positive = ((m_lut * x_abs_scaled) / 4096) + c_lut;
        if (y_positive < 0) y_positive = 0;
        if (y_positive > 1048576) y_positive = 1048576;

            y_final = sign_bit ? (1048576 - y_positive) : y_positive;

            q_out = ((y_final * 2147483647) + (524288)) / 1048576;
            if (q_out < 0) q_out = 0;
            if (q_out > 2147483647) q_out = 2147483647;
            z = q_out[31:0];

    end

endmodule
