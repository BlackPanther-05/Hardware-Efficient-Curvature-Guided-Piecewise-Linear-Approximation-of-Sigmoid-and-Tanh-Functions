// ============================================================
// UINT8 TANH TOP MODULE (PROPOSED LUT — EXTERNAL)
// ============================================================
// Generated from: Work/simulation_results/activation_segment_coefficients.csv
// Architecture: segment select -> external LUT -> multiply/add -> output mapping
// The LUT is a separate module (lut_tanh_uint8) instantiated below.
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
            (x_abs_scaled <= 32'd963): seg = 4'd0;
            (x_abs_scaled <= 32'd1554): seg = 4'd1;
            (x_abs_scaled <= 32'd2078): seg = 4'd2;
            (x_abs_scaled <= 32'd2577): seg = 4'd3;
            (x_abs_scaled <= 32'd3072): seg = 4'd4;
            (x_abs_scaled <= 32'd3577): seg = 4'd5;
            (x_abs_scaled <= 32'd4105): seg = 4'd6;
            (x_abs_scaled <= 32'd4667): seg = 4'd7;
            (x_abs_scaled <= 32'd5281): seg = 4'd8;
            (x_abs_scaled <= 32'd5967): seg = 4'd9;
            (x_abs_scaled <= 32'd6759): seg = 4'd10;
            (x_abs_scaled <= 32'd7707): seg = 4'd11;
            (x_abs_scaled <= 32'd8910): seg = 4'd12;
            (x_abs_scaled <= 32'd10584): seg = 4'd13;
            (x_abs_scaled <= 32'd13417): seg = 4'd14;
            default: seg = 4'd15;
        endcase
    end

    // External LUT — slope and intercept accessed from separate module

    always @(*) begin
            case (seg)
                4'd0:  begin m_lut = 32'd1031471; c_lut = 32'd891;     end
                4'd1:  begin m_lut = 32'd954751;  c_lut = 32'd18566;   end
                4'd2:  begin m_lut = 32'd866453;  c_lut = 32'd52012;   end
                4'd3:  begin m_lut = 32'd771411;  c_lut = 32'd100199;  end
                4'd4:  begin m_lut = 32'd673970;  c_lut = 32'd161499;  end
                4'd5:  begin m_lut = 32'd577285;  c_lut = 32'd234027;  end
                4'd6:  begin m_lut = 32'd483645;  c_lut = 32'd315821;  end
                4'd7:  begin m_lut = 32'd394914;  c_lut = 32'd404763;  end
                4'd8:  begin m_lut = 32'd312647;  c_lut = 32'd498532;  end
                4'd9:  begin m_lut = 32'd238126;  c_lut = 32'd594654;  end
                4'd10: begin m_lut = 32'd172401;  c_lut = 32'd690461;  end
                4'd11: begin m_lut = 32'd116342;  c_lut = 32'd783036;  end
                4'd12: begin m_lut = 32'd70661;   c_lut = 32'd869086;  end
                4'd13: begin m_lut = 32'd35902;   c_lut = 32'd944832;  end
                4'd14: begin m_lut = 32'd12464;   c_lut = 32'd1005611; end
                4'd15: begin m_lut = 32'd318;     c_lut = 32'd1046468; end
                default: begin m_lut = 32'd318;   c_lut = 32'd1046468; end
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
