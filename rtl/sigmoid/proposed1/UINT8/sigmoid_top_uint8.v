// ============================================================
// UINT8 SIGMOID TOP MODULE (PROPOSED LUT — EXTERNAL)
// ============================================================
// Generated from: Work/simulation_results/activation_segment_coefficients.csv
// Architecture: segment select -> external LUT -> multiply/add -> output mapping
// The LUT is a separate module (lut_sigmoid_uint8) instantiated below.
// x scale: x_scaled = x * 4096
// coefficient scale: coeff = real * 1048576
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
    reg signed [63:0] y_signed;
    reg signed [63:0] q_out;
    reg signed [63:0] q_mag;

    always @(*) begin
        casez (1'b1)
            (x_abs_scaled <= 32'd1887): seg = 4'd0;
            (x_abs_scaled <= 32'd3042): seg = 4'd1;
            (x_abs_scaled <= 32'd4064): seg = 4'd2;
            (x_abs_scaled <= 32'd5035): seg = 4'd3;
            (x_abs_scaled <= 32'd5995): seg = 4'd4;
            (x_abs_scaled <= 32'd6970): seg = 4'd5;
            (x_abs_scaled <= 32'd7982): seg = 4'd6;
            (x_abs_scaled <= 32'd9056): seg = 4'd7;
            (x_abs_scaled <= 32'd10218): seg = 4'd8;
            (x_abs_scaled <= 32'd11504): seg = 4'd9;
            (x_abs_scaled <= 32'd12967): seg = 4'd10;
            (x_abs_scaled <= 32'd14686): seg = 4'd11;
            (x_abs_scaled <= 32'd16802): seg = 4'd12;
            (x_abs_scaled <= 32'd19598): seg = 4'd13;
            (x_abs_scaled <= 32'd23802): seg = 4'd14;
            default: seg = 4'd15;
        endcase
    end

    // External LUT — slope and intercept accessed from separate module

    always @(*) begin
            case (seg)
                4'd0:  begin m_lut = 32'd258040; c_lut = 32'd524706;  end
                4'd1:  begin m_lut = 32'd239602; c_lut = 32'd533027;  end
                4'd2:  begin m_lut = 32'd218329; c_lut = 32'd548802;  end
                4'd3:  begin m_lut = 32'd195381; c_lut = 32'd571560;  end
                4'd4:  begin m_lut = 32'd171794; c_lut = 32'd600554;  end
                4'd5:  begin m_lut = 32'd148282; c_lut = 32'd634967;  end
                4'd6:  begin m_lut = 32'd125400; c_lut = 32'd673911;  end
                4'd7:  begin m_lut = 32'd103600; c_lut = 32'd716407;  end
                4'd8:  begin m_lut = 32'd83244;  c_lut = 32'd761427;  end
                4'd9:  begin m_lut = 32'd64641;  c_lut = 32'd807854;  end
                4'd10: begin m_lut = 32'd48038;  c_lut = 32'd854506;  end
                4'd11: begin m_lut = 32'd33649;  c_lut = 32'd900087;  end
                4'd12: begin m_lut = 32'd21649;  c_lut = 32'd943148;  end
                4'd13: begin m_lut = 32'd12174;  c_lut = 32'd982070;  end
                4'd14: begin m_lut = 32'd5327;   c_lut = 32'd1014908; end
                4'd15: begin m_lut = 32'd1179;   c_lut = 32'd1039162; end
                default: begin m_lut = 32'd1179;  c_lut = 32'd1039162; end
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
