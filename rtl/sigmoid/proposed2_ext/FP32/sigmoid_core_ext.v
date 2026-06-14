module sigmoid_core_ext (
    input  [7:0] b,
    input  [31:0] m_lut,
    input  [31:0] c_lut,
    output reg [3:0] seg,
    output reg [7:0] z
);
    wire signed [7:0] b_signed = b;
    wire sign_bit = b_signed < 0;
    wire [8:0] abs_raw = sign_bit ? -b_signed : b_signed;
    wire [31:0] x_abs_scaled = (abs_raw * 4096) / 16;

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
        y_positive = ((m_lut * x_abs_scaled) / 4096) + c_lut;
        if (y_positive < 0) y_positive = 0;
        if (y_positive > 1048576) y_positive = 1048576;
        y_final = sign_bit ? (1048576 - y_positive) : y_positive;
        q_out = ((y_final * 128) + (524288)) / 1048576;
        if (q_out < 0) q_out = 0;
        if (q_out > 127) q_out = 127;
        z = q_out[7:0];
    end
endmodule
