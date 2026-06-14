module cordic_p1_core (
    input clk,
    input signed [31:0] b,
    input is_tanh,
    output reg signed [31:0] z
);
    // Input b is Q6.26? No, b is an integer where x = b/64.
    // So b is Q26.6. We want to convert to Q12.20.
    wire signed [31:0] x_in = (b <<< 14);
    // RHC pipeline to compute exp(-x)
    reg signed [31:0] rhc_x [0:17];
    reg signed [31:0] rhc_y [0:17];
    reg signed [31:0] rhc_z [0:17];
    always @(*) begin
        rhc_x[0] = 1266152;
        rhc_y[0] = 0;
        rhc_z[0] = -x_in;
        if (rhc_z[0] < 0) begin
            rhc_x[1] = rhc_x[0] - (rhc_y[0] >>> 1);
            rhc_y[1] = rhc_y[0] - (rhc_x[0] >>> 1);
            rhc_z[1] = rhc_z[0] + 575989;
        end else begin
            rhc_x[1] = rhc_x[0] + (rhc_y[0] >>> 1);
            rhc_y[1] = rhc_y[0] + (rhc_x[0] >>> 1);
            rhc_z[1] = rhc_z[0] - 575989;
        end
        if (rhc_z[1] < 0) begin
            rhc_x[2] = rhc_x[1] - (rhc_y[1] >>> 2);
            rhc_y[2] = rhc_y[1] - (rhc_x[1] >>> 2);
            rhc_z[2] = rhc_z[1] + 267819;
        end else begin
            rhc_x[2] = rhc_x[1] + (rhc_y[1] >>> 2);
            rhc_y[2] = rhc_y[1] + (rhc_x[1] >>> 2);
            rhc_z[2] = rhc_z[1] - 267819;
        end
        if (rhc_z[2] < 0) begin
            rhc_x[3] = rhc_x[2] - (rhc_y[2] >>> 3);
            rhc_y[3] = rhc_y[2] - (rhc_x[2] >>> 3);
            rhc_z[3] = rhc_z[2] + 131761;
        end else begin
            rhc_x[3] = rhc_x[2] + (rhc_y[2] >>> 3);
            rhc_y[3] = rhc_y[2] + (rhc_x[2] >>> 3);
            rhc_z[3] = rhc_z[2] - 131761;
        end
        if (rhc_z[3] < 0) begin
            rhc_x[4] = rhc_x[3] - (rhc_y[3] >>> 4);
            rhc_y[4] = rhc_y[3] - (rhc_x[3] >>> 4);
            rhc_z[4] = rhc_z[3] + 65621;
        end else begin
            rhc_x[4] = rhc_x[3] + (rhc_y[3] >>> 4);
            rhc_y[4] = rhc_y[3] + (rhc_x[3] >>> 4);
            rhc_z[4] = rhc_z[3] - 65621;
        end
        if (rhc_z[4] < 0) begin
            rhc_x[5] = rhc_x[4] - (rhc_y[4] >>> 4);
            rhc_y[5] = rhc_y[4] - (rhc_x[4] >>> 4);
            rhc_z[5] = rhc_z[4] + 65621;
        end else begin
            rhc_x[5] = rhc_x[4] + (rhc_y[4] >>> 4);
            rhc_y[5] = rhc_y[4] + (rhc_x[4] >>> 4);
            rhc_z[5] = rhc_z[4] - 65621;
        end
        if (rhc_z[5] < 0) begin
            rhc_x[6] = rhc_x[5] - (rhc_y[5] >>> 5);
            rhc_y[6] = rhc_y[5] - (rhc_x[5] >>> 5);
            rhc_z[6] = rhc_z[5] + 32778;
        end else begin
            rhc_x[6] = rhc_x[5] + (rhc_y[5] >>> 5);
            rhc_y[6] = rhc_y[5] + (rhc_x[5] >>> 5);
            rhc_z[6] = rhc_z[5] - 32778;
        end
        if (rhc_z[6] < 0) begin
            rhc_x[7] = rhc_x[6] - (rhc_y[6] >>> 6);
            rhc_y[7] = rhc_y[6] - (rhc_x[6] >>> 6);
            rhc_z[7] = rhc_z[6] + 16385;
        end else begin
            rhc_x[7] = rhc_x[6] + (rhc_y[6] >>> 6);
            rhc_y[7] = rhc_y[6] + (rhc_x[6] >>> 6);
            rhc_z[7] = rhc_z[6] - 16385;
        end
        if (rhc_z[7] < 0) begin
            rhc_x[8] = rhc_x[7] - (rhc_y[7] >>> 7);
            rhc_y[8] = rhc_y[7] - (rhc_x[7] >>> 7);
            rhc_z[8] = rhc_z[7] + 8192;
        end else begin
            rhc_x[8] = rhc_x[7] + (rhc_y[7] >>> 7);
            rhc_y[8] = rhc_y[7] + (rhc_x[7] >>> 7);
            rhc_z[8] = rhc_z[7] - 8192;
        end
        if (rhc_z[8] < 0) begin
            rhc_x[9] = rhc_x[8] - (rhc_y[8] >>> 8);
            rhc_y[9] = rhc_y[8] - (rhc_x[8] >>> 8);
            rhc_z[9] = rhc_z[8] + 4096;
        end else begin
            rhc_x[9] = rhc_x[8] + (rhc_y[8] >>> 8);
            rhc_y[9] = rhc_y[8] + (rhc_x[8] >>> 8);
            rhc_z[9] = rhc_z[8] - 4096;
        end
        if (rhc_z[9] < 0) begin
            rhc_x[10] = rhc_x[9] - (rhc_y[9] >>> 9);
            rhc_y[10] = rhc_y[9] - (rhc_x[9] >>> 9);
            rhc_z[10] = rhc_z[9] + 2048;
        end else begin
            rhc_x[10] = rhc_x[9] + (rhc_y[9] >>> 9);
            rhc_y[10] = rhc_y[9] + (rhc_x[9] >>> 9);
            rhc_z[10] = rhc_z[9] - 2048;
        end
        if (rhc_z[10] < 0) begin
            rhc_x[11] = rhc_x[10] - (rhc_y[10] >>> 10);
            rhc_y[11] = rhc_y[10] - (rhc_x[10] >>> 10);
            rhc_z[11] = rhc_z[10] + 1024;
        end else begin
            rhc_x[11] = rhc_x[10] + (rhc_y[10] >>> 10);
            rhc_y[11] = rhc_y[10] + (rhc_x[10] >>> 10);
            rhc_z[11] = rhc_z[10] - 1024;
        end
        if (rhc_z[11] < 0) begin
            rhc_x[12] = rhc_x[11] - (rhc_y[11] >>> 11);
            rhc_y[12] = rhc_y[11] - (rhc_x[11] >>> 11);
            rhc_z[12] = rhc_z[11] + 512;
        end else begin
            rhc_x[12] = rhc_x[11] + (rhc_y[11] >>> 11);
            rhc_y[12] = rhc_y[11] + (rhc_x[11] >>> 11);
            rhc_z[12] = rhc_z[11] - 512;
        end
        if (rhc_z[12] < 0) begin
            rhc_x[13] = rhc_x[12] - (rhc_y[12] >>> 12);
            rhc_y[13] = rhc_y[12] - (rhc_x[12] >>> 12);
            rhc_z[13] = rhc_z[12] + 256;
        end else begin
            rhc_x[13] = rhc_x[12] + (rhc_y[12] >>> 12);
            rhc_y[13] = rhc_y[12] + (rhc_x[12] >>> 12);
            rhc_z[13] = rhc_z[12] - 256;
        end
        if (rhc_z[13] < 0) begin
            rhc_x[14] = rhc_x[13] - (rhc_y[13] >>> 13);
            rhc_y[14] = rhc_y[13] - (rhc_x[13] >>> 13);
            rhc_z[14] = rhc_z[13] + 128;
        end else begin
            rhc_x[14] = rhc_x[13] + (rhc_y[13] >>> 13);
            rhc_y[14] = rhc_y[13] + (rhc_x[13] >>> 13);
            rhc_z[14] = rhc_z[13] - 128;
        end
        if (rhc_z[14] < 0) begin
            rhc_x[15] = rhc_x[14] - (rhc_y[14] >>> 13);
            rhc_y[15] = rhc_y[14] - (rhc_x[14] >>> 13);
            rhc_z[15] = rhc_z[14] + 128;
        end else begin
            rhc_x[15] = rhc_x[14] + (rhc_y[14] >>> 13);
            rhc_y[15] = rhc_y[14] + (rhc_x[14] >>> 13);
            rhc_z[15] = rhc_z[14] - 128;
        end
        if (rhc_z[15] < 0) begin
            rhc_x[16] = rhc_x[15] - (rhc_y[15] >>> 14);
            rhc_y[16] = rhc_y[15] - (rhc_x[15] >>> 14);
            rhc_z[16] = rhc_z[15] + 64;
        end else begin
            rhc_x[16] = rhc_x[15] + (rhc_y[15] >>> 14);
            rhc_y[16] = rhc_y[15] + (rhc_x[15] >>> 14);
            rhc_z[16] = rhc_z[15] - 64;
        end
        if (rhc_z[16] < 0) begin
            rhc_x[17] = rhc_x[16] - (rhc_y[16] >>> 15);
            rhc_y[17] = rhc_y[16] - (rhc_x[16] >>> 15);
            rhc_z[17] = rhc_z[16] + 32;
        end else begin
            rhc_x[17] = rhc_x[16] + (rhc_y[16] >>> 15);
            rhc_y[17] = rhc_y[16] + (rhc_x[16] >>> 15);
            rhc_z[17] = rhc_z[16] - 32;
        end
    end
    wire signed [31:0] exp_neg_x = rhc_x[17] + rhc_y[17];
    reg signed [31:0] vlc_x [0:15];
    reg signed [31:0] vlc_y [0:15];
    reg signed [31:0] vlc_z [0:15];
    always @(*) begin
        vlc_x[0] = 1048576 + exp_neg_x;
        vlc_y[0] = 1048576;
        vlc_z[0] = 0;
        if (vlc_y[0] < 0) begin
            vlc_x[1] = vlc_x[0];
            vlc_y[1] = vlc_y[0] + (vlc_x[0] >>> 1);
            vlc_z[1] = vlc_z[0] - (1048576 >>> 1);
        end else begin
            vlc_x[1] = vlc_x[0];
            vlc_y[1] = vlc_y[0] - (vlc_x[0] >>> 1);
            vlc_z[1] = vlc_z[0] + (1048576 >>> 1);
        end
        if (vlc_y[1] < 0) begin
            vlc_x[2] = vlc_x[1];
            vlc_y[2] = vlc_y[1] + (vlc_x[1] >>> 2);
            vlc_z[2] = vlc_z[1] - (1048576 >>> 2);
        end else begin
            vlc_x[2] = vlc_x[1];
            vlc_y[2] = vlc_y[1] - (vlc_x[1] >>> 2);
            vlc_z[2] = vlc_z[1] + (1048576 >>> 2);
        end
        if (vlc_y[2] < 0) begin
            vlc_x[3] = vlc_x[2];
            vlc_y[3] = vlc_y[2] + (vlc_x[2] >>> 3);
            vlc_z[3] = vlc_z[2] - (1048576 >>> 3);
        end else begin
            vlc_x[3] = vlc_x[2];
            vlc_y[3] = vlc_y[2] - (vlc_x[2] >>> 3);
            vlc_z[3] = vlc_z[2] + (1048576 >>> 3);
        end
        if (vlc_y[3] < 0) begin
            vlc_x[4] = vlc_x[3];
            vlc_y[4] = vlc_y[3] + (vlc_x[3] >>> 4);
            vlc_z[4] = vlc_z[3] - (1048576 >>> 4);
        end else begin
            vlc_x[4] = vlc_x[3];
            vlc_y[4] = vlc_y[3] - (vlc_x[3] >>> 4);
            vlc_z[4] = vlc_z[3] + (1048576 >>> 4);
        end
        if (vlc_y[4] < 0) begin
            vlc_x[5] = vlc_x[4];
            vlc_y[5] = vlc_y[4] + (vlc_x[4] >>> 5);
            vlc_z[5] = vlc_z[4] - (1048576 >>> 5);
        end else begin
            vlc_x[5] = vlc_x[4];
            vlc_y[5] = vlc_y[4] - (vlc_x[4] >>> 5);
            vlc_z[5] = vlc_z[4] + (1048576 >>> 5);
        end
        if (vlc_y[5] < 0) begin
            vlc_x[6] = vlc_x[5];
            vlc_y[6] = vlc_y[5] + (vlc_x[5] >>> 6);
            vlc_z[6] = vlc_z[5] - (1048576 >>> 6);
        end else begin
            vlc_x[6] = vlc_x[5];
            vlc_y[6] = vlc_y[5] - (vlc_x[5] >>> 6);
            vlc_z[6] = vlc_z[5] + (1048576 >>> 6);
        end
        if (vlc_y[6] < 0) begin
            vlc_x[7] = vlc_x[6];
            vlc_y[7] = vlc_y[6] + (vlc_x[6] >>> 7);
            vlc_z[7] = vlc_z[6] - (1048576 >>> 7);
        end else begin
            vlc_x[7] = vlc_x[6];
            vlc_y[7] = vlc_y[6] - (vlc_x[6] >>> 7);
            vlc_z[7] = vlc_z[6] + (1048576 >>> 7);
        end
        if (vlc_y[7] < 0) begin
            vlc_x[8] = vlc_x[7];
            vlc_y[8] = vlc_y[7] + (vlc_x[7] >>> 8);
            vlc_z[8] = vlc_z[7] - (1048576 >>> 8);
        end else begin
            vlc_x[8] = vlc_x[7];
            vlc_y[8] = vlc_y[7] - (vlc_x[7] >>> 8);
            vlc_z[8] = vlc_z[7] + (1048576 >>> 8);
        end
        if (vlc_y[8] < 0) begin
            vlc_x[9] = vlc_x[8];
            vlc_y[9] = vlc_y[8] + (vlc_x[8] >>> 9);
            vlc_z[9] = vlc_z[8] - (1048576 >>> 9);
        end else begin
            vlc_x[9] = vlc_x[8];
            vlc_y[9] = vlc_y[8] - (vlc_x[8] >>> 9);
            vlc_z[9] = vlc_z[8] + (1048576 >>> 9);
        end
        if (vlc_y[9] < 0) begin
            vlc_x[10] = vlc_x[9];
            vlc_y[10] = vlc_y[9] + (vlc_x[9] >>> 10);
            vlc_z[10] = vlc_z[9] - (1048576 >>> 10);
        end else begin
            vlc_x[10] = vlc_x[9];
            vlc_y[10] = vlc_y[9] - (vlc_x[9] >>> 10);
            vlc_z[10] = vlc_z[9] + (1048576 >>> 10);
        end
        if (vlc_y[10] < 0) begin
            vlc_x[11] = vlc_x[10];
            vlc_y[11] = vlc_y[10] + (vlc_x[10] >>> 11);
            vlc_z[11] = vlc_z[10] - (1048576 >>> 11);
        end else begin
            vlc_x[11] = vlc_x[10];
            vlc_y[11] = vlc_y[10] - (vlc_x[10] >>> 11);
            vlc_z[11] = vlc_z[10] + (1048576 >>> 11);
        end
        if (vlc_y[11] < 0) begin
            vlc_x[12] = vlc_x[11];
            vlc_y[12] = vlc_y[11] + (vlc_x[11] >>> 12);
            vlc_z[12] = vlc_z[11] - (1048576 >>> 12);
        end else begin
            vlc_x[12] = vlc_x[11];
            vlc_y[12] = vlc_y[11] - (vlc_x[11] >>> 12);
            vlc_z[12] = vlc_z[11] + (1048576 >>> 12);
        end
        if (vlc_y[12] < 0) begin
            vlc_x[13] = vlc_x[12];
            vlc_y[13] = vlc_y[12] + (vlc_x[12] >>> 13);
            vlc_z[13] = vlc_z[12] - (1048576 >>> 13);
        end else begin
            vlc_x[13] = vlc_x[12];
            vlc_y[13] = vlc_y[12] - (vlc_x[12] >>> 13);
            vlc_z[13] = vlc_z[12] + (1048576 >>> 13);
        end
        if (vlc_y[13] < 0) begin
            vlc_x[14] = vlc_x[13];
            vlc_y[14] = vlc_y[13] + (vlc_x[13] >>> 14);
            vlc_z[14] = vlc_z[13] - (1048576 >>> 14);
        end else begin
            vlc_x[14] = vlc_x[13];
            vlc_y[14] = vlc_y[13] - (vlc_x[13] >>> 14);
            vlc_z[14] = vlc_z[13] + (1048576 >>> 14);
        end
        if (vlc_y[14] < 0) begin
            vlc_x[15] = vlc_x[14];
            vlc_y[15] = vlc_y[14] + (vlc_x[14] >>> 15);
            vlc_z[15] = vlc_z[14] - (1048576 >>> 15);
        end else begin
            vlc_x[15] = vlc_x[14];
            vlc_y[15] = vlc_y[14] - (vlc_x[14] >>> 15);
            vlc_z[15] = vlc_z[14] + (1048576 >>> 15);
        end
    end
    wire signed [31:0] sigmoid_out = vlc_z[15];
    // Scale output to Q0.31 format for z. 1.0 is 2147483647.
    // sigmoid_out is in Q12.20. We need to shift left by 11 to get Q1.31.
    wire signed [31:0] sig_z = (sigmoid_out <<< 11);
    wire signed [31:0] tanh_z = (sigmoid_out <<< 12) - 32'd2147483647;
    
    always @(*) begin
        z = is_tanh ? tanh_z : sig_z;
    end
endmodule
