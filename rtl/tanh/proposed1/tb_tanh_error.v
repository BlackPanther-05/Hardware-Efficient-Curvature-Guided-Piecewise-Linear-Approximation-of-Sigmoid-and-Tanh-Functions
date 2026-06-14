`timescale 1ns/1ps
// =============================================================
//  tb_tanh_error.v
//  Proposed1 tanh normal error calculation.
// =============================================================
module tb_tanh_error;

    reg  [10:0] b;
    wire [10:0] z;
    tanh_top dut (.b(b), .z(z));

    integer i;
    integer b_int;
    integer z_int;
    integer fail_count;
    real    x_real;
    real    exp_pos;
    real    ref_real;
    real    z_real;
    real    abs_err;
    real    sum_err;
    real    max_err;
    real    avg_err;

    initial begin
        sum_err    = 0.0;
        max_err    = 0.0;
        fail_count = 0;

        $display("==============================================");
        $display(" Proposed1 Tanh RTL Error Test");
        $display(" Range: b_int = -1023 to +1023");
        $display("        x     = -7.9922 to +7.9922");
        $display("==============================================");
        $display("Vectors with error > 0.001:");
        $display("--------------------------------------------------------------");
        $display("  b_int    b_bin         z_int    z_float   err_float");
        $display("--------------------------------------------------------------");

        for (i = -1023; i <= 1023; i = i + 1) begin
            b_int = i;
            b = b_int[10:0];
            #10;

            x_real = $itor(b_int) / 128.0;
            exp_pos = $exp(2.0 * x_real);
            ref_real = (exp_pos - 1.0) / (exp_pos + 1.0);

            z_int = $signed(z);
            z_real = $itor(z_int) / 1024.0;
            abs_err = (z_real > ref_real) ? (z_real - ref_real)
                                          : (ref_real - z_real);

            sum_err = sum_err + abs_err;
            if (abs_err > max_err)
                max_err = abs_err;

            if (abs_err > 0.001) begin
                fail_count = fail_count + 1;
                $display("  %+6d   %b  %+6d  %f  %f",
                    b_int, b, z_int, z_real, abs_err);
            end
        end

        avg_err = sum_err / 2047.0;

        $display("--------------------------------------------------------------");
        $display("RESULTS  (x = -7.9922 to +7.9922, 2047 pts)");
        $display("  Total vectors tested      : 2047");
        $display("  Vectors with err > 0.001  : %0d", fail_count);
        $display("  Max error  (float)        : %f", max_err);
        $display("  Avg error  (float)        : %f", avg_err);
        $display("==============================================");

        $finish;
    end

endmodule
