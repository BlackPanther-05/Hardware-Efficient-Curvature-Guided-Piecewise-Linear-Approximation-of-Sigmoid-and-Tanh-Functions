`timescale 1ns/1ps
// ============================================================
// FP8 TANH ERROR TESTBENCH
// ============================================================
// Tests FP8-style fixed-point tanh over 256 points
// ============================================================

module tb_tanh_fp8_error;

    reg  [7:0] b;
    wire [7:0] z;
    cordic_p1_tanh_top_fp8 dut (.clk(clk), .b(b), .z(z));

    reg clk;
    integer i, fail_count;
    integer b_int;
    real x_real, ref_real, z_real, abs_err;
    real sum_err, max_err, avg_err;

    initial begin
        clk = 0;
        forever #5 clk = ~clk;
    end

    initial begin
        sum_err    = 0.0;
        max_err    = 0.0;
        fail_count = 0;

        $display("=====================================================");
        $display(" FP8 Tanh RTL Error Test");
        $display(" Format: 1s + 3e + 4m style, evaluated as /16 input");
        $display(" Range: x = -8.0 to 7.9375");
        $display("=====================================================");
        $display("Vectors with error > 0.02:");
        $display("-----------------------------------------------------");
        $display("  b_int    z_out   z_float   ref_float   error");
        $display("-----------------------------------------------------");

        for (i = -128; i <= 127; i = i + 1) begin
            b_int = i;
            b = b_int[7:0];
            #10;

            x_real = $itor($signed(b_int)) / 16.0;
            ref_real = (($exp(2.0 * x_real) - 1.0) / ($exp(2.0 * x_real) + 1.0));
            z_real = $itor($signed(z)) / 128.0;
            abs_err = (z_real > ref_real) ? (z_real - ref_real) : (ref_real - z_real);

            sum_err = sum_err + abs_err;
            if (abs_err > max_err)
                max_err = abs_err;

            if (abs_err > 0.02) begin
                fail_count = fail_count + 1;
                $display("  %+4d      %3d    %f  %f  %f",
                    b_int, $signed(z), z_real, ref_real, abs_err);
            end
        end

        avg_err = sum_err / 256.0;

        $display("-----------------------------------------------------");
        $display("RESULTS  (x = -8.0 to 7.9375, 256 pts)");
        $display("  Total vectors tested      : 256");
        $display("  Vectors with err > 0.02   : %0d", fail_count);
        $display("  Max error  (float)        : %f", max_err);
        $display("  Avg error  (float)        : %f", avg_err);
        $display("=====================================================");

        if (avg_err <= 0.02)
            $display("  Avg error <= 0.02: PASS");
        else
            $display("  Avg error <= 0.02: FAIL");

        if (max_err <= 0.10)
            $display("  Max error <= 0.10: PASS");
        else
            $display("  Max error <= 0.10: FAIL");

        $finish;
    end

endmodule
