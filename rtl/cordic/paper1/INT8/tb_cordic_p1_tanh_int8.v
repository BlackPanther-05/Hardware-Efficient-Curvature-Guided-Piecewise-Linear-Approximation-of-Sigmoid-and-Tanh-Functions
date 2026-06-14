`timescale 1ns/1ps
// ============================================================
// INT8 TANH ERROR TESTBENCH
// ============================================================
// Tests INT8 format tanh over all 256 input values
// Signed 8-bit with 1/16 input scaling
// ============================================================

module tb_tanh_int8_error;

    reg  [7:0] b;
    wire [7:0] z;
    cordic_p1_tanh_top_int8 dut (.clk(clk), .b(b), .z(z));
    
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
        $display(" INT8 Tanh RTL Error Test");
        $display(" Format: Signed 8-bit, 1/16 input scaling");
        $display(" Range: x = -8.0 to 7.875");
        $display("=====================================================");
        $display("Vectors with error > 1e-5:");
        $display("-----------------------------------------------------");
        $display("  b_int      z_float   ref_float   error");
        $display("-----------------------------------------------------");

        for (i = -128; i <= 127; i = i + 1) begin
            b_int = i;
            b = b_int[7:0];
            #10;

            x_real = $itor(b_int) / 16.0;
            ref_real = (($exp(2.0 * x_real) - 1.0) / ($exp(2.0 * x_real) + 1.0));
            z_real = $itor($signed(z)) / 128.0;
            
            abs_err = (z_real > ref_real) ? (z_real - ref_real) : (ref_real - z_real);

            sum_err = sum_err + abs_err;
            if (abs_err > max_err)
                max_err = abs_err;

            if (abs_err > 1e-5) begin
                fail_count = fail_count + 1;
                $display("  %+4d(%+6.3f)  %e  %e  %e",
                    b_int, x_real, z_real, ref_real, abs_err);
            end
        end

        avg_err = sum_err / 256.0;

        $display("-----------------------------------------------------");
        $display("RESULTS  (x = -8.0 to 7.9375, 256 pts)");
        $display("  Total vectors tested       : 256");
        $display("  Vectors with err > 1e-5    : %0d", fail_count);
        $display("  Max error  (INT8)          : %e", max_err);
        $display("  Avg error  (INT8)          : %e", avg_err);
        $display("=====================================================");
        
        $display("  Avg error <= 0.03 : %s",
                 (avg_err <= 0.03) ? "PASS" : "FAIL");
        $display("  Max error <= 0.10 : %s",
                 (max_err <= 0.10) ? "PASS" : "FAIL");

        $finish;
    end

endmodule
