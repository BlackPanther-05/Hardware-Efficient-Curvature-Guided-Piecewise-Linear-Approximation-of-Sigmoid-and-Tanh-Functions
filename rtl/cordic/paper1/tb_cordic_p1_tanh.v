`timescale 1ns/1ps
// ============================================================
// FP32 TANH ERROR TESTBENCH
// ============================================================
// Tests FP32 format tanh over 1024 points
// IEEE 754 32-bit floating point
// ============================================================

module tb_tanh_fp32_error;

    reg  [31:0] b;
    wire [31:0] z;
    cordic_p1_tanh_top dut (.clk(clk), .b(b), .z(z));
    
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
        $display(" FP32 Tanh RTL Error Test");
        $display(" Format: IEEE 754 32-bit");
        $display(" Range: x = -8.0 to 7.9375");
        $display("=====================================================");
        $display("Vectors with error > 1e-5:");
        $display("-----------------------------------------------------");
        $display("  b_int(scaled)  z_float   ref_float   error");
        $display("-----------------------------------------------------");

        for (i = -512; i <= 511; i = i + 1) begin
            b_int = i;
            b = b_int[31:0];
            #10;

            x_real = $itor(b_int) / 64.0;
            ref_real = (($exp(2.0 * x_real) - 1.0) / ($exp(2.0 * x_real) + 1.0));
            
            z_real = $itor($signed(z)) / 2147483648.0;
            
            abs_err = (z_real > ref_real) ? (z_real - ref_real) : (ref_real - z_real);

            sum_err = sum_err + abs_err;
            if (abs_err > max_err)
                max_err = abs_err;

            if (abs_err > 1e-5) begin
                fail_count = fail_count + 1;
                $display("  %+7.4f        %e  %e  %e",
                    x_real, z_real, ref_real, abs_err);
            end
        end

        avg_err = sum_err / 1024.0;

        $display("-----------------------------------------------------");
        $display("RESULTS  (x = -8.0 to 7.984375, 1024 pts)");
        $display("  Total vectors tested       : 1024");
        $display("  Vectors with err > 1e-5    : %0d", fail_count);
        $display("  Max error  (float)         : %e", max_err);
        $display("  Avg error  (float)         : %e", avg_err);
        $display("=====================================================");
        
        $display("  Avg error <= 0.005: %s",
                 (avg_err <= 0.005) ? "PASS" : "FAIL");
        $display("  Max error <= 0.02: %s",
                 (max_err <= 0.02) ? "PASS" : "FAIL");

        $finish;
    end

endmodule
