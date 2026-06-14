`timescale 1ns/1ps
// ============================================================
// UINT8 TANH ERROR TESTBENCH
// ============================================================
// Tests UINT8 format tanh over all 256 input values
// Unsigned 8-bit (0-255)
// ============================================================

module tb_tanh_uint8_error;

    reg  [7:0] b;
    wire [7:0] z;
    cordic_p2_tanh_top_uint8 dut (.clk(clk), .b(b), .z(z));
    
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
        $display(" UINT8 Tanh RTL Error Test");
        $display(" Format: Unsigned 8-bit, [0,255]→[-1,1]");
        $display(" Range: x = -1.0 to 0.9961");
        $display("=====================================================");
        $display("Vectors with error > 1e-5:");
        $display("-----------------------------------------------------");
        $display("  b_int(uint8)  x_real  z_float   ref_float   error");
        $display("-----------------------------------------------------");

        for (i = 0; i < 256; i = i + 1) begin
            b_int = i;
            b = b_int[7:0];
            #10;

            // Map [0, 255] to [-1, 1]
            x_real = $itor(b_int) / 127.5 - 1.0;
            ref_real = (($exp(2.0 * x_real) - 1.0) / ($exp(2.0 * x_real) + 1.0));
            
            // Map output back from [0, 255] to [-1, 1]
            z_real = $itor(z) / 127.5 - 1.0;
            
            abs_err = (z_real > ref_real) ? (z_real - ref_real) : (ref_real - z_real);

            sum_err = sum_err + abs_err;
            if (abs_err > max_err)
                max_err = abs_err;

            if (abs_err > 1e-5) begin
                fail_count = fail_count + 1;
                $display("  %3d       %+6.3f  %e  %e  %e",
                    b_int, x_real, z_real, ref_real, abs_err);
            end
        end

        avg_err = sum_err / 256.0;

        $display("-----------------------------------------------------");
        $display("RESULTS  (x = -1.0 to 1.0, 256 pts)");
        $display("  Total vectors tested       : 256");
        $display("  Vectors with err > 1e-5    : %0d", fail_count);
        $display("  Max error  (UINT8)         : %e", max_err);
        $display("  Avg error  (UINT8)         : %e", avg_err);
        $display("=====================================================");
        
        $display("  Avg error <= 0.01 : %s",
                 (avg_err <= 0.01) ? "PASS" : "FAIL");
        $display("  Max error <= 0.05 : %s",
                 (max_err <= 0.05) ? "PASS" : "FAIL");

        $finish;
    end

endmodule
