`timescale 1ns/1ps
// ============================================================
// UINT8 SIGMOID ERROR TESTBENCH
// ============================================================
// Tests UINT8 format sigmoid over 256 points
// Range: 0 to 255 (representing sigmoid output * 255)
// ============================================================

module tb_sigmoid_uint8_error;

    reg  [7:0] b;
    wire [7:0] z;
    sigmoid_top_uint8 dut (.clk(clk), .b(b), .z(z));
    
    reg clk;
    integer i, fail_count;
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
        $display(" UINT8 Sigmoid RTL Error Test");
        $display(" Range: b = 0 to 255");
        $display("        x = -8 to 7.9375 (mapped)");
        $display("=====================================================");
        $display("Vectors with error > 0.01:");
        $display("-----------------------------------------------------");
        $display("  b       z_out   z_float   ref_float   error");
        $display("-----------------------------------------------------");

        for (i = 0; i <= 255; i = i + 1) begin
            b = i[7:0];
            #10;

            // Map [0, 255] to [-8, 8]
            x_real = (($itor(i) / 255.0) * 16.0) - 8.0;
            ref_real = 1.0 / (1.0 + $exp(-x_real));
            z_real = $itor(z) / 255.0;
            abs_err = (z_real > ref_real) ? (z_real - ref_real) : (ref_real - z_real);

            sum_err = sum_err + abs_err;
            if (abs_err > max_err)
                max_err = abs_err;

            if (abs_err > 0.01) begin
                fail_count = fail_count + 1;
                $display("  %3d      %3d    %f  %f  %f",
                    i, z, z_real, ref_real, abs_err);
            end
        end

        avg_err = sum_err / 256.0;

        $display("-----------------------------------------------------");
        $display("RESULTS  (x = -8 to 7.9375, 256 pts)");
        $display("  Total vectors tested      : 256");
        $display("  Vectors with err > 0.01   : %0d", fail_count);
        $display("  Max error  (float)        : %f", max_err);
        $display("  Avg error  (float)        : %f", avg_err);
        $display("=====================================================");
        
        $display("  Avg error <= 0.02 (2e-2): %s",
                 (avg_err <= 0.02) ? "PASS" : "FAIL");
        $display("  Max error <= 0.10 (1e-1): %s",
                 (max_err <= 0.10) ? "PASS" : "FAIL");

        $finish;
    end

endmodule
