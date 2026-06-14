`timescale 1ns/1ps
// ============================================================
// INT8 SIGMOID ERROR TESTBENCH
// ============================================================
// Tests INT8 format sigmoid over 256 points
// Range: -128 to 127 (representing x * 16)
// ============================================================

module tb_sigmoid_int8_error;

    reg  [7:0] b;
    wire [7:0] z;
    cordic_p2_sigmoid_top_int8 dut (.clk(clk), .b(b), .z(z));
    
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
        $display(" INT8 Sigmoid RTL Error Test");
        $display(" Range: b_int = -128 to 127");
        $display("        x     = -8.0 to 7.9375");
        $display("=====================================================");
        $display("Vectors with error > 0.01:");
        $display("-----------------------------------------------------");
        $display("  b_int    z_out   z_float   ref_float   error");
        $display("-----------------------------------------------------");

        for (i = -128; i <= 127; i = i + 1) begin
            b_int = i;
            b = b_int[7:0];
            #10;

            x_real = $itor($signed(b_int)) / 16.0;
            ref_real = 1.0 / (1.0 + $exp(-x_real));
            z_real = $itor($signed(z)) / 128.0;
            abs_err = (z_real > ref_real) ? (z_real - ref_real) : (ref_real - z_real);

            sum_err = sum_err + abs_err;
            if (abs_err > max_err)
                max_err = abs_err;

            if (abs_err > 0.01) begin
                fail_count = fail_count + 1;
                $display("  %+4d      %3d    %f  %f  %f",
                    b_int, $signed(z), z_real, ref_real, abs_err);
            end
        end

        avg_err = sum_err / 256.0;

        $display("-----------------------------------------------------");
        $display("RESULTS  (x = -8.0 to 7.9375, 256 pts)");
        $display("  Total vectors tested      : 256");
        $display("  Vectors with err > 0.01   : %0d", fail_count);
        $display("  Max error  (float)        : %f", max_err);
        $display("  Avg error  (float)        : %f", avg_err);
        $display("=====================================================");
        
        $display("  Avg error <= 0.05: %s",
                 (avg_err <= 0.05) ? "PASS" : "FAIL");
        $display("  Max error <= 0.15: %s",
                 (max_err <= 0.15) ? "PASS" : "FAIL");

        $finish;
    end

endmodule
