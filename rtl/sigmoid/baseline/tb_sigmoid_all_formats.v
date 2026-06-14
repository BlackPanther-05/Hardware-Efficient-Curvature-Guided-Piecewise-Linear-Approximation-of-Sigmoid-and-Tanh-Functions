`timescale 1ns/1ps
// ============================================================
// UNIFIED FULL-COVERAGE TESTBENCH FOR ALL SIGMOID DATA FORMATS
// ============================================================
// INT8/UINT8/FP8: all 256 8-bit input values
// FP32-style fixed test: all scaled points -512 to 511
// ============================================================

module tb_sigmoid_all_formats;

    reg clk;
    integer i;
    integer fail_count;
    real x_real, ref_real, z_real, abs_err;
    real sum_err, max_err, avg_err;

    reg  [7:0] b_int8;
    wire [7:0] z_int8;

    reg  [7:0] b_uint8;
    wire [7:0] z_uint8;

    reg  [7:0] b_fp8;
    wire [7:0] z_fp8;

    reg  [31:0] b_fp32;
    wire [31:0] z_fp32;

    sigmoid_top_int8 dut_int8 (.clk(clk), .b(b_int8), .z(z_int8));
    sigmoid_top_uint8 dut_uint8 (.clk(clk), .b(b_uint8), .z(z_uint8));
    sigmoid_top_fp8 dut_fp8 (.clk(clk), .b(b_fp8), .z(z_fp8));
    sigmoid_top_fp32 dut_fp32 (.clk(clk), .b(b_fp32), .z(z_fp32));

    initial begin
        clk = 0;
        forever #5 clk = ~clk;
    end

    task print_status;
        input [80*8:1] label;
        input real avg_limit;
        input real max_limit;
        begin
            $display("  Avg error <= %e: %s", avg_limit, (avg_err <= avg_limit) ? "PASS" : "FAIL");
            $display("  Max error <= %e: %s", max_limit, (max_err <= max_limit) ? "PASS" : "FAIL");
            $display("  %0s COMPLETE", label);
            $display("-------------------------------------------------------------");
        end
    endtask

    initial begin
        $display("=============================================================");
        $display("UNIFIED SIGMOID FULL-COVERAGE TESTBENCH - ALL FORMATS");
        $display("=============================================================");

        // ========================================================
        // INT8: 256 vectors
        // ========================================================
        sum_err = 0.0;
        max_err = 0.0;
        fail_count = 0;
        $display("");
        $display("TEST 1: INT8 FORMAT");
        $display("Coverage: b_int = -128 to 127 (256 vectors)");
        $display("Vectors with error > 0.01:");

        for (i = -128; i <= 127; i = i + 1) begin
            b_int8 = i[7:0];
            #10;
            x_real = $itor($signed(i)) / 16.0;
            ref_real = 1.0 / (1.0 + $exp(-x_real));
            z_real = $itor($signed(z_int8)) / 128.0;
            abs_err = (z_real > ref_real) ? (z_real - ref_real) : (ref_real - z_real);
            sum_err = sum_err + abs_err;
            if (abs_err > max_err) max_err = abs_err;
            if (abs_err > 0.01) begin
                fail_count = fail_count + 1;
                $display("  %+4d  z=%3d  z_float=%f  ref=%f  err=%f", i, $signed(z_int8), z_real, ref_real, abs_err);
            end
        end

        avg_err = sum_err / 256.0;
        $display("RESULTS  (INT8, 256 pts)");
        $display("  Total vectors tested      : 256");
        $display("  Vectors with err > 0.01   : %0d", fail_count);
        $display("  Max error  (float)        : %f", max_err);
        $display("  Avg error  (float)        : %f", avg_err);
        print_status("INT8", 0.05, 0.15);

        // ========================================================
        // UINT8: 256 vectors
        // ========================================================
        sum_err = 0.0;
        max_err = 0.0;
        fail_count = 0;
        $display("");
        $display("TEST 2: UINT8 FORMAT");
        $display("Coverage: b = 0 to 255 (256 vectors)");
        $display("Vectors with error > 0.01:");

        for (i = 0; i <= 255; i = i + 1) begin
            b_uint8 = i[7:0];
            #10;
            x_real = ($itor(i) / 255.0) * 16.0 - 8.0;
            ref_real = 1.0 / (1.0 + $exp(-x_real));
            z_real = $itor(z_uint8) / 255.0;
            abs_err = (z_real > ref_real) ? (z_real - ref_real) : (ref_real - z_real);
            sum_err = sum_err + abs_err;
            if (abs_err > max_err) max_err = abs_err;
            if (abs_err > 0.01) begin
                fail_count = fail_count + 1;
                $display("  %3d  z=%3d  z_float=%f  ref=%f  err=%f", i, z_uint8, z_real, ref_real, abs_err);
            end
        end

        avg_err = sum_err / 256.0;
        $display("RESULTS  (UINT8, 256 pts)");
        $display("  Total vectors tested      : 256");
        $display("  Vectors with err > 0.01   : %0d", fail_count);
        $display("  Max error  (float)        : %f", max_err);
        $display("  Avg error  (float)        : %f", avg_err);
        print_status("UINT8", 0.02, 0.10);

        // ========================================================
        // FP8: 256 vectors
        // ========================================================
        sum_err = 0.0;
        max_err = 0.0;
        fail_count = 0;
        $display("");
        $display("TEST 3: FP8 FORMAT");
        $display("Coverage: b_int = -128 to 127 (256 vectors)");
        $display("Vectors with error > 0.02:");

        for (i = -128; i <= 127; i = i + 1) begin
            b_fp8 = i[7:0];
            #10;
            x_real = $itor($signed(i)) / 16.0;
            ref_real = 1.0 / (1.0 + $exp(-x_real));
            z_real = $itor($signed(z_fp8)) / 128.0;
            abs_err = (z_real > ref_real) ? (z_real - ref_real) : (ref_real - z_real);
            sum_err = sum_err + abs_err;
            if (abs_err > max_err) max_err = abs_err;
            if (abs_err > 0.02) begin
                fail_count = fail_count + 1;
                $display("  %+4d  z=%3d  z_float=%f  ref=%f  err=%f", i, $signed(z_fp8), z_real, ref_real, abs_err);
            end
        end

        avg_err = sum_err / 256.0;
        $display("RESULTS  (FP8, 256 pts)");
        $display("  Total vectors tested      : 256");
        $display("  Vectors with err > 0.02   : %0d", fail_count);
        $display("  Max error  (float)        : %f", max_err);
        $display("  Avg error  (float)        : %f", avg_err);
        print_status("FP8", 0.05, 0.15);

        // ========================================================
        // FP32-style fixed: 1024 vectors
        // ========================================================
        sum_err = 0.0;
        max_err = 0.0;
        fail_count = 0;
        $display("");
        $display("TEST 4: FP32 FORMAT");
        $display("Coverage: b_int = -512 to 511 (1024 vectors)");
        $display("Vectors with error > 1e-5:");

        for (i = -512; i <= 511; i = i + 1) begin
            b_fp32 = i[31:0];
            #10;
            x_real = $itor(i) / 64.0;
            ref_real = 1.0 / (1.0 + $exp(-x_real));
            z_real = $itor($signed(z_fp32)) / 2147483648.0;
            abs_err = (z_real > ref_real) ? (z_real - ref_real) : (ref_real - z_real);
            sum_err = sum_err + abs_err;
            if (abs_err > max_err) max_err = abs_err;
            if (abs_err > 1e-5) begin
                fail_count = fail_count + 1;
                $display("  %+7.4f  z_float=%e  ref=%e  err=%e", x_real, z_real, ref_real, abs_err);
            end
        end

        avg_err = sum_err / 1024.0;
        $display("RESULTS  (FP32, 1024 pts)");
        $display("  Total vectors tested       : 1024");
        $display("  Vectors with err > 1e-5    : %0d", fail_count);
        $display("  Max error  (float)         : %e", max_err);
        $display("  Avg error  (float)         : %e", avg_err);
        print_status("FP32", 0.005, 0.01);

        $display("");
        $display("=============================================================");
        $display("UNIFIED SIGMOID FULL-COVERAGE TESTBENCH COMPLETE");
        $display("=============================================================");
        $finish;
    end

endmodule
