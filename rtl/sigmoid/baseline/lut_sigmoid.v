// =============================================================
//  lut_sigmoid.v
//
//  Look-Up Table storing aware-quantized slope (m) and
//  intercept (c) for 16 sigmoid segments + tail constant.
//
//  Format: both m and c are S1.F10 (11-bit signed fixed-point)
//    value = integer / 1024.0
//
//  Source: Python aware-quantization script output
//
//  Seg | x range         | m_int | c_int  | m_float   | c_float
//   0  | [0.000, 0.166)  |   256 |    512 | 0.250000  | 0.500000
//   1  | [0.166, 0.344)  |   249 |    514 | 0.243164  | 0.501953
//   2  | [0.344, 0.520)  |   244 |    516 | 0.238281  | 0.503906
//   3  | [0.520, 0.704)  |   231 |    523 | 0.225586  | 0.510742
//   4  | [0.704, 0.888)  |   220 |    531 | 0.214844  | 0.518555
//   5  | [0.888, 1.089)  |   203 |    546 | 0.198242  | 0.533203
//   6  | [1.089, 1.305)  |   181 |    570 | 0.176758  | 0.556641
//   7  | [1.305, 1.537)  |   159 |    599 | 0.155273  | 0.584961
//   8  | [1.537, 1.777)  |   136 |    635 | 0.132812  | 0.620117
//   9  | [1.777, 2.042)  |   119 |    665 | 0.116211  | 0.649414
//  10  | [2.042, 2.354)  |    91 |    722 | 0.088867  | 0.705078
//  11  | [2.354, 2.730)  |    69 |    774 | 0.067383  | 0.755859
//  12  | [2.730, 3.187)  |    51 |    823 | 0.049805  | 0.803711
//  13  | [3.187, 3.811)  |    30 |    889 | 0.029297  | 0.868164
//  14  | [3.811, 4.804)  |    14 |    950 | 0.013672  | 0.927734
//  15  | [4.804, 6.443)  |     4 |    998 | 0.003906  | 0.974609
//  Tail| [6.443, 8.000]  | constant output = 1023/1024 = 0.999023
//
//  is_tail=1 bypasses m,c entirely; top module outputs TAIL_CONST.
// =============================================================
module lut_sigmoid (
    input  wire [ 3:0] xindex,
    output reg  [10:0] m,       // S1.F10
    output reg  [10:0] c        // S1.F10
);

    // Tail constant: 0.999 * 1024 = 1023 → 11'b01111111111
    // Exposed as parameter so top module can mux it in
    localparam [10:0] TAIL_CONST = 11'd1023;

    always @(*) begin
        case (xindex)
            4'd0:  begin m = 11'd256; c = 11'd512;  end
            4'd1:  begin m = 11'd249; c = 11'd514;  end
            4'd2:  begin m = 11'd244; c = 11'd516;  end
            4'd3:  begin m = 11'd231; c = 11'd523;  end
            4'd4:  begin m = 11'd220; c = 11'd531;  end
            4'd5:  begin m = 11'd203; c = 11'd546;  end
            4'd6:  begin m = 11'd181; c = 11'd570;  end
            4'd7:  begin m = 11'd159; c = 11'd599;  end
            4'd8:  begin m = 11'd136; c = 11'd635;  end
            4'd9:  begin m = 11'd119; c = 11'd665;  end
            4'd10: begin m = 11'd91;  c = 11'd722;  end
            4'd11: begin m = 11'd69;  c = 11'd774;  end
            4'd12: begin m = 11'd51;  c = 11'd823;  end
            4'd13: begin m = 11'd30;  c = 11'd889;  end
            4'd14: begin m = 11'd14;  c = 11'd950;  end
            4'd15: begin m = 11'd4;   c = 11'd998;  end
            default: begin m = 11'd0; c = 11'd512;  end
        endcase
    end

endmodule
