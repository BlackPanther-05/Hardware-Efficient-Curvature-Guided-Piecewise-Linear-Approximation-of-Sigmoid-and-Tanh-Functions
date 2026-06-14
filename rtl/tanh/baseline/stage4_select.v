// =============================================================
//  stage4_select.v
//
//  Final tanh sign restoration.
//    sign_bit == 0 -> z =  s
//    sign_bit == 1 -> z = -s
//
//  z is signed two's-complement Q1.10.
// =============================================================
module tanh_stage4_select (
    input  wire [10:0] s,
    input  wire        sign_bit,
    output wire [10:0] z
);

    assign z = sign_bit ? (~s + 11'd1) : s;

endmodule
