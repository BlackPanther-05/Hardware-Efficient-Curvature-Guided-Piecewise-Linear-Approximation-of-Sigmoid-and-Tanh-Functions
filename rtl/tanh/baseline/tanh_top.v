// =============================================================
//  tanh_top.v
//
//  Top-level combinational tanh approximation.
//  Structure mirrors the Sigmoid RTL pipeline:
//
//   b -> Stage1 modulus -> Stage2 comparator/LUT/multiplier
//     -> Stage3 adder -> tail mux -> Stage4 sign select -> z
//
//  Input  b : signed S1.I3.F7, range approximately (-8, 8)
//  Output z : signed S1.F10 two's-complement tanh result
//             actual value = $signed(z) / 1024.0
// =============================================================
module tanh_top (
    input  wire [10:0] b,
    output wire [10:0] z
);

    wire [10:0] xp;
    wire        sign_bit;
    wire [ 3:0] xindex;
    wire        is_tail;
    wire [10:0] m, c;
    wire [10:0] p;
    wire [10:0] s;
    wire [10:0] s_muxed;

    // Largest positive Q1.10 value below 1.0.
    localparam [10:0] TAIL_CONST = 11'd1023;

    tanh_stage1_modulus u_stage1 (
        .b        (b),
        .xp       (xp),
        .sign_bit (sign_bit)
    );

    tanh_stage2_comparator u_comp (
        .xp      (xp),
        .xindex  (xindex),
        .is_tail (is_tail)
    );

    lut_tanh u_lut (
        .xindex (xindex),
        .m      (m),
        .c      (c)
    );

    tanh_stage2_multiplier u_mult (
        .xp (xp),
        .m  (m),
        .p  (p)
    );

    tanh_stage3_adder u_add (
        .p (p),
        .c (c),
        .s (s)
    );

    assign s_muxed = is_tail ? TAIL_CONST : s;

    tanh_stage4_select u_sel (
        .s        (s_muxed),
        .sign_bit (sign_bit),
        .z        (z)
    );

endmodule
