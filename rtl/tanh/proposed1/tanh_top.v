// =============================================================
//  tanh_top.v
//  Proposed1 combinational tanh approximation.
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

    tanh_stage4_select u_sel (
        .s        (s),
        .sign_bit (sign_bit),
        .z        (z)
    );

endmodule
