// =============================================================
//  sigmoid_top.v
//  Proposed1 combinational sigmoid approximation.
// =============================================================
module sigmoid_top (
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

    stage1_modulus u_stage1 (
        .b        (b),
        .xp       (xp),
        .sign_bit (sign_bit)
    );

    stage2_comparator u_comp (
        .xp      (xp),
        .xindex  (xindex),
        .is_tail (is_tail)
    );

    lut_sigmoid u_lut (
        .xindex (xindex),
        .m      (m),
        .c      (c)
    );

    stage2_multiplier u_mult (
        .xp (xp),
        .m  (m),
        .p  (p)
    );

    stage3_adder u_add (
        .p (p),
        .c (c),
        .s (s)
    );

    stage4_select u_sel (
        .s        (s),
        .sign_bit (sign_bit),
        .z        (z)
    );

endmodule
