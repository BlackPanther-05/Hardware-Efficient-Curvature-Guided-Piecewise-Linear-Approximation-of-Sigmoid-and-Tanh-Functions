// =============================================================
//  stage4_select.v
//  Tanh odd symmetry: positive -> s, negative -> -s.
// =============================================================
module tanh_stage4_select (
    input  wire [10:0] s,
    input  wire        sign_bit,
    output wire [10:0] z
);

    assign z = sign_bit ? (~s + 11'd1) : s;

endmodule
