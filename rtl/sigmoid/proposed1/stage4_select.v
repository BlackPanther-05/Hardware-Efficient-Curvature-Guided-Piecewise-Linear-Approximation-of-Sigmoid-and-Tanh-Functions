// =============================================================
//  stage4_select.v
//  Sigmoid symmetry: positive -> s, negative -> 1 - s.
// =============================================================
module stage4_select (
    input  wire [10:0] s,
    input  wire        sign_bit,
    output wire [10:0] z
);

    assign z = sign_bit ? (11'd1024 - s) : s;

endmodule
