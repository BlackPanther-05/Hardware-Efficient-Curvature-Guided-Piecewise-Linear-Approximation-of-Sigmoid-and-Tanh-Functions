// =============================================================
//  stage2_multiplier.v
//  p_int = floor((xp_int * m_int) / 2^7), output Q0.10.
// =============================================================
module stage2_multiplier (
    input  wire [10:0] xp,
    input  wire [10:0] m,
    output wire [10:0] p
);

    wire [21:0] raw_product;
    assign raw_product = xp[9:0] * m[9:0];

    assign p = {1'b0, raw_product[16:7]};

endmodule
