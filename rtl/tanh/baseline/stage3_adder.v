// =============================================================
//  stage3_adder.v
//
//  Computes positive-half tanh magnitude s = p + c in Q0.10.
// =============================================================
module tanh_stage3_adder (
    input  wire [10:0] p,
    input  wire [10:0] c,
    output wire [10:0] s
);

    wire [11:0] sum_full;
    assign sum_full = {1'b0, p} + {1'b0, c};

    assign s = (sum_full > 12'd1023) ? 11'd1023 : sum_full[10:0];

endmodule
