// =============================================================
//  sigmoid_top.v
//
//  Top-level combinational sigmoid approximation.
//  Instantiates all 4 stages.
//
//  Input  b : S1.I3.F7 (11-bit), range (-8, 8)
//             b[10]   = sign bit
//             b[9:7]  = 3-bit integer part
//             b[6:0]  = 7-bit fractional part
//
//  Output z : S1.F10 (11-bit), range (0, 1)
//             z[10]   = sign (always 0 for sigmoid)
//             z[9:0]  = 10-bit fractional part
//             actual value = z / 1024.0
//
//  Pipeline (purely combinational, single clock cycle):
//
//   b ──► [Stage1: modulus] ──► xp ──► [Stage2: comparator] ──► xindex
//                │                         │
//             sign_bit              [LUT] ──► m, c
//                │                         │
//                │               [Stage2: multiplier] ──► p
//                │                         │
//                │               [Stage3: adder p+c] ──► s
//                │                                        │
//                └─────────────► [Stage4: select] ──────► z
//
//  Tail case (xp >= 6.443):
//    is_tail=1 → bypass multiplier/adder, z = TAIL_CONST (0.999)
//    For negative input: z = 1 - TAIL_CONST = 1/1024 ≈ 0.001
// =============================================================
module sigmoid_top (
    input  wire [10:0] b,
    output wire [10:0] z
);

    // ── Internal wires ──────────────────────────────────────
    wire [10:0] xp;
    wire        sign_bit;
    wire [ 3:0] xindex;
    wire        is_tail;
    wire [10:0] m, c;
    wire [10:0] p;
    wire [10:0] s;
    wire [10:0] s_muxed;     // s after tail mux
    wire [10:0] z_out;

    // Tail constant: 0.999 * 1024 = 1023
    localparam [10:0] TAIL_CONST = 11'd1023;

    // ── Stage 1: Modulus ────────────────────────────────────
    stage1_modulus u_stage1 (
        .b        (b),
        .xp       (xp),
        .sign_bit (sign_bit)
    );

    // ── Stage 2a: Comparator ────────────────────────────────
    stage2_comparator u_comp (
        .xp      (xp),
        .xindex  (xindex),
        .is_tail (is_tail)
    );

    // ── Stage 2b: LUT ───────────────────────────────────────
    lut_sigmoid u_lut (
        .xindex (xindex),
        .m      (m),
        .c      (c)
    );

    // ── Stage 2c: Multiplier (xp * m → p) ──────────────────
    stage2_multiplier u_mult (
        .xp (xp),
        .m  (m),
        .p  (p)
    );

    // ── Stage 3: Adder (p + c → s) ──────────────────────────
    stage3_adder u_add (
        .p (p),
        .c (c),
        .s (s)
    );

    // ── Tail mux: bypass s with TAIL_CONST when is_tail=1 ───
    assign s_muxed = is_tail ? TAIL_CONST : s;

    // ── Stage 4: Select (handle negative input) ─────────────
    stage4_select u_sel (
        .s        (s_muxed),
        .sign_bit (sign_bit),
        .z        (z)
    );

endmodule
