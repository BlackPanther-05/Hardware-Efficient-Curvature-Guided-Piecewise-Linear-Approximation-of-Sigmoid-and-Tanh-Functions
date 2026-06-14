// =============================================================
//  stage2_multiplier.v
//
//  Computes p = floor(m * xp) in fixed-point, keeping 10
//  fractional bits in the 11-bit output.
//
//  Operand formats:
//    xp : S1.I3.F7  (11-bit, always positive from Stage 1)
//           bit[10]=0 (sign), bits[9:7]=integer, bits[6:0]=frac
//           actual value = xp_int / 128.0
//
//    m  : S1.F10   (11-bit, always positive, range 0..1)
//           actual value = m_int / 1024.0
//
//  Raw product:
//    xp_int * m_int  = integer product (22 bits max for 11x11)
//    In real units  : (xp/128) * (m/1024) = product / 131072
//                   = product / 2^17
//
//  We want p in S1.F10 format, i.e., p_int / 1024.0
//    p_real = xp_real * m_real = (xp_int * m_int) / 2^17
//    p_int (Q.10) = p_real * 1024 = (xp_int * m_int) / 2^7
//
//  So: p_int = floor( (xp_int * m_int) >> 7 )
//
//  Product word-length: 11 * 11 = 22 bits raw
//  After >> 7: 22 - 7 = 15 bits needed, we keep lower 11 bits
//  for S1.F10 (since p_real < 1 always for these segments,
//  bit[10] = sign = 0, bits[9:0] = 10 frac bits).
//
//  Output p is 11 bits S1.F10, floor-truncated (no rounding).
// =============================================================
module stage2_multiplier (
    input  wire [10:0] xp,   // S1.I3.F7 (positive)
    input  wire [10:0] m,    // S1.F10   (positive)
    output wire [10:0] p     // S1.F10   (positive, floor truncated)
);

    // Full 22-bit product of unsigned magnitudes
    // xp[10]=0 always (positive), m[10]=0 always (0 < m < 1)
    wire [21:0] raw_product;
    assign raw_product = xp[9:0] * m[9:0];   // 10b × 10b = 20b, zero-extended

    // Right-shift by 7 to align to Q1.10
    // raw_product >> 7 gives the Q1.10 integer representation of p
    // We take bits [16:7] of raw_product (floor = drop lower 7 bits)
    // Output is capped at 11 bits; sign bit = 0 (always positive product)
    assign p = {1'b0, raw_product[16:7]};

endmodule
