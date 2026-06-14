import numpy as np

# ============================================================
# USER CONFIG
# ============================================================
SEG_BITS  = 4               # 4 → 16 segments (matches paper)
NUM_SEG   = 2**SEG_BITS
X_MAX     = 8.0
SEARCH_RANGE = 8            # ±LSBs to search in quant-aware regression

# Hardware pipeline mode (paper uses QUANTIZE_PRODUCT = True):
#   True  → truncate m*x product back to 11 bits before adding c  (paper)
#   False → keep full precision product, only quantize final output
QUANTIZE_PRODUCT = True

# ============================================================
# Paper-exact fixed-point formats  (all 11-bit total)
#
#  x   : 1 sign + 3 integer + 7 fractional   → SCALE_X  = 2^7  = 128
#  m,c : 1 sign + 0 integer + 10 fractional  → SCALE_MC = 2^10 = 1024
#  y   : 1 sign + 0 integer + 10 fractional  → SCALE_Y  = 2^10 = 1024
#
#  Product p = m(10f) × x(7f) = 17 fractional bits raw
#    paper keeps 10 frac bits → discard bottom 7 (truncate, not round)
# ============================================================
INT_BITS_X   = 3
FRAC_BITS_X  = 7     # x resolution = 1/128  = 0.0078125
FRAC_BITS_MC = 10    # m,c resolution = 1/1024 = 0.0009765625
FRAC_BITS_Y  = 10    # y resolution = 1/1024

SCALE_X  = 2**FRAC_BITS_X    # 128
SCALE_MC = 2**FRAC_BITS_MC   # 1024
SCALE_Y  = 2**FRAC_BITS_Y    # 1024

LSB_X  = 1.0 / SCALE_X       # 0.0078125
LSB_MC = 1.0 / SCALE_MC      # 0.0009765625

# ============================================================
# Quantizers — one per signal type
# ============================================================
def quantize_x(val):
    """11-bit: 1s+3i+7f  resolution=1/128"""
    return np.round(np.array(val, dtype=float) * SCALE_X) / SCALE_X

def quantize_mc(val):
    """11-bit: 1s+0i+10f  resolution=1/1024"""
    return np.round(np.array(val, dtype=float) * SCALE_MC) / SCALE_MC

def quantize_y(val):
    """11-bit: 1s+0i+10f  resolution=1/1024"""
    return np.round(np.array(val, dtype=float) * SCALE_Y) / SCALE_Y

def hw_pipeline(m_q, x_q, c_q):
    """
    Paper hardware pipeline (Stage 2 + Stage 3):
      p = trunc(m_q * x_q)   [17 frac bits → 10 frac bits, drop bottom 7]
      s = quantize_y(p + c_q)

    With QUANTIZE_PRODUCT=False: skip truncation (useful for comparison).
    """
    if QUANTIZE_PRODUCT:
        # m_q: 10 frac bits, x_q: 7 frac bits → product: 17 frac bits
        # Keep top 10 frac bits: floor(product * 2^10) / 2^10
        #   = floor(product * SCALE_MC) / SCALE_MC
        # But product * SCALE_MC = m_q * x_q * 1024
        # We want floor not round → use np.floor
        p = np.floor(np.array(m_q, dtype=float) *
                     np.array(x_q, dtype=float) * SCALE_MC) / SCALE_MC
        return quantize_y(p + c_q)
    else:
        return quantize_y(np.array(m_q, dtype=float) *
                          np.array(x_q, dtype=float) + c_q)

# ============================================================
# True sigmoid and curvature
# ============================================================
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def second_derivative(x):
    s = sigmoid(x)
    return s * (1 - s) * (1 - 2*s)

# ============================================================
# Step 1: Curvature-weighted segmentation
# ============================================================
x_dense    = np.linspace(0, X_MAX, 50000)
weight     = np.sqrt(np.abs(second_derivative(x_dense)))
weight[weight < 1e-14] = 1e-14

cumulative = np.cumsum(weight)
cumulative /= cumulative[-1]

boundaries = [0.0]
for k in range(1, NUM_SEG):
    idx = np.searchsorted(cumulative, k / NUM_SEG)
    boundaries.append(float(x_dense[idx]))
boundaries.append(X_MAX)

# ============================================================
# Step 2: Float regression
# ============================================================
def float_regression(x_seg, y_seg):
    X_bar = np.mean(x_seg)
    Y_bar = np.mean(y_seg)
    m = np.sum((x_seg - X_bar) * (y_seg - Y_bar)) / np.sum((x_seg - X_bar)**2)
    c = Y_bar - m * X_bar
    return m, c

# ============================================================
# Step 3: Quantization-aware regression
#   Searches ±SEARCH_RANGE LSBs of m and c (in MC resolution)
#   using the exact hardware pipeline as objective.
# ============================================================
def quant_aware_regression(x_seg, y_seg, m_float, c_float):
    x_seg_q = quantize_x(x_seg)        # x as hardware sees it (7 frac bits)

    best_err = float('inf')
    best_mq  = quantize_mc(m_float)
    best_cq  = quantize_mc(c_float)
    m_base   = quantize_mc(m_float)

    for dm in range(-SEARCH_RANGE, SEARCH_RANGE + 1):
        m_candidate = m_base + dm * LSB_MC

        # Re-derive optimal c accounting for the full hardware pipeline:
        # After product truncation, the effective signal reaching the adder is p:
        if QUANTIZE_PRODUCT:
            p = np.floor(m_candidate * x_seg_q * SCALE_MC) / SCALE_MC
        else:
            p = m_candidate * x_seg_q

        # Best c (in float) minimises mean(|y_seg - quantize_y(p + c)|)
        # Good closed-form starting point: c* = mean(y_seg - p)
        c_optimal = np.mean(y_seg - p)
        c_base    = quantize_mc(c_optimal)

        for dc in range(-SEARCH_RANGE, SEARCH_RANGE + 1):
            c_candidate = c_base + dc * LSB_MC

            y_hat = hw_pipeline(m_candidate, x_seg_q, c_candidate)
            # Optimise mean absolute error (matches paper's Eavg metric)
            err   = np.mean(np.abs(y_seg - y_hat))

            if err < best_err:
                best_err = err
                best_mq  = m_candidate
                best_cq  = c_candidate

    return best_mq, best_cq

# ============================================================
# Step 4: Build segment tables
# ============================================================
segments_float       = []
segments_quant_naive = []
segments_quant_aware = []

print("\n" + "="*68)
print(f"CONFIG: {NUM_SEG} segments | paper 11-bit exact | X_MAX={X_MAX}")
print(f"  x   : 1s + {INT_BITS_X}i + {FRAC_BITS_X}f  (LSB = {LSB_X})")
print(f"  m,c : 1s + 0i + {FRAC_BITS_MC}f  (LSB = {LSB_MC})")
print(f"  y   : 1s + 0i + {FRAC_BITS_Y}f")
pipeline_str = "trunc(m*x) + c → quantize_y" if QUANTIZE_PRODUCT else "quantize_y(m*x + c)"
print(f"  pipeline: {pipeline_str}")
print("="*68)

for i in range(NUM_SEG):
    x1 = boundaries[i]
    x2 = boundaries[i+1]

    x_seg = np.linspace(x1, x2, 2048)
    y_seg = sigmoid(x_seg)

    m_f, c_f     = float_regression(x_seg, y_seg)
    m_naive      = quantize_mc(m_f)
    c_naive      = quantize_mc(c_f)
    m_aware, c_aware = quant_aware_regression(x_seg, y_seg, m_f, c_f)

    # errors
    y_hat_f      = m_f * x_seg + c_f
    err_f        = np.abs(y_seg - y_hat_f)

    x_seg_q      = quantize_x(x_seg)
    y_hat_naive  = hw_pipeline(m_naive, x_seg_q, c_naive)
    err_naive    = np.abs(y_seg - y_hat_naive)

    y_hat_aware  = hw_pipeline(m_aware, x_seg_q, c_aware)
    err_aware    = np.abs(y_seg - y_hat_aware)

    print(f"\nSeg {i+1:2d}  [{x1:.4f}, {x2:.4f}]")
    print(f"  {'':28s}  {'Float':>12}  {'Naive Q':>12}  {'Aware Q':>12}")
    print(f"  {'m':28s}  {m_f:12.8f}  {m_naive:12.8f}  {m_aware:12.8f}")
    print(f"  {'c':28s}  {c_f:12.8f}  {c_naive:12.8f}  {c_aware:12.8f}")
    print(f"  {'Avg Error':28s}  {np.mean(err_f):12.3e}  {np.mean(err_naive):12.3e}  {np.mean(err_aware):12.3e}")
    print(f"  {'Max Error':28s}  {np.max(err_f):12.3e}  {np.max(err_naive):12.3e}  {np.max(err_aware):12.3e}")

    segments_float.append((x1, x2, m_f,     c_f))
    segments_quant_naive.append((x1, x2, m_naive, c_naive))
    segments_quant_aware.append((x1, x2, m_aware, c_aware))

# ============================================================
# Step 5: Approximation functions  (exploit σ(-x) = 1 - σ(x))
# ============================================================
def _lookup(xp, table):
    for (x1, x2, m, c) in table:
        if x1 <= xp <= x2:
            return m, c
    return table[-1][2], table[-1][3]   # clamp to last segment

def approx_float(x):
    sign = x < 0
    xp   = abs(x)
    m, c = _lookup(xp, segments_float)
    y    = m * xp + c
    return 1.0 - y if sign else y

def approx_quant_naive(x):
    sign = x < 0
    xp   = quantize_x(abs(x))
    m, c = _lookup(float(xp), segments_quant_naive)
    y    = hw_pipeline(m, xp, c)
    return float(quantize_y(1.0 - y)) if sign else float(y)

def approx_quant_aware(x):
    sign = x < 0
    xp   = quantize_x(abs(x))
    m, c = _lookup(float(xp), segments_quant_aware)
    y    = hw_pipeline(m, xp, c)
    return float(quantize_y(1.0 - y)) if sign else float(y)

# ============================================================
# Step 6: Global evaluation
# ============================================================
x_full  = np.linspace(-X_MAX, X_MAX, 10000)
y_true  = sigmoid(x_full)

y_f     = np.array([approx_float(x)       for x in x_full])
y_naive = np.array([approx_quant_naive(x)  for x in x_full])
y_aware = np.array([approx_quant_aware(x)  for x in x_full])

err_f     = np.abs(y_true - y_f)
err_naive = np.abs(y_true - y_naive)
err_aware = np.abs(y_true - y_aware)

print("\n\n" + "="*68)
print("GLOBAL ERROR SUMMARY")
print("="*68)
print(f"\n{'Model':30s}  {'Avg Error':>12}  {'Max Error':>12}")
print("-"*55)
print(f"{'Float regression':30s}  {np.mean(err_f):12.4e}  {np.max(err_f):12.4e}")
print(f"{'Naive quantized':30s}  {np.mean(err_naive):12.4e}  {np.max(err_naive):12.4e}")
print(f"{'Quant-aware':30s}  {np.mean(err_aware):12.4e}  {np.max(err_aware):12.4e}")
print(f"\n{'Paper result (sigmoid)':30s}  {'4.0e-04':>12}  {'1.7e-03':>12}")

naive_ratio = np.mean(err_naive) / np.mean(err_f)
aware_ratio = np.mean(err_aware) / np.mean(err_f)
print(f"\nNaive Q overhead vs float : {naive_ratio:.2f}x")
print(f"Aware Q overhead vs float : {aware_ratio:.2f}x")

if aware_ratio < 1.3:
    print("✓ Quantization overhead near-zero — matches paper quality!")
elif aware_ratio < naive_ratio * 0.6:
    print(f"✓ Good improvement ({naive_ratio:.2f}x → {aware_ratio:.2f}x)")
else:
    print(f"⚠ Try increasing SEARCH_RANGE (currently {SEARCH_RANGE})")