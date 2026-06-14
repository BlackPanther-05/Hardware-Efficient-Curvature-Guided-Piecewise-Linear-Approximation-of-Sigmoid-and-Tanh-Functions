import numpy as np

# ==========================================
# Fixed-point configuration
#
#  Original pipeline (baseline):
#    x, m, c, y all use SCALE = 2^10 = 1024
#    quantize = round(val * 1024) / 1024
#
#  Naive / Aware pipeline (paper-correct):
#    x   : 1s + 3i + 7f  → SCALE_X  = 128
#    m,c : 1s + 0i + 10f → SCALE_MC = 1024
#    y   : 1s + 0i + 10f → SCALE_Y  = 1024
#    product: floor(m * xp * SCALE_MC) / SCALE_MC
# ==========================================
SCALE    = 2**10        # original baseline scale
SCALE_X  = 2**7         # x  : 7 frac bits
SCALE_MC = 2**10        # m,c: 10 frac bits
SCALE_Y  = 2**10        # y  : 10 frac bits
LSB_MC   = 1.0 / SCALE_MC

def quantize(val):                          # original single-scale
    return np.round(np.array(val, dtype=float) * SCALE) / SCALE

def quantize_x(val):
    return np.round(np.array(val, dtype=float) * SCALE_X) / SCALE_X

def quantize_mc(val):
    return np.round(np.array(val, dtype=float) * SCALE_MC) / SCALE_MC

def quantize_y(val):
    return np.round(np.array(val, dtype=float) * SCALE_Y) / SCALE_Y


# ==========================================
# True sigmoid
# ==========================================
def true_sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


# ==========================================
# ORIGINAL segments — paper Table I float
# m,c values (from paper image)
# ==========================================
SEGMENTS_ORIGINAL = [
    (0.000, 0.166, 0.2494, 0.5000),
    (0.166, 0.344, 0.2458, 0.5006),
    (0.344, 0.520, 0.2385, 0.5032),
    (0.520, 0.704, 0.2278, 0.5088),
    (0.704, 0.888, 0.2141, 0.5185),
    (0.888, 1.089, 0.1975, 0.5333),
    (1.089, 1.305, 0.1781, 0.5545),
    (1.305, 1.537, 0.1566, 0.5826),
    (1.537, 1.777, 0.1344, 0.6167),
    (1.777, 2.042, 0.1124, 0.6560),
    (2.042, 2.354, 0.0900, 0.7018),
    (2.354, 2.730, 0.0677, 0.7544),
    (2.730, 3.187, 0.0470, 0.8110),
    (3.187, 3.811, 0.0286, 0.8698),
    (3.811, 4.804, 0.0134, 0.9284),
    (4.804, 6.443, 0.0038, 0.9744),
]

TAIL_START = 6.443
CONST_VAL  = 0.999


# ==========================================
# NAIVE segments — round paper float m,c
# to Q1.10 (10 frac bits)
# ==========================================
SEGMENTS_NAIVE = [
    (x1, x2, quantize_mc(m), quantize_mc(c))
    for (x1, x2, m, c) in SEGMENTS_ORIGINAL
]


# ==========================================
# AWARE segments — search best m,c on
# paper boundaries using exact hw pipeline
# ==========================================
SEARCH_RANGE = 8

def float_regression(x_seg, y_seg):
    X_bar = np.mean(x_seg)
    Y_bar = np.mean(y_seg)
    m = np.sum((x_seg - X_bar) * (y_seg - Y_bar)) / np.sum((x_seg - X_bar)**2)
    c = Y_bar - m * X_bar
    return m, c

def quant_aware_regression(x_seg, y_seg, m_float, c_float):
    x_seg_q  = quantize_x(x_seg)
    best_err = float('inf')
    best_mq  = quantize_mc(m_float)
    best_cq  = quantize_mc(c_float)
    m_base   = quantize_mc(m_float)

    for dm in range(-SEARCH_RANGE, SEARCH_RANGE + 1):
        m_cand = m_base + dm * LSB_MC
        p      = np.floor(m_cand * x_seg_q * SCALE_MC) / SCALE_MC
        c_base = quantize_mc(np.mean(y_seg - p))

        for dc in range(-SEARCH_RANGE, SEARCH_RANGE + 1):
            c_cand = c_base + dc * LSB_MC
            p_hw   = np.floor(m_cand * x_seg_q * SCALE_MC) / SCALE_MC
            y_hat  = quantize_y(p_hw + c_cand)
            err    = np.mean(np.abs(y_seg - y_hat))
            if err < best_err:
                best_err = err
                best_mq  = m_cand
                best_cq  = c_cand

    return best_mq, best_cq

SEGMENTS_AWARE = []
for (x1, x2, m, c) in SEGMENTS_ORIGINAL:
    x_seg = np.linspace(x1, x2, 2048)
    y_seg = true_sigmoid(x_seg)
    m_a, c_a = quant_aware_regression(x_seg, y_seg, m, c)
    SEGMENTS_AWARE.append((x1, x2, m_a, c_a))


# ==========================================
# Hardware sigmoid — ORIGINAL
# Single SCALE=1024 everywhere (your baseline)
# ==========================================
def approx_sigmoid_original(b):
    sign = b < 0
    xp   = quantize(abs(b))

    if xp <= TAIL_START:
        for (x1, x2, m, c) in SEGMENTS_ORIGINAL:
            if xp >= x1 and xp <= x2:
                mult  = quantize(m * xp)
                y_pos = quantize(mult + c)
                break
    else:
        y_pos = quantize(CONST_VAL)

    return float(quantize(1.0 - y_pos)) if sign else float(y_pos)


# ==========================================
# Hardware sigmoid — NAIVE
# Rounded m,c + correct split quantization
# ==========================================
def approx_sigmoid_naive(b):
    sign = b < 0
    xp   = quantize_x(abs(b))

    if xp <= TAIL_START:
        for (x1, x2, m, c) in SEGMENTS_NAIVE:
            if xp >= x1 and xp <= x2:
                p     = np.floor(m * xp * SCALE_MC) / SCALE_MC
                y_pos = float(quantize_y(p + c))
                break
    else:
        y_pos = float(quantize_y(CONST_VAL))

    return float(quantize_y(1.0 - y_pos)) if sign else float(y_pos)


# ==========================================
# Hardware sigmoid — AWARE
# Optimal m,c + correct split quantization
# ==========================================
def approx_sigmoid_aware(b):
    sign = b < 0
    xp   = quantize_x(abs(b))

    if xp <= TAIL_START:
        for (x1, x2, m, c) in SEGMENTS_AWARE:
            if xp >= x1 and xp <= x2:
                p     = np.floor(m * xp * SCALE_MC) / SCALE_MC
                y_pos = float(quantize_y(p + c))
                break
    else:
        y_pos = float(quantize_y(CONST_VAL))

    return float(quantize_y(1.0 - y_pos)) if sign else float(y_pos)


# ==========================================
# Segment-wise error — all three models
# ==========================================
print("\n" + "="*70)
print("SEGMENT ERROR")
print("="*70)
print(f"{'Seg':>4}  {'Range':>16}  {'Model':>10}  {'Avg Error':>13}  {'Max Error':>13}")
print("-"*70)

for i, (x1, x2, _, _) in enumerate(SEGMENTS_ORIGINAL):
    x_seg  = np.linspace(x1, x2, 10000)
    y_true = true_sigmoid(x_seg)
    rng    = f"[{x1:.3f}, {x2:.3f}]"

    for label, fn in [("Original", approx_sigmoid_original),
                      ("Naive Q",  approx_sigmoid_naive),
                      ("Aware Q",  approx_sigmoid_aware)]:
        y_hat = np.array([fn(x) for x in x_seg])
        err   = np.abs(y_true - y_hat)
        print(f"  {i+1:2d}  {rng:>16}  {label:>10}  {np.mean(err):13.4e}  {np.max(err):13.4e}")
    print()


# ==========================================
# Tail error — all three models
# ==========================================
x_tail      = np.linspace(TAIL_START, 8, 20000)
y_true_tail = true_sigmoid(x_tail)

print("Segment 17 (Tail)")
for label, fn in [("Original", approx_sigmoid_original),
                  ("Naive Q",  approx_sigmoid_naive),
                  ("Aware Q",  approx_sigmoid_aware)]:
    y_hat = np.array([fn(x) for x in x_tail])
    err   = np.abs(y_true_tail - y_hat)
    print(f"  {label:>10}  Avg: {np.mean(err):.4e}  Max: {np.max(err):.4e}")
print()


# ==========================================
# Global error (-8 to 8)
# ==========================================
x_full = np.linspace(-8, 8, 20000)
y_true = true_sigmoid(x_full)

y_orig = np.array([approx_sigmoid_original(x) for x in x_full])
y_naiv = np.array([approx_sigmoid_naive(x)    for x in x_full])
y_awar = np.array([approx_sigmoid_aware(x)    for x in x_full])

err_orig = np.abs(y_true - y_orig)
err_naiv = np.abs(y_true - y_naiv)
err_awar = np.abs(y_true - y_awar)

print("\n" + "="*50)
print("GLOBAL ERROR  (-8 to 8)")
print("="*50)
print(f"\n{'Model':>10}  {'Avg Error':>14}  {'Max Error':>14}")
print("-"*42)
print(f"{'Original':>10}  {np.mean(err_orig):14.4e}  {np.max(err_orig):14.4e}")
print(f"{'Naive Q':>10}  {np.mean(err_naiv):14.4e}  {np.max(err_naiv):14.4e}")
print(f"{'Aware Q':>10}  {np.mean(err_awar):14.4e}  {np.max(err_awar):14.4e}")

print(f"\nNaive Q vs Original : {np.mean(err_naiv)/np.mean(err_orig):.2f}x")
print(f"Aware Q vs Original : {np.mean(err_awar)/np.mean(err_orig):.2f}x")