import numpy as np

# ============================================================
# USER CONFIG
# ============================================================
SEG_BITS     = 4
NUM_SEG      = 2**SEG_BITS   # 16 segments
X_MAX        = 8.0
SEARCH_RANGE = 8

# ============================================================
# Fixed-point formats
#
#  11-bit (paper):
#    x   : 1s + 3i + 7f   → SCALE_X  = 2^7  = 128
#    m,c : 1s + 0i + 10f  → SCALE_MC = 2^10 = 1024
#    y   : 1s + 0i + 10f  → SCALE_Y  = 2^10 = 1024
#    product: m(10f) x x(7f) = 17f → keep 10f → floor >> 7
#
#  15-bit (new):
#    x   : 1s + 3i + 11f  → SCALE_X  = 2^11 = 2048
#    m,c : 1s + 0i + 14f  → SCALE_MC = 2^14 = 16384
#    y   : 1s + 0i + 14f  → SCALE_Y  = 2^14 = 16384
#    product: m(14f) x x(11f) = 25f → keep 14f → floor >> 11
# ============================================================
SCALE_X11  = 2**7;   SCALE_MC11 = 2**10;  SCALE_Y11  = 2**10
SCALE_X15  = 2**11;  SCALE_MC15 = 2**14;  SCALE_Y15  = 2**14
LSB11      = 1.0 / SCALE_MC11
LSB15      = 1.0 / SCALE_MC15

# ============================================================
# Quantizers
# ============================================================
def qx11(v):  return np.round(np.array(v,float) * SCALE_X11)  / SCALE_X11
def qmc11(v): return np.round(np.array(v,float) * SCALE_MC11) / SCALE_MC11
def qy11(v):  return np.round(np.array(v,float) * SCALE_Y11)  / SCALE_Y11

def qx15(v):  return np.round(np.array(v,float) * SCALE_X15)  / SCALE_X15
def qmc15(v): return np.round(np.array(v,float) * SCALE_MC15) / SCALE_MC15
def qy15(v):  return np.round(np.array(v,float) * SCALE_Y15)  / SCALE_Y15

# ============================================================
# Hardware pipelines
#   p = floor(m * xp * SCALE_MC) / SCALE_MC   [truncate product]
#   s = quantize_y(p + c)
# ============================================================
def hw11(m, xp, c):
    p = np.floor(m * xp * SCALE_MC11) / SCALE_MC11
    return qy11(p + c)

def hw15(m, xp, c):
    p = np.floor(m * xp * SCALE_MC15) / SCALE_MC15
    return qy15(p + c)

# ============================================================
# True sigmoid
# ============================================================
def sigmoid(x): return 1.0 / (1.0 + np.exp(-x))
def second_derivative(x):
    s = sigmoid(x); return s * (1 - s) * (1 - 2*s)

# ============================================================
# Curvature-weighted segmentation
#   More segments in high-curvature regions → lower PWL error
#   This is what gives float avg = 1.70e-4 vs paper's ~5e-4
# ============================================================
x_dense    = np.linspace(0, X_MAX, 50000)
weight     = np.sqrt(np.abs(second_derivative(x_dense)))
weight[weight < 1e-14] = 1e-14
cumulative = np.cumsum(weight); cumulative /= cumulative[-1]

boundaries = [0.0]
for k in range(1, NUM_SEG):
    idx = np.searchsorted(cumulative, k / NUM_SEG)
    boundaries.append(float(x_dense[idx]))
boundaries.append(X_MAX)

# ============================================================
# Float regression (least squares)
# ============================================================
def float_regression(xs, ys):
    Xb = np.mean(xs); Yb = np.mean(ys)
    m  = np.sum((xs - Xb) * (ys - Yb)) / np.sum((xs - Xb)**2)
    return m, Yb - m * Xb

# ============================================================
# Quantization-aware regression (generic)
# ============================================================
def quant_aware_reg(xs, ys, mf, cf, qx_fn, qmc_fn, hw_fn, lsb):
    xq   = qx_fn(xs)
    best = float('inf')
    bm   = qmc_fn(mf)
    bc   = qmc_fn(cf)
    mb   = qmc_fn(mf)

    for dm in range(-SEARCH_RANGE, SEARCH_RANGE + 1):
        mc = mb + dm * lsb
        p  = np.floor(mc * xq / lsb) * lsb   # truncated product in float units
        cb = qmc_fn(np.mean(ys - p))

        for dc in range(-SEARCH_RANGE, SEARCH_RANGE + 1):
            cc  = cb + dc * lsb
            yh  = hw_fn(mc, xq, cc)
            err = np.mean(np.abs(ys - yh))
            if err < best:
                best = err; bm = mc; bc = cc

    return bm, bc

# ============================================================
# Build all segment tables
# ============================================================
segs11_float = []; segs11_naive = []; segs11_aware = []
segs15_float = []; segs15_naive = []; segs15_aware = []

print("\n" + "="*72)
print(f"Curvature-weighted boundaries ({NUM_SEG} segments):")
for i in range(NUM_SEG):
    print(f"  Seg{i+1:2d}: [{boundaries[i]:.4f}, {boundaries[i+1]:.4f}]")

print("\nBuilding segment tables...")
print("="*72)

for i in range(NUM_SEG):
    x1, x2 = boundaries[i], boundaries[i+1]
    xs = np.linspace(x1, x2, 2048)
    ys = sigmoid(xs)
    mf, cf = float_regression(xs, ys)

    # 11-bit
    m11n = qmc11(mf); c11n = qmc11(cf)
    m11a, c11a = quant_aware_reg(xs, ys, mf, cf, qx11, qmc11, hw11, LSB11)

    # 15-bit
    m15n = qmc15(mf); c15n = qmc15(cf)
    m15a, c15a = quant_aware_reg(xs, ys, mf, cf, qx15, qmc15, hw15, LSB15)

    segs11_float.append((x1, x2, mf,   cf))
    segs11_naive.append((x1, x2, m11n, c11n))
    segs11_aware.append((x1, x2, m11a, c11a))

    segs15_float.append((x1, x2, mf,   cf))
    segs15_naive.append((x1, x2, m15n, c15n))
    segs15_aware.append((x1, x2, m15a, c15a))

    # Per-segment errors (on dense linspace, to show pure fit quality)
    xq11 = qx11(xs); xq15 = qx15(xs)
    ef   = np.abs(ys - (mf*xs + cf))
    en11 = np.abs(ys - hw11(m11n, xq11, c11n))
    ea11 = np.abs(ys - hw11(m11a, xq11, c11a))
    en15 = np.abs(ys - hw15(m15n, xq15, c15n))
    ea15 = np.abs(ys - hw15(m15a, xq15, c15a))

    print(f"\nSeg{i+1:2d} [{x1:.4f},{x2:.4f}]")
    print(f"  {'Model':18s}  {'Avg Error':>12}  {'Max Error':>12}  {'m_int(11b)':>11}  {'c_int(11b)':>11}  {'m_int(15b)':>11}  {'c_int(15b)':>11}")
    print(f"  {'Float':18s}  {np.mean(ef):12.3e}  {np.max(ef):12.3e}  {'N/A':>11}  {'N/A':>11}  {'N/A':>11}  {'N/A':>11}")
    print(f"  {'11-bit Naive':18s}  {np.mean(en11):12.3e}  {np.max(en11):12.3e}  {int(round(m11n*SCALE_MC11)):>11}  {int(round(c11n*SCALE_MC11)):>11}  {'—':>11}  {'—':>11}")
    print(f"  {'11-bit Aware':18s}  {np.mean(ea11):12.3e}  {np.max(ea11):12.3e}  {int(round(m11a*SCALE_MC11)):>11}  {int(round(c11a*SCALE_MC11)):>11}  {'—':>11}  {'—':>11}")
    print(f"  {'15-bit Naive':18s}  {np.mean(en15):12.3e}  {np.max(en15):12.3e}  {'—':>11}  {'—':>11}  {int(round(m15n*SCALE_MC15)):>11}  {int(round(c15n*SCALE_MC15)):>11}")
    print(f"  {'15-bit Aware':18s}  {np.mean(ea15):12.3e}  {np.max(ea15):12.3e}  {'—':>11}  {'—':>11}  {int(round(m15a*SCALE_MC15)):>11}  {int(round(c15a*SCALE_MC15)):>11}")

# ============================================================
# Approximation functions
# ============================================================
def lookup(xp, segs):
    for x1, x2, m, c in segs:
        if x1 <= xp <= x2: return m, c
    return segs[-1][2], segs[-1][3]

def make_approx(segs, qx_fn, qy_fn, hw_fn):
    def approx(b):
        sign = b < 0
        xp   = float(qx_fn(abs(b)))
        m, c = lookup(xp, segs)
        y    = float(hw_fn(m, xp, c))
        return float(qy_fn(1.0 - y)) if sign else float(y)
    return approx

approx11_float = make_approx(segs11_float, qx11, qy11, hw11)
approx11_naive = make_approx(segs11_naive, qx11, qy11, hw11)
approx11_aware = make_approx(segs11_aware, qx11, qy11, hw11)
approx15_float = make_approx(segs15_float, qx15, qy15, hw15)
approx15_naive = make_approx(segs15_naive, qx15, qy15, hw15)
approx15_aware = make_approx(segs15_aware, qx15, qy15, hw15)

# ============================================================
# Global evaluation — hardware-representable x ONLY
#
#  11-bit: b_int = -1023..+1023  →  x = b_int/128   (2047 pts)
#  15-bit: b_int = -16383..+16383 → x = b_int/2048  (32767 pts)
#
#  WHY: linspace hits non-grid x values → quantize_x snaps them
#  to the grid → error is measured vs raw x, not the x hardware
#  actually received → inflated avg error
#  Using b_int/scale ensures x IS on the grid → fair measurement
# ============================================================
x_hw11 = np.array([b / SCALE_X11 for b in range(-1023,  1024)])
x_hw15 = np.array([b / SCALE_X15 for b in range(-16383, 16384)])

def eval_model(approx_fn, x_hw):
    y_hw   = np.array([approx_fn(x) for x in x_hw])
    y_true = sigmoid(x_hw)
    err    = np.abs(y_true - y_hw)
    return np.mean(err), np.max(err)

print("\n\n" + "="*72)
print("GLOBAL ERROR — hardware-representable x only")
print("  11-bit: 2047 points  (b_int=-1023..+1023,  step=1/128)")
print("  15-bit: 32767 points (b_int=-16383..+16383, step=1/2048)")
print("="*72)
print(f"\n{'Model':25s}  {'Pts':>7}  {'Avg Error':>12}  {'Max Error':>12}")
print("-"*60)

results = [
    ("11-bit Float",  approx11_float, x_hw11),
    ("11-bit Naive",  approx11_naive, x_hw11),
    ("11-bit Aware",  approx11_aware, x_hw11),
    ("15-bit Float",  approx15_float, x_hw15),
    ("15-bit Naive",  approx15_naive, x_hw15),
    ("15-bit Aware",  approx15_aware, x_hw15),
]

avg_11a = avg_15a = 0
for label, fn, x_hw in results:
    avg, mx = eval_model(fn, x_hw)
    print(f"  {label:23s}  {len(x_hw):7d}  {avg:12.4e}  {mx:12.4e}")
    if label == "11-bit Aware": avg_11a = avg
    if label == "15-bit Aware": avg_15a = avg

print()
print(f"  Improvement (11-bit Aware → 15-bit Aware): {avg_11a/avg_15a:.1f}x better avg error")
print(f"  15-bit y LSB = 1/16384 = {1/16384:.2e}  (quant noise floor)")
print(f"  15-bit PWL fitting error dominates → more segments needed for further gains")

# ============================================================
# LUT integer values for RTL
# ============================================================
print("\n\n" + "="*72)
print("LUT INTEGER VALUES FOR RTL")
print("="*72)

print(f"\n11-bit Aware  (m * 1024, c * 1024):")
print(f"{'Seg':>4}  {'x range':>14}  {'m_int':>7}  {'c_int':>7}")
for i,(x1,x2,m,c) in enumerate(segs11_aware):
    print(f"  {i+1:2d}  [{x1:.4f},{x2:.4f}]  {int(round(m*SCALE_MC11)):7d}  {int(round(c*SCALE_MC11)):7d}")

print(f"\n15-bit Aware  (m * 16384, c * 16384):")
print(f"{'Seg':>4}  {'x range':>14}  {'m_int':>8}  {'c_int':>8}")
for i,(x1,x2,m,c) in enumerate(segs15_aware):
    print(f"  {i+1:2d}  [{x1:.4f},{x2:.4f}]  {int(round(m*SCALE_MC15)):8d}  {int(round(c*SCALE_MC15)):8d}")