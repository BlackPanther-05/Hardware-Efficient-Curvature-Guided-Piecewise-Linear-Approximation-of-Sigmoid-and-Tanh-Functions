import numpy as np

# ============================================================
# INT8 FORMAT SIGMOID APPROXIMATION
# ============================================================
# INT8: Signed 8-bit integer with scaling
# Range: -128 to 127
# Mapped domain: sigmoid x in [-8, 8] → INT8 [-128, 127]
# Scale factor: 128/8 = 16  (i.e., x_int = x_float * 16)
# ============================================================

INT8_SCALE = 16.0  # x_float * 16 = x_int

def quantize_int8(val):
    """
    Convert float to INT8 range with scaling
    val: float value in sigmoid domain (roughly -8 to 8)
    returns: quantized value (still in float, but int8-grid representable)
    """
    val_arr = np.array(val, dtype=float)
    val_scaled = np.round(val_arr * INT8_SCALE) / INT8_SCALE
    val_clamped = np.clip(val_scaled, -128.0/INT8_SCALE, 127.0/INT8_SCALE)
    return val_clamped

def dequant_int8(val):
    """Convert from INT8 back to float"""
    return float(val)

# ============================================================
# True sigmoid and second derivative
# ============================================================
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def second_derivative(x):
    s = sigmoid(x)
    return s * (1 - s) * (1 - 2*s)

# ============================================================
# Curvature-weighted segmentation (16 segments)
# ============================================================
X_MAX = 8.0
NUM_SEG = 16

x_dense = np.linspace(0, X_MAX, 50000)
weight = np.sqrt(np.abs(second_derivative(x_dense)))
weight[weight < 1e-14] = 1e-14

cumulative = np.cumsum(weight)
cumulative /= cumulative[-1]

boundaries = [0.0]
for k in range(1, NUM_SEG):
    idx = np.searchsorted(cumulative, k / NUM_SEG)
    boundaries.append(float(x_dense[idx]))
boundaries.append(X_MAX)

# ============================================================
# Linear regression per segment
# ============================================================
def float_regression(xs, ys):
    Xb = np.mean(xs)
    Yb = np.mean(ys)
    m = np.sum((xs - Xb) * (ys - Yb)) / np.sum((xs - Xb)**2)
    return m, Yb - m * Xb

# ============================================================
# INT8-aware regression
# ============================================================
def int8_aware_regression(x_seg, y_seg, m_float, c_float):
    xq = quantize_int8(x_seg)
    
    best_err = float('inf')
    best_m = quantize_int8(m_float)
    best_c = quantize_int8(c_float)
    
    # Search range: ±8 INT8 steps (8 * 1/16 = 0.5 float units)
    search_range = 0.5 / INT8_SCALE
    
    for dm in np.linspace(-search_range, search_range, 17):
        m_cand = quantize_int8(quantize_int8(m_float) + dm)
        
        # Compute effective c
        p = m_cand * xq
        c_opt = np.mean(y_seg - p)
        c_base = quantize_int8(c_opt)
        
        for dc in np.linspace(-search_range, search_range, 17):
            c_cand = quantize_int8(c_base + dc)
            y_hat = quantize_int8(m_cand * xq + c_cand)
            err = np.mean(np.abs(y_seg - y_hat))
            
            if err < best_err:
                best_err = err
                best_m = m_cand
                best_c = c_cand
    
    return best_m, best_c

# ============================================================
# Build segment tables
# ============================================================
segs_float = []
segs_int8_naive = []
segs_int8_aware = []

print("\n" + "="*70)
print("INT8 SIGMOID APPROXIMATION (Signed 8-bit with scaling)")
print("="*70)
print(f"Segments: {NUM_SEG}")
print(f"INT8 Range: -128 to 127")
print(f"Scale Factor: {INT8_SCALE} (x_int = x_float * {INT8_SCALE})")
print(f"INT8 Resolution: 1/{INT8_SCALE} = {1/INT8_SCALE:.4f}")
print("="*70)

for i in range(NUM_SEG):
    x1, x2 = boundaries[i], boundaries[i+1]
    xs = np.linspace(x1, x2, 2048)
    ys = sigmoid(xs)
    
    mf, cf = float_regression(xs, ys)
    m_naive = quantize_int8(mf)
    c_naive = quantize_int8(cf)
    m_aware, c_aware = int8_aware_regression(xs, ys, mf, cf)
    
    # Errors
    xq = quantize_int8(xs)
    ef = np.abs(ys - (mf*xs + cf))
    en = np.abs(ys - quantize_int8(m_naive*xq + c_naive))
    ea = np.abs(ys - quantize_int8(m_aware*xq + c_aware))
    
    segs_float.append((x1, x2, mf, cf))
    segs_int8_naive.append((x1, x2, m_naive, c_naive))
    segs_int8_aware.append((x1, x2, m_aware, c_aware))
    
    print(f"\nSeg {i+1:2d}  [{x1:.4f}, {x2:.4f}]")
    print(f"  {'Model':20s}  {'Avg Error':>12}  {'Max Error':>12}")
    print(f"  {'Float':20s}  {np.mean(ef):12.4e}  {np.max(ef):12.4e}")
    print(f"  {'INT8 Naive':20s}  {np.mean(en):12.4e}  {np.max(en):12.4e}")
    print(f"  {'INT8 Aware':20s}  {np.mean(ea):12.4e}  {np.max(ea):12.4e}")

# ============================================================
# Approximation functions
# ============================================================
def lookup(xp, segs):
    for x1, x2, m, c in segs:
        if x1 <= xp <= x2:
            return m, c
    return segs[-1][2], segs[-1][3]

def approx_float(x):
    sign = x < 0
    xp = abs(x)
    m, c = lookup(xp, segs_float)
    return (1.0 - (m*xp + c)) if sign else (m*xp + c)

def approx_int8_naive(x):
    sign = x < 0
    xp = float(quantize_int8(abs(x)))
    m, c = lookup(xp, segs_int8_naive)
    y = float(quantize_int8(m*xp + c))
    return (1.0 - y) if sign else y

def approx_int8_aware(x):
    sign = x < 0
    xp = float(quantize_int8(abs(x)))
    m, c = lookup(xp, segs_int8_aware)
    y = float(quantize_int8(m*xp + c))
    return (1.0 - y) if sign else y

# ============================================================
# Global evaluation (INT8-representable x only)
# ============================================================
x_hw = np.array([quantize_int8(float(i)/INT8_SCALE) for i in range(-128, 128)])
y_true = sigmoid(x_hw)

y_float = np.array([approx_float(x) for x in x_hw])
y_naive = np.array([approx_int8_naive(x) for x in x_hw])
y_aware = np.array([approx_int8_aware(x) for x in x_hw])

err_float = np.abs(y_true - y_float)
err_naive = np.abs(y_true - y_naive)
err_aware = np.abs(y_true - y_aware)

print("\n\n" + "="*70)
print("GLOBAL ERROR (INT8-representable x only)")
print(f"Points: {len(x_hw)}")
print(f"Range: {x_hw[0]:.4f} to {x_hw[-1]:.4f}")
print("="*70)
print(f"\n{'Model':25s}  {'Avg Error':>12}  {'Max Error':>12}")
print("-"*52)
print(f"{'Float regression':25s}  {np.mean(err_float):12.4e}  {np.max(err_float):12.4e}")
print(f"{'INT8 Naive':25s}  {np.mean(err_naive):12.4e}  {np.max(err_naive):12.4e}")
print(f"{'INT8 Aware':25s}  {np.mean(err_aware):12.4e}  {np.max(err_aware):12.4e}")

ratio_naive = np.mean(err_naive) / np.mean(err_float) if np.mean(err_float) > 0 else 1.0
ratio_aware = np.mean(err_aware) / np.mean(err_float) if np.mean(err_float) > 0 else 1.0
print(f"\nNaive INT8 overhead: {ratio_naive:.2f}x")
print(f"Aware INT8 overhead: {ratio_aware:.2f}x")

print("\n" + "="*70)
print("LUT VALUES FOR RTL (INT8 as signed bytes)")
print("="*70)
print(f"\nINT8 Aware Coefficients:")
print(f"{'Seg':>4}  {'Range':>14}  {'m_int8':>8}  {'c_int8':>8}")
for i, (x1, x2, m, c) in enumerate(segs_int8_aware):
    m_int8 = int(np.round(m * INT8_SCALE))
    c_int8 = int(np.round(c * INT8_SCALE))
    # Clamp to INT8 range
    m_int8 = np.clip(m_int8, -128, 127)
    c_int8 = np.clip(c_int8, -128, 127)
    print(f"  {i+1:2d}  [{x1:.4f},{x2:.4f}]  {m_int8:8d}  {c_int8:8d}")
