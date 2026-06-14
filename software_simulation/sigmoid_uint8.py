import numpy as np

# ============================================================
# UINT8 FORMAT SIGMOID APPROXIMATION
# ============================================================
# UINT8: Unsigned 8-bit integer with scaling
# Range: 0 to 255
# Mapped domain: sigmoid output in [0, 1] → UINT8 [0, 255]
# Scale factor: 255  (i.e., y_int = y_float * 255)
# Note: For symmetric sigmoid input domain [-8, 8], we use:
#   x_scaled = (x + 8) / 16 to map [-8, 8] → [0, 1]
#   then x_uint8 = x_scaled * 255
# ============================================================

UINT8_OUTPUT_SCALE = 255.0  # y: float [0, 1] → UINT8 [0, 255]
UINT8_INPUT_SCALE = 255.0 / 16.0  # x: [-8, 8] → [0, 255] (offset+scale)
X_OFFSET = 8.0  # Map [-8, 8] to [0, 16]

def quantize_uint8_output(val):
    """
    Convert sigmoid output (typically [0, 1]) to UINT8 [0, 255]
    """
    val_arr = np.array(val, dtype=float)
    val_clipped = np.clip(val_arr, 0.0, 1.0)
    val_scaled = np.round(val_clipped * UINT8_OUTPUT_SCALE) / UINT8_OUTPUT_SCALE
    return val_scaled

def quantize_uint8_input(val):
    """
    Convert sigmoid input [-8, 8] to UINT8-representable space [0, 255]
    Then back to float for computation
    """
    val_arr = np.array(val, dtype=float)
    # Map [-8, 8] to [0, 255]
    val_offset = val_arr + X_OFFSET  # [-8, 8] → [0, 16]
    val_scaled = (val_offset / 16.0) * UINT8_INPUT_SCALE  # [0, 16] → [0, 255]
    val_clipped = np.clip(val_scaled, 0, 255)
    # Quantize
    val_quantized = np.round(val_clipped) / UINT8_INPUT_SCALE  # Back to [0, 255] in float
    # Unmap back to [-8, 8]
    val_unmapped = (val_quantized * 16.0 / UINT8_INPUT_SCALE) - X_OFFSET
    return val_unmapped

def dequant_uint8_output(val):
    """Convert from UINT8 output back to float [0, 1]"""
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
# UINT8-aware regression (output quantization)
# ============================================================
def uint8_aware_regression(x_seg, y_seg, m_float, c_float):
    xq = quantize_uint8_input(x_seg)
    
    best_err = float('inf')
    best_m = m_float
    best_c = c_float
    
    # Search range: ±8 steps in UINT8 space (~3% perturbation)
    search_range = 0.01
    
    for dm in np.linspace(-search_range, search_range, 17):
        m_cand = m_float * (1 + dm)
        
        # Compute effective c
        p = m_cand * xq
        c_opt = np.mean(y_seg - p)
        c_base = c_opt
        
        for dc in np.linspace(-search_range, search_range, 17):
            c_cand = c_base * (1 + dc)
            y_raw = m_cand * xq + c_cand
            y_hat = quantize_uint8_output(y_raw)  # Quantize output only
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
segs_uint8_naive = []
segs_uint8_aware = []

print("\n" + "="*70)
print("UINT8 SIGMOID APPROXIMATION (Unsigned 8-bit with output scaling)")
print("="*70)
print(f"Segments: {NUM_SEG}")
print(f"UINT8 Output Range: 0 to 255")
print(f"Output Scale Factor: {UINT8_OUTPUT_SCALE}")
print(f"Output Resolution: 1/{UINT8_OUTPUT_SCALE:.1f} = {1/UINT8_OUTPUT_SCALE:.4f}")
print(f"Input mapping: [-8, 8] → [0, 255]")
print("="*70)

for i in range(NUM_SEG):
    x1, x2 = boundaries[i], boundaries[i+1]
    xs = np.linspace(x1, x2, 2048)
    ys = sigmoid(xs)
    
    mf, cf = float_regression(xs, ys)
    m_naive = mf
    c_naive = cf
    m_aware, c_aware = uint8_aware_regression(xs, ys, mf, cf)
    
    # Errors
    xq = quantize_uint8_input(xs)
    ef = np.abs(ys - (mf*xs + cf))
    y_naive_hat = quantize_uint8_output(m_naive*xq + c_naive)
    en = np.abs(ys - y_naive_hat)
    y_aware_hat = quantize_uint8_output(m_aware*xq + c_aware)
    ea = np.abs(ys - y_aware_hat)
    
    segs_float.append((x1, x2, mf, cf))
    segs_uint8_naive.append((x1, x2, m_naive, c_naive))
    segs_uint8_aware.append((x1, x2, m_aware, c_aware))
    
    print(f"\nSeg {i+1:2d}  [{x1:.4f}, {x2:.4f}]")
    print(f"  {'Model':20s}  {'Avg Error':>12}  {'Max Error':>12}")
    print(f"  {'Float':20s}  {np.mean(ef):12.4e}  {np.max(ef):12.4e}")
    print(f"  {'UINT8 Naive':20s}  {np.mean(en):12.4e}  {np.max(en):12.4e}")
    print(f"  {'UINT8 Aware':20s}  {np.mean(ea):12.4e}  {np.max(ea):12.4e}")

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

def approx_uint8_naive(x):
    sign = x < 0
    xp = float(quantize_uint8_input(abs(x)))
    m, c = lookup(xp, segs_uint8_naive)
    y = float(quantize_uint8_output(m*xp + c))
    return (1.0 - y) if sign else y

def approx_uint8_aware(x):
    sign = x < 0
    xp = float(quantize_uint8_input(abs(x)))
    m, c = lookup(xp, segs_uint8_aware)
    y = float(quantize_uint8_output(m*xp + c))
    return (1.0 - y) if sign else y

# ============================================================
# Global evaluation (UINT8-representable x only)
# ============================================================
x_hw = np.array([quantize_uint8_input(float(i)/UINT8_INPUT_SCALE * 16.0 - 8.0) for i in range(256)])
y_true = sigmoid(x_hw)

y_float = np.array([approx_float(x) for x in x_hw])
y_naive = np.array([approx_uint8_naive(x) for x in x_hw])
y_aware = np.array([approx_uint8_aware(x) for x in x_hw])

err_float = np.abs(y_true - y_float)
err_naive = np.abs(y_true - y_naive)
err_aware = np.abs(y_true - y_aware)

print("\n\n" + "="*70)
print("GLOBAL ERROR (UINT8-representable x/y only)")
print(f"Points: {len(x_hw)}")
print(f"Range: {x_hw[0]:.4f} to {x_hw[-1]:.4f}")
print("="*70)
print(f"\n{'Model':25s}  {'Avg Error':>12}  {'Max Error':>12}")
print("-"*52)
print(f"{'Float regression':25s}  {np.mean(err_float):12.4e}  {np.max(err_float):12.4e}")
print(f"{'UINT8 Naive':25s}  {np.mean(err_naive):12.4e}  {np.max(err_naive):12.4e}")
print(f"{'UINT8 Aware':25s}  {np.mean(err_aware):12.4e}  {np.max(err_aware):12.4e}")

ratio_naive = np.mean(err_naive) / np.mean(err_float) if np.mean(err_float) > 0 else 1.0
ratio_aware = np.mean(err_aware) / np.mean(err_float) if np.mean(err_float) > 0 else 1.0
print(f"\nNaive UINT8 overhead: {ratio_naive:.2f}x")
print(f"Aware UINT8 overhead: {ratio_aware:.2f}x")

print("\n" + "="*70)
print("LUT VALUES FOR RTL (UINT8 as unsigned bytes)")
print("="*70)
print(f"\nUINT8 Aware Coefficients (×256 for fixed-point storage):")
print(f"{'Seg':>4}  {'Range':>14}  {'m_fp8':>8}  {'c_fp8':>8}")
for i, (x1, x2, m, c) in enumerate(segs_uint8_aware):
    m_uint8 = int(np.round(m * 256)) & 0xFF
    c_uint8 = int(np.round(c * 256)) & 0xFF
    print(f"  {i+1:2d}  [{x1:.4f},{x2:.4f}]  {m_uint8:8d}  {c_uint8:8d}")
