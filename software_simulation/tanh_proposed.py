import numpy as np

# ============================================================
# TANH PROPOSED APPROXIMATION - ALL FORMATS
# ============================================================
# Proposed approach: curvature-weighted segmentation via 2nd derivative
# Implements: FP32, INT8, UINT8, FP8
# ============================================================

X_MAX = 8.0
NUM_SEG = 16

# ============================================================
# True tanh and second derivative
# ============================================================
def tanh_func(x):
    return np.tanh(x)

def second_derivative_tanh(x):
    t = np.tanh(x)
    return -2.0 * t * (1.0 - t**2)

# ============================================================
# CURVATURE-WEIGHTED SEGMENTATION (Proposed)
# ============================================================
x_dense = np.linspace(0, X_MAX, 50000)
weight = np.sqrt(np.abs(second_derivative_tanh(x_dense)))
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
# FORMAT-SPECIFIC QUANTIZERS
# ============================================================

# FP32
def quantize_fp32(val):
    return np.float32(val)

# INT8 (x scaled by 16, output by 128)
def quantize_int8(val):
    val_arr = np.array(val, dtype=float)
    val_scaled = np.round(val_arr * 16.0) / 16.0
    return np.clip(val_scaled, -8.0, 8.0)

# UINT8
def quantize_uint8_input(val):
    val_arr = np.array(val, dtype=float)
    val_offset = (val_arr + 8.0) / 16.0 * 255.0
    val_clipped = np.clip(val_offset, 0, 255)
    val_quantized = np.round(val_clipped) / 255.0 * 16.0 - 8.0
    return val_quantized

def quantize_uint8_output(val):
    val_arr = np.array(val, dtype=float)
    val_clipped = np.clip(val_arr, -1.0, 1.0)
    return np.round((val_clipped + 1.0) / 2.0 * 255.0) / 255.0 * 2.0 - 1.0

# FP8
def quantize_fp8(val):
    val_arr = np.array(val, dtype=float)
    scale_fp8 = 16.0
    val_quantized = np.round(val_arr * scale_fp8) / scale_fp8
    return np.clip(val_quantized, -8.0, 8.0)

# ============================================================
# BUILD PROPOSED SEGMENT TABLES
# ============================================================
print("\n" + "="*70)
print("TANH PROPOSED APPROXIMATION - ALL FORMATS")
print("="*70)
print(f"Segments: {NUM_SEG}")
print("Segmentation: Curvature-weighted (via 2nd derivative)")
print("="*70)

segs_float = []
segs_fp32 = []
segs_int8 = []
segs_uint8 = []
segs_fp8 = []

for i in range(NUM_SEG):
    x1, x2 = boundaries[i], boundaries[i+1]
    xs = np.linspace(x1, x2, 2048)
    ys = tanh_func(xs)
    
    mf, cf = float_regression(xs, ys)
    
    # Format-specific quantization
    m_fp32 = quantize_fp32(mf)
    c_fp32 = quantize_fp32(cf)
    
    m_int8 = quantize_int8(mf)
    c_int8 = quantize_int8(cf)
    
    m_uint8 = mf
    c_uint8 = cf
    
    m_fp8 = quantize_fp8(mf)
    c_fp8 = quantize_fp8(cf)
    
    segs_float.append((x1, x2, mf, cf))
    segs_fp32.append((x1, x2, m_fp32, c_fp32))
    segs_int8.append((x1, x2, m_int8, c_int8))
    segs_uint8.append((x1, x2, m_uint8, c_uint8))
    segs_fp8.append((x1, x2, m_fp8, c_fp8))
    
    # Errors
    xq_int8 = quantize_int8(xs)
    xq_fp8 = quantize_fp8(xs)
    
    ef = np.abs(ys - (mf*xs + cf))
    efp32 = np.abs(ys - (m_fp32*xs + c_fp32))
    eint8 = np.abs(ys - (m_int8*xq_int8 + c_int8))
    euint8 = np.abs(ys - quantize_uint8_output(m_uint8*quantize_uint8_input(xs) + c_uint8))
    efp8 = np.abs(ys - (m_fp8*xq_fp8 + c_fp8))
    
    print(f"\nSeg {i+1:2d}  [{x1:.4f}, {x2:.4f}]")
    print(f"  {'Format':20s}  {'Avg Error':>12}  {'Max Error':>12}")
    print(f"  {'Float':20s}  {np.mean(ef):12.4e}  {np.max(ef):12.4e}")
    print(f"  {'FP32':20s}  {np.mean(efp32):12.4e}  {np.max(efp32):12.4e}")
    print(f"  {'INT8':20s}  {np.mean(eint8):12.4e}  {np.max(eint8):12.4e}")
    print(f"  {'UINT8':20s}  {np.mean(euint8):12.4e}  {np.max(euint8):12.4e}")
    print(f"  {'FP8':20s}  {np.mean(efp8):12.4e}  {np.max(efp8):12.4e}")

# ============================================================
# GLOBAL EVALUATION
# ============================================================
print("\n\n" + "="*70)
print("GLOBAL ERROR SUMMARY - TANH PROPOSED")
print("="*70)

x_test = np.linspace(-8.0, 8.0, 1000)
y_true = tanh_func(x_test)

# Simple lookup function
def lookup(xp, segs):
    xp_abs = abs(xp)
    for x1, x2, m, c in segs:
        if x1 <= xp_abs <= x2:
            return m, c
    return segs[-1][2], segs[-1][3]

def approx_float(x):
    sign = x < 0
    xp = abs(x)
    m, c = lookup(xp, segs_float)
    return -(m*xp + c) if sign else (m*xp + c)

def approx_fp32(x):
    sign = x < 0
    xp = float(abs(x))
    m, c = lookup(xp, segs_fp32)
    return -(m*xp + c) if sign else (m*xp + c)

def approx_int8(x):
    sign = x < 0
    xp = float(quantize_int8(abs(x)))
    m, c = lookup(xp, segs_int8)
    return -(m*xp + c) if sign else (m*xp + c)

def approx_uint8(x):
    sign = x < 0
    xp = float(quantize_uint8_input(abs(x)))
    m, c = lookup(xp, segs_uint8)
    y = m*xp + c
    y_q = quantize_uint8_output(y)
    return -y_q if sign else y_q

def approx_fp8(x):
    sign = x < 0
    xp = float(quantize_fp8(abs(x)))
    m, c = lookup(xp, segs_fp8)
    return -(m*xp + c) if sign else (m*xp + c)

y_float = np.array([approx_float(x) for x in x_test])
y_fp32 = np.array([approx_fp32(x) for x in x_test])
y_int8 = np.array([approx_int8(x) for x in x_test])
y_uint8 = np.array([approx_uint8(x) for x in x_test])
y_fp8 = np.array([approx_fp8(x) for x in x_test])

err_float = np.abs(y_true - y_float)
err_fp32 = np.abs(y_true - y_fp32)
err_int8 = np.abs(y_true - y_int8)
err_uint8 = np.abs(y_true - y_uint8)
err_fp8 = np.abs(y_true - y_fp8)

print(f"\n{'Format':25s}  {'Avg Error':>12}  {'Max Error':>12}")
print("-"*52)
print(f"{'Float regression':25s}  {np.mean(err_float):12.4e}  {np.max(err_float):12.4e}")
print(f"{'FP32':25s}  {np.mean(err_fp32):12.4e}  {np.max(err_fp32):12.4e}")
print(f"{'INT8':25s}  {np.mean(err_int8):12.4e}  {np.max(err_int8):12.4e}")
print(f"{'UINT8':25s}  {np.mean(err_uint8):12.4e}  {np.max(err_uint8):12.4e}")
print(f"{'FP8':25s}  {np.mean(err_fp8):12.4e}  {np.max(err_fp8):12.4e}")

print("\n" + "="*70)
print("TANH PROPOSED - COMPLETE")
print("="*70)
