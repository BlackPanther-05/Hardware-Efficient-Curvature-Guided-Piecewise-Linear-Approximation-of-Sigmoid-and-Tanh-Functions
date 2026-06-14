import numpy as np

# ==========================================
# USER CONFIG (CHANGE HERE)

# ==========================================
FRAC_BITS = 10        # 10 → 11-bit, 15 → 16-bit
SEG_BITS = 4          # 3→8 segments, 4→16, 5→32
NUM_SEG = 2**SEG_BITS

X_MAX = 8.0

# ==========================================
# Fixed-point
# ==========================================
SCALE = 2**FRAC_BITS

def quantize(val):
    return np.round(val * SCALE) / SCALE

# ==========================================
# Functions
# ==========================================
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def second_derivative(x):
    s = sigmoid(x)
    return s * (1 - s) * (1 - 2*s)

# ==========================================
# Step 1: Segmentation (√|f''|)
# ==========================================
x_dense = np.linspace(0, X_MAX, 50000)
weight = np.sqrt(np.abs(second_derivative(x_dense)))
weight[weight < 1e-14] = 1e-14

cumulative = np.cumsum(weight)
cumulative /= cumulative[-1]

boundaries = [0.0]
for k in range(1, NUM_SEG):
    idx = np.searchsorted(cumulative, k / NUM_SEG)
    boundaries.append(x_dense[idx])
boundaries.append(X_MAX)

# ==========================================
# Step 2: Compute segments (float + quant)
# ==========================================
segments_float = []
segments_quant = []

print("\n=====================================")
print(f"CONFIG: {NUM_SEG} segments | Q1.{FRAC_BITS}")
print("=====================================\n")

for i in range(NUM_SEG):

    x1 = boundaries[i]
    x2 = boundaries[i+1]

    x_seg = np.linspace(x1, x2, 2048)
    y_seg = sigmoid(x_seg)

    # Linear regression
    X_bar = np.mean(x_seg)
    Y_bar = np.mean(y_seg)

    m = np.sum((x_seg - X_bar)*(y_seg - Y_bar)) / np.sum((x_seg - X_bar)**2)
    c = Y_bar - m * X_bar

    # Quantize m, c
    m_q = quantize(m)
    c_q = quantize(c)

    # Float error
    y_hat_f = m * x_seg + c
    err_f = np.abs(y_seg - y_hat_f)

    # Quantized pipeline error
    y_hat_q = []
    for x in x_seg:
        x_q = quantize(x)
        mult = quantize(m_q * x_q)
        y_q = quantize(mult + c_q)
        y_hat_q.append(y_q)

    y_hat_q = np.array(y_hat_q)
    err_q = np.abs(y_seg - y_hat_q)

    print(f"Segment {i+1}")
    print(f"  Range        : [{x1:.6f}, {x2:.6f}]")

    print(f"  m float      : {m:.8f}")
    print(f"  m quant      : {m_q:.8f}")
    print(f"  c float      : {c:.8f}")
    print(f"  c quant      : {c_q:.8f}")

    print(f"  Avg Err (F)  : {np.mean(err_f):.3e}")
    print(f"  Max Err (F)  : {np.max(err_f):.3e}")

    print(f"  Avg Err (Q)  : {np.mean(err_q):.3e}")
    print(f"  Max Err (Q)  : {np.max(err_q):.3e}")
    print()

    segments_float.append((x1, x2, m, c))
    segments_quant.append((x1, x2, m_q, c_q))

# ==========================================
# Approx functions
# ==========================================
def approx_float(x):
    sign = x < 0
    xp = abs(x)

    for (x1, x2, m, c) in segments_float:
        if xp >= x1 and xp <= x2:
            y = m * xp + c
            break

    return 1-y if sign else y


def approx_quant(x):
    sign = x < 0
    xp = quantize(abs(x))

    for (x1, x2, m, c) in segments_quant:
        if xp >= x1 and xp <= x2:
            mult = quantize(m * xp)
            y = quantize(mult + c)
            break

    y = quantize(1 - y) if sign else y
    return y

# ==========================================
# Global evaluation
# ==========================================
x_full = np.linspace(-8, 8, 10000)
y_true = sigmoid(x_full)

y_float = np.array([approx_float(x) for x in x_full])
y_quant = np.array([approx_quant(x) for x in x_full])

err_float = np.abs(y_true - y_float)
err_quant = np.abs(y_true - y_quant)

print("\n=====================================")
print("GLOBAL ERROR")
print("=====================================")

print("\nFLOAT MODEL")
print("Avg Error :", np.mean(err_float))
print("Max Error :", np.max(err_float))

print("\nQUANTIZED MODEL")
print("Avg Error :", np.mean(err_quant))
print("Max Error :", np.max(err_quant))