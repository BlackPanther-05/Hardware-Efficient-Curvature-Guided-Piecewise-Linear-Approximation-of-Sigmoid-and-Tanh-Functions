import numpy as np

# ==========================================
# True tanh and second derivative
# ==========================================
def true_tanh(x):
    return np.tanh(x)

def second_derivative(x):
    t = np.tanh(x)
    return np.abs(-2 * t * (1 - t**2))


# ==========================================
# Parameters
# ==========================================
X_MAX = 3.251
NUM_SEG = 16
CONST_VAL = 0.999


# ==========================================
# Step 1: Curvature-based segmentation
# ==========================================
x_dense = np.linspace(0, X_MAX, 20000)
curv = second_derivative(x_dense)

cumulative = np.cumsum(curv)
cumulative = cumulative / cumulative[-1]

boundaries = [0.0]

for k in range(1, NUM_SEG):
    idx = np.searchsorted(cumulative, k / NUM_SEG)
    boundaries.append(x_dense[idx])

boundaries.append(X_MAX)


# ==========================================
# Step 2: Linear regression per segment
# ==========================================
segments = []

print("\n=====================================")
print("SEGMENT DETAILS (Second Derivative)")
print("=====================================\n")

for i in range(NUM_SEG):

    x1 = boundaries[i]
    x2 = boundaries[i+1]

    x_seg = np.linspace(x1, x2, 2048)
    y_seg = true_tanh(x_seg)

    X_bar = np.mean(x_seg)
    Y_bar = np.mean(y_seg)

    m = np.sum((x_seg - X_bar)*(y_seg - Y_bar)) / np.sum((x_seg - X_bar)**2)
    c = Y_bar - m * X_bar

    y_hat = m * x_seg + c
    abs_error = np.abs(y_seg - y_hat)

    seg_avg = np.mean(abs_error)
    seg_max = np.max(abs_error)

    print(f"Segment {i+1}")
    print(f"  Range      : [{x1:.6f}, {x2:.6f}]")
    print(f"  Slope (m)  : {m:.8f}")
    print(f"  Intercept  : {c:.8f}")
    print(f"  Avg Error  : {seg_avg:.10e}")
    print(f"  Max Error  : {seg_max:.10e}")
    print()

    segments.append((x1, x2, m, c))


# ==========================================
# Segment 17: Tail region
# ==========================================
x_tail = np.linspace(X_MAX, 8, 20000)
y_true_tail = true_tanh(x_tail)
y_hat_tail = np.full_like(x_tail, CONST_VAL)

abs_error_tail = np.abs(y_true_tail - y_hat_tail)

print("Segment 17 (Tail)")
print(f"  Range      : [{X_MAX:.6f}, 8.000000]")
print(f"  Approx Val : {CONST_VAL}")
print(f"  Avg Error  : {np.mean(abs_error_tail):.10e}")
print(f"  Max Error  : {np.max(abs_error_tail):.10e}")
print()


# ==========================================
# Approximation Function (Full Range)
# ==========================================
def approx_tanh(b):

    sign = b < 0
    xp = abs(b)

    if xp > X_MAX:
        y_pos = CONST_VAL
    else:
        for (x1, x2, m, c) in segments:
            if xp >= x1 and xp <= x2:
                y_pos = m * xp + c
                break

    return -y_pos if sign else y_pos


# ==========================================
# Global error over (-8, 8)
# ==========================================
x_full = np.linspace(-8, 8, 200000)

y_true = true_tanh(x_full)
y_hat = np.array([approx_tanh(x) for x in x_full])

abs_error = np.abs(y_true - y_hat)

print("\n=====================================")
print("GLOBAL ERROR (-8 to 8)")
print("=====================================")
print("Global Avg Error :", np.mean(abs_error))
print("Global Max Error :", np.max(abs_error))