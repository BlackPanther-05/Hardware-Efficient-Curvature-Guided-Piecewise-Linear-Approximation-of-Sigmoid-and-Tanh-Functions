import numpy as np

# ==========================================
# True sigmoid and second derivative
# ==========================================
def true_sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def second_derivative(x):
    s = true_sigmoid(x)
    return s * (1 - s) * (1 - 2*s)


# ==========================================
# Parameters
# ==========================================
X_MAX = 8.0
NUM_SEG = 16


# ==========================================
# Step 1: Dense sampling over 0→8
# ==========================================
x_dense = np.linspace(0, X_MAX, 50000)

# √|f''(x)| optimal density
weight = np.sqrt(np.abs(second_derivative(x_dense)))

# Avoid numerical issues
weight[weight < 1e-14] = 1e-14

# ==========================================
# Step 2: Cumulative integral
# ==========================================
cumulative = np.cumsum(weight)
cumulative = cumulative / cumulative[-1]

# ==========================================
# Step 3: Equal-weight segmentation
# ==========================================
boundaries = [0.0]

for k in range(1, NUM_SEG):
    idx = np.searchsorted(cumulative, k / NUM_SEG)
    boundaries.append(x_dense[idx])

boundaries.append(X_MAX)

# ==========================================
# Step 4: Linear regression per segment
# ==========================================
segments = []

print("\n=====================================")
print("√|f''| SEGMENTS (Sigmoid 0 → 8, 16 total)")
print("=====================================\n")

for i in range(NUM_SEG):

    x1 = boundaries[i]
    x2 = boundaries[i+1]

    x_seg = np.linspace(x1, x2, 2048)
    y_seg = true_sigmoid(x_seg)

    # Least squares linear fit
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
    print(f"  Slope (m)  : {m:.10f}")
    print(f"  Intercept  : {c:.10f}")
    print(f"  Avg Error  : {seg_avg:.10e}")
    print(f"  Max Error  : {seg_max:.10e}")
    print()

    segments.append((x1, x2, m, c))


# ==========================================
# Approximation function (full domain)
# ==========================================
def approx_sigmoid(b):

    sign = b < 0
    xp = abs(b)

    for (x1, x2, m, c) in segments:
        if xp >= x1 and xp <= x2:
            y_pos = m * xp + c
            break

    # Sigmoid symmetry
    if sign:
        return 1 - y_pos
    else:
        return y_pos


# ==========================================
# Global error over (-8, 8)
# ==========================================
x_full = np.linspace(-8, 8, 200000)

y_true = true_sigmoid(x_full)
y_hat = np.array([approx_sigmoid(x) for x in x_full])

abs_error = np.abs(y_true - y_hat)

print("\n=====================================")
print("GLOBAL ERROR (-8 to 8)")
print("=====================================")
print("Global Avg Error :", np.mean(abs_error))
print("Global Max Error :", np.max(abs_error))