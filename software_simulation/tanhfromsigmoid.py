import numpy as np

# ==========================================
# True functions
# ==========================================
def true_sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def true_tanh(x):
    return np.tanh(x)

def sigmoid_second_derivative(x):
    s = true_sigmoid(x)
    return s * (1 - s) * (1 - 2*s)


# ==========================================
# Parameters
# ==========================================
X_MAX = 16.0      # because we need σ(2x), x∈[0,8]
NUM_SEG = 16


# ==========================================
# Step 1: Dense sampling for sigmoid [0,16]
# ==========================================
x_dense = np.linspace(0, X_MAX, 60000)

# √|f''(x)| spacing
weight = np.sqrt(np.abs(sigmoid_second_derivative(x_dense)))
weight[weight < 1e-14] = 1e-14

cumulative = np.cumsum(weight)
cumulative = cumulative / cumulative[-1]

# ==========================================
# Step 2: Boundaries
# ==========================================
boundaries = [0.0]

for k in range(1, NUM_SEG):
    idx = np.searchsorted(cumulative, k / NUM_SEG)
    boundaries.append(x_dense[idx])

boundaries.append(X_MAX)

# ==========================================
# Step 3: Linear regression per segment
# ==========================================
segments = []

print("\n=====================================")
print("SIGMOID SEGMENTS (0 → 16)")
print("=====================================\n")

for i in range(NUM_SEG):

    x1 = boundaries[i]
    x2 = boundaries[i+1]

    x_seg = np.linspace(x1, x2, 2048)
    y_seg = true_sigmoid(x_seg)

    X_bar = np.mean(x_seg)
    Y_bar = np.mean(y_seg)

    m = np.sum((x_seg - X_bar)*(y_seg - Y_bar)) / np.sum((x_seg - X_bar)**2)
    c = Y_bar - m * X_bar

    y_hat = m * x_seg + c
    abs_error = np.abs(y_seg - y_hat)

    print(f"Segment {i+1}")
    print(f"  Range      : [{x1:.6f}, {x2:.6f}]")
    print(f"  Slope (m)  : {m:.10f}")
    print(f"  Intercept  : {c:.10f}")
    print(f"  Max Error  : {np.max(abs_error):.10e}")
    print()

    segments.append((x1, x2, m, c))


# ==========================================
# Approximate sigmoid
# ==========================================
def approx_sigmoid(x):

    xp = abs(x)

    for (x1, x2, m, c) in segments:
        if xp >= x1 and xp <= x2:
            return m * xp + c

    return 1.0


# ==========================================
# Approximate tanh using sigmoid identity
# ==========================================
def approx_tanh(x):

    sign = x < 0
    xp = abs(x)

    sig_val = approx_sigmoid(2*xp)
    tanh_val = 2*sig_val - 1

    return -tanh_val if sign else tanh_val


# ==========================================
# Global error over (-8, 8)
# ==========================================
x_full = np.linspace(-8, 8, 200000)

y_true = true_tanh(x_full)
y_hat = np.array([approx_tanh(x) for x in x_full])

abs_error = np.abs(y_true - y_hat)

print("\n=====================================")
print("GLOBAL TANH ERROR (via sigmoid)")
print("=====================================")
print("Global Avg Error :", np.mean(abs_error))
print("Global Max Error :", np.max(abs_error))