import os
from pathlib import Path

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

def quantize(val):
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
# NAIVE segments
# ==========================================
SEGMENTS_NAIVE = [
    (x1, x2, quantize_mc(m), quantize_mc(c))
    for (x1, x2, m, c) in SEGMENTS_ORIGINAL
]


# ==========================================
# AWARE segments
# ==========================================
SEARCH_RANGE = 8

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
# Hardware-representable x values ONLY
#
#  OLD (wrong): x_full = np.linspace(-8, 8, 20000)
#    - 20000 raw float points, most NOT on the 1/128 grid
#    - hardware internally snaps them to 1/128 via quantize_x
#    - but error was measured vs the raw unsnapped x
#    - → inflated avg error (4.08e-4 instead of 3.57e-4)
#
#  NEW (correct): x_hw = b_int / 128  for b_int = -1023..+1023
#    - 2047 points, ALL exact multiples of 1/128
#    - these are the ONLY x values the hardware can represent
#    - b_int=±1024 (x=±8.0) excluded: overflows S1.I3.F7 format
#    - error measured vs true_sigmoid(x_hw) → no x quantization bias
#    - matches RTL testbench measurement exactly
# ==========================================
x_hw   = np.array([b_int / 128.0 for b_int in range(-1023, 1024)])
y_true = true_sigmoid(x_hw)

y_orig = np.array([approx_sigmoid_original(x) for x in x_hw])
y_naiv = np.array([approx_sigmoid_naive(x)    for x in x_hw])
y_awar = np.array([approx_sigmoid_aware(x)    for x in x_hw])

err_orig = np.abs(y_true - y_orig)
err_naiv = np.abs(y_true - y_naiv)
err_awar = np.abs(y_true - y_awar)

print("\n" + "="*60)
print("GLOBAL ERROR  (hardware-representable x only)")
print("x = b_int/128,  b_int = -1023 to +1023  (2047 points)")
print("="*60)
print(f"\n{'Model':>10}  {'Avg Error':>14}  {'Max Error':>14}")
print("-"*42)
print(f"{'Original':>10}  {np.mean(err_orig):14.4e}  {np.max(err_orig):14.4e}")
print(f"{'Naive Q':>10}  {np.mean(err_naiv):14.4e}  {np.max(err_naiv):14.4e}")
print(f"{'Aware Q':>10}  {np.mean(err_awar):14.4e}  {np.max(err_awar):14.4e}")

print(f"\nNaive Q vs Original : {np.mean(err_naiv)/np.mean(err_orig):.2f}x")
print(f"Aware Q vs Original : {np.mean(err_awar)/np.mean(err_orig):.2f}x")


# ==========================================
# RTL-matching tanh model
# ==========================================
TANH_THRESHOLDS = [11, 24, 35, 47, 60, 74, 88, 104,
                   121, 139, 161, 188, 219, 265, 341]
TANH_M = [1022, 1005, 972, 925, 862, 787, 703, 609,
          513, 419, 327, 238, 158, 90, 36, 7]
TANH_C = [0, 2, 8, 20, 44, 80, 128, 193,
          272, 360, 461, 573, 690, 808, 919, 997]
TANH_TAIL_START = 490       # 3.827 in Q3.7, rounded by the RTL
TANH_TAIL_VALUE = 1023      # Largest positive Q1.10 value below 1


def approx_tanh_rtl(x):
    """Match the fixed-point operations in rtl/tanh."""
    sign = x < 0
    xp_int = int(round(abs(x) * SCALE_X))

    if xp_int >= TANH_TAIL_START:
        y_int = TANH_TAIL_VALUE
    else:
        segment_index = int(np.searchsorted(TANH_THRESHOLDS, xp_int, side="right"))
        product_int = (xp_int * TANH_M[segment_index]) >> 7
        y_int = min(product_int + TANH_C[segment_index], 1023)

    y = y_int / SCALE_Y
    return -y if sign else y


# ==========================================
# Function and scaled-error images
# ==========================================
def save_error_graph(x, y_reference, y_approx, activation_name, output_path,
                     error_scale=300):
    """Plot the true function, approximation, and visible scaled error."""
    cache_dir = output_path.parent / ".matplotlib"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    error = np.abs(y_reference - y_approx)
    scaled_error = error * error_scale
    max_index = int(np.argmax(error))
    high_error_limit = np.percentile(error, 95)

    fig, axis = plt.subplots(figsize=(8.0, 5.2))
    axis.plot(x, y_reference, color="blue", linewidth=1.5,
              label=activation_name.title())
    axis.plot(x, y_approx, color="black", linewidth=1.2,
              label=f"Approx. {activation_name.title()}")
    axis.plot(x, scaled_error, color="red", linewidth=0.8, alpha=0.9,
              label=f"Absolute Error x {error_scale}")
    axis.fill_between(
        x,
        0,
        scaled_error,
        where=error >= high_error_limit,
        color="red",
        alpha=0.22,
        label="Highest 5% error region",
    )
    axis.scatter(
        x[max_index],
        scaled_error[max_index],
        color="darkred",
        s=38,
        zorder=5,
    )
    axis.annotate(
        f"Max error = {error[max_index]:.4e}\nat x = {x[max_index]:.4f}",
        xy=(x[max_index], scaled_error[max_index]),
        xytext=(12, 14),
        textcoords="offset points",
        fontsize=8,
        color="darkred",
        arrowprops={"arrowstyle": "->", "color": "darkred", "lw": 0.8},
    )

    axis.set_title(f"Approximate {activation_name.title()} Error")
    axis.set_xlabel("Input x")
    axis.set_ylabel(f"{activation_name.title()} / Scaled Absolute Error")
    axis.set_xlim(float(x[0]), float(x[-1]))
    axis.grid(True, color="#80aaff", linewidth=0.6, alpha=0.75)
    axis.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    return error, max_index


true_tanh_values = np.tanh(x_hw)
approx_tanh_values = np.array([approx_tanh_rtl(x) for x in x_hw])

images_dir = Path(__file__).resolve().parent / "generated_images"
images_dir.mkdir(parents=True, exist_ok=True)

tanh_image = images_dir / "tanh_approximation_error.png"
sigmoid_image = images_dir / "sigmoid_approximation_error.png"

tanh_error, tanh_max_index = save_error_graph(
    x_hw, true_tanh_values, approx_tanh_values, "tanh", tanh_image
)
sigmoid_error, sigmoid_max_index = save_error_graph(
    x_hw, y_true, y_awar, "sigmoid", sigmoid_image
)

print("\n" + "="*60)
print("TANH ERROR  (RTL-matching fixed-point model)")
print("="*60)
print(f"Average error : {np.mean(tanh_error):.4e}")
print(f"Maximum error : {tanh_error[tanh_max_index]:.4e}")
print(f"Max-error x   : {x_hw[tanh_max_index]:.6f}")

print("\nHighest-error graph locations")
print(f"Tanh    : x = {x_hw[tanh_max_index]:.6f}, "
      f"error = {tanh_error[tanh_max_index]:.4e}")
print(f"Sigmoid : x = {x_hw[sigmoid_max_index]:.6f}, "
      f"error = {sigmoid_error[sigmoid_max_index]:.4e}")
print(f"\nTanh graph    : {tanh_image}")
print(f"Sigmoid graph : {sigmoid_image}")
