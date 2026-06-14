import sys
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# We can import functions from run_activation_simulations
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_activation_simulations import build_segments, approximate, activation_function

def plot_overlay(activation, arch_name, segments, x_max=8.0, samples=1000, out_dir=Path("images")):
    out_dir.mkdir(parents=True, exist_ok=True)
    
    x_values = np.linspace(-x_max, x_max, samples)
    fn = activation_function(activation)
    y_true_values = np.array(fn(x_values), dtype=float)
    
    # Get segment table
    seg_count = len(segments)
    
    y_approx_values = np.array(
        [approximate(activation, "float", segments, float(x)) for x in x_values],
        dtype=float,
    )
    errors = np.abs(y_true_values - y_approx_values)
    max_index = int(np.argmax(errors))
    max_error = float(errors[max_index])
    error_scale = 0.8 / max_error if max_error > 0.0 else 1.0
    scaled_error = errors * error_scale
    
    fig, axis = plt.subplots(figsize=(6, 4.5))
    
    axis.plot(x_values, y_true_values, color="black", linewidth=1.8, label=f"True {activation.title()}")
    axis.plot(x_values, y_approx_values, color="#1455ff", linewidth=1.2, label=f"{arch_name} Approx")
    axis.plot(x_values, scaled_error, color="red", linewidth=1.0, alpha=0.85, label=f"Abs Error x {error_scale:.0f}")
    
    axis.scatter([x_values[max_index]], [scaled_error[max_index]], color="darkred", s=30, zorder=5)
    axis.annotate(
        f"max={max_error:.2e}\nx={x_values[max_index]:.2f}",
        xy=(x_values[max_index], scaled_error[max_index]),
        xytext=(5, 7),
        textcoords="offset points",
        fontsize=9,
        color="darkred",
    )
    
    axis.set_title(f"{activation.title()} - {arch_name} ({seg_count} segments)", fontsize=12, fontweight="bold")
    axis.set_ylim(-1.1, 1.1)
    axis.set_xlabel("Input x")
    axis.set_ylabel("Output")
    axis.grid(True, color="#80aaff", linewidth=0.55, alpha=0.8)
    axis.legend(loc="lower right", fontsize=9, frameon=True)
    
    fig.tight_layout()
    # E.g. images/sigmoid_proposed1_overlay.png
    out_filename = out_dir / f"{activation}_{arch_name.replace(' ', '').lower()}_overlay.png"
    fig.savefig(out_filename, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Generated {out_filename}")

def main():
    # Proposed 1: 16 segments
    plot_overlay("sigmoid", "Proposed 1", build_segments("sigmoid", "proposed", "float", 16, 8.0, 2048))
    plot_overlay("tanh", "Proposed 1", build_segments("tanh", "proposed", "float", 16, 8.0, 2048))
    
    # Proposed 2: 14 for sigmoid, 9 for tanh
    plot_overlay("sigmoid", "Proposed 2", build_segments("sigmoid", "proposed", "float", 14, 8.0, 2048))
    plot_overlay("tanh", "Proposed 2", build_segments("tanh", "proposed", "float", 9, 8.0, 2048))

if __name__ == "__main__":
    main()
