#!/usr/bin/env python3
"""Run Sigmoid/Tanh approximation simulations and export CSV reports.

This script is intentionally standalone and lives outside the Sigmoid/Tanh
implementation folders. It reproduces the baseline and proposed segmentation
flows in a reusable, argument-driven form and writes structured CSV files.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np


X_MAX = 8.0
DEFAULT_SEGMENTS = 16
DEFAULT_SAMPLES = 1000
FORMATS = ("float", "fp32", "int8", "uint8", "fp8")
ACTIVATIONS = ("sigmoid", "tanh")
METHODS = ("baseline", "proposed")
FORMAT_LABELS = {
    "float": "Float",
    "fp32": "FP32",
    "int8": "INT8",
    "uint8": "UINT8",
    "fp8": "FP8",
}
FORMAT_COLORS = {
    "float": "#1f77b4",
    "fp32": "#17becf",
    "int8": "#d62728",
    "uint8": "#2ca02c",
    "fp8": "#9467bd",
}


@dataclass(frozen=True)
class Segment:
    index: int
    x_start: float
    x_end: float
    slope: float
    intercept: float


@dataclass(frozen=True)
class SimulationRow:
    activation: str
    method: str
    data_format: str
    sample_index: int
    x: float
    y_true: float
    y_approx: float
    abs_error: float


def sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-x))


def tanh_func(x: np.ndarray | float) -> np.ndarray | float:
    return np.tanh(x)


def sigmoid_second_derivative(x: np.ndarray) -> np.ndarray:
    s = sigmoid(x)
    return s * (1.0 - s) * (1.0 - 2.0 * s)


def tanh_second_derivative(x: np.ndarray) -> np.ndarray:
    t = tanh_func(x)
    return -2.0 * t * (1.0 - t**2)


def activation_function(name: str) -> Callable[[np.ndarray | float], np.ndarray | float]:
    if name == "sigmoid":
        return sigmoid
    if name == "tanh":
        return tanh_func
    raise ValueError(f"Unsupported activation: {name}")


def second_derivative_function(name: str) -> Callable[[np.ndarray], np.ndarray]:
    if name == "sigmoid":
        return sigmoid_second_derivative
    if name == "tanh":
        return tanh_second_derivative
    raise ValueError(f"Unsupported activation: {name}")


def build_boundaries(activation: str, method: str, segments: int, x_max: float) -> np.ndarray:
    if method == "baseline":
        return np.linspace(0.0, x_max, segments + 1)

    x_dense = np.linspace(0.0, x_max, 50_000)
    curvature = np.sqrt(np.abs(second_derivative_function(activation)(x_dense)))
    curvature[curvature < 1e-14] = 1e-14

    cumulative = np.cumsum(curvature)
    cumulative /= cumulative[-1]

    boundaries = [0.0]
    for index in range(1, segments):
        dense_index = np.searchsorted(cumulative, index / segments)
        boundaries.append(float(x_dense[dense_index]))
    boundaries.append(float(x_max))
    return np.array(boundaries, dtype=float)


def float_regression(xs: np.ndarray, ys: np.ndarray) -> tuple[float, float]:
    x_mean = float(np.mean(xs))
    y_mean = float(np.mean(ys))
    denominator = float(np.sum((xs - x_mean) ** 2))
    if denominator == 0.0:
        return 0.0, y_mean
    slope = float(np.sum((xs - x_mean) * (ys - y_mean)) / denominator)
    return slope, y_mean - slope * x_mean


def quantize_int8(value: np.ndarray | float) -> np.ndarray | float:
    arr = np.array(value, dtype=float)
    quantized = np.round(arr * 16.0) / 16.0
    clipped = np.clip(quantized, -X_MAX, X_MAX)
    return float(clipped) if np.isscalar(value) else clipped


def quantize_fp8(value: np.ndarray | float) -> np.ndarray | float:
    arr = np.array(value, dtype=float)
    quantized = np.round(arr * 16.0) / 16.0
    clipped = np.clip(quantized, -X_MAX, X_MAX)
    return float(clipped) if np.isscalar(value) else clipped


def quantize_uint8_input(value: np.ndarray | float) -> np.ndarray | float:
    arr = np.array(value, dtype=float)
    offset = (arr + X_MAX) / (2.0 * X_MAX) * 255.0
    clipped = np.clip(offset, 0.0, 255.0)
    quantized = np.round(clipped) / 255.0 * (2.0 * X_MAX) - X_MAX
    return float(quantized) if np.isscalar(value) else quantized


def quantize_output(activation: str, data_format: str, value: float) -> float:
    if data_format == "float":
        return float(value)
    if data_format == "fp32":
        return float(np.float32(value))
    if data_format in {"int8", "fp8"}:
        return float(np.clip(np.round(value * 128.0) / 128.0, -1.0, 1.0))
    if data_format == "uint8":
        if activation == "sigmoid":
            clipped = np.clip(value, 0.0, 1.0)
            return float(np.round(clipped * 255.0) / 255.0)
        clipped = np.clip(value, -1.0, 1.0)
        return float(np.round((clipped + 1.0) * 127.5) / 127.5 - 1.0)
    raise ValueError(f"Unsupported data format: {data_format}")


def quantize_coefficients(data_format: str, slope: float, intercept: float) -> tuple[float, float]:
    if data_format == "float":
        return slope, intercept
    if data_format == "fp32":
        return float(np.float32(slope)), float(np.float32(intercept))
    if data_format == "int8":
        return float(quantize_int8(slope)), float(quantize_int8(intercept))
    if data_format == "uint8":
        return slope, intercept
    if data_format == "fp8":
        return float(quantize_fp8(slope)), float(quantize_fp8(intercept))
    raise ValueError(f"Unsupported data format: {data_format}")


def build_segments(
    activation: str,
    method: str,
    data_format: str,
    segments: int,
    x_max: float,
    points_per_segment: int,
) -> list[Segment]:
    fn = activation_function(activation)
    boundaries = build_boundaries(activation, method, segments, x_max)
    segment_table: list[Segment] = []

    for index in range(segments):
        x_start = float(boundaries[index])
        x_end = float(boundaries[index + 1])
        xs = np.linspace(x_start, x_end, points_per_segment)
        ys = fn(xs)
        slope, intercept = float_regression(xs, ys)
        slope, intercept = quantize_coefficients(data_format, slope, intercept)
        segment_table.append(Segment(index + 1, x_start, x_end, slope, intercept))

    return segment_table


def find_segment(segments: list[Segment], x_abs: float) -> Segment:
    for segment in segments:
        if segment.x_start <= x_abs <= segment.x_end:
            return segment
    return segments[-1]


def quantize_input(data_format: str, x_abs: float) -> float:
    if data_format in {"float", "fp32"}:
        return x_abs
    if data_format == "int8":
        return float(quantize_int8(x_abs))
    if data_format == "uint8":
        return float(quantize_uint8_input(x_abs))
    if data_format == "fp8":
        return float(quantize_fp8(x_abs))
    raise ValueError(f"Unsupported data format: {data_format}")


def approximate(activation: str, data_format: str, segments: list[Segment], x: float) -> float:
    x_abs = abs(float(x))
    x_quantized = quantize_input(data_format, x_abs)
    segment = find_segment(segments, x_quantized)
    y_positive = segment.slope * x_quantized + segment.intercept

    if activation == "sigmoid":
        y = 1.0 - y_positive if x < 0.0 else y_positive
    else:
        y = -y_positive if x < 0.0 else y_positive

    return quantize_output(activation, data_format, y)


def run_simulation(
    activations: Iterable[str],
    methods: Iterable[str],
    formats: Iterable[str],
    samples: int,
    segments: int,
    x_max: float,
    points_per_segment: int,
) -> tuple[list[SimulationRow], list[dict[str, float | int | str]], list[dict[str, float | int | str]]]:
    detailed_rows: list[SimulationRow] = []
    summary_rows: list[dict[str, float | int | str]] = []
    segment_rows: list[dict[str, float | int | str]] = []
    x_values = np.linspace(-x_max, x_max, samples)

    for activation in activations:
        fn = activation_function(activation)
        y_true_values = np.array(fn(x_values), dtype=float)

        for method in methods:
            for data_format in formats:
                segment_table = build_segments(
                    activation,
                    method,
                    data_format,
                    segments,
                    x_max,
                    points_per_segment,
                )

                for segment in segment_table:
                    segment_rows.append(
                        {
                            "activation": activation,
                            "method": method,
                            "data_format": data_format,
                            "segment": segment.index,
                            "x_start": segment.x_start,
                            "x_end": segment.x_end,
                            "slope": segment.slope,
                            "intercept": segment.intercept,
                        }
                    )

                y_approx_values = np.array(
                    [approximate(activation, data_format, segment_table, float(x)) for x in x_values],
                    dtype=float,
                )
                errors = np.abs(y_true_values - y_approx_values)

                for sample_index, (x, y_true, y_approx, error) in enumerate(
                    zip(x_values, y_true_values, y_approx_values, errors),
                    start=1,
                ):
                    detailed_rows.append(
                        SimulationRow(
                            activation=activation,
                            method=method,
                            data_format=data_format,
                            sample_index=sample_index,
                            x=float(x),
                            y_true=float(y_true),
                            y_approx=float(y_approx),
                            abs_error=float(error),
                        )
                    )

                summary_rows.append(
                    {
                        "activation": activation,
                        "method": method,
                        "data_format": data_format,
                        "samples": samples,
                        "segments": segments,
                        "avg_error": float(np.mean(errors)),
                        "max_error": float(np.max(errors)),
                        "rmse": float(math.sqrt(np.mean(errors**2))),
                    }
                )

    return detailed_rows, summary_rows, segment_rows


def write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_outputs(
    output_dir: Path,
    detailed_rows: list[SimulationRow],
    summary_rows: list[dict[str, float | int | str]],
    segment_rows: list[dict[str, float | int | str]],
) -> tuple[Path, Path, Path]:
    detail_path = output_dir / "activation_simulation_results.csv"
    summary_path = output_dir / "activation_simulation_summary.csv"
    segments_path = output_dir / "activation_segment_coefficients.csv"

    write_csv(
        detail_path,
        ["activation", "method", "data_format", "sample_index", "x", "y_true", "y_approx", "abs_error"],
        [row.__dict__ for row in detailed_rows],
    )
    write_csv(
        summary_path,
        ["activation", "method", "data_format", "samples", "segments", "avg_error", "max_error", "rmse"],
        summary_rows,
    )
    write_csv(
        segments_path,
        ["activation", "method", "data_format", "segment", "x_start", "x_end", "slope", "intercept"],
        segment_rows,
    )
    return detail_path, summary_path, segments_path


def rows_to_series(detailed_rows: list[SimulationRow]) -> dict[tuple[str, str, str], dict[str, np.ndarray]]:
    grouped: dict[tuple[str, str, str], dict[str, list[float]]] = {}
    for row in detailed_rows:
        key = (row.activation, row.method, row.data_format)
        if key not in grouped:
            grouped[key] = {"x": [], "y_true": [], "y_approx": [], "abs_error": []}
        grouped[key]["x"].append(row.x)
        grouped[key]["y_true"].append(row.y_true)
        grouped[key]["y_approx"].append(row.y_approx)
        grouped[key]["abs_error"].append(row.abs_error)

    return {
        key: {
            "x": np.array(values["x"], dtype=float),
            "y_true": np.array(values["y_true"], dtype=float),
            "y_approx": np.array(values["y_approx"], dtype=float),
            "abs_error": np.array(values["abs_error"], dtype=float),
        }
        for key, values in grouped.items()
    }


def plot_error_curves(
    detailed_rows: list[SimulationRow],
    activations: Iterable[str],
    methods: Iterable[str],
    formats: Iterable[str],
    images_dir: Path,
) -> Path:
    images_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = images_dir / ".matplotlib"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))

    import matplotlib.pyplot as plt

    activation_list = list(activations)
    method_list = list(methods)
    format_list = list(formats)
    series = rows_to_series(detailed_rows)

    fig, axes = plt.subplots(
        len(activation_list),
        len(method_list),
        figsize=(6.4 * len(method_list), 4.4 * len(activation_list)),
        squeeze=False,
        sharex=True,
    )
    fig.suptitle("Approximation Error Curves by Activation, Method, and Data Format", fontsize=16, fontweight="bold")

    for row_index, activation in enumerate(activation_list):
        for col_index, method in enumerate(method_list):
            axis = axes[row_index][col_index]
            for data_format in format_list:
                key = (activation, method, data_format)
                if key not in series:
                    continue
                axis.plot(
                    series[key]["x"],
                    np.maximum(series[key]["abs_error"], 1e-12),
                    label=FORMAT_LABELS.get(data_format, data_format.upper()),
                    linewidth=1.7,
                    color=FORMAT_COLORS.get(data_format),
                )

            axis.set_title(f"{activation.title()} - {method.title()}", fontsize=12, fontweight="bold")
            axis.set_yscale("log")
            axis.grid(True, which="both", linestyle="--", linewidth=0.45, alpha=0.5)
            axis.set_xlabel("Input x")
            axis.set_ylabel("Absolute error (log scale)")
            axis.set_ylim(bottom=1e-12)
            axis.legend(loc="upper right", fontsize=8, frameon=True)

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    output_path = images_dir / "activation_error_curves.png"
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_error_summary(
    summary_rows: list[dict[str, float | int | str]],
    activations: Iterable[str],
    methods: Iterable[str],
    formats: Iterable[str],
    images_dir: Path,
) -> Path:
    images_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = images_dir / ".matplotlib"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))

    import matplotlib.pyplot as plt

    activation_list = list(activations)
    method_list = list(methods)
    format_list = list(formats)
    summary = {
        (str(row["activation"]), str(row["method"]), str(row["data_format"])): row
        for row in summary_rows
    }

    categories = [(activation, method) for activation in activation_list for method in method_list]
    x_positions = np.arange(len(categories))
    bar_width = 0.15 if len(format_list) >= 5 else 0.18

    fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True)
    fig.suptitle("Approximation Error Summary", fontsize=16, fontweight="bold")

    for axis, metric, title in (
        (axes[0], "avg_error", "Average Absolute Error"),
        (axes[1], "max_error", "Maximum Absolute Error"),
    ):
        for index, data_format in enumerate(format_list):
            offset = (index - (len(format_list) - 1) / 2.0) * bar_width
            values = []
            for activation, method in categories:
                row = summary.get((activation, method, data_format))
                values.append(float(row[metric]) if row else np.nan)

            axis.bar(
                x_positions + offset,
                values,
                width=bar_width,
                label=FORMAT_LABELS.get(data_format, data_format.upper()),
                color=FORMAT_COLORS.get(data_format),
                alpha=0.9,
            )

        axis.set_title(title, fontsize=12, fontweight="bold")
        axis.set_yscale("log")
        axis.set_ylabel("Error (log scale)")
        axis.grid(True, axis="y", which="both", linestyle="--", linewidth=0.45, alpha=0.5)
        axis.legend(ncol=len(format_list), fontsize=8, loc="upper right")

    axes[-1].set_xticks(x_positions)
    axes[-1].set_xticklabels(
        [f"{activation.title()}\n{method.title()}" for activation, method in categories],
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    output_path = images_dir / "activation_error_summary.png"
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_activation_error_overlays(
    detailed_rows: list[SimulationRow],
    activations: Iterable[str],
    methods: Iterable[str],
    formats: Iterable[str],
    images_dir: Path,
) -> list[Path]:
    """Plot true/approximated activations with scaled absolute error."""
    images_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = images_dir / ".matplotlib"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))

    import matplotlib.pyplot as plt

    series = rows_to_series(detailed_rows)
    method_list = list(methods)
    format_list = list(formats)
    output_paths: list[Path] = []

    for activation in activations:
        fig, axes = plt.subplots(
            len(method_list),
            len(format_list),
            figsize=(4.4 * len(format_list), 3.6 * len(method_list)),
            squeeze=False,
            sharex=True,
        )
        fig.suptitle(
            f"{activation.title()} Approximation and Scaled Absolute Error",
            fontsize=16,
            fontweight="bold",
        )

        for row_index, method in enumerate(method_list):
            for col_index, data_format in enumerate(format_list):
                axis = axes[row_index][col_index]
                key = (activation, method, data_format)
                if key not in series:
                    axis.set_visible(False)
                    continue

                values = series[key]
                x_values = values["x"]
                errors = values["abs_error"]
                max_index = int(np.argmax(errors))
                max_error = float(errors[max_index])
                error_scale = 0.8 / max_error if max_error > 0.0 else 1.0
                scaled_error = errors * error_scale

                axis.plot(x_values, values["y_true"], color="black", linewidth=1.6, label=activation.title())
                axis.plot(
                    x_values,
                    values["y_approx"],
                    color="#1455ff",
                    linewidth=1.0,
                    label=f"Approx. {activation.title()}",
                )
                axis.plot(
                    x_values,
                    scaled_error,
                    color="red",
                    linewidth=0.8,
                    alpha=0.85,
                    label=f"Absolute error x {error_scale:.0f}",
                )
                axis.scatter(
                    [x_values[max_index]],
                    [scaled_error[max_index]],
                    color="darkred",
                    s=20,
                    zorder=5,
                )
                axis.annotate(
                    f"max={max_error:.2e}\nx={x_values[max_index]:.2f}",
                    xy=(x_values[max_index], scaled_error[max_index]),
                    xytext=(5, 7),
                    textcoords="offset points",
                    fontsize=7,
                    color="darkred",
                )

                axis.set_title(
                    f"{method.title()} - {FORMAT_LABELS.get(data_format, data_format.upper())}",
                    fontsize=10,
                    fontweight="bold",
                )
                axis.set_ylim(-1.1, 1.1)
                axis.set_xlabel("Input x")
                if col_index == 0:
                    axis.set_ylabel(activation.title())
                axis.grid(True, color="#80aaff", linewidth=0.55, alpha=0.8)
                axis.legend(loc="lower right", fontsize=6, frameon=True)

        fig.tight_layout(rect=(0, 0, 1, 0.95))
        output_path = images_dir / f"{activation}_approximation_error_all_formats.png"
        fig.savefig(output_path, dpi=220, bbox_inches="tight")
        plt.close(fig)
        output_paths.append(output_path)

    return output_paths


def write_images(
    images_dir: Path,
    detailed_rows: list[SimulationRow],
    summary_rows: list[dict[str, float | int | str]],
    activations: Iterable[str],
    methods: Iterable[str],
    formats: Iterable[str],
) -> tuple[Path, Path, list[Path]]:
    curve_path = plot_error_curves(detailed_rows, activations, methods, formats, images_dir)
    summary_path = plot_error_summary(summary_rows, activations, methods, formats, images_dir)
    overlay_paths = plot_activation_error_overlays(
        detailed_rows,
        activations,
        methods,
        formats,
        images_dir,
    )
    return curve_path, summary_path, overlay_paths


def parse_multi(value: str, allowed: tuple[str, ...], label: str) -> list[str]:
    if value.lower() == "all":
        return list(allowed)
    selected = [item.strip().lower() for item in value.split(",") if item.strip()]
    invalid = sorted(set(selected) - set(allowed))
    if invalid:
        raise argparse.ArgumentTypeError(f"Invalid {label}: {', '.join(invalid)}")
    return selected


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run production-style Sigmoid/Tanh simulations and export CSV files.",
    )
    parser.add_argument(
        "--activations",
        default="all",
        help="Comma-separated activations to run: sigmoid,tanh or all.",
    )
    parser.add_argument(
        "--methods",
        default="all",
        help="Comma-separated methods to run: baseline,proposed or all.",
    )
    parser.add_argument(
        "--formats",
        default="all",
        help="Comma-separated formats to run: float,fp32,int8,uint8,fp8 or all.",
    )
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES, help="Number of x samples.")
    parser.add_argument("--segments", type=int, default=DEFAULT_SEGMENTS, help="Number of segments.")
    parser.add_argument("--x-max", type=float, default=X_MAX, help="Symmetric input range is [-x-max, x-max].")
    parser.add_argument(
        "--points-per-segment",
        type=int,
        default=2048,
        help="Regression points used to fit each segment.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "simulation_results",
        help="Directory where CSV files are written.",
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "images",
        help="Directory where PNG graph images are written.",
    )
    parser.add_argument(
        "--no-images",
        action="store_true",
        help="Skip graph generation and only write CSV files.",
    )
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.samples < 2:
        raise ValueError("--samples must be at least 2")
    if args.segments < 1:
        raise ValueError("--segments must be at least 1")
    if args.x_max <= 0.0:
        raise ValueError("--x-max must be positive")
    if args.points_per_segment < 2:
        raise ValueError("--points-per-segment must be at least 2")


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    validate_args(args)

    activations = parse_multi(args.activations, ACTIVATIONS, "activation")
    methods = parse_multi(args.methods, METHODS, "method")
    formats = parse_multi(args.formats, FORMATS, "format")

    detailed_rows, summary_rows, segment_rows = run_simulation(
        activations=activations,
        methods=methods,
        formats=formats,
        samples=args.samples,
        segments=args.segments,
        x_max=args.x_max,
        points_per_segment=args.points_per_segment,
    )

    detail_path, summary_path, segments_path = write_outputs(
        output_dir=args.output_dir,
        detailed_rows=detailed_rows,
        summary_rows=summary_rows,
        segment_rows=segment_rows,
    )

    print("Activation simulation complete.")
    print(f"Detailed results : {detail_path}")
    print(f"Summary results  : {summary_path}")
    print(f"Segment table    : {segments_path}")
    if not args.no_images:
        curve_path, image_summary_path, overlay_paths = write_images(
            images_dir=args.images_dir,
            detailed_rows=detailed_rows,
            summary_rows=summary_rows,
            activations=activations,
            methods=methods,
            formats=formats,
        )
        print(f"Error curves     : {curve_path}")
        print(f"Error summary    : {image_summary_path}")
        for overlay_path in overlay_paths:
            print(f"Function overlay : {overlay_path}")
    print("")
    print(f"{'Activation':10s} {'Method':10s} {'Format':8s} {'Avg Error':>12s} {'Max Error':>12s} {'RMSE':>12s}")
    print("-" * 70)
    for row in summary_rows:
        print(
            f"{str(row['activation']):10s} "
            f"{str(row['method']):10s} "
            f"{str(row['data_format']):8s} "
            f"{float(row['avg_error']):12.4e} "
            f"{float(row['max_error']):12.4e} "
            f"{float(row['rmse']):12.4e}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
