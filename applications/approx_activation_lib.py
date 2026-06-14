from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

import torch
import torch.nn as nn


DTYPES = ("fp32", "fp8", "int8", "uint8")
SCHEMES = ("baseline", "proposed", "proposed_2")
ACTIVATIONS = ("sigmoid", "tanh")


def require_device(name: str, allow_cpu: bool = False) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        if allow_cpu:
            return torch.device("cpu")
        raise RuntimeError("CUDA was requested but is not available. Use --allow-cpu for a CPU dry run.")
    return torch.device(name)


def quantize_like(x: torch.Tensor, dtype: str, *, activation: str, role: str) -> torch.Tensor:
    dtype = dtype.lower()
    if dtype == "fp32":
        return x.float()
    if dtype in {"fp8", "int8"}:
        return torch.clamp(torch.round(x * 16.0) / 16.0, -8.0, 8.0)
    if dtype == "uint8":
        if role == "output":
            if activation == "sigmoid":
                return torch.round(torch.clamp(x, 0.0, 1.0) * 255.0) / 255.0
            return torch.round((torch.clamp(x, -1.0, 1.0) + 1.0) * 127.5) / 127.5 - 1.0
        return torch.round(torch.clamp((x + 8.0) * (255.0 / 16.0), 0.0, 255.0)) * (16.0 / 255.0) - 8.0
    raise ValueError(f"Unsupported dtype: {dtype}")


def _curvature_weighted_boundaries(activation: str, x_max: float, segments: int) -> torch.Tensor:
    x = torch.linspace(0.0, x_max, 50_000, dtype=torch.float64)
    if activation == "sigmoid":
        s = torch.sigmoid(x)
        second = s * (1.0 - s) * (1.0 - 2.0 * s)
    elif activation == "tanh":
        t = torch.tanh(x)
        second = -2.0 * t * (1.0 - t.square())
    else:
        raise ValueError(f"Unsupported activation: {activation}")

    weight = torch.sqrt(second.abs()).clamp_min(1e-14)
    cumulative = torch.cumsum(weight, dim=0)
    cumulative = cumulative / cumulative[-1]

    boundaries = [0.0]
    for k in range(1, segments):
        idx = torch.searchsorted(cumulative, torch.tensor(k / segments, dtype=torch.float64)).item()
        boundaries.append(float(x[idx]))
    boundaries.append(float(x_max))
    return torch.tensor(boundaries, dtype=torch.float32)


def _equal_boundaries(x_max: float, segments: int) -> torch.Tensor:
    return torch.linspace(0.0, x_max, segments + 1, dtype=torch.float32)


def _fit_segments(activation: str, scheme: str, dtype: str, x_max: float, segments: int):
    if scheme == "baseline":
        boundaries = _equal_boundaries(x_max, segments)
    elif scheme in ("proposed", "proposed_2"):
        boundaries = _curvature_weighted_boundaries(activation, x_max, segments)
    else:
        raise ValueError(f"Unsupported scheme: {scheme}")

    slopes, offsets = [], []
    for i in range(segments):
        x1, x2 = float(boundaries[i]), float(boundaries[i + 1])
        xs = torch.linspace(x1, x2, 2048, dtype=torch.float64)
        ys = torch.sigmoid(xs) if activation == "sigmoid" else torch.tanh(xs)
        xb, yb = xs.mean(), ys.mean()
        slope = torch.sum((xs - xb) * (ys - yb)) / torch.sum((xs - xb).square()).clamp_min(1e-18)
        offset = yb - slope * xb
        slopes.append(float(slope))
        offsets.append(float(offset))

    slopes_t = quantize_like(torch.tensor(slopes, dtype=torch.float32), dtype, activation=activation, role="coef")
    offsets_t = quantize_like(torch.tensor(offsets, dtype=torch.float32), dtype, activation=activation, role="coef")
    return boundaries, slopes_t.float(), offsets_t.float()


class ApproxActivation(nn.Module):
    """Piecewise-linear baseline/proposed sigmoid or tanh for fp32/fp8/int8/uint8."""

    def __init__(
        self,
        activation: str,
        scheme: str,
        dtype: str,
        *,
        x_max: float = 8.0,
        segments: int = 16,
    ) -> None:
        super().__init__()
        activation = activation.lower()
        scheme = scheme.lower()
        dtype = dtype.lower()
        if activation not in ACTIVATIONS:
            raise ValueError(f"activation must be one of {ACTIVATIONS}")
        if scheme not in SCHEMES:
            raise ValueError(f"scheme must be one of {SCHEMES}")
        if dtype not in DTYPES:
            raise ValueError(f"dtype must be one of {DTYPES}")
            
        if scheme == "proposed_2":
            if activation == "sigmoid":
                segments = 14
            elif activation == "tanh":
                segments = 9
                
        boundaries, slopes, offsets = _fit_segments(activation, scheme, dtype, x_max, segments)
        self.activation = activation
        self.scheme = scheme
        self.dtype = dtype
        self.x_max = float(x_max)
        self.segments = int(segments)
        self.register_buffer("boundaries", boundaries)
        self.register_buffer("slopes", slopes)
        self.register_buffer("offsets", offsets)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        sign = x < 0
        xp = quantize_like(x.abs(), self.dtype, activation=self.activation, role="input")
        xp = xp.clamp(0.0, self.x_max)
        idx = torch.bucketize(xp.reshape(-1), self.boundaries[1:-1].to(xp.device), right=False)
        idx = idx.reshape_as(xp).clamp_(0, self.segments - 1)
        y = self.slopes.to(xp.device)[idx] * xp + self.offsets.to(xp.device)[idx]
        if self.activation == "sigmoid":
            y = torch.where(sign, 1.0 - y, y)
        else:
            y = torch.where(sign, -y, y)
        return quantize_like(y, self.dtype, activation=self.activation, role="output")


class ApproxGELU(nn.Module):
    """GELU expressed through the selected approximate tanh or sigmoid."""

    def __init__(self, activation: str, scheme: str, dtype: str) -> None:
        super().__init__()
        self.activation = activation.lower()
        self.inner = ApproxActivation(self.activation, scheme, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.activation == "sigmoid":
            return x * self.inner(1.702 * x)
        z = math.sqrt(2.0 / math.pi) * (x + 0.044715 * x.pow(3))
        return 0.5 * x * (1.0 + self.inner(z))


class ExactActivation(nn.Module):
    """Exact FP32 sigmoid or tanh reference used before any approximate activation unit."""

    def __init__(self, activation: str) -> None:
        super().__init__()
        activation = activation.lower()
        if activation not in ACTIVATIONS:
            raise ValueError(f"activation must be one of {ACTIVATIONS}")
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        return torch.sigmoid(x) if self.activation == "sigmoid" else torch.tanh(x)


class ExactGELU(nn.Module):
    """Exact FP32 GELU formulation matching the activation family under test."""

    def __init__(self, activation: str) -> None:
        super().__init__()
        activation = activation.lower()
        if activation not in ACTIVATIONS:
            raise ValueError(f"activation must be one of {ACTIVATIONS}")
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        if self.activation == "sigmoid":
            return x * torch.sigmoid(1.702 * x)
        z = math.sqrt(2.0 / math.pi) * (x + 0.044715 * x.pow(3))
        return 0.5 * x * (1.0 + torch.tanh(z))


class ApproxReLUReplacement(nn.Module):
    """Explicit activation-substitution mode for ReLU networks."""

    def __init__(self, activation: str, scheme: str, dtype: str) -> None:
        super().__init__()
        self.inner = ApproxActivation(activation, scheme, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.inner(x)


@dataclass
class ClassificationResult:
    activation: str
    scheme: str
    dtype: str
    accuracy: float
    delta_pp: float
    latency_s: float
    status: str


def format_table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    rendered = [[str(cell) for cell in row] for row in rows]
    widths = [len(h) for h in headers]
    for row in rendered:
        widths = [max(width, len(cell)) for width, cell in zip(widths, row)]
    sep = "+-" + "-+-".join("-" * width for width in widths) + "-+"
    out = [sep, "| " + " | ".join(h.ljust(width) for h, width in zip(headers, widths)) + " |", sep]
    for row in rendered:
        out.append("| " + " | ".join(cell.ljust(width) for cell, width in zip(row, widths)) + " |")
    out.append(sep)
    return "\n".join(out)


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    parser.add_argument("--allow-cpu", action="store_true", help="Permit CPU execution when CUDA is unavailable.")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-samples", type=int, default=None, help="Optional quick-run sample cap.")
    parser.add_argument("--tolerance-pp", type=float, default=1.0, help="Allowed accuracy drop in percentage points.")
    parser.add_argument("--csv", type=Path, default=None, help="Optional CSV path for the summary table.")


def selected_variants(args) -> List[tuple[str, str, str]]:
    activations = getattr(args, "activations", None) or ACTIVATIONS
    schemes = getattr(args, "schemes", None) or SCHEMES
    dtypes = getattr(args, "dtypes", None) or DTYPES
    return [(act, scheme, dtype) for act in activations for scheme in schemes for dtype in dtypes]


def write_csv(path: Path, rows: Iterable[dict]) -> None:
    import csv

    rows = list(rows)
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


class CudaTimer:
    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.start = 0.0

    def __enter__(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self.elapsed = time.perf_counter() - self.start
