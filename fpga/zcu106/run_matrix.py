#!/usr/bin/env python3
"""Run the ZCU106 Vivado implementation matrix and aggregate metrics."""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path


ACTIVATIONS = ("sigmoid", "tanh")
METHODS = ("baseline", "proposed")
FORMATS = ("int8", "uint8", "fp8", "fp32")
ROOT = Path(__file__).resolve().parent
TCL_SCRIPT = ROOT / "scripts" / "run_design.tcl"
BUILD_DIR = ROOT / "build"
RESULTS_DIR = ROOT / "results"


def selections(value: str, allowed: tuple[str, ...]) -> list[str]:
    if value == "all":
        return list(allowed)
    values = [item.strip().lower() for item in value.split(",") if item.strip()]
    invalid = sorted(set(values) - set(allowed))
    if invalid:
        raise argparse.ArgumentTypeError(f"Unsupported values: {', '.join(invalid)}")
    return values


def find_vivado(explicit: str | None) -> str:
    candidates = [
        explicit,
        os.environ.get("VIVADO"),
        # Update this path to your Vivado installation
        # Set VIVADO env var or pass --vivado to specify your Vivado path
        os.environ.get("VIVADO", "vivado"),
        "vivado",
    ]
    for candidate in candidates:
        if not candidate:
            continue
        if candidate == "vivado" or Path(candidate).is_file():
            return candidate
    raise FileNotFoundError("Vivado was not found; pass --vivado or set VIVADO")


def collect_metrics() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for metrics_path in sorted(BUILD_DIR.glob("*/metrics.csv")):
        with metrics_path.open(newline="", encoding="utf-8") as handle:
            rows.extend(csv.DictReader(handle))
    return rows


def write_summary(rows: list[dict[str, str]]) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output = RESULTS_DIR / "implementation_summary.csv"
    if not rows:
        return output
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--activations", default="all")
    parser.add_argument("--methods", default="all")
    parser.add_argument("--formats", default="all")
    parser.add_argument("--action", choices=("synth", "impl"), default="impl")
    parser.add_argument("--period-ns", type=float, default=1.0)
    parser.add_argument("--jobs", type=int, default=max(1, min(8, os.cpu_count() or 1)))
    parser.add_argument("--vivado", help="Path to the Vivado executable")
    parser.add_argument("--keep-going", action="store_true", help="Continue after a Vivado failure")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.period_ns <= 0.0:
        raise SystemExit("--period-ns must be positive")
    if args.jobs < 1:
        raise SystemExit("--jobs must be positive")

    vivado = find_vivado(args.vivado)
    activations = selections(args.activations, ACTIVATIONS)
    methods = selections(args.methods, METHODS)
    formats = selections(args.formats, FORMATS)
    designs = [(a, m, f) for a in activations for m in methods for f in formats]

    failures: list[str] = []
    for index, (activation, method, data_format) in enumerate(designs, start=1):
        design = f"{activation}_{method}_{data_format}"
        run_dir = BUILD_DIR / design
        run_dir.mkdir(parents=True, exist_ok=True)
        log_path = run_dir / "vivado.log"
        command = [
            vivado,
            "-mode",
            "batch",
            "-nojournal",
            "-notrace",
            "-log",
            str(log_path),
            "-source",
            str(TCL_SCRIPT),
            "-tclargs",
            activation,
            method,
            data_format,
            args.action,
            str(args.period_ns),
            str(args.jobs),
        ]
        print(f"[{index:02d}/{len(designs):02d}] {design}: {args.action} at {args.period_ns:.3f} ns", flush=True)
        completed = subprocess.run(command, cwd=ROOT, check=False)
        if completed.returncode != 0:
            failures.append(design)
            print(f"  FAILED: see {log_path}", file=sys.stderr)
            if not args.keep_going:
                break

    rows = collect_metrics()
    summary = write_summary(rows)
    timing_failures = [
        f"{row['activation']}_{row['method']}_{row['data_format']}"
        for row in rows
        if row["timing_met"] != "1"
    ]

    print(f"\nSummary: {summary}")
    print(f"Completed metrics: {len(rows)}/{len(designs)}")
    if timing_failures:
        print(f"Timing failures: {', '.join(timing_failures)}")
    if failures:
        print(f"Vivado failures: {', '.join(failures)}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
