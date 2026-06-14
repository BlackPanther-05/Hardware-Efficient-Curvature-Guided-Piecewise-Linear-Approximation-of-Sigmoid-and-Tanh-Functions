#!/usr/bin/env python3
"""Compile and run RTL simulations, then export CSV reports.

The script is intentionally independent of the bash wrappers in the Sigmoid and
Tanh folders. It runs the configured Verilog testbenches with iverilog/vvp,
stores logs in one output directory, and converts the simulation summaries into
CSV tables that are easy to compare or import into a spreadsheet.
"""

from __future__ import annotations

import argparse
import csv
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


RTL_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = RTL_ROOT / "simulation_results"
ACTIVATIONS = ("sigmoid", "tanh")
FORMATS = ("int8", "uint8", "fp8", "fp32")
METHODS = ("baseline", "proposed")


@dataclass(frozen=True)
class RtlTest:
    method: str
    activation: str
    data_format: str
    work_dir: Path
    testbench_dir: Path
    top_file: str
    testbench_file: str
    executable_name: str
    log_name: str

    @property
    def test_id(self) -> str:
        return f"{self.activation}_{self.method}_{self.data_format}"


RTL_TESTS = (
    RtlTest("baseline", "sigmoid", "int8", RTL_ROOT / "sigmoid" / "INT8", RTL_ROOT / "sigmoid" / "INT8", "sigmoid_top_int8.v", "tb_sigmoid_int8_error.v", "tb_sigmoid_baseline_int8", "sigmoid_baseline_int8.log"),
    RtlTest("baseline", "sigmoid", "uint8", RTL_ROOT / "sigmoid" / "UINT8", RTL_ROOT / "sigmoid" / "UINT8", "sigmoid_top_uint8.v", "tb_sigmoid_uint8_error.v", "tb_sigmoid_baseline_uint8", "sigmoid_baseline_uint8.log"),
    RtlTest("baseline", "sigmoid", "fp8", RTL_ROOT / "sigmoid" / "FP8", RTL_ROOT / "sigmoid" / "FP8", "sigmoid_top_fp8.v", "tb_sigmoid_fp8_error.v", "tb_sigmoid_baseline_fp8", "sigmoid_baseline_fp8.log"),
    RtlTest("baseline", "sigmoid", "fp32", RTL_ROOT / "sigmoid" / "FP32", RTL_ROOT / "sigmoid" / "FP32", "sigmoid_top_fp32.v", "tb_sigmoid_fp32_error.v", "tb_sigmoid_baseline_fp32", "sigmoid_baseline_fp32.log"),
    RtlTest("proposed", "sigmoid", "int8", RTL_ROOT / "sigmoid" / "proposed1" / "INT8", RTL_ROOT / "sigmoid" / "INT8", "sigmoid_top_int8.v", "tb_sigmoid_int8_error.v", "tb_sigmoid_proposed_int8", "sigmoid_proposed_int8.log"),
    RtlTest("proposed", "sigmoid", "uint8", RTL_ROOT / "sigmoid" / "proposed1" / "UINT8", RTL_ROOT / "sigmoid" / "UINT8", "sigmoid_top_uint8.v", "tb_sigmoid_uint8_error.v", "tb_sigmoid_proposed_uint8", "sigmoid_proposed_uint8.log"),
    RtlTest("proposed", "sigmoid", "fp8", RTL_ROOT / "sigmoid" / "proposed1" / "FP8", RTL_ROOT / "sigmoid" / "FP8", "sigmoid_top_fp8.v", "tb_sigmoid_fp8_error.v", "tb_sigmoid_proposed_fp8", "sigmoid_proposed_fp8.log"),
    RtlTest("proposed", "sigmoid", "fp32", RTL_ROOT / "sigmoid" / "proposed1" / "FP32", RTL_ROOT / "sigmoid" / "FP32", "sigmoid_top_fp32.v", "tb_sigmoid_fp32_error.v", "tb_sigmoid_proposed_fp32", "sigmoid_proposed_fp32.log"),
    RtlTest("baseline", "tanh", "int8", RTL_ROOT / "tanh" / "INT8", RTL_ROOT / "tanh" / "INT8", "tanh_top_int8.v", "tb_tanh_int8_error.v", "tb_tanh_baseline_int8", "tanh_baseline_int8.log"),
    RtlTest("baseline", "tanh", "uint8", RTL_ROOT / "tanh" / "UINT8", RTL_ROOT / "tanh" / "UINT8", "tanh_top_uint8.v", "tb_tanh_uint8_error.v", "tb_tanh_baseline_uint8", "tanh_baseline_uint8.log"),
    RtlTest("baseline", "tanh", "fp8", RTL_ROOT / "tanh" / "FP8", RTL_ROOT / "tanh" / "FP8", "tanh_top_fp8.v", "tb_tanh_fp8_error.v", "tb_tanh_baseline_fp8", "tanh_baseline_fp8.log"),
    RtlTest("baseline", "tanh", "fp32", RTL_ROOT / "tanh" / "FP32", RTL_ROOT / "tanh" / "FP32", "tanh_top_fp32.v", "tb_tanh_fp32_error.v", "tb_tanh_baseline_fp32", "tanh_baseline_fp32.log"),
    RtlTest("proposed", "tanh", "int8", RTL_ROOT / "tanh" / "proposed1" / "INT8", RTL_ROOT / "tanh" / "INT8", "tanh_top_int8.v", "tb_tanh_int8_error.v", "tb_tanh_proposed_int8", "tanh_proposed_int8.log"),
    RtlTest("proposed", "tanh", "uint8", RTL_ROOT / "tanh" / "proposed1" / "UINT8", RTL_ROOT / "tanh" / "UINT8", "tanh_top_uint8.v", "tb_tanh_uint8_error.v", "tb_tanh_proposed_uint8", "tanh_proposed_uint8.log"),
    RtlTest("proposed", "tanh", "fp8", RTL_ROOT / "tanh" / "proposed1" / "FP8", RTL_ROOT / "tanh" / "FP8", "tanh_top_fp8.v", "tb_tanh_fp8_error.v", "tb_tanh_proposed_fp8", "tanh_proposed_fp8.log"),
    RtlTest("proposed", "tanh", "fp32", RTL_ROOT / "tanh" / "proposed1" / "FP32", RTL_ROOT / "tanh" / "FP32", "tanh_top_fp32.v", "tb_tanh_fp32_error.v", "tb_tanh_proposed_fp32", "tanh_proposed_fp32.log"),
)


RESULT_PATTERNS = {
    "total_vectors": re.compile(r"Total vectors tested\s*:\s*(?P<value>\d+)", re.IGNORECASE),
    "vectors_over_threshold": re.compile(r"Vectors with err >\s*(?P<threshold>[0-9.eE+-]+)\s*:\s*(?P<value>\d+)", re.IGNORECASE),
    "max_error": re.compile(r"Max error\s*\([^)]*\)\s*:\s*(?P<value>[0-9.eE+-]+)", re.IGNORECASE),
    "avg_error": re.compile(r"Avg error\s*\([^)]*\)\s*:\s*(?P<value>[0-9.eE+-]+)", re.IGNORECASE),
}
CHECK_PATTERN = re.compile(r"^\s*(?P<check>.+?)\s*:\s*(?P<status>PASS|FAIL)\s*$", re.IGNORECASE)
NUMBER_PATTERN = re.compile(r"[+-]?(?:\d+\.\d+|\d+|\.\d+)(?:[eE][+-]?\d+)?")


def parse_multi(value: str, allowed: tuple[str, ...], label: str) -> list[str]:
    if value.lower() == "all":
        return list(allowed)
    selected = [item.strip().lower() for item in value.split(",") if item.strip()]
    invalid = sorted(set(selected) - set(allowed))
    if invalid:
        raise argparse.ArgumentTypeError(f"Invalid {label}: {', '.join(invalid)}")
    return selected


def select_tests(activations: Iterable[str], methods: Iterable[str], formats: Iterable[str]) -> list[RtlTest]:
    activation_set = set(activations)
    method_set = set(methods)
    format_set = set(formats)
    return [
        test
        for test in RTL_TESTS
        if test.activation in activation_set and test.method in method_set and test.data_format in format_set
    ]


def ensure_tools() -> None:
    missing = [tool for tool in ("iverilog", "vvp") if shutil.which(tool) is None]
    if missing:
        raise RuntimeError(
            "Missing required RTL simulation tool(s): "
            + ", ".join(missing)
            + ". Install Icarus Verilog or add the tools to PATH."
        )


def run_command(command: list[str], cwd: Path) -> tuple[int, str, str, float]:
    started = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=str(cwd),
        capture_output=True,
        text=True,
        check=False,
    )
    elapsed = time.perf_counter() - started
    return completed.returncode, completed.stdout, completed.stderr, elapsed


def parse_log(log_text: str) -> tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]]]:
    metrics: dict[str, object] = {
        "total_vectors": "",
        "error_threshold": "",
        "vectors_over_threshold": "",
        "max_error": "",
        "avg_error": "",
    }
    checks: list[dict[str, object]] = []
    vector_rows: list[dict[str, object]] = []
    in_vector_table = False

    for line_number, line in enumerate(log_text.splitlines(), start=1):
        stripped = line.strip()

        for key, pattern in RESULT_PATTERNS.items():
            match = pattern.search(stripped)
            if not match:
                continue
            if key == "vectors_over_threshold":
                metrics["error_threshold"] = match.group("threshold")
                metrics[key] = int(match.group("value"))
            elif key == "total_vectors":
                metrics[key] = int(match.group("value"))
            else:
                metrics[key] = float(match.group("value"))

        check_match = CHECK_PATTERN.match(stripped)
        if check_match and "error <=" in stripped.lower():
            checks.append(
                {
                    "check_name": check_match.group("check").strip(),
                    "status": check_match.group("status").upper(),
                }
            )

        if stripped.startswith("Vectors with error >"):
            in_vector_table = True
            continue
        if stripped.startswith("RESULTS"):
            in_vector_table = False

        if not in_vector_table:
            continue
        if not stripped or stripped.startswith("-") or "z_float" in stripped:
            continue

        numbers = NUMBER_PATTERN.findall(stripped)
        if len(numbers) < 4:
            continue
        vector_rows.append(
            {
                "line_number": line_number,
                "raw_line": stripped,
                "input_value": numbers[0],
                "z_output": numbers[1] if len(numbers) > 1 else "",
                "z_float": numbers[-3],
                "reference": numbers[-2],
                "abs_error": numbers[-1],
            }
        )

    return metrics, checks, vector_rows


def write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_test(test: RtlTest, output_dir: Path) -> tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]]]:
    build_dir = output_dir / "build"
    log_dir = output_dir / "logs"
    build_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    executable = build_dir / test.executable_name
    log_path = log_dir / test.log_name

    compile_command = [
        "iverilog",
        "-o",
        str(executable),
        str(test.work_dir / test.top_file),
        str(test.testbench_dir / test.testbench_file),
    ]
    compile_code, compile_stdout, compile_stderr, compile_seconds = run_command(compile_command, RTL_ROOT)

    simulation_stdout = ""
    simulation_stderr = ""
    simulation_seconds = 0.0
    simulation_code = ""

    if compile_code == 0:
        simulation_code_int, simulation_stdout, simulation_stderr, simulation_seconds = run_command(
            ["vvp", str(executable)],
            RTL_ROOT,
        )
        simulation_code = simulation_code_int

    log_text = simulation_stdout
    if simulation_stderr:
        log_text += "\nSTDERR:\n" + simulation_stderr
    log_path.write_text(log_text, encoding="utf-8")

    metrics, checks, vector_rows = parse_log(log_text)
    check_statuses = [str(check["status"]) for check in checks]
    overall_status = "PASS" if compile_code == 0 and simulation_code == 0 and "FAIL" not in check_statuses else "FAIL"
    if compile_code != 0:
        overall_status = "COMPILE_FAIL"
    elif simulation_code not in (0, ""):
        overall_status = "SIM_FAIL"

    summary = {
        "activation": test.activation,
        "method": test.method,
        "data_format": test.data_format,
        "test_id": test.test_id,
        "status": overall_status,
        "compile_return_code": compile_code,
        "simulation_return_code": simulation_code,
        "compile_seconds": f"{compile_seconds:.6f}",
        "simulation_seconds": f"{simulation_seconds:.6f}",
        "total_vectors": metrics["total_vectors"],
        "error_threshold": metrics["error_threshold"],
        "vectors_over_threshold": metrics["vectors_over_threshold"],
        "avg_error": metrics["avg_error"],
        "max_error": metrics["max_error"],
        "checks_passed": sum(1 for status in check_statuses if status == "PASS"),
        "checks_failed": sum(1 for status in check_statuses if status == "FAIL"),
        "log_path": str(log_path),
        "compile_stdout": compile_stdout.strip(),
        "compile_stderr": compile_stderr.strip(),
    }

    for check in checks:
        check.update(
            {
                "activation": test.activation,
                "method": test.method,
                "data_format": test.data_format,
                "test_id": test.test_id,
            }
        )
    for row in vector_rows:
        row.update(
            {
                "activation": test.activation,
                "method": test.method,
                "data_format": test.data_format,
                "test_id": test.test_id,
            }
        )

    return summary, checks, vector_rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compile/run RTL simulations and export CSV reports.",
    )
    parser.add_argument(
        "--methods",
        default="all",
        help="Comma-separated methods to run: baseline,proposed or all.",
    )
    parser.add_argument(
        "--activations",
        default="all",
        help="Comma-separated activations to run: sigmoid,tanh or all.",
    )
    parser.add_argument(
        "--formats",
        default="all",
        help="Comma-separated formats to run: int8,uint8,fp8,fp32 or all.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for CSV files, logs, and build artifacts.",
    )
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="Continue remaining simulations if one test fails.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    ensure_tools()
    activations = parse_multi(args.activations, ACTIVATIONS, "activation")
    methods = parse_multi(args.methods, METHODS, "method")
    formats = parse_multi(args.formats, FORMATS, "format")
    tests = select_tests(activations, methods, formats)
    if not tests:
        raise RuntimeError("No RTL tests matched the requested activation/format filters.")

    output_dir = args.output_dir.resolve()
    summary_rows: list[dict[str, object]] = []
    check_rows: list[dict[str, object]] = []
    vector_rows: list[dict[str, object]] = []

    print(f"Running {len(tests)} RTL simulation(s)...")
    for test in tests:
        print(f"  - {test.activation.upper()} {test.method.upper()} {test.data_format.upper()}")
        summary, checks, vectors = run_test(test, output_dir)
        summary_rows.append(summary)
        check_rows.extend(checks)
        vector_rows.extend(vectors)
        if summary["status"] != "PASS" and not args.keep_going:
            break

    summary_path = output_dir / "rtl_simulation_summary.csv"
    checks_path = output_dir / "rtl_simulation_checks.csv"
    vectors_path = output_dir / "rtl_simulation_vectors.csv"

    write_csv(
        summary_path,
        [
            "activation",
            "method",
            "data_format",
            "test_id",
            "status",
            "compile_return_code",
            "simulation_return_code",
            "compile_seconds",
            "simulation_seconds",
            "total_vectors",
            "error_threshold",
            "vectors_over_threshold",
            "avg_error",
            "max_error",
            "checks_passed",
            "checks_failed",
            "log_path",
            "compile_stdout",
            "compile_stderr",
        ],
        summary_rows,
    )
    write_csv(
        checks_path,
        ["activation", "method", "data_format", "test_id", "check_name", "status"],
        check_rows,
    )
    write_csv(
        vectors_path,
        [
            "activation",
            "method",
            "data_format",
            "test_id",
            "line_number",
            "input_value",
            "z_output",
            "z_float",
            "reference",
            "abs_error",
            "raw_line",
        ],
        vector_rows,
    )

    print("")
    print("RTL simulation CSV files written:")
    print(f"  Summary : {summary_path}")
    print(f"  Checks  : {checks_path}")
    print(f"  Vectors : {vectors_path}")
    print("")
    print(f"{'Activation':10s} {'Method':10s} {'Format':8s} {'Status':12s} {'Vectors':>8s} {'Avg Error':>12s} {'Max Error':>12s}")
    print("-" * 84)
    for row in summary_rows:
        print(
            f"{str(row['activation']):10s} "
            f"{str(row['method']):10s} "
            f"{str(row['data_format']):8s} "
            f"{str(row['status']):12s} "
            f"{str(row['total_vectors']):>8s} "
            f"{str(row['avg_error']):>12s} "
            f"{str(row['max_error']):>12s}"
        )

    return 0 if all(row["status"] == "PASS" for row in summary_rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
