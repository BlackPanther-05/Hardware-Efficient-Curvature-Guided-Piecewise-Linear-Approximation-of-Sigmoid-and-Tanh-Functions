#!/usr/bin/env python3
"""Generate data-format RTL modules from simulation coefficient CSV.

The generated modules keep the existing testbench-facing top-module names, but
replace behavioral $exp models with integer piecewise-linear LUT logic:

    segment select -> slope/intercept LUT -> multiply/add -> symmetry/output

Baseline modules are written to the existing format folders. Proposed modules
are written under proposed1/<FORMAT>/ and use proposed segment boundaries.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RTL_ROOT = ROOT / "rtl"
COEFF_CSV = ROOT / "results" / "activation_segment_coefficients.csv"
X_SCALE = 4096
COEFF_SCALE = 1 << 20
OUT_FP32_SCALE = 2_147_483_647

ACTIVATIONS = ("sigmoid", "tanh")
METHODS = ("baseline", "proposed")
FORMATS = ("int8", "uint8", "fp8", "fp32")


def load_coefficients() -> dict[tuple[str, str, str], list[dict[str, float]]]:
    tables: dict[tuple[str, str, str], list[dict[str, float]]] = defaultdict(list)
    with COEFF_CSV.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            key = (row["activation"], row["method"], row["data_format"])
            if key[0] in ACTIVATIONS and key[1] in METHODS and key[2] in FORMATS:
                tables[key].append(
                    {
                        "segment": int(row["segment"]),
                        "x_start": float(row["x_start"]),
                        "x_end": float(row["x_end"]),
                        "slope": float(row["slope"]),
                        "intercept": float(row["intercept"]),
                    }
                )

    for key, rows in tables.items():
        rows.sort(key=lambda item: item["segment"])
    return tables


def q(value: float, scale: int) -> int:
    return int(round(value * scale))


def module_name(activation: str, data_format: str) -> str:
    return f"{activation}_top_{data_format}"


def input_width(data_format: str) -> int:
    return 32 if data_format == "fp32" else 8


def output_width(data_format: str) -> int:
    return 32 if data_format == "fp32" else 8


def output_signed_decl(data_format: str) -> str:
    return "reg [31:0]" if data_format == "fp32" else "reg [7:0]"


def input_abs_logic(activation: str, data_format: str) -> str:
    if data_format in {"int8", "fp8"}:
        return f"""
    wire signed [7:0] b_signed = b;
    wire sign_bit = b_signed < 0;
    wire [8:0] abs_raw = sign_bit ? -b_signed : b_signed;
    wire [31:0] x_abs_scaled = (abs_raw * {X_SCALE}) / 16;
"""
    if data_format == "fp32":
        return f"""
    wire signed [31:0] b_signed = b;
    wire sign_bit = b_signed < 0;
    wire [31:0] abs_raw = sign_bit ? -b_signed : b_signed;
    wire [31:0] x_abs_scaled = (abs_raw * {X_SCALE}) / 64;
"""
    if activation == "sigmoid":
        return f"""
    wire signed [31:0] centered = ($signed({{1'b0, b}}) * 16) - 32'sd2040;
    wire sign_bit = centered < 0;
    wire [31:0] abs_raw = sign_bit ? -centered : centered;
    wire [31:0] x_abs_scaled = ((abs_raw * {X_SCALE}) + 127) / 255;
"""
    return f"""
    wire signed [31:0] centered = ($signed({{1'b0, b}}) * 2) - 32'sd255;
    wire sign_bit = centered < 0;
    wire [31:0] abs_raw = sign_bit ? -centered : centered;
    wire [31:0] x_abs_scaled = ((abs_raw * {X_SCALE}) + 127) / 255;
"""


def output_logic(activation: str, data_format: str) -> str:
    if activation == "sigmoid":
        positive_logic = f"""
            y_final = sign_bit ? ({COEFF_SCALE} - y_positive) : y_positive;
"""
        if data_format in {"int8", "fp8"}:
            return positive_logic + f"""
            q_out = ((y_final * 128) + ({COEFF_SCALE // 2})) / {COEFF_SCALE};
            if (q_out < 0) q_out = 0;
            if (q_out > 127) q_out = 127;
            z = q_out[7:0];
"""
        if data_format == "uint8":
            return positive_logic + f"""
            q_out = ((y_final * 255) + ({COEFF_SCALE // 2})) / {COEFF_SCALE};
            if (q_out < 0) q_out = 0;
            if (q_out > 255) q_out = 255;
            z = q_out[7:0];
"""
        return positive_logic + f"""
            q_out = ((y_final * {OUT_FP32_SCALE}) + ({COEFF_SCALE // 2})) / {COEFF_SCALE};
            if (q_out < 0) q_out = 0;
            if (q_out > {OUT_FP32_SCALE}) q_out = {OUT_FP32_SCALE};
            z = q_out[31:0];
"""

    if data_format in {"int8", "fp8"}:
        return f"""
            q_mag = ((y_positive * 128) + ({COEFF_SCALE // 2})) / {COEFF_SCALE};
            if (q_mag < 0) q_mag = 0;
            if (q_mag > 127) q_mag = 127;
            q_out = sign_bit ? -q_mag : q_mag;
            z = q_out[7:0];
"""
    if data_format == "uint8":
        return f"""
            y_signed = sign_bit ? -y_positive : y_positive;
            q_out = (((y_signed + {COEFF_SCALE}) * 255) + {COEFF_SCALE}) / ({2 * COEFF_SCALE});
            if (q_out < 0) q_out = 0;
            if (q_out > 255) q_out = 255;
            z = q_out[7:0];
"""
    return f"""
            y_signed = sign_bit ? -y_positive : y_positive;
            q_out = ((y_signed * {OUT_FP32_SCALE}) + (sign_bit ? -{COEFF_SCALE // 2} : {COEFF_SCALE // 2})) / {COEFF_SCALE};
            z = q_out[31:0];
"""


def render_module(activation: str, method: str, data_format: str, rows: list[dict[str, float]]) -> str:
    mod = module_name(activation, data_format)
    in_width = input_width(data_format) - 1
    out_decl = output_signed_decl(data_format)
    boundaries = [q(item["x_end"], X_SCALE) for item in rows]
    slopes = [q(item["slope"], COEFF_SCALE) for item in rows]
    intercepts = [q(item["intercept"], COEFF_SCALE) for item in rows]

    segment_cases = "\n".join(
        f"            (x_abs_scaled <= 32'd{boundaries[index]}): seg = 4'd{index};"
        for index in range(15)
    )
    lut_cases = "\n".join(
        f"            4'd{index}: begin m_lut = 32'd{slopes[index]}; c_lut = 32'd{intercepts[index]}; end"
        for index in range(16)
    )

    return f"""// ============================================================
// {data_format.upper()} {activation.upper()} TOP MODULE ({method.upper()} LUT)
// ============================================================
// Architecture: segment select -> slope/intercept LUT -> multiply/add -> output mapping
// x scale: x_scaled = x * {X_SCALE}
// coefficient scale: coeff = real * {COEFF_SCALE}
// ============================================================

module {mod} (
    input        clk,
    input  [{in_width}:0] b,
    output {out_decl} z
);

{input_abs_logic(activation, data_format)}
    reg [3:0] seg;
    reg [31:0] m_lut;
    reg [31:0] c_lut;
    reg signed [63:0] y_positive;
    reg signed [63:0] y_final;
    reg signed [63:0] y_signed;
    reg signed [63:0] q_out;
    reg signed [63:0] q_mag;

    always @(*) begin
        casez (1'b1)
{segment_cases}
            default: seg = 4'd15;
        endcase
    end

    always @(*) begin
        case (seg)
{lut_cases}
            default: begin m_lut = 32'd{slopes[-1]}; c_lut = 32'd{intercepts[-1]}; end
        endcase
    end

    always @(*) begin
        y_positive = ((m_lut * x_abs_scaled) / {X_SCALE}) + c_lut;
        if (y_positive < 0) y_positive = 0;
        if (y_positive > {COEFF_SCALE}) y_positive = {COEFF_SCALE};
{output_logic(activation, data_format)}
    end

endmodule
"""


def output_path(activation: str, method: str, data_format: str) -> Path:
    family = "sigmoid" if activation == "sigmoid" else "tanh"
    filename = f"{activation}_top_{data_format}.v"
    if method == "baseline":
        return RTL_ROOT / family / "baseline" / data_format.upper() / filename
    return RTL_ROOT / family / "proposed1" / data_format.upper() / filename


def main() -> int:
    tables = load_coefficients()
    for activation in ACTIVATIONS:
        for method in METHODS:
            for data_format in FORMATS:
                rows = tables[(activation, method, data_format)]
                if len(rows) != 16:
                    raise RuntimeError(f"Expected 16 rows for {(activation, method, data_format)}, got {len(rows)}")
                path = output_path(activation, method, data_format)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(render_module(activation, method, data_format, rows), encoding="utf-8")
                print(f"Wrote {path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
