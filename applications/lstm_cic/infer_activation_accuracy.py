from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

APP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(APP_DIR.parent))

from approx_activation_lib import (  # noqa: E402
    ACTIVATIONS,
    DTYPES,
    SCHEMES,
    ApproxActivation,
    CudaTimer,
    ExactActivation,
    add_common_args,
    format_table,
    require_device,
    selected_variants,
    write_csv,
)
from cic_lstm_common import (  # noqa: E402
    ActivationOnlyLSTMClassifier,
    LSTMCICClassifier,
    TensorSequenceDataset,
)


def load_checkpoint(weights: Path) -> tuple[dict, dict]:
    """Load checkpoint and return (model state_dict, metadata)."""
    ckpt = torch.load(weights, map_location="cpu")
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        return ckpt["model_state"], ckpt.get("metadata", {})
    return ckpt, {}


def build_baseline_model(state: dict, meta: dict) -> LSTMCICClassifier:
    """Reconstruct the trained FP32 LSTM model from checkpoint."""
    model = LSTMCICClassifier(
        input_size=meta["input_size"],
        hidden_size=meta["hidden_size"],
        num_classes=meta["num_classes"],
        num_layers=meta.get("num_layers", 1),
        dropout=0.0,  # dropout disabled for inference
    )
    model.load_state_dict(state)
    return model.eval()


def build_activation_model(
    baseline: LSTMCICClassifier,
    *,
    sigmoid_act: nn.Module,
    tanh_act: nn.Module,
) -> ActivationOnlyLSTMClassifier:
    """Build an ActivationOnlyLSTMClassifier with custom gate activations."""
    model = ActivationOnlyLSTMClassifier(
        baseline,
        sigmoid_activation=sigmoid_act,
        tanh_activation=tanh_act,
    )
    return model.eval()


def load_test_data(path: Path) -> TensorSequenceDataset:
    """Load pre-saved test sequences."""
    data = torch.load(path, map_location="cpu")
    return TensorSequenceDataset(data["x"], data["y"])


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    desc: str,
) -> tuple[float, float, float, float]:
    """Evaluate model and return (accuracy%, macro_f1, loss, elapsed_seconds)."""
    model.to(device).eval()
    criterion = nn.CrossEntropyLoss()
    preds_all = []
    labels_all = []
    losses = []
    with CudaTimer(device) as timer:
        for x, y in tqdm(loader, desc=desc, leave=False, ncols=110):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x)
            loss = criterion(logits, y)
            losses.append(loss.item())
            preds_all.append(logits.argmax(dim=1).cpu().numpy())
            labels_all.append(y.cpu().numpy())
    preds = np.concatenate(preds_all)
    labels = np.concatenate(labels_all)
    acc = accuracy_score(labels, preds) * 100.0
    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
    avg_loss = float(np.mean(losses))
    return acc, macro_f1, avg_loss, timer.elapsed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LSTM CIC-IDS2018 activation approximation inference benchmark."
    )
    add_common_args(parser)
    parser.add_argument(
        "--weights",
        type=Path,
        default=APP_DIR / "artifacts" / "lstm_cic_fp32.pth",
        help="Path to trained LSTM checkpoint.",
    )
    parser.add_argument(
        "--test-data",
        type=Path,
        default=APP_DIR / "artifacts" / "test_sequences.pt",
        help="Path to pre-saved test sequences.",
    )
    parser.add_argument(
        "--activations",
        nargs="+",
        choices=ACTIVATIONS,
        default=list(ACTIVATIONS),
    )
    parser.add_argument(
        "--schemes",
        nargs="+",
        choices=SCHEMES,
        default=list(SCHEMES),
    )
    parser.add_argument(
        "--dtypes",
        nargs="+",
        choices=DTYPES,
        default=list(DTYPES),
    )
    args = parser.parse_args()

    device = require_device(args.device, args.allow_cpu)

    # ── Load checkpoint and test data ──
    state, meta = load_checkpoint(args.weights)
    test_ds = load_test_data(args.test_data)
    if args.max_samples is not None:
        test_ds = Subset(test_ds, range(min(args.max_samples, len(test_ds))))
    loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # ── 1. Baseline: original FP32 LSTM (PyTorch native sigmoid/tanh) ──
    baseline_model = build_baseline_model(state, meta)
    baseline_acc, baseline_f1, baseline_loss, baseline_time = evaluate(
        baseline_model, loader, device, "baseline FP32 (native LSTM)"
    )
    print(
        f"\nBaseline FP32 | acc={baseline_acc:.4f}% | macro_f1={baseline_f1:.4f} | "
        f"loss={baseline_loss:.4f} | time={baseline_time:.2f}s"
    )

    rows = [
        {
            "reference": "trained",
            "activation": "native",
            "scheme": "exact_fp32",
            "dtype": "fp32",
            "accuracy": baseline_acc,
            "macro_f1": baseline_f1,
            "delta_pp": 0.0,
            "latency_s": baseline_time,
            "status": "TRAINED_REF",
        }
    ]

    # ── 2. Exact FP32 activation references ──
    # For LSTM, both sigmoid and tanh are used internally in gates.
    # The "activation" selector determines which approximation family we test.
    # - "sigmoid" family: approximate sigmoid gates, exact tanh gates
    # - "tanh" family: approximate tanh gates, exact sigmoid gates
    # - The exact reference for each family uses the exact (torch) version for all.
    activation_refs: dict[str, float] = {}
    for activation in args.activations:
        exact_sigmoid = ExactActivation("sigmoid").to(device)
        exact_tanh = ExactActivation("tanh").to(device)
        exact_model = build_activation_model(
            baseline_model,
            sigmoid_act=exact_sigmoid,
            tanh_act=exact_tanh,
        )
        exact_acc, exact_f1, exact_loss, exact_time = evaluate(
            exact_model, loader, device, f"exact {activation} fp32"
        )
        activation_refs[activation] = exact_acc
        rows.append(
            {
                "reference": activation,
                "activation": activation,
                "scheme": "exact_fp32",
                "dtype": "fp32",
                "accuracy": exact_acc,
                "macro_f1": exact_f1,
                "delta_pp": 0.0,
                "latency_s": exact_time,
                "status": "ACT_REF",
            }
        )

    # ── 3. Approximate activation variants ──
    num_layers = meta.get("num_layers", 1)
    for activation, scheme, dtype in selected_variants(args):
        # Build approximate activations for the LSTM gates.
        # - Baseline: each layer gets its OWN LUT copy (embedded per layer)
        # - Proposed: all layers share ONE LUT instance (single shared block)
        # For each "activation family":
        #   - "sigmoid": approximate sigmoid for gates, exact tanh for cell
        #   - "tanh": exact sigmoid for gates, approximate tanh for cell
        if scheme == "proposed":
            # ── Proposed: single shared LUT across all layers ──
            if activation == "sigmoid":
                sig_act = ApproxActivation("sigmoid", scheme, dtype).to(device)
                tanh_act = ExactActivation("tanh").to(device)
            else:
                sig_act = ExactActivation("sigmoid").to(device)
                tanh_act = ApproxActivation("tanh", scheme, dtype).to(device)
        else:
            # ── Baseline: separate LUT copy per layer ──
            if activation == "sigmoid":
                sig_act = [ApproxActivation("sigmoid", scheme, dtype).to(device) for _ in range(num_layers)]
                tanh_act = ExactActivation("tanh").to(device)
            else:
                sig_act = ExactActivation("sigmoid").to(device)
                tanh_act = [ApproxActivation("tanh", scheme, dtype).to(device) for _ in range(num_layers)]

        model = build_activation_model(
            baseline_model,
            sigmoid_act=sig_act,
            tanh_act=tanh_act,
        )
        acc, macro_f1, loss, latency = evaluate(
            model, loader, device, f"{activation} {scheme} {dtype}"
        )
        delta = acc - activation_refs[activation]
        rows.append(
            {
                "reference": activation,
                "activation": activation,
                "scheme": scheme,
                "dtype": dtype,
                "accuracy": acc,
                "macro_f1": macro_f1,
                "delta_pp": delta,
                "latency_s": latency,
                "status": "PASS" if delta >= -args.tolerance_pp else "DROP>1pp",
            }
        )

    # ── Print summary table ──
    table_rows = [
        [
            r["reference"],
            r["activation"],
            r["scheme"],
            r["dtype"],
            f'{r["accuracy"]:.4f}%',
            f'{r["macro_f1"]:.4f}',
            f'{r["delta_pp"]:+.4f}',
            f'{r["latency_s"]:.2f}',
            r["status"],
        ]
        for r in rows
    ]
    print("\nLSTM CIC-IDS2018 activation approximation results")
    print(
        f"Device: {device} | Samples: {len(loader.dataset)} | "
        f"Task: {meta.get('task', 'binary')} | Classes: {meta.get('classes', 'N/A')}"
    )
    print(
        "The trained reference is original FP32 inference with native PyTorch LSTM (built-in sigmoid/tanh)."
    )
    print(
        "Each activation family has its own exact FP32 ACT_REF (manual LSTM with exact torch sigmoid & tanh)."
    )
    print(
        "Approximate rows compare against the matching activation family ACT_REF."
    )
    print(
        "For 'sigmoid' family: sigmoid gates use approximate, tanh gates remain exact."
    )
    print(
        "For 'tanh' family: tanh gates use approximate, sigmoid gates remain exact."
    )
    print(
        "Only the activation functions are replaced; all weight matrices and the classifier head remain FP32."
    )
    print(
        format_table(
            [
                "Reference",
                "Activation",
                "Act Unit",
                "DType",
                "Accuracy",
                "Macro F1",
                "Delta pp",
                "Seconds",
                "Status",
            ],
            table_rows,
        )
    )
    if args.csv:
        write_csv(args.csv, rows)


if __name__ == "__main__":
    main()
