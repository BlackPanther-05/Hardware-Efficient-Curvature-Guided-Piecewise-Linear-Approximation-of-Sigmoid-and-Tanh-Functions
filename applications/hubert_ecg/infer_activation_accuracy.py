from __future__ import annotations

import argparse
import ast
import copy
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import wfdb
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

APP_DIR = Path(__file__).resolve().parent
import sys
sys.path.insert(0, str(APP_DIR.parent))

from approx_activation_lib import (  # noqa: E402
    ACTIVATIONS,
    DTYPES,
    SCHEMES,
    ApproxActivation,
    ApproxGELU,
    CudaTimer,
    ExactActivation,
    ExactGELU,
    add_common_args,
    format_table,
    require_device,
    selected_variants,
    write_csv,
)


SAMPLING_RATE = 100
SAMPLES_PER_LEAD = 500
NUM_LEADS = 12
NUM_CLASSES = 5
SUPERCLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]
THRESHOLD = 0.5


class LocalHuBERTECGClassifier(nn.Module):
    """HuBERT-ECG classifier reconstructed from the local checkpoint state dict."""

    def __init__(self, num_classes: int = 5) -> None:
        super().__init__()
        from transformers import HubertConfig, HubertModel

        config = HubertConfig(
            conv_dim=[512, 512, 512, 512, 512],
            conv_kernel=[10, 3, 3, 2, 2],
            conv_stride=[5, 2, 2, 2, 2],
            feat_extract_norm="group",
            hidden_size=768,
            intermediate_size=3072,
            num_attention_heads=12,
            num_hidden_layers=12,
            mask_time_prob=0.0,
            mask_feature_prob=0.0,
            hidden_dropout=0.0,
            attention_dropout=0.0,
            activation_dropout=0.0,
            feat_proj_dropout=0.0,
            final_dropout=0.0,
        )
        self.backbone = HubertModel(config)
        self.head = nn.Sequential(
            nn.LayerNorm(config.hidden_size),
            nn.Dropout(0.0),
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.GELU(),
            nn.Dropout(0.0),
            nn.Linear(config.hidden_size // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.backbone(input_values=x).last_hidden_state
        return self.head(hidden.mean(dim=1))


def load_local_model(weights: Path) -> nn.Module:
    ckpt = torch.load(weights, map_location="cpu")
    state = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
    num_classes = ckpt.get("config", {}).get("num_classes", 5) if isinstance(ckpt, dict) else 5
    model = LocalHuBERTECGClassifier(num_classes=num_classes)
    model_state = model.state_dict()
    compatible = {}
    skipped = []
    for key, value in state.items():
        if key.startswith("backbone.final_proj") or key.startswith("backbone.label_embedding"):
            skipped.append(key)
            continue
        local_key = key
        if local_key in model_state and tuple(model_state[local_key].shape) == tuple(value.shape):
            compatible[local_key] = value
        else:
            skipped.append(key)
    missing, unexpected = model.load_state_dict(compatible, strict=False)
    if missing:
        print(f"[LOAD] Missing local keys: {len(missing)} (first 5: {missing[:5]})")
    if unexpected:
        print(f"[LOAD] Unexpected local keys: {len(unexpected)} (first 5: {unexpected[:5]})")
    if skipped:
        print(f"[LOAD] Skipped checkpoint keys not used for inference: {len(skipped)}")
    return model.eval()


def _get_gelu_types() -> tuple:
    """Return all GELU module types to replace (nn.GELU + HuggingFace GELUActivation)."""
    gelu_types = [nn.GELU]
    try:
        from transformers.activations import GELUActivation
        gelu_types.append(GELUActivation)
    except ImportError:
        pass
    return tuple(gelu_types)


def replace_gelu(module: nn.Module, activation: str, scheme: str, dtype: str) -> int:
    """Replace all GELU variants with ApproxGELU instances.

    - Baseline: each GELU site gets its OWN LUT copy (embedded per layer).
    - Proposed: all GELU sites share ONE LUT instance (single shared block).
    """
    target_types = _get_gelu_types()
    if scheme == "proposed":
        shared_gelu = ApproxGELU(activation, scheme, dtype)
        return _replace_gelu_shared(module, target_types, shared_gelu)
    else:
        # Baseline: each site gets a fresh copy
        return _replace_gelu_per_copy(module, target_types, activation, scheme, dtype)


def replace_gelu_exact(module: nn.Module, activation: str) -> int:
    """Replace all GELU variants with a SINGLE shared ExactGELU instance.

    Exact reference is stateless (no LUT), so sharing is always correct.
    """
    shared_gelu = ExactGELU(activation)
    return _replace_gelu_shared(module, _get_gelu_types(), shared_gelu)


def _replace_gelu_shared(module: nn.Module, target_types: tuple, replacement: nn.Module) -> int:
    """Recursively replace all instances of target_types with the SAME shared module."""
    replaced = 0
    for name, child in list(module.named_children()):
        if isinstance(child, target_types):
            setattr(module, name, replacement)
            replaced += 1
        else:
            replaced += _replace_gelu_shared(child, target_types, replacement)
    return replaced


def _replace_gelu_per_copy(
    module: nn.Module, target_types: tuple, activation: str, scheme: str, dtype: str
) -> int:
    """Recursively replace each GELU with its OWN separate ApproxGELU instance (baseline)."""
    replaced = 0
    for name, child in list(module.named_children()):
        if isinstance(child, target_types):
            setattr(module, name, ApproxGELU(activation, scheme, dtype))
            replaced += 1
        else:
            replaced += _replace_gelu_per_copy(child, target_types, activation, scheme, dtype)
    return replaced


def preprocess_ecg(signal: np.ndarray) -> np.ndarray:
    if signal.ndim == 1:
        signal = signal[np.newaxis, :]
    if signal.shape[0] != NUM_LEADS:
        signal = signal.T

    mean = signal.mean(axis=1, keepdims=True)
    std = signal.std(axis=1, keepdims=True) + 1e-8
    signal = np.clip((signal - mean) / std, -6.0, 6.0)

    if signal.shape[1] >= SAMPLES_PER_LEAD:
        signal = signal[:, :SAMPLES_PER_LEAD]
    else:
        pad = np.zeros((NUM_LEADS, SAMPLES_PER_LEAD - signal.shape[1]), dtype=np.float32)
        signal = np.concatenate([signal, pad], axis=1)
    return signal.reshape(-1).astype(np.float32)


class PTBXLDataset(torch.utils.data.Dataset):
    def __init__(self, df: pd.DataFrame, ptbxl_dir: Path) -> None:
        col = "filename_hr" if SAMPLING_RATE == 500 else "filename_lr"
        self.records = df.reset_index()[[col, "label"]].values.tolist()
        self.ptbxl_dir = ptbxl_dir

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        rel_path, label = self.records[idx]
        record = wfdb.rdrecord(os.path.join(self.ptbxl_dir, rel_path))
        signal = preprocess_ecg(record.p_signal.astype(np.float32))
        return torch.tensor(signal), torch.tensor(label, dtype=torch.float32)


def load_metadata(ptbxl_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(ptbxl_dir / "ptbxl_database.csv", index_col="ecg_id")
    scp = pd.read_csv(ptbxl_dir / "scp_statements.csv", index_col=0)
    df["scp_codes"] = df["scp_codes"].apply(ast.literal_eval)
    diag_map = scp[scp["diagnostic"] == 1]["diagnostic_class"].dropna().to_dict()

    def make_label(codes: dict) -> np.ndarray:
        label = np.zeros(NUM_CLASSES, dtype=np.float32)
        for code, likelihood in codes.items():
            diagnostic_class = diag_map.get(code)
            if diagnostic_class in SUPERCLASSES and likelihood > 0:
                label[SUPERCLASSES.index(diagnostic_class)] = 1.0
        return label

    df["label"] = df["scp_codes"].apply(make_label)
    return df[df["label"].apply(lambda x: x.sum() > 0)].copy()


def get_split_loader(ptbxl_dir: Path, split: str, batch_size: int, num_workers: int) -> DataLoader:
    df = load_metadata(ptbxl_dir)
    if split == "test":
        split_df = df[df["strat_fold"] == 10].copy()
    elif split == "val":
        split_df = df[df["strat_fold"] == 9].copy()
    else:
        split_df = df[df["strat_fold"] <= 8].copy()
    return DataLoader(
        PTBXLDataset(split_df, ptbxl_dir),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )


def make_loader(args) -> DataLoader:
    loader = get_split_loader(args.ptbxl_dir, args.split, args.batch_size, args.num_workers)
    if args.max_samples is None:
        return loader
    subset = Subset(loader.dataset, range(min(args.max_samples, len(loader.dataset))))
    return DataLoader(
        subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )


def compute_metrics(labels: np.ndarray, probs: np.ndarray) -> dict:
    preds = (probs >= THRESHOLD).astype(int)
    auroc_per, acc_per, f1_per = [], [], []
    for class_idx in range(NUM_CLASSES):
        if len(np.unique(labels[:, class_idx])) > 1:
            auroc_per.append(roc_auc_score(labels[:, class_idx], probs[:, class_idx]))
        else:
            auroc_per.append(float("nan"))
        acc_per.append(accuracy_score(labels[:, class_idx], preds[:, class_idx]) * 100.0)
        f1_per.append(f1_score(labels[:, class_idx], preds[:, class_idx], zero_division=0))
    return {
        "auroc_per": auroc_per,
        "macro_auroc": float(np.nanmean(auroc_per)) if not np.all(np.isnan(auroc_per)) else float("nan"),
        "acc_per": acc_per,
        "macro_acc": float(np.mean(acc_per)),
        "f1_per": f1_per,
        "macro_f1": float(np.mean(f1_per)),
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    desc: str,
    output_sigmoid: nn.Module | None = None,
) -> tuple[dict, float]:
    model.to(device).eval()
    labels_all = []
    probs_all = []
    with CudaTimer(device) as timer:
        for signals, labels in tqdm(loader, desc=desc, leave=False, ncols=110):
            signals = signals.to(device, non_blocking=True)
            logits = model(signals)
            probs = torch.sigmoid(logits) if output_sigmoid is None else output_sigmoid(logits)
            labels_all.append(labels.cpu().numpy())
            probs_all.append(probs.detach().cpu().numpy())
    labels_np = np.concatenate(labels_all, axis=0)
    probs_np = np.concatenate(probs_all, axis=0)
    return compute_metrics(labels_np, probs_np), timer.elapsed


def main() -> None:
    parser = argparse.ArgumentParser(description="HuBERT PTB-XL activation approximation inference benchmark.")
    add_common_args(parser)
    parser.add_argument("--weights", type=Path, default=APP_DIR / "hubert_ecg_ptbxl.pth")
    parser.add_argument("--ptbxl-dir", type=Path, default=APP_DIR / "ptb-xl-1.0.3")
    parser.add_argument("--split", choices=("test", "val", "train"), default="test")
    parser.add_argument("--activations", nargs="+", choices=ACTIVATIONS, default=list(ACTIVATIONS))
    parser.add_argument("--schemes", nargs="+", choices=SCHEMES, default=list(SCHEMES))
    parser.add_argument("--dtypes", nargs="+", choices=DTYPES, default=list(DTYPES))
    args = parser.parse_args()

    device = require_device(args.device, args.allow_cpu)
    loader = make_loader(args)
    base_model = load_local_model(args.weights)
    baseline_metrics, baseline_time = evaluate(base_model, loader, device, "exact GELU + exact sigmoid")
    baseline_acc = baseline_metrics["macro_acc"]
    baseline_auroc = baseline_metrics["macro_auroc"]
    baseline_f1 = baseline_metrics["macro_f1"]

    rows = [{
        "reference": "trained",
        "activation": "gelu/sigmoid",
        "scheme": "exact_fp32",
        "dtype": "fp32",
        "macro_auroc": baseline_auroc,
        "macro_acc": baseline_acc,
        "macro_f1": baseline_f1,
        "delta_acc_pp": 0.0,
        "latency_s": baseline_time,
        "patched_gelu": 0,
        "status": "TRAINED_REF",
    }]

    activation_refs = {}
    for activation in args.activations:
        model = copy.deepcopy(base_model)
        patched = replace_gelu_exact(model, activation)
        output_sigmoid = ExactActivation("sigmoid").to(device) if activation == "sigmoid" else None
        metrics, latency = evaluate(model, loader, device, f"exact {activation} fp32", output_sigmoid)
        activation_refs[activation] = metrics["macro_acc"]
        rows.append({
            "reference": activation,
            "activation": activation,
            "scheme": "exact_fp32",
            "dtype": "fp32",
            "macro_auroc": metrics["macro_auroc"],
            "macro_acc": metrics["macro_acc"],
            "macro_f1": metrics["macro_f1"],
            "delta_acc_pp": 0.0,
            "latency_s": latency,
            "patched_gelu": patched,
            "status": "ACT_REF",
        })

    for activation, scheme, dtype in selected_variants(args):
        model = copy.deepcopy(base_model)
        patched = replace_gelu(model, activation, scheme, dtype)
        output_sigmoid = None
        if activation == "sigmoid":
            output_sigmoid = ApproxActivation("sigmoid", scheme, dtype).to(device)
        metrics, latency = evaluate(model, loader, device, f"{activation} {scheme} {dtype}", output_sigmoid)
        delta = metrics["macro_acc"] - activation_refs[activation]
        rows.append({
            "reference": activation,
            "activation": activation,
            "scheme": scheme,
            "dtype": dtype,
            "macro_auroc": metrics["macro_auroc"],
            "macro_acc": metrics["macro_acc"],
            "macro_f1": metrics["macro_f1"],
            "delta_acc_pp": delta,
            "latency_s": latency,
            "patched_gelu": patched,
            "status": "PASS" if delta >= -args.tolerance_pp else "DROP>1pp",
        })

    table_rows = [
        [
            r["reference"],
            r["activation"],
            r["scheme"],
            r["dtype"],
            f'{r["macro_auroc"]:.4f}',
            f'{r["macro_acc"]:.4f}%',
            f'{r["macro_f1"]:.4f}',
            f'{r["delta_acc_pp"]:+.4f}',
            str(r["patched_gelu"]),
            f'{r["latency_s"]:.2f}',
            r["status"],
        ]
        for r in rows
    ]
    print("\nHuBERT PTB-XL activation approximation results")
    print(f"Device: {device} | Split: {args.split} | Samples: {len(loader.dataset)} | Classes: {', '.join(SUPERCLASSES)}")
    print("The trained reference is original FP32 inference with no approximate/custom unit.")
    print("Each activation has its own exact FP32 ACT_REF; approximate rows are compared against that matching activation reference.")
    print("Only activation functions are replaced; weights, linear layers, convolutions and attention remain FP32.")
    print("Tanh rows approximate GELU's tanh form. Sigmoid rows approximate GELU's sigmoid form and the final multi-label sigmoid.")
    print(format_table(["Reference", "Activation", "Act Unit", "DType", "AUROC", "Macro Acc", "Macro F1", "Delta pp", "GELUs", "Seconds", "Status"], table_rows))
    if args.csv:
        write_csv(args.csv, rows)


if __name__ == "__main__":
    main()
