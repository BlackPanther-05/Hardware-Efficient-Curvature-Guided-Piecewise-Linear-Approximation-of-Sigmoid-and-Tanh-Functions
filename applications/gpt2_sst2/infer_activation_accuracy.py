from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

APP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(APP_DIR.parent))

from approx_activation_lib import (  # noqa: E402
    ACTIVATIONS,
    DTYPES,
    SCHEMES,
    ApproxGELU,
    CudaTimer,
    ExactGELU,
    add_common_args,
    format_table,
    require_device,
    selected_variants,
    write_csv,
)

# ─────────────────────────────────────────────────────────────
# Dataset configuration (mirrors training script)
# ─────────────────────────────────────────────────────────────
DATASET_CONFIG = {
    "sst2": {
        "task": "glue", "subset": "sst2", "num_classes": 2,
        "text_cols": ["sentence"], "label_col": "label",
        "metric": "accuracy", "regression": False,
    },
    "cola": {
        "task": "glue", "subset": "cola", "num_classes": 2,
        "text_cols": ["sentence"], "label_col": "label",
        "metric": "matthews_corrcoef", "regression": False,
    },
    "rte": {
        "task": "glue", "subset": "rte", "num_classes": 2,
        "text_cols": ["sentence1", "sentence2"], "label_col": "label",
        "metric": "accuracy", "regression": False,
    },
    "stsb": {
        "task": "glue", "subset": "stsb", "num_classes": 1,
        "text_cols": ["sentence1", "sentence2"], "label_col": "label",
        "metric": "pearson_spearman", "regression": True,
    },
    "qqp": {
        "task": "glue", "subset": "qqp", "num_classes": 2,
        "text_cols": ["question1", "question2"], "label_col": "label",
        "metric": "f1", "regression": False,
    },
    "mnli": {
        "task": "glue", "subset": "mnli", "num_classes": 3,
        "text_cols": ["premise", "hypothesis"], "label_col": "label",
        "metric": "accuracy", "regression": False,
    },
    "mrpc": {
        "task": "glue", "subset": "mrpc", "num_classes": 2,
        "text_cols": ["sentence1", "sentence2"], "label_col": "label",
        "metric": "f1", "regression": False,
    },
    "qnli": {
        "task": "glue", "subset": "qnli", "num_classes": 2,
        "text_cols": ["question", "sentence"], "label_col": "label",
        "metric": "accuracy", "regression": False,
    },
}


# ─────────────────────────────────────────────────────────────
# Dataset loading
# ─────────────────────────────────────────────────────────────
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=256, regression=False):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.regression = regression

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        enc = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        if self.regression:
            label_tensor = torch.tensor(label, dtype=torch.float32)
        else:
            label_tensor = torch.tensor(int(label), dtype=torch.long)
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": label_tensor,
        }


def load_val_dataset(name: str, tokenizer, max_length: int = 256) -> tuple:
    """Load validation split for a GLUE task."""
    config = DATASET_CONFIG[name]
    ds = load_dataset("glue", config["subset"])

    def get_text(example):
        if len(config["text_cols"]) == 1:
            return str(example[config["text_cols"][0]])
        return f"{example[config['text_cols'][0]]} [SEP] {example[config['text_cols'][1]]}"

    if name == "mnli":
        val_split = ds["validation_matched"]
    else:
        val_split = ds["validation"]

    val_texts = [get_text(ex) for ex in val_split]
    val_labels = [ex[config["label_col"]] for ex in val_split]

    # Filter out examples with -1 label
    val_data = [(t, l) for t, l in zip(val_texts, val_labels) if l != -1]
    val_texts, val_labels = zip(*val_data) if val_data else ([], [])

    dataset = TextDataset(val_texts, val_labels, tokenizer, max_length, config["regression"])
    return dataset, config


# ─────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────
def load_model(checkpoint_dir: Path) -> tuple[nn.Module, dict]:
    """Load GPT-2 model from a saved checkpoint (with or without LoRA)."""
    from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

    meta_path = checkpoint_dir / "metadata.json"
    if meta_path.exists():
        with meta_path.open() as f:
            metadata = json.load(f)
    else:
        metadata = {}

    model_name = metadata.get("model_name", "gpt2")
    num_classes = metadata.get("num_classes", 2)
    is_regression = metadata.get("is_regression", False)
    use_lora = metadata.get("use_lora", False)

    # Load tokenizer from checkpoint directory
    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint_dir))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Build base model configuration
    config = AutoConfig.from_pretrained(model_name)
    config.num_labels = 1 if is_regression else num_classes
    config.problem_type = "regression" if is_regression else "single_label_classification"
    config.pad_token_id = tokenizer.pad_token_id

    if use_lora:
        # Load base model, then apply LoRA adapter
        base_model = AutoModelForSequenceClassification.from_pretrained(
            model_name, config=config, ignore_mismatched_sizes=True
        )
        base_model.config.pad_token_id = tokenizer.pad_token_id
        from peft import PeftModel
        model = PeftModel.from_pretrained(base_model, str(checkpoint_dir))
        model = model.merge_and_unload()  # Merge LoRA weights into base model
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            str(checkpoint_dir), config=config, ignore_mismatched_sizes=True
        )
        model.config.pad_token_id = tokenizer.pad_token_id

    return model.eval(), tokenizer, metadata


# ─────────────────────────────────────────────────────────────
# GELU replacement — baseline vs proposed
# ─────────────────────────────────────────────────────────────
def _get_gelu_types() -> tuple:
    """Return all GELU module types to replace (nn.GELU + HuggingFace variants)."""
    gelu_types = [nn.GELU]
    try:
        from transformers.activations import GELUActivation
        gelu_types.append(GELUActivation)
    except ImportError:
        pass
    try:
        from transformers.activations import NewGELUActivation
        gelu_types.append(NewGELUActivation)
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


# ─────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────
def compute_metrics(predictions, labels, metric_type: str, regression: bool = False) -> dict:
    """Compute appropriate metrics based on task type."""
    if regression:
        pearson_val = pearsonr(predictions, labels)[0]
        spearman_val = spearmanr(predictions, labels)[0]
        return {
            "pearson": pearson_val * 100,
            "spearman": spearman_val * 100,
            "combined": ((pearson_val + spearman_val) / 2) * 100,
            "primary": ((pearson_val + spearman_val) / 2) * 100,
        }
    else:
        accuracy = accuracy_score(labels, predictions) * 100
        if metric_type == "matthews_corrcoef":
            mcc = matthews_corrcoef(labels, predictions) * 100
            return {"accuracy": accuracy, "matthews_corrcoef": mcc, "primary": mcc}
        elif metric_type == "f1":
            f1_val = f1_score(labels, predictions, average="binary") * 100
            return {"accuracy": accuracy, "f1": f1_val, "primary": f1_val}
        else:
            return {"accuracy": accuracy, "primary": accuracy}


# ─────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    desc: str,
    regression: bool = False,
) -> tuple[dict, float]:
    """Evaluate model on a dataset and return (metrics_dict, elapsed_seconds)."""
    model.to(device).eval()
    all_preds = []
    all_labels = []
    with CudaTimer(device) as timer:
        for batch in tqdm(loader, desc=desc, leave=False, ncols=110):
            ids = batch["input_ids"].to(device, non_blocking=True)
            mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"]
            outputs = model(input_ids=ids, attention_mask=mask)
            logits = outputs.logits
            if regression:
                all_preds.extend(logits.squeeze(-1).cpu().numpy().tolist())
            else:
                all_preds.extend(logits.argmax(dim=1).cpu().numpy().tolist())
            all_labels.extend(labels.numpy().tolist())
    return all_preds, all_labels, timer.elapsed


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="GPT-2 GLUE activation approximation inference benchmark."
    )
    add_common_args(parser)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=APP_DIR / "gpt2_sst2_best_lora",
        help="Path to saved model checkpoint directory.",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        choices=list(DATASET_CONFIG.keys()),
        help="GLUE dataset name. Auto-detected from metadata if omitted.",
    )
    parser.add_argument("--max-length", type=int, default=256)
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

    # ── Load model and tokenizer ──
    base_model, tokenizer, metadata = load_model(args.checkpoint)

    # Determine dataset
    dataset_name = args.dataset or metadata.get("dataset", "sst2")
    ds_config = DATASET_CONFIG[dataset_name]
    metric_type = ds_config["metric"]
    is_regression = ds_config["regression"]

    print(f"\nModel: {metadata.get('model_name', 'gpt2')} | Dataset: {dataset_name.upper()}")
    print(f"LoRA: {metadata.get('use_lora', False)} | Metric: {metric_type}")
    print(f"Training best metric: {metadata.get('best_metric', 'N/A')}")

    # ── Load validation data ──
    val_ds, _ = load_val_dataset(dataset_name, tokenizer, args.max_length)
    if args.max_samples is not None:
        val_ds = Subset(val_ds, range(min(args.max_samples, len(val_ds))))
    loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # ── 1. Baseline: original FP32 (no activation replacement) ──
    baseline_preds, baseline_labels, baseline_time = evaluate(
        base_model, loader, device, "baseline FP32 (original GELU)", is_regression
    )
    baseline_metrics = compute_metrics(baseline_preds, baseline_labels, metric_type, is_regression)
    baseline_primary = baseline_metrics["primary"]
    print(
        f"\nBaseline FP32 | primary={baseline_primary:.4f} | "
        f"time={baseline_time:.2f}s"
    )
    for k, v in baseline_metrics.items():
        if k != "primary":
            print(f"  {k}: {v:.4f}")

    rows = [
        {
            "reference": "trained",
            "activation": "gelu_new",
            "scheme": "exact_fp32",
            "dtype": "fp32",
            "primary_metric": baseline_primary,
            "delta_pp": 0.0,
            "latency_s": baseline_time,
            "patched_gelu": 0,
            "status": "TRAINED_REF",
        }
    ]

    # ── 2. Exact FP32 activation references ──
    activation_refs: dict[str, float] = {}
    for activation in args.activations:
        model = copy.deepcopy(base_model)
        patched = replace_gelu_exact(model, activation)
        preds, labels, latency = evaluate(
            model, loader, device, f"exact {activation} fp32", is_regression
        )
        metrics = compute_metrics(preds, labels, metric_type, is_regression)
        activation_refs[activation] = metrics["primary"]
        rows.append(
            {
                "reference": activation,
                "activation": activation,
                "scheme": "exact_fp32",
                "dtype": "fp32",
                "primary_metric": metrics["primary"],
                "delta_pp": 0.0,
                "latency_s": latency,
                "patched_gelu": patched,
                "status": "ACT_REF",
            }
        )

    # ── 3. Approximate activation variants ──
    for activation, scheme, dtype in selected_variants(args):
        model = copy.deepcopy(base_model)
        patched = replace_gelu(model, activation, scheme, dtype)
        preds, labels, latency = evaluate(
            model, loader, device, f"{activation} {scheme} {dtype}", is_regression
        )
        metrics = compute_metrics(preds, labels, metric_type, is_regression)
        delta = metrics["primary"] - activation_refs[activation]
        rows.append(
            {
                "reference": activation,
                "activation": activation,
                "scheme": scheme,
                "dtype": dtype,
                "primary_metric": metrics["primary"],
                "delta_pp": delta,
                "latency_s": latency,
                "patched_gelu": patched,
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
            f'{r["primary_metric"]:.4f}',
            f'{r["delta_pp"]:+.4f}',
            str(r["patched_gelu"]),
            f'{r["latency_s"]:.2f}',
            r["status"],
        ]
        for r in rows
    ]
    print(f"\nGPT-2 {dataset_name.upper()} activation approximation results")
    print(
        f"Device: {device} | Samples: {len(loader.dataset)} | "
        f"Metric: {metric_type} | Regression: {is_regression}"
    )
    print(
        "The trained reference is original FP32 inference with native GPT-2 NewGELUActivation."
    )
    print(
        "Each activation family has its own exact FP32 ACT_REF (manual GELU formulation)."
    )
    print(
        "Approximate rows compare against the matching activation family ACT_REF."
    )
    print(
        "Tanh rows approximate GELU's tanh form. "
        "Sigmoid rows approximate GELU's sigmoid form."
    )
    print(
        "Baseline: each GELU site gets its OWN LUT copy. "
        "Proposed: all sites share ONE LUT."
    )
    print(
        "Only activation functions are replaced; "
        "attention, embeddings, linear layers, and LayerNorm remain FP32."
    )
    print(
        format_table(
            [
                "Reference",
                "Activation",
                "Act Unit",
                "DType",
                f"Primary ({metric_type})",
                "Delta pp",
                "GELUs",
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
