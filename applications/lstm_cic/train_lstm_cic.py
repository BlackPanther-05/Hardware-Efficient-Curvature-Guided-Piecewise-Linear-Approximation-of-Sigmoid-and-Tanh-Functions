from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from cic_lstm_common import (
    LSTMCICClassifier,
    TensorSequenceDataset,
    discover_data_files,
    encode_labels,
    find_label_column,
    load_cic_dataframe,
    make_binary_labels,
    make_sequences,
    prepare_features,
    save_json,
    standardize_features,
)


APP_DIR = Path(__file__).resolve().parent


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def choose_device(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return torch.device(name)


def stratified_split_indices(y: torch.Tensor, test_size: float, seed: int):
    indices = np.arange(len(y))
    labels = y.cpu().numpy()
    unique, counts = np.unique(labels, return_counts=True)
    stratify = labels if len(unique) > 1 and counts.min() >= 2 else None
    return train_test_split(indices, test_size=test_size, random_state=seed, stratify=stratify)


def make_loader(dataset: TensorSequenceDataset, batch_size: int, num_workers: int, weighted: bool) -> DataLoader:
    sampler = None
    shuffle = True
    if weighted:
        labels = dataset.y.numpy()
        classes, counts = np.unique(labels, return_counts=True)
        weights = {cls: 1.0 / count for cls, count in zip(classes, counts)}
        sample_weights = torch.tensor([weights[label] for label in labels], dtype=torch.double)
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
        shuffle = False
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
    )


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float, float]:
    model.eval()
    losses = []
    preds_all = []
    labels_all = []
    criterion = nn.CrossEntropyLoss()
    for x, y in loader:
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
    return float(np.mean(losses)), acc, macro_f1


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a normal FP32 LSTM on local CIC-IDS2018 flow data.")
    parser.add_argument("--data-dir", type=Path, default=APP_DIR / "data", help="Directory containing CIC CSV/parquet files.")
    parser.add_argument("--csv", type=Path, nargs="*", default=None, help="Explicit CIC CSV/parquet files.")
    parser.add_argument("--label-column", default=None)
    parser.add_argument("--task", choices=("binary", "multiclass"), default="binary")
    parser.add_argument("--sequence-length", type=int, default=8)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--val-size", type=float, default=0.1)
    parser.add_argument("--max-rows-per-file", type=int, default=None)
    parser.add_argument("--sample-frac", type=float, default=None)
    parser.add_argument("--weighted-sampler", action="store_true")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=APP_DIR / "artifacts")
    args = parser.parse_args()

    set_seed(args.seed)
    device = choose_device(args.device)
    files = discover_data_files(args.data_dir, args.csv)
    df = load_cic_dataframe(
        files,
        max_rows_per_file=args.max_rows_per_file,
        sample_frac=args.sample_frac,
        seed=args.seed,
    )
    label_column = find_label_column(df, args.label_column)
    features_df, labels_raw = prepare_features(df, label_column)
    labels_for_encoding = make_binary_labels(labels_raw) if args.task == "binary" else labels_raw.astype(str).str.strip()
    labels, classes = encode_labels(labels_for_encoding)
    features, mean, std = standardize_features(features_df)
    x_seq, y_seq = make_sequences(features, labels, args.sequence_length)

    train_idx, test_idx = stratified_split_indices(y_seq, args.test_size, args.seed)
    train_labels = y_seq[train_idx]
    train_sub_idx, val_sub_idx = stratified_split_indices(train_labels, args.val_size, args.seed)
    train_idx_final = train_idx[train_sub_idx]
    val_idx = train_idx[val_sub_idx]

    train_ds = TensorSequenceDataset(x_seq[train_idx_final], y_seq[train_idx_final])
    val_ds = TensorSequenceDataset(x_seq[val_idx], y_seq[val_idx])
    test_ds = TensorSequenceDataset(x_seq[test_idx], y_seq[test_idx])
    train_loader = make_loader(train_ds, args.batch_size, args.num_workers, args.weighted_sampler)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    model = LSTMCICClassifier(
        input_size=x_seq.shape[-1],
        hidden_size=args.hidden_size,
        num_classes=len(classes),
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=3, factor=0.5)

    best_state = None
    best_val_acc = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for x, y in tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}", ncols=100):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            running_loss += loss.item()

        val_loss, val_acc, val_f1 = evaluate(model, val_loader, device)
        scheduler.step(val_acc)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
        print(
            f"Epoch {epoch:03d} | train_loss={running_loss / max(len(train_loader), 1):.4f} "
            f"| val_loss={val_loss:.4f} | val_acc={val_acc:.4f}% | val_macro_f1={val_f1:.4f}"
        )

    if best_state is not None:
        model.load_state_dict(best_state)
    test_loss, test_acc, test_f1 = evaluate(model, test_loader, device)
    print(f"\nBest FP32 test results | loss={test_loss:.4f} | accuracy={test_acc:.4f}% | macro_f1={test_f1:.4f}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "task": args.task,
        "classes": classes,
        "label_column": label_column,
        "feature_columns": list(features_df.columns),
        "feature_mean": mean.tolist(),
        "feature_std": std.tolist(),
        "sequence_length": args.sequence_length,
        "input_size": int(x_seq.shape[-1]),
        "hidden_size": args.hidden_size,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "num_classes": len(classes),
        "train_files": [str(path) for path in files],
    }
    checkpoint = {
        "model_state": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "metadata": metadata,
        "test_accuracy": test_acc,
        "test_macro_f1": test_f1,
    }
    torch.save(checkpoint, args.output_dir / "lstm_cic_fp32.pth")
    torch.save({"x": test_ds.x, "y": test_ds.y}, args.output_dir / "test_sequences.pt")
    save_json(args.output_dir / "preprocess.json", metadata)
    print(f"Saved checkpoint: {args.output_dir / 'lstm_cic_fp32.pth'}")
    print(f"Saved test tensors: {args.output_dir / 'test_sequences.pt'}")


if __name__ == "__main__":
    main()
