from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset


DEFAULT_LABEL_COLUMNS = ("Label", "label", "Class", "class", "Attack", "attack")
DEFAULT_DROP_COLUMNS = (
    "Flow ID",
    "Src IP",
    "Src Port",
    "Dst IP",
    "Dst Port",
    "Protocol",
    "Timestamp",
)


def discover_data_files(data_dir: Path | None, csv_files: Sequence[Path] | None = None) -> List[Path]:
    files: List[Path] = []
    if csv_files:
        files.extend(Path(p) for p in csv_files)
    if data_dir:
        files.extend(sorted(data_dir.glob("*.csv")))
        files.extend(sorted(data_dir.glob("*.parquet")))
    unique = []
    seen = set()
    for file in files:
        resolved = file.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(file)
    if not unique:
        raise FileNotFoundError("No CIC-IDS2018 CSV/parquet files found. Pass --data-dir or --csv.")
    return unique


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, low_memory=False)


def load_cic_dataframe(
    files: Sequence[Path],
    *,
    max_rows_per_file: int | None = None,
    sample_frac: float | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    frames = []
    for path in files:
        frame = read_table(path)
        frame.columns = [str(col).strip() for col in frame.columns]
        if max_rows_per_file is not None and len(frame) > max_rows_per_file:
            frame = frame.sample(n=max_rows_per_file, random_state=seed)
        elif sample_frac is not None and 0.0 < sample_frac < 1.0:
            frame = frame.sample(frac=sample_frac, random_state=seed)
        frames.append(frame)
    if not frames:
        raise ValueError("No dataframes were loaded.")
    return pd.concat(frames, ignore_index=True)


def find_label_column(df: pd.DataFrame, label_column: str | None = None) -> str:
    if label_column:
        if label_column not in df.columns:
            raise ValueError(f"Label column '{label_column}' not found.")
        return label_column
    for column in DEFAULT_LABEL_COLUMNS:
        if column in df.columns:
            return column
    raise ValueError(f"Could not infer label column. Tried: {', '.join(DEFAULT_LABEL_COLUMNS)}")


def make_binary_labels(labels: pd.Series) -> pd.Series:
    normalized = labels.astype(str).str.strip()
    return normalized.apply(lambda value: "BENIGN" if value.upper() == "BENIGN" else "ATTACK")


def encode_labels(labels: pd.Series) -> tuple[np.ndarray, List[str]]:
    classes = sorted(labels.astype(str).unique().tolist())
    mapping = {label: idx for idx, label in enumerate(classes)}
    encoded = labels.astype(str).map(mapping).to_numpy(dtype=np.int64)
    return encoded, classes


def prepare_features(
    df: pd.DataFrame,
    label_column: str,
    *,
    feature_columns: Sequence[str] | None = None,
    drop_columns: Iterable[str] = DEFAULT_DROP_COLUMNS,
) -> tuple[pd.DataFrame, pd.Series]:
    labels = df[label_column].copy()
    if feature_columns:
        missing = [col for col in feature_columns if col not in df.columns]
        if missing:
            raise ValueError(f"Missing expected feature columns: {missing[:8]}")
        features = df[list(feature_columns)].copy()
    else:
        drop = {label_column, *drop_columns}
        candidates = [col for col in df.columns if col not in drop]
        features = df[candidates].copy()
        numeric_columns = []
        for col in features.columns:
            converted = pd.to_numeric(features[col], errors="coerce")
            if converted.notna().any():
                features[col] = converted
                numeric_columns.append(col)
        features = features[numeric_columns]

    if features.empty:
        raise ValueError("No numeric feature columns remain after preprocessing.")
    features = features.replace([np.inf, -np.inf], np.nan)
    features = features.fillna(features.median(numeric_only=True)).fillna(0.0)
    return features.astype(np.float32), labels


def standardize_features(
    features: pd.DataFrame,
    *,
    mean: np.ndarray | None = None,
    std: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = features.to_numpy(dtype=np.float32)
    if mean is None:
        mean = values.mean(axis=0)
    if std is None:
        std = values.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    values = (values - mean) / std
    return values.astype(np.float32), mean.astype(np.float32), std.astype(np.float32)


def make_sequences(features: np.ndarray, labels: np.ndarray, sequence_length: int) -> tuple[torch.Tensor, torch.Tensor]:
    if sequence_length < 1:
        raise ValueError("sequence_length must be >= 1")
    usable = (len(features) // sequence_length) * sequence_length
    if usable == 0:
        raise ValueError("Not enough rows to create one sequence.")
    features = features[:usable]
    labels = labels[:usable]
    x = features.reshape(-1, sequence_length, features.shape[1])
    y = labels.reshape(-1, sequence_length)[:, -1]
    return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        json.dump(payload, fh, indent=2)


def load_json(path: Path) -> dict:
    with path.open() as fh:
        return json.load(fh)


class TensorSequenceDataset(Dataset):
    def __init__(self, x: torch.Tensor, y: torch.Tensor) -> None:
        self.x = x.float()
        self.y = y.long()

    def __len__(self) -> int:
        return int(self.y.numel())

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]


class LSTMCICClassifier(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_classes: int,
        *,
        num_layers: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.lstm(x)
        return self.classifier(output[:, -1, :])


class ActivationOnlyLSTMClassifier(nn.Module):
    """LSTM inference module where only gate tanh/sigmoid functions are swappable.

    Accepts either a single nn.Module (shared across all layers — proposed scheme)
    or a list of nn.Module (one per layer — baseline scheme with per-layer LUT copies).
    """

    def __init__(
        self,
        trained_model: LSTMCICClassifier,
        *,
        sigmoid_activation: nn.Module | List[nn.Module],
        tanh_activation: nn.Module | List[nn.Module],
    ) -> None:
        super().__init__()
        self.hidden_size = trained_model.lstm.hidden_size
        self.num_layers = trained_model.lstm.num_layers
        if trained_model.lstm.bidirectional:
            raise ValueError("ActivationOnlyLSTMClassifier supports unidirectional LSTMs only.")

        # Per-layer activation lists: baseline gives each layer its own LUT copy,
        # proposed shares one LUT instance across all layers.
        self.sigmoid_activations = self._wrap_activations(sigmoid_activation)
        self.tanh_activations = self._wrap_activations(tanh_activation)

        self.classifier = nn.Linear(
            trained_model.classifier.in_features,
            trained_model.classifier.out_features,
        )
        self.classifier.load_state_dict(trained_model.classifier.state_dict())

        for layer_idx in range(self.num_layers):
            for name in ("weight_ih", "weight_hh", "bias_ih", "bias_hh"):
                param = getattr(trained_model.lstm, f"{name}_l{layer_idx}").detach().clone()
                self.register_buffer(f"{name}_l{layer_idx}", param)

    def _wrap_activations(self, act: nn.Module | List[nn.Module]) -> nn.ModuleList:
        if isinstance(act, (list, nn.ModuleList)):
            if len(act) != self.num_layers:
                raise ValueError(
                    f"Expected {self.num_layers} activations (one per layer), got {len(act)}"
                )
            return nn.ModuleList(act)
        # Single shared instance — reference it for every layer
        return nn.ModuleList([act] * self.num_layers)

    def _layer_forward(self, x: torch.Tensor, layer_idx: int) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        h = x.new_zeros(batch_size, self.hidden_size)
        c = x.new_zeros(batch_size, self.hidden_size)
        weight_ih = getattr(self, f"weight_ih_l{layer_idx}")
        weight_hh = getattr(self, f"weight_hh_l{layer_idx}")
        bias_ih = getattr(self, f"bias_ih_l{layer_idx}")
        bias_hh = getattr(self, f"bias_hh_l{layer_idx}")
        sig = self.sigmoid_activations[layer_idx]
        tanh = self.tanh_activations[layer_idx]
        outputs = []
        for step in range(seq_len):
            gates = x[:, step, :].matmul(weight_ih.t()) + bias_ih + h.matmul(weight_hh.t()) + bias_hh
            i_gate, f_gate, g_gate, o_gate = gates.chunk(4, dim=1)
            i_gate = sig(i_gate)
            f_gate = sig(f_gate)
            g_gate = tanh(g_gate)
            o_gate = sig(o_gate)
            c = f_gate * c + i_gate * g_gate
            h = o_gate * tanh(c)
            outputs.append(h.unsqueeze(1))
        return torch.cat(outputs, dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer_idx in range(self.num_layers):
            x = self._layer_forward(x, layer_idx)
        return self.classifier(x[:, -1, :])
