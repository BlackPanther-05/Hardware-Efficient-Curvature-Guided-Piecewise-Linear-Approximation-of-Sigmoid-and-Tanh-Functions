from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from approx_activation_lib import (  # noqa: E402
    ACTIVATIONS,
    DTYPES,
    SCHEMES,
    ApproxReLUReplacement,
    CudaTimer,
    ExactActivation,
    add_common_args,
    format_table,
    require_device,
    selected_variants,
    write_csv,
)


APP_DIR = Path(__file__).resolve().parent


class Cifar100PickleDataset(Dataset):
    def __init__(self, root: Path, split: str, img_size: int = 32) -> None:
        batch_path = root / "cifar-100-python" / split
        with batch_path.open("rb") as fh:
            batch = pickle.load(fh, encoding="latin1")
        data = batch["data"].reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
        images = torch.from_numpy(data)
        if img_size != 32:
            images = torch.nn.functional.interpolate(
                images,
                size=(img_size, img_size),
                mode="bilinear",
                align_corners=False,
            )
        self.images = (images - 0.5) / 0.5
        self.labels = torch.tensor(batch["fine_labels"], dtype=torch.long)

    def __len__(self) -> int:
        return int(self.labels.numel())

    def __getitem__(self, idx: int):
        return self.images[idx], self.labels[idx]


class VGG11CIFAR100(nn.Module):
    def __init__(self, num_classes: int = 100) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=False)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(64, 128, 3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU(inplace=False)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(128, 256, 3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.relu3 = nn.ReLU(inplace=False)

        self.conv4 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.relu4 = nn.ReLU(inplace=False)
        self.pool4 = nn.MaxPool2d(2, 2)

        self.conv5 = nn.Conv2d(256, 512, 3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(512)
        self.relu5 = nn.ReLU(inplace=False)

        self.conv6 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.bn6 = nn.BatchNorm2d(512)
        self.relu6 = nn.ReLU(inplace=False)
        self.pool6 = nn.MaxPool2d(2, 2)

        self.conv7 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.bn7 = nn.BatchNorm2d(512)
        self.relu7 = nn.ReLU(inplace=False)

        self.conv8 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.bn8 = nn.BatchNorm2d(512)
        self.relu8 = nn.ReLU(inplace=False)
        self.pool8 = nn.MaxPool2d(2, 2)

        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(512, 4096)
        self.relu_fc1 = nn.ReLU(inplace=False)
        self.fc2 = nn.Linear(4096, 4096)
        self.relu_fc2 = nn.ReLU(inplace=False)
        self.fc3 = nn.Linear(4096, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.pool4(self.relu4(self.bn4(self.conv4(x))))
        x = self.relu5(self.bn5(self.conv5(x)))
        x = self.pool6(self.relu6(self.bn6(self.conv6(x))))
        x = self.relu7(self.bn7(self.conv7(x)))
        x = self.pool8(self.relu8(self.bn8(self.conv8(x))))
        x = self.gap(x)
        x = self.flatten(x)
        x = self.relu_fc1(self.fc1(x))
        x = self.relu_fc2(self.fc2(x))
        return self.fc3(x)


RELU_NAMES = ("relu1", "relu2", "relu3", "relu4", "relu5", "relu6", "relu7", "relu8", "relu_fc1", "relu_fc2")
NUM_ACT_SITES = len(RELU_NAMES)  # 10


def replace_relu_modules(model: nn.Module, activation: nn.Module | list[nn.Module]) -> None:
    """Replace all ReLU modules with custom activations.

    - Single nn.Module (proposed): shared across all sites.
    - List of nn.Module (baseline): one per site — each layer gets its own LUT copy.
    """
    if isinstance(activation, (list, nn.ModuleList)):
        assert len(activation) == NUM_ACT_SITES, (
            f"Expected {NUM_ACT_SITES} activations, got {len(activation)}"
        )
        for name, act in zip(RELU_NAMES, activation):
            setattr(model, name, act)
    else:
        for name in RELU_NAMES:
            setattr(model, name, activation)


def load_model(
    weights: Path,
    device: torch.device,
    activation: nn.Module | list[nn.Module] | None = None,
) -> nn.Module:
    model = VGG11CIFAR100(num_classes=100)
    state = torch.load(weights, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    if activation is not None:
        replace_relu_modules(model, activation)
    return model.to(device).eval()


def get_loader(split: str, batch_size: int, num_workers: int, max_samples: int | None, img_size: int) -> DataLoader:
    dataset = Cifar100PickleDataset(APP_DIR / "cifar100", split, img_size=img_size)
    if max_samples is not None:
        dataset = Subset(dataset, range(min(max_samples, len(dataset))))
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, desc: str) -> tuple[float, float]:
    correct = 0
    total = 0
    with CudaTimer(device) as timer:
        for images, labels in tqdm(loader, desc=desc, leave=False, ncols=100):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            pred = model(images).argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += labels.numel()
    return 100.0 * correct / max(total, 1), timer.elapsed


def main() -> None:
    parser = argparse.ArgumentParser(description="VGG11 CIFAR-100 activation approximation inference benchmark.")
    add_common_args(parser)
    parser.add_argument("--weights", type=Path, default=APP_DIR / "vgg11_cifar100_1.pth")
    parser.add_argument("--split", choices=("test", "train"), default="test")
    parser.add_argument("--img-size", type=int, default=32)
    parser.add_argument("--activations", nargs="+", choices=ACTIVATIONS, default=list(ACTIVATIONS))
    parser.add_argument("--schemes", nargs="+", choices=SCHEMES, default=list(SCHEMES))
    parser.add_argument("--dtypes", nargs="+", choices=DTYPES, default=list(DTYPES))
    args = parser.parse_args()

    device = require_device(args.device, args.allow_cpu)
    loader = get_loader(args.split, args.batch_size, args.num_workers, args.max_samples, args.img_size)
    baseline = load_model(args.weights, device)
    baseline_acc, baseline_time = evaluate(baseline, loader, device, "exact ReLU")

    rows = [{
        "reference": "trained",
        "activation": "relu",
        "scheme": "exact_fp32",
        "dtype": "fp32",
        "accuracy": baseline_acc,
        "delta_pp": 0.0,
        "latency_s": baseline_time,
        "status": "TRAINED_REF",
    }]

    activation_refs = {}
    for activation in args.activations:
        exact_act = ExactActivation(activation).to(device)
        exact_model = load_model(args.weights, device, activation=exact_act)
        exact_acc, exact_latency = evaluate(exact_model, loader, device, f"exact {activation} fp32")
        activation_refs[activation] = exact_acc
        rows.append({
            "reference": activation,
            "activation": activation,
            "scheme": "exact_fp32",
            "dtype": "fp32",
            "accuracy": exact_acc,
            "delta_pp": 0.0,
            "latency_s": exact_latency,
            "status": "ACT_REF",
        })

    for activation, scheme, dtype in selected_variants(args):
        # Baseline: each ReLU site gets its OWN LUT copy (embedded per layer)
        # Proposed: all sites share ONE LUT instance (single shared block)
        if scheme == "proposed":
            act = ApproxReLUReplacement(activation, scheme, dtype).to(device)
        else:
            # Baseline: create a separate LUT copy per activation site
            act = [ApproxReLUReplacement(activation, scheme, dtype).to(device) for _ in range(NUM_ACT_SITES)]
        model = load_model(args.weights, device, activation=act)
        acc, latency = evaluate(model, loader, device, f"{activation} {scheme} {dtype}")
        delta = acc - activation_refs[activation]
        rows.append({
            "reference": activation,
            "activation": activation,
            "scheme": scheme,
            "dtype": dtype,
            "accuracy": acc,
            "delta_pp": delta,
            "latency_s": latency,
            "status": "PASS" if delta >= -args.tolerance_pp else "DROP>1pp",
        })

    table_rows = [
        [r["reference"], r["activation"], r["scheme"], r["dtype"], f'{r["accuracy"]:.4f}%', f'{r["delta_pp"]:+.4f}', f'{r["latency_s"]:.2f}', r["status"]]
        for r in rows
    ]
    print("\nVGG11 CIFAR-100 activation approximation results")
    print(f"Device: {device} | Split: {args.split} | Samples: {len(loader.dataset)} | Image size: {args.img_size}")
    print("Preprocessing: Resize then ToTensor then Normalize(mean=0.5, std=0.5), matching training.")
    print("The trained reference is original FP32 inference with no approximate/custom unit.")
    print("Each activation has its own exact FP32 ACT_REF; approximate rows are compared against that matching activation reference.")
    print("Only the activation function is replaced; conv, batchnorm, pooling and linear layers remain FP32.")
    print(format_table(["Reference", "Activation", "Act Unit", "DType", "Accuracy", "Delta pp", "Seconds", "Status"], table_rows))
    if args.csv:
        write_csv(args.csv, rows)


if __name__ == "__main__":
    main()
