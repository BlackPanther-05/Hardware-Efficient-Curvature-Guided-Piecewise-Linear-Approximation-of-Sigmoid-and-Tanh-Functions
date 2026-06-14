from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
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


NUM_ACT_SITES = 5  # conv1, conv2, conv3, fc1, fc2


class LeNet5_3Conv(nn.Module):
    def __init__(self, activation: nn.Module | list[nn.Module] | None = None) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.conv3 = nn.Conv2d(16, 32, 3)
        self.fc1 = nn.Linear(32 * 1 * 1, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        # Accept either a single nn.Module (proposed: shared across all sites)
        # or a list of nn.Module (baseline: one per activation site).
        if activation is None:
            self.activations = None
        elif isinstance(activation, (list, nn.ModuleList)):
            assert len(activation) == NUM_ACT_SITES, (
                f"Expected {NUM_ACT_SITES} activations, got {len(activation)}"
            )
            self.activations = nn.ModuleList(activation)
        else:
            # Single shared instance — reference it for every site
            self.activations = nn.ModuleList([activation] * NUM_ACT_SITES)

    def _act(self, x: torch.Tensor, idx: int) -> torch.Tensor:
        if self.activations is None:
            return F.relu(x)
        return self.activations[idx](x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._act(self.conv1(x), 0)
        x = F.max_pool2d(x, 2)
        x = self._act(self.conv2(x), 1)
        x = F.max_pool2d(x, 2)
        x = self._act(self.conv3(x), 2)
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = self._act(self.fc1(x), 3)
        x = self._act(self.fc2(x), 4)
        return self.fc3(x)


def load_model(
    weights: Path,
    device: torch.device,
    activation: nn.Module | list[nn.Module] | None = None,
) -> nn.Module:
    model = LeNet5_3Conv()
    state = torch.load(weights, map_location="cpu")
    model.load_state_dict(state)
    # Re-construct with custom activations after loading weights
    if activation is not None:
        if isinstance(activation, (list, nn.ModuleList)):
            assert len(activation) == NUM_ACT_SITES
            model.activations = nn.ModuleList(activation)
        else:
            model.activations = nn.ModuleList([activation] * NUM_ACT_SITES)
    return model.to(device).eval()


def get_loader(split: str, batch_size: int, num_workers: int, max_samples: int | None) -> DataLoader:
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    dataset = datasets.MNIST(root=str(APP_DIR / "data"), train=False, download=False, transform=transform)
    indices = range(6000) if split == "test" else range(6000, 10000)
    if max_samples is not None:
        indices = list(indices)[:max_samples]
    return DataLoader(
        Subset(dataset, indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )


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
    parser = argparse.ArgumentParser(description="LeNet-5 MNIST activation approximation inference benchmark.")
    add_common_args(parser)
    parser.add_argument("--weights", type=Path, default=APP_DIR / "lenet5_3conv_weights.pth")
    parser.add_argument("--split", choices=("test", "val"), default="test")
    parser.add_argument("--activations", nargs="+", choices=ACTIVATIONS, default=list(ACTIVATIONS))
    parser.add_argument("--schemes", nargs="+", choices=SCHEMES, default=list(SCHEMES))
    parser.add_argument("--dtypes", nargs="+", choices=DTYPES, default=list(DTYPES))
    args = parser.parse_args()

    device = require_device(args.device, args.allow_cpu)
    loader = get_loader(args.split, args.batch_size, args.num_workers, args.max_samples)
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
        # Baseline: each activation site gets its OWN LUT copy (embedded per layer)
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
    print("\nLeNet-5 MNIST activation approximation results")
    print(f"Device: {device} | Split: {args.split} | Samples: {len(loader.dataset)}")
    print("The trained reference is original FP32 inference with no approximate/custom unit.")
    print("Each activation has its own exact FP32 ACT_REF; approximate rows are compared against that matching activation reference.")
    print("Only the activation function is replaced; conv, pooling and linear layers remain FP32.")
    print(format_table(["Reference", "Activation", "Act Unit", "DType", "Accuracy", "Delta pp", "Seconds", "Status"], table_rows))
    if args.csv:
        write_csv(args.csv, rows)


if __name__ == "__main__":
    main()
