# scripts/benchmark.py
"""
Throughput and latency benchmarking script for LightningMasterPro.

Measures:
- Samples/second throughput (train step, forward pass)
- Memory usage (peak GPU/CPU)
- Average forward + backward pass time

Usage::
    python scripts/benchmark.py \\
        --config configs/vision/classifier.yaml \\
        [--batch_sizes 32 64 128] \\
        [--n_warmup 10] [--n_steps 100]
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark LightningMasterPro model throughput"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to a YAML config file.",
    )
    parser.add_argument(
        "--batch_sizes",
        type=int,
        nargs="+",
        default=[32, 64, 128],
        help="List of batch sizes to benchmark.",
    )
    parser.add_argument(
        "--input_shape",
        type=int,
        nargs="+",
        default=[3, 64, 64],
        help="Input shape (excluding batch dim), e.g. 3 64 64.",
    )
    parser.add_argument(
        "--n_warmup",
        type=int,
        default=10,
        help="Number of warmup steps before timing.",
    )
    parser.add_argument(
        "--n_steps",
        type=int,
        default=100,
        help="Number of timed steps per benchmark.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda", "mps"],
        help="Device to benchmark on.",
    )
    parser.add_argument(
        "--forward_only",
        action="store_true",
        help="Benchmark forward pass only (no backward).",
    )
    return parser.parse_args()


def get_device(device_str: str) -> torch.device:
    if device_str == "cuda" and not torch.cuda.is_available():
        print("[Warning] CUDA not available, falling back to CPU.")
        return torch.device("cpu")
    if device_str == "mps" and not torch.backends.mps.is_available():
        print("[Warning] MPS not available, falling back to CPU.")
        return torch.device("cpu")
    return torch.device(device_str)


def benchmark_batch(
    model: torch.nn.Module,
    device: torch.device,
    batch_size: int,
    input_shape: List[int],
    n_warmup: int,
    n_steps: int,
    forward_only: bool,
) -> Dict[str, float]:
    model = model.to(device)
    model.train()

    criterion = torch.nn.CrossEntropyLoss()

    def make_batch():
        x = torch.randn(batch_size, *input_shape, device=device)
        # Dummy label
        y = torch.zeros(batch_size, dtype=torch.long, device=device)
        return x, y

    # Warmup
    for _ in range(n_warmup):
        x, y = make_batch()
        with torch.no_grad() if forward_only else torch.enable_grad():
            out = model(x)
            if not forward_only:
                loss = criterion(out, y)
                loss.backward()
                model.zero_grad()

    # Synchronize before timing
    if device.type == "cuda":
        torch.cuda.synchronize()

    # Peak memory reset
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    start = time.perf_counter()
    for _ in range(n_steps):
        x, y = make_batch()
        if forward_only:
            with torch.no_grad():
                _ = model(x)
        else:
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            model.zero_grad()

    if device.type == "cuda":
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start
    total_samples = n_steps * batch_size
    throughput = total_samples / elapsed
    latency_ms = (elapsed / n_steps) * 1000

    result = {
        "batch_size": batch_size,
        "n_steps": n_steps,
        "elapsed_s": round(elapsed, 3),
        "throughput_samples_per_s": round(throughput, 1),
        "latency_per_step_ms": round(latency_ms, 2),
    }

    if device.type == "cuda":
        peak_mb = torch.cuda.max_memory_allocated(device) / 1024 ** 2
        result["peak_memory_mb"] = round(peak_mb, 1)

    return result


def build_default_model(input_shape: List[int]) -> torch.nn.Module:
    """Build a simple convolutional model for benchmarking if no config provided."""
    in_channels = input_shape[0]
    model = torch.nn.Sequential(
        torch.nn.Conv2d(in_channels, 32, 3, padding=1),
        torch.nn.ReLU(),
        torch.nn.AdaptiveAvgPool2d(4),
        torch.nn.Flatten(),
        torch.nn.Linear(32 * 4 * 4, 10),
    )
    return model


def print_table(results: List[Dict[str, float]]) -> None:
    if not results:
        return
    headers = list(results[0].keys())
    widths = [max(len(h), max(len(str(r.get(h, ""))) for r in results)) for h in headers]
    header_line = " | ".join(h.ljust(w) for h, w in zip(headers, widths))
    sep = "-+-".join("-" * w for w in widths)
    print("\n=== Benchmark Results ===")
    print(header_line)
    print(sep)
    for r in results:
        print(" | ".join(str(r.get(h, "")).ljust(w) for h, w in zip(headers, widths)))
    print()


def main():
    args = parse_args()
    device = get_device(args.device)
    print(f"Device: {device}")

    model = build_default_model(args.input_shape)
    model_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {model_params:,}")
    print(f"Input shape: {args.input_shape}")
    print(f"Pass type: {'forward only' if args.forward_only else 'forward + backward'}")
    print(f"Steps: {args.n_warmup} warmup + {args.n_steps} timed\n")

    all_results = []
    for bs in args.batch_sizes:
        print(f"Benchmarking batch_size={bs} ...")
        result = benchmark_batch(
            model=model,
            device=device,
            batch_size=bs,
            input_shape=args.input_shape,
            n_warmup=args.n_warmup,
            n_steps=args.n_steps,
            forward_only=args.forward_only,
        )
        all_results.append(result)

    print_table(all_results)


if __name__ == "__main__":
    main()
