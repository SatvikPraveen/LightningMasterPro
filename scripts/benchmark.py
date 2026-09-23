# scripts/benchmark.py
"""Throughput / latency benchmark for any model in a training config.

Usage::

    python scripts/benchmark.py --config configs/vision/classifier.yaml --batch_sizes 16 32 64
    python scripts/benchmark.py --batch_sizes 32 64           # built-in toy CNN, no config

Measures samples/second and mean step time for forward-only or forward+backward,
plus peak memory on CUDA.  With ``--config`` the real LightningModule's
``training_step`` is timed on batches drawn from its own datamodule.
"""

import argparse
import time
from pathlib import Path
from typing import Any, Callable, Dict, List

import torch
import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark model throughput")
    parser.add_argument("--config", default=None, help="Training YAML config. Omit to benchmark a toy CNN.")
    parser.add_argument("--batch_sizes", type=int, nargs="+", default=[32, 64, 128])
    parser.add_argument("--input_shape", type=int, nargs="+", default=[3, 64, 64], help="Toy model input shape.")
    parser.add_argument("--n_warmup", type=int, default=5)
    parser.add_argument("--n_steps", type=int, default=20)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--forward_only", action="store_true")
    parser.add_argument("--compile", action="store_true", help="Wrap the model in torch.compile before timing.")
    parser.add_argument("--output", default=None, help="Optional JSON/YAML path for the results table.")
    return parser.parse_args()


def get_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but not available")
    if name == "mps" and not torch.backends.mps.is_available():
        raise SystemExit("MPS requested but not available")
    return torch.device(name)


def _to_device(batch: Any, device: torch.device) -> Any:
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    if isinstance(batch, (tuple, list)):
        return type(batch)(_to_device(b, device) for b in batch)
    if isinstance(batch, dict):
        return {k: _to_device(v, device) for k, v in batch.items()}
    return batch


def _resize_batch(batch: Any, batch_size: int) -> Any:
    """Tile/slice every tensor in the batch along dim 0 to the requested size."""

    def fix(t: torch.Tensor) -> torch.Tensor:
        reps = -(-batch_size // t.shape[0])
        return t.repeat(reps, *([1] * (t.dim() - 1)))[:batch_size]

    if isinstance(batch, torch.Tensor):
        return fix(batch)
    if isinstance(batch, (tuple, list)):
        return type(batch)(_resize_batch(b, batch_size) for b in batch)
    if isinstance(batch, dict):
        return {k: _resize_batch(v, batch_size) for k, v in batch.items()}
    return batch


def build_toy_model(input_shape: List[int]) -> torch.nn.Module:
    return torch.nn.Sequential(
        torch.nn.Conv2d(input_shape[0], 32, 3, padding=1),
        torch.nn.ReLU(),
        torch.nn.AdaptiveAvgPool2d(4),
        torch.nn.Flatten(),
        torch.nn.Linear(32 * 16, 10),
    )


def make_step_fn(model: torch.nn.Module, forward_only: bool, is_lightning: bool) -> Callable[[Any], None]:
    def step(batch: Any) -> None:
        if forward_only:
            with torch.no_grad():
                x = batch[0] if isinstance(batch, (tuple, list)) else batch
                model(x)
            return
        if is_lightning:
            out = model.training_step(batch, 0)
            loss = out["loss"] if isinstance(out, dict) else out
        else:
            x, y = batch
            loss = torch.nn.functional.cross_entropy(model(x), y)
        loss.backward()
        model.zero_grad(set_to_none=True)

    return step


def benchmark(
    step: Callable[[Any], None], batch: Any, device: torch.device, n_warmup: int, n_steps: int
) -> Dict[str, float]:
    batch_size = (batch[0] if isinstance(batch, (tuple, list)) else batch).shape[0]
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for _ in range(n_warmup):
        step(batch)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    for _ in range(n_steps):
        step(batch)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    result = {
        "batch_size": batch_size,
        "step_ms": round(1000 * elapsed / n_steps, 3),
        "samples_per_s": round(batch_size * n_steps / elapsed, 1),
    }
    if device.type == "cuda":
        result["peak_mem_mb"] = round(torch.cuda.max_memory_allocated(device) / 2**20, 1)
    return result


def print_table(rows: List[Dict[str, float]]) -> None:
    if not rows:
        return
    headers = list(rows[0])
    widths = [max(len(h), *(len(str(r.get(h, ""))) for r in rows)) for h in headers]
    print(" | ".join(h.ljust(w) for h, w in zip(headers, widths)))
    print("-+-".join("-" * w for w in widths))
    for r in rows:
        print(" | ".join(str(r.get(h, "")).ljust(w) for h, w in zip(headers, widths)))


def main() -> List[Dict[str, float]]:
    args = parse_args()
    device = get_device(args.device)

    if args.config:
        from lmpro.cli import LightningMasterCLI

        cli = LightningMasterCLI.from_config(args.config, "--trainer.logger=false", "--trainer.callbacks=[]")
        model = cli.model
        cli.datamodule.setup("fit")
        base_batch = next(iter(cli.datamodule.train_dataloader()))
        is_lightning = True
    else:
        model = build_toy_model(args.input_shape)
        base_batch = (torch.randn(8, *args.input_shape), torch.randint(0, 10, (8,)))
        is_lightning = False

    model = model.to(device).train(not args.forward_only)
    if args.compile:
        model = torch.compile(model)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"device={device} params={n_params:,} mode={'forward' if args.forward_only else 'forward+backward'}")

    step = make_step_fn(model, args.forward_only, is_lightning)
    rows: List[Dict[str, float]] = []
    for bs in args.batch_sizes:
        batch = _to_device(_resize_batch(base_batch, bs), device)
        rows.append(benchmark(step, batch, device, args.n_warmup, args.n_steps))
    print_table(rows)

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            yaml.safe_dump(rows) if out.suffix in (".yml", ".yaml") else __import__("json").dumps(rows, indent=2)
        )
        print(f"Saved to {out}")
    return rows


if __name__ == "__main__":
    main()
