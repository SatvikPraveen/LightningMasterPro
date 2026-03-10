# scripts/generate_data.py
"""
Synthetic data generation script for LightningMasterPro.

Pre-generates and caches synthetic datasets to disk so notebooks and training
scripts can load them quickly without re-generating each run.

Usage::
    python scripts/generate_data.py \\
        [--domains vision nlp tabular timeseries] \\
        [--num_samples 1000] \\
        [--output_dir data/synthetic/]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch


SUPPORTED_DOMAINS = ["vision", "nlp", "tabular", "timeseries"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate and cache synthetic datasets"
    )
    parser.add_argument(
        "--domains",
        nargs="+",
        default=SUPPORTED_DOMAINS,
        choices=SUPPORTED_DOMAINS,
        help="Which domains to generate data for.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1000,
        help="Number of samples per dataset.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/synthetic",
        help="Directory to save generated datasets.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    return parser.parse_args()


def generate_vision(num_samples: int, output_dir: Path) -> None:
    from lmpro.data.synth_vision import VisionDatasetConfig, SyntheticImageDataset

    cfg = VisionDatasetConfig(num_samples=num_samples, image_size=(64, 64), num_classes=10)
    for split in ("train", "val", "test"):
        ds = SyntheticImageDataset(config=cfg, split=split)
        pt_file = output_dir / f"vision_{split}.pt"
        data = [ds[i] for i in range(len(ds))]
        torch.save(data, str(pt_file))
        print(f"  Saved {len(data)} vision samples → {pt_file}")


def generate_nlp(num_samples: int, output_dir: Path) -> None:
    from lmpro.data.synth_nlp import NLPDatasetConfig, SyntheticTextDataset

    cfg = NLPDatasetConfig(num_samples=num_samples, vocab_size=5000, max_sequence_length=128)
    for split in ("train", "val", "test"):
        ds = SyntheticTextDataset(config=cfg, split=split)
        pt_file = output_dir / f"nlp_{split}.pt"
        data = [ds[i] for i in range(len(ds))]
        torch.save(data, str(pt_file))
        print(f"  Saved {len(data)} NLP samples → {pt_file}")


def generate_tabular(num_samples: int, output_dir: Path) -> None:
    from lmpro.data.synth_tabular import TabularDatasetConfig, SyntheticTabularDataset

    cfg = TabularDatasetConfig(num_samples=num_samples, num_features=20, num_classes=3)
    for split in ("train", "val", "test"):
        ds = SyntheticTabularDataset(config=cfg, task="classification", split=split)
        pt_file = output_dir / f"tabular_clf_{split}.pt"
        data = [ds[i] for i in range(len(ds))]
        torch.save(data, str(pt_file))
        print(f"  Saved {len(data)} tabular samples → {pt_file}")


def generate_timeseries(num_samples: int, output_dir: Path) -> None:
    from lmpro.data.synth_timeseries import (
        TimeSeriesDatasetConfig,
        SyntheticTimeSeriesDataset,
    )

    cfg = TimeSeriesDatasetConfig(
        num_samples=num_samples, sequence_length=100, prediction_horizon=10
    )
    for split in ("train", "val", "test"):
        ds = SyntheticTimeSeriesDataset(config=cfg, task="forecasting", split=split)
        pt_file = output_dir / f"timeseries_{split}.pt"
        data = [ds[i] for i in range(len(ds))]
        torch.save(data, str(pt_file))
        print(f"  Saved {len(data)} timeseries samples → {pt_file}")


GENERATORS = {
    "vision": generate_vision,
    "nlp": generate_nlp,
    "tabular": generate_tabular,
    "timeseries": generate_timeseries,
}


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir.resolve()}")

    for domain in args.domains:
        print(f"\nGenerating {domain} data ({args.num_samples} samples)...")
        GENERATORS[domain](args.num_samples, output_dir)

    print("\nDone. All datasets cached to disk.")


if __name__ == "__main__":
    main()
