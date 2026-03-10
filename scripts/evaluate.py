# scripts/evaluate.py
"""
Evaluation script for LightningMasterPro.

Loads a checkpoint and runs evaluation on a test or validation set,
printing a summary of metrics and optionally saving results to disk.

Usage::
    python scripts/evaluate.py \\
        --checkpoint models/best.ckpt \\
        --config configs/vision/classifier.yaml \\
        [--split test] [--output_dir results/]
"""

import argparse
import json
import sys
from pathlib import Path

# Add src to path when running directly
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from lightning import Trainer
from lightning.pytorch.utilities.rank_zero import rank_zero_info


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a LightningMasterPro checkpoint"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the .ckpt file to evaluate.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional YAML config path (overrides checkpoint hparams).",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Dataset split to evaluate on.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save evaluation results (JSON).",
    )
    parser.add_argument(
        "--accelerator",
        type=str,
        default="auto",
        help="Accelerator to use: 'cpu', 'gpu', 'auto'.",
    )
    parser.add_argument(
        "--devices",
        type=int,
        default=1,
        help="Number of devices to use.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for evaluation.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of DataLoader workers.",
    )
    return parser.parse_args()


def load_module_and_datamodule(args):
    """
    Restore the LightningModule from a checkpoint.
    Returns (model, datamodule_class_or_none).
    """
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    # Load state to inspect hparams and determine model class
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    hparams = ckpt.get("hyper_parameters", {})
    class_name = hparams.get("model_class", None)

    rank_zero_info(f"Loading checkpoint: {ckpt_path}")
    rank_zero_info(f"Checkpoint hparams: {list(hparams.keys())}")

    # Dynamic import based on checkpoint metadata
    # Falls back to a generic import path if not specified
    from lmpro.modules.vision.classifier import VisionClassifier

    model = VisionClassifier.load_from_checkpoint(str(ckpt_path))
    model.eval()
    return model


def run_evaluation(args, model):
    """Run Trainer.test() on the loaded model and return metric dict."""
    from lmpro.datamodules.vision_dm import VisionDataModule

    dm = VisionDataModule(batch_size=args.batch_size, num_workers=args.num_workers)
    dm.setup(stage="test")

    trainer = Trainer(
        accelerator=args.accelerator,
        devices=args.devices,
        logger=False,
        enable_progress_bar=True,
    )

    results = trainer.test(model, datamodule=dm)
    return results[0] if results else {}


def save_results(results: dict, output_dir: str) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    out_file = out / "eval_results.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    rank_zero_info(f"Evaluation results saved to {out_file}")


def main():
    args = parse_args()

    model = load_module_and_datamodule(args)
    results = run_evaluation(args, model)

    rank_zero_info("\n=== Evaluation Results ===")
    for k, v in results.items():
        rank_zero_info(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    if args.output_dir:
        save_results(results, args.output_dir)

    return results


if __name__ == "__main__":
    main()
