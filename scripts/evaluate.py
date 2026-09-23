# scripts/evaluate.py
"""Evaluate a checkpoint on the validation or test split defined by a training config.

Usage::

    python scripts/evaluate.py --config configs/vision/classifier.yaml \
        --checkpoint checkpoints/vision/classifier/last.ckpt --split test --output_dir evaluation_results

Extra ``--key value`` arguments are forwarded to the CLI as overrides
(e.g. ``--trainer.accelerator cpu``).
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

from lmpro.cli import LightningMasterCLI


def parse_args() -> "tuple[argparse.Namespace, List[str]]":
    parser = argparse.ArgumentParser(description="Evaluate a LightningMasterPro checkpoint")
    parser.add_argument("--config", required=True, help="Training YAML config (model + data + trainer).")
    parser.add_argument("--checkpoint", required=True, help="Path to the .ckpt file.")
    parser.add_argument("--split", default="test", choices=["val", "test"], help="Which split to evaluate.")
    parser.add_argument("--output_dir", default=None, help="Optional directory for eval_results.json")
    return parser.parse_known_args()


def evaluate(
    config: str, checkpoint: str, split: str = "test", overrides: Optional[List[str]] = None
) -> Dict[str, float]:
    cli = LightningMasterCLI.from_config(
        config,
        *(overrides or []),
        "--trainer.logger=false",
        "--trainer.callbacks=[]",
    )
    model = type(cli.model).load_from_checkpoint(checkpoint, map_location="cpu")
    runner = cli.trainer.test if split == "test" else cli.trainer.validate
    results = runner(model, datamodule=cli.datamodule, verbose=True)
    return {k: float(v) for k, v in (results[0] if results else {}).items()}


def main() -> Dict[str, float]:
    args, overrides = parse_args()
    results = evaluate(args.config, args.checkpoint, args.split, overrides)

    print(f"\n=== {args.split} results for {args.checkpoint} ===")
    for key, value in sorted(results.items()):
        print(f"  {key}: {value:.4f}")

    if args.output_dir:
        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        out_file = out / f"eval_{args.split}.json"
        out_file.write_text(json.dumps({"checkpoint": args.checkpoint, "split": args.split, **results}, indent=2))
        print(f"Saved to {out_file}")
    return results


if __name__ == "__main__":
    main()
