# scripts/scale_batch.py
"""Find the largest batch size that fits in memory with ``Tuner.scale_batch_size``.

Usage::

    python scripts/scale_batch.py --config configs/vision/classifier.yaml \
        [--tuning_config configs/tuning/batch_scaler.yaml] [--output_dir batch_scale_results]

The ``batch_scaler`` section of the tuning config is passed verbatim to
``Tuner.scale_batch_size``.  The tuner mutates ``datamodule.batch_size`` in place
(``batch_arg_name``), which is why every datamodule keeps ``batch_size`` as a plain attribute.
"""

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Dict, List

import yaml
from lightning.pytorch.tuner import Tuner

from lmpro.cli import LightningMasterCLI


def parse_args() -> "tuple[argparse.Namespace, List[str]]":
    parser = argparse.ArgumentParser(description="Scale batch size with Lightning's Tuner")
    parser.add_argument("--config", required=True, help="Training YAML config.")
    parser.add_argument(
        "--tuning_config", default="configs/tuning/batch_scaler.yaml", help="YAML with a batch_scaler section."
    )
    parser.add_argument("--output_dir", default="batch_scale_results", help="Where to write results.")
    return parser.parse_known_args()


def load_section(path: str, section: str) -> Dict[str, Any]:
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    return dict(data.get(section) or {})


def run_batch_scaler(config: str, scaler_kwargs: Dict[str, Any], overrides: List[str], output_dir: Path) -> int:
    cli = LightningMasterCLI.from_config(
        config,
        *overrides,
        "--trainer.logger=false",
        "--trainer.callbacks=[]",
    )
    tuner = Tuner(cli.trainer)
    new_batch_size = tuner.scale_batch_size(cli.model, datamodule=cli.datamodule, **scaler_kwargs)
    if new_batch_size is None:
        raise RuntimeError("scale_batch_size returned nothing (are you running on rank > 0?)")

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "batch_scale_results.json").write_text(
        json.dumps({"optimal_batch_size": int(new_batch_size), "scale_batch_size_kwargs": scaler_kwargs}, indent=2)
    )

    with open(config) as f:
        updated = copy.deepcopy(yaml.safe_load(f))
    updated.setdefault("data", {}).setdefault("init_args", {})["batch_size"] = int(new_batch_size)
    (output_dir / "updated_config.yaml").write_text(yaml.safe_dump(updated, sort_keys=False))
    return int(new_batch_size)


def main() -> int:
    args, overrides = parse_args()
    scaler_kwargs = load_section(args.tuning_config, "batch_scaler")
    output_dir = Path(args.output_dir)
    batch_size = run_batch_scaler(args.config, scaler_kwargs, overrides, output_dir)
    print(f"Optimal batch size: {batch_size}")
    print(f"Results written to {output_dir}")
    return batch_size


if __name__ == "__main__":
    main()
