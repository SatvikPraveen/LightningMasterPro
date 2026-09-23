# scripts/tune_lr.py
"""Learning-rate range test with ``lightning.pytorch.tuner.Tuner.lr_find``.

Usage::

    python scripts/tune_lr.py --config configs/vision/classifier.yaml \
        [--tuning_config configs/tuning/lr_finder.yaml] [--output_dir lr_finder_results]

The ``lr_finder`` section of the tuning config is passed verbatim to ``Tuner.lr_find``.
Outputs: ``lr_finder_results.json`` (lr/loss curve + suggestion), ``lr_finder_plot.png``
and ``updated_config.yaml`` with the suggested learning rate written into
``model.init_args.learning_rate``.
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
    parser = argparse.ArgumentParser(description="Find a learning rate with Lightning's Tuner")
    parser.add_argument("--config", required=True, help="Training YAML config.")
    parser.add_argument(
        "--tuning_config", default="configs/tuning/lr_finder.yaml", help="YAML with an lr_finder section."
    )
    parser.add_argument("--output_dir", default="lr_finder_results", help="Where to write results.")
    return parser.parse_known_args()


def load_section(path: str, section: str) -> Dict[str, Any]:
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    return dict(data.get(section) or {})


def run_lr_finder(config: str, lr_kwargs: Dict[str, Any], overrides: List[str], output_dir: Path) -> float:
    cli = LightningMasterCLI.from_config(
        config,
        *overrides,
        "--trainer.logger=false",
        "--trainer.callbacks=[]",
    )
    tuner = Tuner(cli.trainer)
    lr_finder = tuner.lr_find(cli.model, datamodule=cli.datamodule, **lr_kwargs)
    if lr_finder is None:
        raise RuntimeError("lr_find returned nothing (are you running on rank > 0?)")

    suggested = lr_finder.suggestion()
    if suggested is None:
        raise RuntimeError("lr_find could not suggest a learning rate; try more steps or a wider range")

    output_dir.mkdir(parents=True, exist_ok=True)
    results = {
        "suggested_lr": float(suggested),
        "lr": [float(x) for x in lr_finder.results["lr"]],
        "loss": [float(x) for x in lr_finder.results["loss"]],
        "lr_find_kwargs": lr_kwargs,
    }
    (output_dir / "lr_finder_results.json").write_text(json.dumps(results, indent=2))

    fig = lr_finder.plot(suggest=True)
    fig.savefig(output_dir / "lr_finder_plot.png", dpi=150, bbox_inches="tight")

    with open(config) as f:
        updated = copy.deepcopy(yaml.safe_load(f))
    updated.setdefault("model", {}).setdefault("init_args", {})["learning_rate"] = float(suggested)
    (output_dir / "updated_config.yaml").write_text(yaml.safe_dump(updated, sort_keys=False))
    return float(suggested)


def main() -> float:
    args, overrides = parse_args()
    lr_kwargs = load_section(args.tuning_config, "lr_finder")
    output_dir = Path(args.output_dir)
    suggested = run_lr_finder(args.config, lr_kwargs, overrides, output_dir)
    print(f"Suggested learning rate: {suggested:.3e}")
    print(f"Results written to {output_dir}")
    return suggested


if __name__ == "__main__":
    main()
