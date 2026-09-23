# scripts/run_ablation.py
"""Grid ablation over any config keys, one full LightningCLI ``fit`` per combination.

Usage::

    python scripts/run_ablation.py --config configs/vision/classifier.yaml \
        [--ablation_config configs/tuning/ablation_study.yaml] [--output_dir ablation_results] [--max_combinations 8]

The ablation YAML lists dotted config paths and the values to sweep, e.g.::

    ablation:
      experiment_name: classifier_lr_wd
      parameters:
        model.init_args.learning_rate: [1e-4, 1e-3]
        model.init_args.weight_decay: [0.0, 1e-4]
      trainer_overrides:
        max_epochs: 2
      metrics: [val/loss, val/acc]

Each run gets its own config file, TensorBoard logger name and checkpoint directory.
Results are aggregated into ``ablation_summary.csv`` plus per-parameter box plots.
"""

import argparse
import copy
import itertools
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import matplotlib
import pandas as pd
import yaml

from lmpro.cli import LightningMasterCLI

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a grid ablation with LightningCLI")
    parser.add_argument("--config", required=True, help="Base training YAML config.")
    parser.add_argument(
        "--ablation_config", default="configs/tuning/ablation_study.yaml", help="YAML with an ablation section."
    )
    parser.add_argument("--output_dir", default="ablation_results", help="Root output directory.")
    parser.add_argument("--max_combinations", type=int, default=16, help="Random subset size if the grid is larger.")
    parser.add_argument("--seed", type=int, default=42, help="Seed for sampling the grid subset.")
    return parser.parse_args()


def set_nested(config: Dict[str, Any], dotted_key: str, value: Any) -> None:
    node = config
    *parents, leaf = dotted_key.split(".")
    for key in parents:
        node = node.setdefault(key, {})
    node[leaf] = value


def build_experiment_config(
    base: Dict[str, Any], params: Dict[str, Any], overrides: Dict[str, Any], name: str
) -> Dict[str, Any]:
    cfg = copy.deepcopy(base)
    for key, value in params.items():
        set_nested(cfg, key, value)
    for key, value in overrides.items():
        set_nested(cfg, f"trainer.{key}", value)

    logger = cfg.get("trainer", {}).get("logger")
    if isinstance(logger, dict) and "init_args" in logger:
        logger["init_args"]["name"] = name
    for callback in cfg.get("trainer", {}).get("callbacks", []) or []:
        if isinstance(callback, dict) and callback.get("class_path", "").endswith("ModelCheckpoint"):
            callback.setdefault("init_args", {})["dirpath"] = f"checkpoints/ablation/{name}"
    return cfg


def run_ablation(
    base_config_path: str, ablation: Dict[str, Any], output_dir: Path, max_combinations: int, seed: int
) -> pd.DataFrame:
    with open(base_config_path) as f:
        base_config = yaml.safe_load(f)

    experiment_name = ablation.get("experiment_name", "ablation")
    parameters: Dict[str, List[Any]] = ablation.get("parameters", {})
    trainer_overrides: Dict[str, Any] = ablation.get("trainer_overrides", {})
    metrics: List[str] = ablation.get("metrics", ["val/loss"])

    names = list(parameters)
    grid = list(itertools.product(*(parameters[n] for n in names)))
    if len(grid) > max_combinations:
        random.Random(seed).shuffle(grid)
        grid = grid[:max_combinations]
    print(f"{experiment_name}: {len(grid)} runs over {names}")

    run_dir = output_dir / experiment_name
    run_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    for i, combo in enumerate(grid):
        params = dict(zip(names, combo))
        run_name = f"{experiment_name}_{i:03d}"
        cfg = build_experiment_config(base_config, params, trainer_overrides, run_name)
        cfg_path = run_dir / f"{run_name}.yaml"
        cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))

        row: Dict[str, Any] = {"run": run_name, **params}
        start = time.time()
        try:
            saved_argv, sys.argv = sys.argv, sys.argv[:1]  # keep LightningCLI from seeing this script's argv
            try:
                cli = LightningMasterCLI(args=["fit", "--config", str(cfg_path)])
            finally:
                sys.argv = saved_argv
            row["train_time_s"] = round(time.time() - start, 2)
            for metric in metrics:
                if metric in cli.trainer.callback_metrics:
                    row[metric] = float(cli.trainer.callback_metrics[metric])
            row["status"] = "ok"
        except Exception as exc:  # noqa: BLE001 - one failing run must not kill the sweep
            row["train_time_s"] = round(time.time() - start, 2)
            row["status"] = f"error: {exc}"
        print(f"[{i + 1}/{len(grid)}] {row}")
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(run_dir / "ablation_summary.csv", index=False)
    (run_dir / "ablation_results.json").write_text(json.dumps(rows, indent=2, default=str))

    primary = metrics[0] if metrics else None
    if primary and primary in df.columns and df[primary].notna().any():
        best = df.loc[df[primary].idxmin()]
        best_params = {n: (best[n].item() if hasattr(best[n], "item") else best[n]) for n in names}
        (run_dir / "best_config.yaml").write_text(
            yaml.safe_dump(
                build_experiment_config(base_config, best_params, trainer_overrides, "best"), sort_keys=False
            )
        )
        print(f"Best {primary}: {best[primary]:.4f} with {best_params}")

    _plot(df, names, [m for m in metrics if m in df.columns], run_dir / "plots")
    return df


def _plot(df: pd.DataFrame, names: List[str], metrics: List[str], plots_dir: Path) -> None:
    if not metrics or df.empty:
        return
    plots_dir.mkdir(exist_ok=True)
    for name in names:
        fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 4), squeeze=False)
        for ax, metric in zip(axes[0], metrics):
            df.boxplot(column=metric, by=name, ax=ax)
            ax.set_title(f"{metric} vs {name}")
        fig.suptitle("")
        fig.tight_layout()
        fig.savefig(plots_dir / f"{name.replace('.', '_')}.png", dpi=120)
        plt.close(fig)


def main() -> pd.DataFrame:
    args = parse_args()
    with open(args.ablation_config) as f:
        ablation = (yaml.safe_load(f) or {}).get("ablation", {})
    df = run_ablation(args.config, ablation, Path(args.output_dir), args.max_combinations, args.seed)
    print(df.to_string(index=False))
    return df


if __name__ == "__main__":
    main()
