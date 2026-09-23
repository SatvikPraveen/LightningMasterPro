# scripts/train.py
"""Train, validate, test or predict with any config via LightningCLI subcommands.

Examples::

    python scripts/train.py fit --config configs/vision/classifier.yaml
    python scripts/train.py fit --config configs/nlp/sentiment.yaml --trainer.max_epochs 3
    python scripts/train.py test --config configs/tabular/mlp.yaml --ckpt_path checkpoints/tabular/mlp/last.ckpt
    python scripts/train.py fit --config configs/vision/classifier.yaml --print_config

The model and datamodule classes come from the ``class_path`` entries in the YAML,
so a single entry point serves every domain.
"""

from lmpro.cli import LightningMasterCLI


def main() -> LightningMasterCLI:
    return LightningMasterCLI(description="LightningMasterPro training CLI")


if __name__ == "__main__":
    main()
