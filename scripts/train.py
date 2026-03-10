# scripts/train.py
"""Thin wrapper around LightningCLI for training models."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lmpro.cli import LightningMasterCLI


def main():
    """Main training function using LightningCLI."""
    # The LightningCLI with run=True (default) automatically executes the training
    # when instantiated, so no need to explicitly call cli.fit()
    cli = LightningMasterCLI(
        save_config_callback=True,
        auto_configure_optimizers=False,
        parser_kwargs={"parser_mode": "omegaconf"}
    )
    return cli
    

if __name__ == "__main__":
    main()