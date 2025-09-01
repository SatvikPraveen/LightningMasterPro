# scripts/train.py
"""Thin wrapper around LightningCLI for training models."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lmpro.cli import LightningMasterProCLI


def main():
    """Main training function using LightningCLI."""
    cli = LightningMasterProCLI(
        save_config_callback=True,
        auto_configure_optimizers=False,
        parser_kwargs={"parser_mode": "omegaconf"}
    )
    

if __name__ == "__main__":
    main()