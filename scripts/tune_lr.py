# scripts/tune_lr.py
"""Script for finding optimal learning rates using Lightning's LR finder."""

import sys
from pathlib import Path
import json
import matplotlib.pyplot as plt
import torch
import lightning.pytorch as L
from lightning.pytorch.cli import LightningArgumentParser
from lightning.pytorch.tuner import Tuner

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lmpro.cli import LightningMasterProCLI


def main():
    """Main LR tuning function."""
    parser = LightningArgumentParser()
    parser.add_argument("--config", type=str, required=True,
                       help="Path to the model configuration file")
    parser.add_argument("--min_lr", type=float, default=1e-8,
                       help="Minimum learning rate to test")
    parser.add_argument("--max_lr", type=float, default=1e-1,
                       help="Maximum learning rate to test")
    parser.add_argument("--num_training_steps", type=int, default=200,
                       help="Number of training steps for LR finder")
    parser.add_argument("--mode", type=str, default="exponential",
                       choices=["exponential", "linear"],
                       help="LR progression mode")
    parser.add_argument("--output_dir", type=str, default="lr_finder_results/",
                       help="Directory to save results")
    
    args = parser.parse_args()
    
    # Load model and datamodule from config
    cli = LightningMasterProCLI(
        args=["--config", args.config, "--trainer.max_epochs", "1"],
        run=False
    )
    
    model = cli.model
    datamodule = cli.datamodule
    
    # Create trainer for LR finding
    trainer = L.Trainer(
        accelerator="auto",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=True,
        max_epochs=1
    )
    
    # Create tuner
    tuner = Tuner(trainer)
    
    # Run LR finder
    print("Running learning rate finder...")
    lr_finder = tuner.lr_find(
        model,
        datamodule=datamodule,
        min_lr=args.min_lr,
        max_lr=args.max_lr,
        num_training_steps=args.num_training_steps,
        mode=args.mode,
        early_stop_threshold=4.0
    )
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get suggested LR
    suggested_lr = lr_finder.suggestion()
    print(f"Suggested learning rate: {suggested_lr}")
    
    # Save results
    results = {
        "suggested_lr": float(suggested_lr),
        "min_lr": args.min_lr,
        "max_lr": args.max_lr,
        "num_training_steps": args.num_training_steps,
        "mode": args.mode,
        "lr_schedule": lr_finder.results.tolist(),
        "losses": [float(x) for x in lr_finder.results["loss"]]
    }
    
    with open(output_dir / "lr_finder_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # Plot results
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(lr_finder.results["lr"], lr_finder.results["loss"])
    ax.set_xscale("log")
    ax.set_xlabel("Learning Rate")
    ax.set_ylabel("Loss")
    ax.set_title("Learning Rate Finder Results")
    ax.axvline(x=suggested_lr, color='red', linestyle='--', 
               label=f'Suggested LR: {suggested_lr:.2e}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "lr_finder_plot.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save updated config with suggested LR
    updated_config_path = output_dir / "updated_config.yaml"
    
    # Read original config and update learning rate
    import yaml
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Update learning rate in config
    if 'model' in config and 'init_args' in config['model']:
        config['model']['init_args']['learning_rate'] = float(suggested_lr)
    elif 'model' in config:
        config['model']['learning_rate'] = float(suggested_lr)
    
    with open(updated_config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)
    
    print(f"Results saved to: {output_dir}")
    print(f"Plot saved to: {output_dir / 'lr_finder_plot.png'}")
    print(f"Updated config saved to: {updated_config_path}")
    print(f"Suggested learning rate: {suggested_lr:.2e}")


if __name__ == "__main__":
    main()