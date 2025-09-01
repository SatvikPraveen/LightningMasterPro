# scripts/scale_batch.py
"""Script for finding optimal batch sizes using Lightning's batch size finder."""

import sys
from pathlib import Path
import json
import lightning.pytorch as L
from lightning.pytorch.cli import LightningArgumentParser
from lightning.pytorch.tuner import Tuner

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lmpro.cli import LightningMasterProCLI


def main():
    """Main batch size scaling function."""
    parser = LightningArgumentParser()
    parser.add_argument("--config", type=str, required=True,
                       help="Path to the model configuration file")
    parser.add_argument("--mode", type=str, default="power_scaling",
                       choices=["power_scaling", "binsearch"],
                       help="Batch size scaling mode")
    parser.add_argument("--steps_per_trial", type=int, default=10,
                       help="Number of steps per batch size trial")
    parser.add_argument("--init_val", type=int, default=2,
                       help="Initial batch size value")
    parser.add_argument("--max_trials", type=int, default=25,
                       help="Maximum number of trials")
    parser.add_argument("--output_dir", type=str, default="batch_scale_results/",
                       help="Directory to save results")
    
    args = parser.parse_args()
    
    # Load model and datamodule from config
    cli = LightningMasterProCLI(
        args=["--config", args.config, "--trainer.max_epochs", "2"],
        run=False
    )
    
    model = cli.model
    datamodule = cli.datamodule
    
    # Create trainer for batch scaling
    trainer = L.Trainer(
        accelerator="auto",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=True,
        max_epochs=2
    )
    
    # Create tuner
    tuner = Tuner(trainer)
    
    # Run batch size scaling
    print("Running batch size scaling...")
    try:
        new_batch_size = tuner.scale_batch_size(
            model,
            datamodule=datamodule,
            mode=args.mode,
            steps_per_trial=args.steps_per_trial,
            init_val=args.init_val,
            max_trials=args.max_trials,
            batch_arg_name="batch_size"
        )
        
        print(f"Optimal batch size found: {new_batch_size}")
        
    except RuntimeError as e:
        print(f"Error during batch scaling: {e}")
        # Try to get the last working batch size
        if hasattr(datamodule, 'batch_size'):
            new_batch_size = datamodule.batch_size
            print(f"Using current batch size: {new_batch_size}")
        else:
            new_batch_size = args.init_val
            print(f"Using initial batch size: {new_batch_size}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save results
    results = {
        "optimal_batch_size": int(new_batch_size),
        "mode": args.mode,
        "steps_per_trial": args.steps_per_trial,
        "init_val": args.init_val,
        "max_trials": args.max_trials,
        "config_file": args.config
    }
    
    with open(output_dir / "batch_scale_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # Save max batch size to simple text file
    with open(output_dir / "max_batch_size.txt", "w") as f:
        f.write(str(new_batch_size))
    
    # Save updated config with optimal batch size
    updated_config_path = output_dir / "updated_config.yaml"
    
    # Read original config and update batch size
    import yaml
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Update batch size in config
    if 'data' in config and 'init_args' in config['data']:
        config['data']['init_args']['batch_size'] = int(new_batch_size)
    elif 'data' in config:
        config['data']['batch_size'] = int(new_batch_size)
    
    with open(updated_config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)
    
    print(f"Results saved to: {output_dir}")
    print(f"Updated config saved to: {updated_config_path}")
    print(f"Optimal batch size: {new_batch_size}")


if __name__ == "__main__":
    main()