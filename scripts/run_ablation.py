# scripts/run_ablation.py
"""Script for running systematic ablation studies."""

import sys
from pathlib import Path
import json
import yaml
import time
import itertools
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import lightning.pytorch as L
from lightning.pytorch.cli import LightningArgumentParser
from lightning.pytorch.loggers import TensorBoardLogger

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lmpro.cli import LightningMasterProCLI


def main():
    """Main ablation study function."""
    parser = LightningArgumentParser()
    parser.add_argument("--config", type=str, required=True,
                       help="Path to the base configuration file")
    parser.add_argument("--ablation_config", type=str, required=True,
                       help="Path to the ablation configuration file")
    parser.add_argument("--output_dir", type=str, default="ablation_results/",
                       help="Directory to save results")
    parser.add_argument("--max_combinations", type=int, default=50,
                       help="Maximum number of parameter combinations to test")
    
    args = parser.parse_args()
    
    # Load base config
    with open(args.config, 'r') as f:
        base_config = yaml.safe_load(f)
    
    # Load ablation config
    with open(args.ablation_config, 'r') as f:
        ablation_config = yaml.safe_load(f)
    
    # Extract ablation parameters
    ablation_params = ablation_config.get('ablation', {})
    experiment_name = ablation_params.get('experiment_name', 'default_ablation')
    parameters = ablation_params.get('parameters', {})
    fixed_params = ablation_params.get('fixed', {})
    metrics_to_track = ablation_params.get('metrics', ['val_loss', 'val_acc'])
    
    # Generate parameter combinations
    param_names = list(parameters.keys())
    param_values = list(parameters.values())
    
    # Create all combinations
    all_combinations = list(itertools.product(*param_values))
    
    # Limit combinations if too many
    if len(all_combinations) > args.max_combinations:
        print(f"Too many combinations ({len(all_combinations)}), sampling {args.max_combinations}")
        import random
        random.seed(42)
        all_combinations = random.sample(all_combinations, args.max_combinations)
    
    print(f"Running ablation study: {experiment_name}")
    print(f"Testing {len(all_combinations)} parameter combinations")
    print(f"Parameters: {param_names}")
    
    # Create output directory
    output_dir = Path(args.output_dir) / experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Results storage
    results = []
    
    # Run experiments
    for i, combination in enumerate(all_combinations):
        print(f"\n=== Experiment {i+1}/{len(all_combinations)} ===")
        
        # Create experiment config
        experiment_config = base_config.copy()
        
        # Update with current parameter combination
        param_dict = dict(zip(param_names, combination))
        print(f"Parameters: {param_dict}")
        
        # Update config with current parameters
        for param_name, param_value in param_dict.items():
            if param_name == 'learning_rate':
                if 'model' in experiment_config:
                    if 'init_args' in experiment_config['model']:
                        experiment_config['model']['init_args']['learning_rate'] = param_value
                    else:
                        experiment_config['model']['learning_rate'] = param_value
            elif param_name == 'weight_decay':
                if 'model' in experiment_config:
                    if 'init_args' in experiment_config['model']:
                        experiment_config['model']['init_args']['weight_decay'] = param_value
                    else:
                        experiment_config['model']['weight_decay'] = param_value
            elif param_name == 'dropout_rate':
                if 'model' in experiment_config:
                    if 'init_args' in experiment_config['model']:
                        experiment_config['model']['init_args']['dropout_rate'] = param_value
                    else:
                        experiment_config['model']['dropout_rate'] = param_value
            elif param_name == 'batch_size':
                if 'data' in experiment_config:
                    if 'init_args' in experiment_config['data']:
                        experiment_config['data']['init_args']['batch_size'] = param_value
                    else:
                        experiment_config['data']['batch_size'] = param_value
        
        # Add fixed parameters
        for param_name, param_value in fixed_params.items():
            if param_name in ['num_workers', 'pin_memory']:
                if 'data' in experiment_config:
                    if 'init_args' in experiment_config['data']:
                        experiment_config['data']['init_args'][param_name] = param_value
                    else:
                        experiment_config['data'][param_name] = param_value
        
        # Save experiment config
        exp_config_path = output_dir / f"experiment_{i:03d}_config.yaml"
        with open(exp_config_path, 'w') as f:
            yaml.dump(experiment_config, f, default_flow_style=False, indent=2)
        
        # Run experiment
        start_time = time.time()
        
        try:
            # Create CLI with current config
            config_args = ["--config", str(exp_config_path)]
            
            # Update logger name
            if 'logger' in experiment_config:
                for logger_config in experiment_config['logger']:
                    if 'init_args' in logger_config:
                        logger_config['init_args']['name'] = f"{experiment_name}_exp_{i:03d}"
            
            cli = LightningMasterProCLI(args=config_args, run=True)
            
            # Get metrics from trainer
            trainer_state = cli.trainer.state
            train_time = time.time() - start_time
            
            # Extract metrics from callbacks/loggers
            experiment_results = {
                'experiment_id': i,
                'train_time': train_time,
                'trainer_state': str(trainer_state),
                **param_dict
            }
            
            # Try to get validation metrics
            if hasattr(cli.trainer, 'logged_metrics'):
                logged_metrics = cli.trainer.logged_metrics
                for metric in metrics_to_track:
                    if metric in logged_metrics:
                        experiment_results[metric] = float(logged_metrics[metric])
            
            # Try to get from callback metrics
            if hasattr(cli.trainer, 'callback_metrics'):
                callback_metrics = cli.trainer.callback_metrics
                for metric in metrics_to_track:
                    if metric in callback_metrics:
                        experiment_results[metric] = float(callback_metrics[metric])
            
            # Estimate GPU memory usage (if available)
            if hasattr(cli.trainer, 'strategy') and hasattr(cli.trainer.strategy, 'root_device'):
                try:
                    import torch
                    if torch.cuda.is_available():
                        experiment_results['gpu_memory'] = torch.cuda.max_memory_allocated() / 1024**3  # GB
                except:
                    pass
            
            results.append(experiment_results)
            print(f"Experiment {i+1} completed successfully")
            
        except Exception as e:
            print(f"Experiment {i+1} failed: {e}")
            experiment_results = {
                'experiment_id': i,
                'train_time': time.time() - start_time,
                'error': str(e),
                **param_dict
            }
            results.append(experiment_results)
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_dir / "ablation_summary.csv", index=False)
    
    # Save detailed results
    with open(output_dir / "ablation_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    # Find best configuration
    if 'val_loss' in results_df.columns:
        best_idx = results_df['val_loss'].idxmin()
        best_config = results_df.loc[best_idx]
        
        print(f"\n=== Best Configuration ===")
        print(best_config)
        
        # Save best config
        with open(output_dir / "best_config.yaml", 'w') as f:
            yaml.dump(best_config.to_dict(), f, default_flow_style=False, indent=2)
    
    # Create visualizations
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    
    # Plot results for each parameter
    for param_name in param_names:
        if param_name in results_df.columns:
            fig, axes = plt.subplots(1, len(metrics_to_track), figsize=(15, 5))
            if len(metrics_to_track) == 1:
                axes = [axes]
            
            for j, metric in enumerate(metrics_to_track):
                if metric in results_df.columns:
                    ax = axes[j]
                    results_df.boxplot(column=metric, by=param_name, ax=ax)
                    ax.set_title(f'{metric} vs {param_name}')
                    ax.set_xlabel(param_name)
                    ax.set_ylabel(metric)
            
            plt.tight_layout()
            plt.savefig(plots_dir / f"{param_name}_analysis.png", dpi=300, bbox_inches='tight')
            plt.close()
    
    # Create correlation heatmap
    if len(results_df.select_dtypes(include=['number']).columns) > 1:
        plt.figure(figsize=(10, 8))
        numeric_cols = results_df.select_dtypes(include=['number']).columns
        corr_matrix = results_df[numeric_cols].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title('Parameter Correlation Matrix')
        plt.tight_layout()
        plt.savefig(plots_dir / "correlation_matrix.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"\n=== Ablation Study Complete ===")
    print(f"Results saved to: {output_dir}")
    print(f"Summary: {output_dir / 'ablation_summary.csv'}")
    print(f"Plots: {plots_dir}")


if __name__ == "__main__":
    main()