# File: src/lmpro/loops/kfold_loop.py

"""
K-Fold Cross-Validation Loop for robust model evaluation
"""

import torch
from torch.utils.data import DataLoader, Subset
from lightning.pytorch.loops import Loop
from lightning import LightningModule, Trainer
from sklearn.model_selection import KFold, StratifiedKFold
from typing import Any, Dict, List, Optional, Union, Tuple
import numpy as np
from pathlib import Path
import json
from lightning.pytorch.utilities.rank_zero import rank_zero_info, rank_zero_only
from copy import deepcopy


class KFoldLoop(Loop):
    """
    K-Fold Cross-Validation Loop
    
    Implements k-fold cross-validation for robust model evaluation.
    Supports both regular and stratified k-fold splitting.
    """
    
    def __init__(
        self,
        num_folds: int = 5,
        stratified: bool = False,
        shuffle: bool = True,
        random_state: int = 42,
        save_fold_results: bool = True,
        results_dir: str = "kfold_results",
        **kwargs
    ):
        super().__init__()
        
        self.num_folds = num_folds
        self.stratified = stratified
        self.shuffle = shuffle
        self.random_state = random_state
        self.save_fold_results = save_fold_results
        self.results_dir = Path(results_dir)
        
        # Internal state
        self.current_fold = 0
        self.fold_results = []
        self.fold_datasets = []
        self.original_train_dataloader = None
        self.original_val_dataloader = None
        self.kf_splitter = None
        
        # Results storage
        self.all_fold_metrics = []
        self.best_fold_metrics = {}
        self.fold_predictions = []
        
    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        """Setup k-fold cross-validation"""
        super().setup(trainer, pl_module, stage)
        
        # Store original dataloaders
        self.original_train_dataloader = trainer.datamodule.train_dataloader()
        self.original_val_dataloader = getattr(trainer.datamodule, 'val_dataloader', lambda: None)()
        
        # Get full dataset for k-fold splitting
        train_dataset = self.original_train_dataloader.dataset
        
        # Create k-fold splitter
        if self.stratified:
            # Extract labels for stratified splitting
            labels = self._extract_labels(train_dataset)
            self.kf_splitter = StratifiedKFold(
                n_splits=self.num_folds,
                shuffle=self.shuffle,
                random_state=self.random_state
            )
            self.fold_splits = list(self.kf_splitter.split(range(len(train_dataset)), labels))
        else:
            self.kf_splitter = KFold(
                n_splits=self.num_folds,
                shuffle=self.shuffle,
                random_state=self.random_state
            )
            self.fold_splits = list(self.kf_splitter.split(range(len(train_dataset))))
        
        # Create results directory
        if self.save_fold_results:
            self.results_dir.mkdir(parents=True, exist_ok=True)
        
        rank_zero_info(f"K-Fold CV setup: {self.num_folds} folds, stratified={self.stratified}")
    
    def _extract_labels(self, dataset) -> np.ndarray:
        """Extract labels from dataset for stratified splitting"""
        labels = []
        
        # Try different ways to extract labels
        if hasattr(dataset, 'targets'):
            labels = dataset.targets
        elif hasattr(dataset, 'labels'):
            labels = dataset.labels
        else:
            # Iterate through dataset to extract labels
            for i in range(len(dataset)):
                try:
                    _, label = dataset[i]
                    labels.append(label.item() if hasattr(label, 'item') else label)
                except Exception:
                    # If we can't extract labels, fall back to regular k-fold
                    rank_zero_info("Could not extract labels for stratified k-fold, using regular k-fold")
                    return np.arange(len(dataset))
        
        return np.array(labels)
    
    @property
    def done(self) -> bool:
        """Check if all folds are completed"""
        return self.current_fold >= self.num_folds
    
    def reset(self) -> None:
        """Reset loop state"""
        self.current_fold = 0
        self.fold_results = []
        self.all_fold_metrics = []
        self.fold_predictions = []
    
    def advance(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Advance to next fold"""
        if self.done:
            return
        
        rank_zero_info(f"\n{'='*50}")
        rank_zero_info(f"Starting Fold {self.current_fold + 1}/{self.num_folds}")
        rank_zero_info(f"{'='*50}")
        
        # Get current fold indices
        train_indices, val_indices = self.fold_splits[self.current_fold]
        
        # Create fold datasets
        train_dataset = self.original_train_dataloader.dataset
        fold_train_dataset = Subset(train_dataset, train_indices)
        fold_val_dataset = Subset(train_dataset, val_indices)
        
        # Create fold dataloaders
        fold_train_dataloader = DataLoader(
            fold_train_dataset,
            batch_size=self.original_train_dataloader.batch_size,
            shuffle=True,
            num_workers=self.original_train_dataloader.num_workers,
            pin_memory=getattr(self.original_train_dataloader, 'pin_memory', False),
            persistent_workers=getattr(self.original_train_dataloader, 'persistent_workers', False)
        )
        
        fold_val_dataloader = DataLoader(
            fold_val_dataset,
            batch_size=self.original_val_dataloader.batch_size if self.original_val_dataloader else self.original_train_dataloader.batch_size,
            shuffle=False,
            num_workers=self.original_train_dataloader.num_workers,
            pin_memory=getattr(self.original_train_dataloader, 'pin_memory', False),
            persistent_workers=getattr(self.original_train_dataloader, 'persistent_workers', False)
        )
        
        # Replace trainer dataloaders
        trainer.fit_loop.setup_data()
        trainer.datamodule.train_dataloader = lambda: fold_train_dataloader
        trainer.datamodule.val_dataloader = lambda: fold_val_dataloader
        
        # Reset model weights for each fold
        self._reset_model_weights(pl_module)
        
        # Run training for this fold
        trainer.fit_loop.run()
        
        # Collect fold results
        fold_metrics = self._collect_fold_metrics(trainer, pl_module)
        self.all_fold_metrics.append(fold_metrics)
        
        # Save fold results
        if self.save_fold_results:
            self._save_fold_results(fold_metrics, self.current_fold)
        
        # Run validation/test on this fold
        if hasattr(trainer.datamodule, 'test_dataloader') and trainer.datamodule.test_dataloader() is not None:
            test_results = trainer.test(pl_module, dataloaders=trainer.datamodule.test_dataloader())
            fold_metrics['test_results'] = test_results
        
        self.fold_results.append({
            'fold': self.current_fold,
            'train_size': len(train_indices),
            'val_size': len(val_indices),
            'metrics': fold_metrics
        })
        
        rank_zero_info(f"Completed Fold {self.current_fold + 1}")
        rank_zero_info(f"Fold {self.current_fold + 1} Metrics: {fold_metrics}")
        
        self.current_fold += 1
    
    def _reset_model_weights(self, pl_module: LightningModule) -> None:
        """Reset model weights for each fold"""
        # Reinitialize model weights
        def init_weights(m):
            if hasattr(m, 'reset_parameters'):
                m.reset_parameters()
            elif isinstance(m, (torch.nn.Linear, torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d)):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
            elif isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)):
                torch.nn.init.ones_(m.weight)
                torch.nn.init.zeros_(m.bias)
        
        pl_module.apply(init_weights)
        
        # Reset optimizer states if they exist
        if hasattr(pl_module, 'trainer') and pl_module.trainer is not None:
            for optimizer in pl_module.trainer.optimizers:
                optimizer.state = {}
    
    def _collect_fold_metrics(self, trainer: Trainer, pl_module: LightningModule) -> Dict[str, float]:
        """Collect metrics from the current fold"""
        fold_metrics = {}
        
        # Get logged metrics
        if trainer.logged_metrics:
            for key, value in trainer.logged_metrics.items():
                if isinstance(value, torch.Tensor):
                    fold_metrics[key] = value.item()
                else:
                    fold_metrics[key] = value
        
        # Get callback metrics
        if trainer.callback_metrics:
            for key, value in trainer.callback_metrics.items():
                if isinstance(value, torch.Tensor):
                    fold_metrics[key] = value.item()
                else:
                    fold_metrics[key] = value
        
        return fold_metrics
    
    def _save_fold_results(self, fold_metrics: Dict[str, float], fold_idx: int) -> None:
        """Save results for current fold"""
        fold_file = self.results_dir / f"fold_{fold_idx}_results.json"
        
        # Convert tensors to float for JSON serialization
        serializable_metrics = {}
        for key, value in fold_metrics.items():
            if isinstance(value, torch.Tensor):
                serializable_metrics[key] = value.item()
            elif isinstance(value, (np.ndarray, np.number)):
                serializable_metrics[key] = float(value)
            else:
                serializable_metrics[key] = value
        
        with open(fold_file, 'w') as f:
            json.dump(serializable_metrics, f, indent=2)
    
    def on_run_end(self, trainer: Trainer, pl_module: LightningModule) -> Any:
        """Called when k-fold loop completes"""
        # Compute summary statistics
        summary_stats = self._compute_summary_statistics()
        
        # Save overall results
        if self.save_fold_results:
            self._save_summary_results(summary_stats)
        
        # Log summary
        self._log_summary(summary_stats)
        
        # Restore original dataloaders
        trainer.datamodule.train_dataloader = lambda: self.original_train_dataloader
        if self.original_val_dataloader is not None:
            trainer.datamodule.val_dataloader = lambda: self.original_val_dataloader
        
        return summary_stats
    
    def _compute_summary_statistics(self) -> Dict[str, Dict[str, float]]:
        """Compute summary statistics across all folds"""
        if not self.all_fold_metrics:
            return {}
        
        # Collect all metric names
        all_metric_names = set()
        for fold_metrics in self.all_fold_metrics:
            all_metric_names.update(fold_metrics.keys())
        
        summary_stats = {}
        
        for metric_name in all_metric_names:
            metric_values = []
            for fold_metrics in self.all_fold_metrics:
                if metric_name in fold_metrics:
                    value = fold_metrics[metric_name]
                    if isinstance(value, (int, float)):
                        metric_values.append(value)
            
            if metric_values:
                summary_stats[metric_name] = {
                    'mean': np.mean(metric_values),
                    'std': np.std(metric_values),
                    'min': np.min(metric_values),
                    'max': np.max(metric_values),
                    'values': metric_values
                }
        
        return summary_stats
    
    def _save_summary_results(self, summary_stats: Dict[str, Dict[str, float]]) -> None:
        """Save summary statistics"""
        summary_file = self.results_dir / "kfold_summary.json"
        
        # Prepare serializable summary
        serializable_summary = {}
        for metric_name, stats in summary_stats.items():
            serializable_summary[metric_name] = {
                'mean': float(stats['mean']),
                'std': float(stats['std']),
                'min': float(stats['min']),
                'max': float(stats['max']),
                'values': [float(v) for v in stats['values']]
            }
        
        # Add metadata
        metadata = {
            'num_folds': self.num_folds,
            'stratified': self.stratified,
            'shuffle': self.shuffle,
            'random_state': self.random_state,
            'fold_results': self.fold_results
        }
        
        final_results = {
            'metadata': metadata,
            'summary_statistics': serializable_summary,
            'individual_fold_results': self.fold_results
        }
        
        with open(summary_file, 'w') as f:
            json.dump(final_results, f, indent=2)
        
        rank_zero_info(f"K-Fold results saved to {summary_file}")
    
    @rank_zero_only
    def _log_summary(self, summary_stats: Dict[str, Dict[str, float]]) -> None:
        """Log summary statistics"""
        rank_zero_info(f"\n{'='*60}")
        rank_zero_info(f"K-FOLD CROSS-VALIDATION SUMMARY ({self.num_folds} folds)")
        rank_zero_info(f"{'='*60}")
        
        for metric_name, stats in summary_stats.items():
            rank_zero_info(
                f"{metric_name:20s}: "
                f"{stats['mean']:.4f} ± {stats['std']:.4f} "
                f"(min: {stats['min']:.4f}, max: {stats['max']:.4f})"
            )
        
        rank_zero_info(f"{'='*60}\n")
    
    def get_summary_statistics(self) -> Dict[str, Dict[str, float]]:
        """Get summary statistics"""
        return self._compute_summary_statistics()
    
    def get_best_fold(self, metric_name: str, mode: str = 'max') -> Tuple[int, Dict[str, float]]:
        """Get best fold based on a specific metric"""
        if not self.all_fold_metrics:
            return -1, {}
        
        metric_values = []
        for i, fold_metrics in enumerate(self.all_fold_metrics):
            if metric_name in fold_metrics:
                metric_values.append((i, fold_metrics[metric_name]))
        
        if not metric_values:
            return -1, {}
        
        # Find best fold
        if mode == 'max':
            best_fold_idx, best_value = max(metric_values, key=lambda x: x[1])
        else:  # mode == 'min'
            best_fold_idx, best_value = min(metric_values, key=lambda x: x[1])
        
        return best_fold_idx, self.all_fold_metrics[best_fold_idx]


# Convenience function
def create_kfold_loop(
    num_folds: int = 5,
    stratified: bool = False,
    random_state: int = 42,
    **kwargs
) -> KFoldLoop:
    """Create k-fold loop with common settings"""
    return KFoldLoop(
        num_folds=num_folds,
        stratified=stratified,
        random_state=random_state,
        **kwargs
    )