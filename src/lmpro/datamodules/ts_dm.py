# File: src/lmpro/datamodules/ts_dm.py

"""
Time Series DataModule for forecasting and classification tasks
"""

import torch
from torch.utils.data import DataLoader
from lightning import LightningDataModule
from typing import Optional, Dict, Any, List, Tuple
import numpy as np
import matplotlib.pyplot as plt

from ..data.synth_timeseries import (
    TimeSeriesDatasetConfig,
    SyntheticTimeSeriesDataset,
    MultiVariateTimeSeriesDataset,
    AnomalyTimeSeriesDataset,
    create_synthetic_timeseries_dataset,
    create_synthetic_forecasting_dataset
)
from ..utils.seed import worker_init_fn


class TimeSeriesDataModule(LightningDataModule):
    """
    Lightning DataModule for time series tasks (forecasting, classification, anomaly detection)
    """
    
    def __init__(
        self,
        task: str = "forecasting",
        data_config: Optional[TimeSeriesDatasetConfig] = None,
        batch_size: int = 32,
        num_workers: int = 4,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        dataset_type: str = "univariate",
        split_ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15),
        **kwargs
    ):
        super().__init__()
        
        # Save hyperparameters
        self.save_hyperparameters(logger=False)
        
        self.task = task
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.dataset_type = dataset_type
        self.split_ratios = split_ratios
        
        # Data configuration
        self.data_config = data_config or TimeSeriesDatasetConfig()
        
        # Datasets
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        
        # Time series info
        self.sequence_length = self.data_config.sequence_length
        self.prediction_horizon = self.data_config.prediction_horizon
        self.num_features = self.data_config.num_features
        
    def prepare_data(self) -> None:
        """Download and prepare data (called once per node)"""
        # Generate synthetic time series data
        self.datasets = create_synthetic_timeseries_dataset(
            self.data_config,
            task=self.task,
            dataset_type=self.dataset_type,
            splits=["train", "val", "test"],
            split_ratios=self.split_ratios
        )
    
    def setup(self, stage: Optional[str] = None) -> None:
        """Setup datasets for each stage"""
        if not hasattr(self, 'datasets'):
            self.prepare_data()
        
        if stage == "fit" or stage is None:
            self.train_dataset = self.datasets["train"]
            self.val_dataset = self.datasets["val"]
            
            # Get actual dimensions from data
            sample_seq, sample_target = self.train_dataset[0]
            if isinstance(sample_seq, torch.Tensor):
                if sample_seq.dim() == 1:
                    self.sequence_length = sample_seq.shape[0]
                    self.num_features = 1
                elif sample_seq.dim() == 2:
                    self.sequence_length = sample_seq.shape[0]
                    self.num_features = sample_seq.shape[1]
        
        if stage == "test" or stage is None:
            self.test_dataset = self.datasets["test"]
        
        if stage == "predict" or stage is None:
            if not hasattr(self, 'test_dataset') or self.test_dataset is None:
                self.test_dataset = self.datasets["test"]
    
    def train_dataloader(self) -> DataLoader:
        """Create training dataloader"""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            worker_init_fn=worker_init_fn,
            drop_last=False,
        )
    
    def val_dataloader(self) -> DataLoader:
        """Create validation dataloader"""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            worker_init_fn=worker_init_fn,
        )
    
    def test_dataloader(self) -> DataLoader:
        """Create test dataloader"""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            worker_init_fn=worker_init_fn,
        )
    
    def predict_dataloader(self) -> DataLoader:
        """Create prediction dataloader"""
        return self.test_dataloader()
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """Get information about the dataset"""
        return {
            "task": self.task,
            "dataset_type": self.dataset_type,
            "sequence_length": self.sequence_length,
            "prediction_horizon": self.prediction_horizon,
            "num_features": self.num_features,
            "train_size": len(self.train_dataset) if self.train_dataset else 0,
            "val_size": len(self.val_dataset) if self.val_dataset else 0,
            "test_size": len(self.test_dataset) if self.test_dataset else 0,
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
        }
    
    def visualize_batch(self, stage: str = "train", num_samples: int = 3) -> None:
        """Visualize a batch of time series data"""
        if stage == "train":
            dataloader = self.train_dataloader()
        elif stage == "val":
            dataloader = self.val_dataloader()
        else:
            dataloader = self.test_dataloader()
        
        batch = next(iter(dataloader))
        sequences, targets = batch
        
        print(f"\n{stage.upper()} Batch Visualization:")
        print(f"Sequences shape: {sequences.shape}")
        print(f"Targets shape: {targets.shape}")
        
        # Create plots
        fig, axes = plt.subplots(num_samples, 1, figsize=(15, 4 * num_samples))
        if num_samples == 1:
            axes = [axes]
        
        for i in range(min(num_samples, len(sequences))):
            sequence = sequences[i]
            target = targets[i]
            
            # Plot sequence
            if sequence.dim() == 1:  # Univariate
                axes[i].plot(sequence.numpy(), 'b-', label='Input Sequence', linewidth=2)
            else:  # Multivariate
                for feature_idx in range(min(3, sequence.shape[1])):  # Plot up to 3 features
                    axes[i].plot(sequence[:, feature_idx].numpy(), 
                               label=f'Feature {feature_idx}', alpha=0.7)
            
            # Plot target based on task
            if self.task == "forecasting":
                # Plot forecast target
                forecast_start = sequence.shape[0]
                if target.dim() == 0:  # Single point forecast
                    axes[i].axvline(x=forecast_start, color='r', linestyle='--', alpha=0.5)
                    axes[i].scatter([forecast_start], [target.item()], 
                                  color='r', s=50, label='Target')
                else:  # Multi-step forecast
                    forecast_indices = range(forecast_start, forecast_start + len(target))
                    axes[i].plot(forecast_indices, target.numpy(), 
                               'r--', linewidth=2, label='Target Forecast')
            else:  # Classification or anomaly detection
                class_label = target.item() if target.dim() == 0 else target.argmax().item()
                axes[i].set_title(f'Sample {i} - Class: {class_label}')
            
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
            axes[i].set_xlabel('Time Steps')
            axes[i].set_ylabel('Value')
            
            if self.task != "forecasting":
                axes[i].set_title(f'Sample {i} - Label: {target.item() if target.dim() == 0 else target.argmax().item()}')
        
        plt.tight_layout()
        plt.show()
        
        # Print statistics
        print(f"\nSequence Statistics:")
        if sequences.dim() == 2:  # Univariate batch
            print(f"  Mean: {sequences.mean():.4f}")
            print(f"  Std: {sequences.std():.4f}")
            print(f"  Min: {sequences.min():.4f}")
            print(f"  Max: {sequences.max():.4f}")
        elif sequences.dim() == 3:  # Multivariate batch
            for feature_idx in range(min(3, sequences.shape[2])):
                feature_data = sequences[:, :, feature_idx]
                print(f"  Feature {feature_idx} - Mean: {feature_data.mean():.4f}, Std: {feature_data.std():.4f}")
        
        if self.task == "forecasting":
            print(f"\nTarget Statistics:")
            print(f"  Mean: {targets.mean():.4f}")
            print(f"  Std: {targets.std():.4f}")
            print(f"  Shape: {targets.shape}")
        elif self.task in ["classification", "anomaly"]:
            print(f"\nTarget Distribution:")
            unique_targets, counts = torch.unique(targets, return_counts=True)
            for target_val, count in zip(unique_targets, counts):
                print(f"  Class {target_val.item()}: {count.item()} samples")
    
    def compute_sequence_statistics(self) -> Dict[str, Any]:
        """Compute statistics across all sequences"""
        if self.train_dataset is None:
            return {}
        
        all_sequences = []
        all_targets = []
        
        # Sample a subset for efficiency
        sample_size = min(100, len(self.train_dataset))
        for i in range(sample_size):
            seq, target = self.train_dataset[i]
            all_sequences.append(seq)
            all_targets.append(target)
        
        # Stack sequences
        sequences_tensor = torch.stack(all_sequences)
        targets_tensor = torch.stack(all_targets) if self.task == "forecasting" else torch.tensor([t.item() for t in all_targets])
        
        stats = {
            "sequence_length": self.sequence_length,
            "num_features": self.num_features,
            "num_samples_analyzed": sample_size,
        }
        
        # Sequence statistics
        if sequences_tensor.dim() == 2:  # Univariate
            stats.update({
                "seq_mean": sequences_tensor.mean().item(),
                "seq_std": sequences_tensor.std().item(),
                "seq_min": sequences_tensor.min().item(),
                "seq_max": sequences_tensor.max().item(),
            })
        elif sequences_tensor.dim() == 3:  # Multivariate
            stats["per_feature_stats"] = {}
            for i in range(min(5, sequences_tensor.shape[2])):  # Limit to first 5 features
                feature_data = sequences_tensor[:, :, i]
                stats["per_feature_stats"][f"feature_{i}"] = {
                    "mean": feature_data.mean().item(),
                    "std": feature_data.std().item(),
                    "min": feature_data.min().item(),
                    "max": feature_data.max().item(),
                }
        
        # Target statistics
        if self.task == "forecasting":
            stats.update({
                "target_mean": targets_tensor.mean().item(),
                "target_std": targets_tensor.std().item(),
                "prediction_horizon": self.prediction_horizon,
            })
        elif self.task in ["classification", "anomaly"]:
            unique_classes, counts = torch.unique(targets_tensor, return_counts=True)
            stats["class_distribution"] = {
                f"class_{cls.item()}": count.item() 
                for cls, count in zip(unique_classes, counts)
            }
        
        return stats
    
    def create_forecast_visualization(self, predictions: torch.Tensor, targets: torch.Tensor, 
                                    inputs: torch.Tensor, num_samples: int = 3) -> None:
        """Create visualization comparing predictions vs targets for forecasting"""
        if self.task != "forecasting":
            print("Forecast visualization only available for forecasting tasks")
            return
        
        fig, axes = plt.subplots(num_samples, 1, figsize=(15, 4 * num_samples))
        if num_samples == 1:
            axes = [axes]
        
        for i in range(min(num_samples, len(predictions))):
            input_seq = inputs[i]
            pred_seq = predictions[i]
            target_seq = targets[i]
            
            # Plot input sequence
            input_length = len(input_seq)
            if input_seq.dim() == 1:  # Univariate
                axes[i].plot(range(input_length), input_seq.numpy(), 
                           'b-', label='Input', linewidth=2)
            else:  # Plot first feature for multivariate
                axes[i].plot(range(input_length), input_seq[:, 0].numpy(), 
                           'b-', label='Input (Feature 0)', linewidth=2)
            
            # Plot predictions and targets
            forecast_indices = range(input_length, input_length + len(pred_seq))
            
            if pred_seq.dim() == 0:  # Single point
                axes[i].scatter([input_length], [pred_seq.item()], 
                              color='red', s=50, label='Prediction')
                axes[i].scatter([input_length], [target_seq.item()], 
                              color='green', s=50, label='Target')
            else:  # Multi-step
                axes[i].plot(forecast_indices, pred_seq.numpy(), 
                           'r--', label='Prediction', linewidth=2)
                axes[i].plot(forecast_indices, target_seq.numpy(), 
                           'g--', label='Target', linewidth=2)
            
            axes[i].axvline(x=input_length, color='black', linestyle=':', alpha=0.5)
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
            axes[i].set_title(f'Forecast Sample {i}')
            axes[i].set_xlabel('Time Steps')
            axes[i].set_ylabel('Value')
        
        plt.tight_layout()
        plt.show()
    
    def __repr__(self) -> str:
        """String representation"""
        info = self.get_dataset_info()
        return (
            f"TimeSeriesDataModule(\n"
            f"  task={info['task']},\n"
            f"  dataset_type={info['dataset_type']},\n"
            f"  sequence_length={info['sequence_length']},\n"
            f"  prediction_horizon={info['prediction_horizon']},\n"
            f"  num_features={info['num_features']},\n"
            f"  train_size={info['train_size']},\n"
            f"  val_size={info['val_size']},\n"
            f"  test_size={info['test_size']},\n"
            f"  batch_size={info['batch_size']}\n"
            f")"
        )


# Convenience functions
def get_forecasting_datamodule(
    sequence_length: int = 100,
    prediction_horizon: int = 10,
    num_features: int = 1,
    batch_size: int = 32,
    dataset_type: str = "univariate",
    **kwargs
) -> TimeSeriesDataModule:
    """Get a time series datamodule for forecasting"""
    config = TimeSeriesDatasetConfig(
        sequence_length=sequence_length,
        prediction_horizon=prediction_horizon,
        num_features=num_features,
        **kwargs
    )
    
    return TimeSeriesDataModule(
        task="forecasting",
        data_config=config,
        batch_size=batch_size,
        dataset_type=dataset_type
    )


def get_anomaly_detection_datamodule(
    sequence_length: int = 50,
    num_features: int = 1,
    batch_size: int = 32,
    **kwargs
) -> TimeSeriesDataModule:
    """Get a time series datamodule for anomaly detection"""
    config = TimeSeriesDatasetConfig(
        sequence_length=sequence_length,
        num_features=num_features,
        **kwargs
    )
    
    return TimeSeriesDataModule(
        task="anomaly",
        data_config=config,
        batch_size=batch_size,
        dataset_type="anomaly"
    )


def get_timeseries_classification_datamodule(
    sequence_length: int = 50,
    num_features: int = 1,
    batch_size: int = 32,
    dataset_type: str = "univariate",
    **kwargs
) -> TimeSeriesDataModule:
    """Get a time series datamodule for classification"""
    config = TimeSeriesDatasetConfig(
        sequence_length=sequence_length,
        num_features=num_features,
        **kwargs
    )
    
    return TimeSeriesDataModule(
        task="classification",
        data_config=config,
        batch_size=batch_size,
        dataset_type=dataset_type
    )