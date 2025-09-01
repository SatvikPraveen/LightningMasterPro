# File: src/lmpro/datamodules/tabular_dm.py

"""
Tabular DataModule for regression and classification with structured data
"""

import torch
from torch.utils.data import DataLoader
from lightning import LightningDataModule
from typing import Optional, Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from ..data.synth_tabular import (
    TabularDatasetConfig,
    SyntheticTabularDataset,
    ComplexTabularDataset,
    TimeVaryingTabularDataset,
    create_synthetic_tabular_dataset,
    create_synthetic_regression_dataset,
    create_synthetic_classification_dataset
)
from ..utils.seed import worker_init_fn


class TabularDataModule(LightningDataModule):
    """
    Lightning DataModule for tabular data tasks (regression, classification)
    """
    
    def __init__(
        self,
        task: str = "classification",
        data_config: Optional[TabularDatasetConfig] = None,
        batch_size: int = 64,
        num_workers: int = 4,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        normalize_features: bool = True,
        dataset_type: str = "simple",
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
        self.normalize_features = normalize_features
        self.dataset_type = dataset_type
        self.split_ratios = split_ratios
        
        # Data configuration
        self.data_config = data_config or TabularDatasetConfig()
        
        # Datasets
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        
        # Feature info
        self.num_features = self.data_config.num_features
        self.num_classes = self.data_config.num_classes if task == "classification" else None
        self.feature_names = None
        self.scaler = None
        
    def prepare_data(self) -> None:
        """Download and prepare data (called once per node)"""
        # Generate synthetic data based on task and type
        self.datasets = create_synthetic_tabular_dataset(
            self.data_config,
            task=self.task,
            splits=["train", "val", "test"],
            split_ratios=self.split_ratios,
            dataset_type=self.dataset_type
        )
    
    def setup(self, stage: Optional[str] = None) -> None:
        """Setup datasets for each stage"""
        if not hasattr(self, 'datasets'):
            self.prepare_data()
        
        if stage == "fit" or stage is None:
            self.train_dataset = self.datasets["train"]
            self.val_dataset = self.datasets["val"]
            
            # Get feature information
            if hasattr(self.train_dataset, 'feature_names'):
                self.feature_names = self.train_dataset.feature_names
            else:
                self.feature_names = [f"feature_{i}" for i in range(self.num_features)]
            
            # Update feature count based on actual data
            sample_x, _ = self.train_dataset[0]
            if isinstance(sample_x, torch.Tensor):
                if sample_x.dim() == 1:
                    self.num_features = sample_x.shape[0]
                elif sample_x.dim() == 2:  # Time-varying data
                    self.num_features = sample_x.shape[1]
        
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
    
    def get_feature_names(self) -> List[str]:
        """Get feature names"""
        return self.feature_names or [f"feature_{i}" for i in range(self.num_features)]
    
    def get_class_names(self) -> List[str]:
        """Get class names for classification tasks"""
        if self.task == "classification":
            return [f"class_{i}" for i in range(self.num_classes)]
        else:
            return []
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """Get information about the dataset"""
        info = {
            "task": self.task,
            "dataset_type": self.dataset_type,
            "num_features": self.num_features,
            "train_size": len(self.train_dataset) if self.train_dataset else 0,
            "val_size": len(self.val_dataset) if self.val_dataset else 0,
            "test_size": len(self.test_dataset) if self.test_dataset else 0,
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "normalize_features": self.normalize_features,
        }
        
        if self.task == "classification":
            info["num_classes"] = self.num_classes
        
        return info
    
    def visualize_batch(self, stage: str = "train", num_samples: int = 5) -> None:
        """Visualize a batch of data"""
        import matplotlib.pyplot as plt
        
        if stage == "train":
            dataloader = self.train_dataloader()
        elif stage == "val":
            dataloader = self.val_dataloader()
        else:
            dataloader = self.test_dataloader()
        
        batch = next(iter(dataloader))
        features, targets = batch
        
        print(f"\n{stage.upper()} Batch Visualization:")
        print(f"Features shape: {features.shape}")
        print(f"Targets shape: {targets.shape}")
        
        # Show feature statistics
        print(f"\nFeature Statistics:")
        for i, feature_name in enumerate(self.get_feature_names()[:min(10, self.num_features)]):
            if features.dim() == 2:  # Regular tabular data
                feature_values = features[:, i]
            elif features.dim() == 3:  # Time-varying data
                feature_values = features[:, -1, i]  # Last time step
            else:
                continue
                
            print(f"  {feature_name}: mean={feature_values.mean():.3f}, std={feature_values.std():.3f}")
        
        # Show target distribution
        if self.task == "classification":
            print(f"\nTarget Distribution in Batch:")
            unique_targets, counts = torch.unique(targets, return_counts=True)
            for target, count in zip(unique_targets, counts):
                print(f"  Class {target.item()}: {count.item()} samples")
        else:
            print(f"\nTarget Statistics:")
            print(f"  Mean: {targets.mean():.3f}")
            print(f"  Std: {targets.std():.3f}")
            print(f"  Min: {targets.min():.3f}")
            print(f"  Max: {targets.max():.3f}")
        
        # Plot first few features if 2D data
        if features.dim() == 2 and self.num_features >= 4:
            fig, axes = plt.subplots(2, 2, figsize=(10, 8))
            axes = axes.flatten()
            
            for i in range(min(4, self.num_features)):
                feature_values = features[:num_samples, i].numpy()
                target_values = targets[:num_samples].numpy()
                
                if self.task == "classification":
                    scatter = axes[i].scatter(range(len(feature_values)), feature_values, 
                                            c=target_values, cmap='tab10', alpha=0.7)
                    plt.colorbar(scatter, ax=axes[i])
                else:
                    axes[i].scatter(feature_values, target_values, alpha=0.7)
                    axes[i].set_xlabel(f'{self.get_feature_names()[i]}')
                    axes[i].set_ylabel('Target')
                
                axes[i].set_title(f'{self.get_feature_names()[i]}')
                axes[i].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
        
        # Show sample data points
        print(f"\nSample Data Points:")
        for i in range(min(num_samples, len(features))):
            if features.dim() == 2:
                feature_str = ", ".join([f"{val:.3f}" for val in features[i][:5].tolist()])
                if self.num_features > 5:
                    feature_str += "..."
            else:
                feature_str = f"Shape: {features[i].shape}"
            
            target_str = f"{targets[i].item():.3f}" if self.task == "regression" else f"Class {targets[i].item()}"
            print(f"  Sample {i}: [{feature_str}] -> {target_str}")
    
    def compute_class_weights(self) -> torch.Tensor:
        """Compute class weights for imbalanced classification datasets"""
        if self.task != "classification" or self.train_dataset is None:
            return torch.ones(self.num_classes) if self.num_classes else torch.tensor([1.0])
        
        # Count class frequencies
        class_counts = torch.zeros(self.num_classes)
        
        for _, target in self.train_dataset:
            if isinstance(target, torch.Tensor):
                class_counts[target.item()] += 1
        
        # Compute inverse frequency weights
        total_samples = class_counts.sum()
        class_weights = total_samples / (self.num_classes * class_counts + 1e-8)
        
        return class_weights
    
    def get_feature_importance_data(self) -> Dict[str, torch.Tensor]:
        """Get data for feature importance analysis"""
        if self.train_dataset is None:
            return {}
        
        # Collect all features and targets
        all_features = []
        all_targets = []
        
        for features, target in self.train_dataset:
            if features.dim() == 2:  # Time-varying data - use last timestep
                features = features[-1]
            all_features.append(features)
            all_targets.append(target)
        
        features_tensor = torch.stack(all_features)
        targets_tensor = torch.stack(all_targets)
        
        return {
            "features": features_tensor,
            "targets": targets_tensor,
            "feature_names": self.get_feature_names()
        }
    
    def create_dataframe_sample(self, num_samples: int = 100) -> pd.DataFrame:
        """Create a pandas DataFrame sample for analysis"""
        if self.train_dataset is None:
            return pd.DataFrame()
        
        samples = []
        targets = []
        
        for i, (features, target) in enumerate(self.train_dataset):
            if i >= num_samples:
                break
            
            if features.dim() == 2:  # Time-varying data
                features = features[-1]  # Last timestep
            
            samples.append(features.numpy())
            targets.append(target.item())
        
        # Create DataFrame
        df = pd.DataFrame(samples, columns=self.get_feature_names())
        df['target'] = targets
        
        return df
    
    def __repr__(self) -> str:
        """String representation"""
        info = self.get_dataset_info()
        return (
            f"TabularDataModule(\n"
            f"  task={info['task']},\n"
            f"  dataset_type={info['dataset_type']},\n"
            f"  num_features={info['num_features']},\n"
            f"  num_classes={info.get('num_classes', 'N/A')},\n"
            f"  train_size={info['train_size']},\n"
            f"  val_size={info['val_size']},\n"
            f"  test_size={info['test_size']},\n"
            f"  batch_size={info['batch_size']}\n"
            f")"
        )


# Convenience functions
def get_regression_datamodule(
    num_features: int = 20,
    batch_size: int = 64,
    dataset_type: str = "simple",
    **kwargs
) -> TabularDataModule:
    """Get a tabular datamodule for regression"""
    config = TabularDatasetConfig(
        num_features=num_features,
        **kwargs
    )
    
    return TabularDataModule(
        task="regression",
        data_config=config,
        batch_size=batch_size,
        dataset_type=dataset_type
    )


def get_classification_datamodule(
    num_features: int = 20,
    num_classes: int = 3,
    batch_size: int = 64,
    dataset_type: str = "simple",
    **kwargs
) -> TabularDataModule:
    """Get a tabular datamodule for classification"""
    config = TabularDatasetConfig(
        num_features=num_features,
        num_classes=num_classes,
        **kwargs
    )
    
    return TabularDataModule(
        task="classification",
        data_config=config,
        batch_size=batch_size,
        dataset_type=dataset_type
    )