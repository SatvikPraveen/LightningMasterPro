# File: src/lmpro/datamodules/vision_dm.py

"""
Vision DataModule for image classification and segmentation tasks
"""

import torch
from torch.utils.data import DataLoader, random_split
from lightning import LightningDataModule
from torchvision import transforms
from typing import Optional, Dict, Any, Callable, Tuple
import numpy as np

from ..data.synth_vision import (
    VisionDatasetConfig, 
    SyntheticImageDataset, 
    SyntheticSegmentationDataset,
    create_synthetic_image_dataset,
    create_synthetic_segmentation_dataset
)
from ..utils.seed import worker_init_fn


class VisionDataModule(LightningDataModule):
    """
    Lightning DataModule for vision tasks (classification, segmentation)
    """
    
    def __init__(
        self,
        task: str = "classification",
        data_config: Optional[VisionDatasetConfig] = None,
        batch_size: int = 32,
        num_workers: int = 4,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        train_transforms: Optional[Callable] = None,
        val_transforms: Optional[Callable] = None,
        test_transforms: Optional[Callable] = None,
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
        self.split_ratios = split_ratios
        
        # Data configuration
        self.data_config = data_config or VisionDatasetConfig()
        
        # Transforms
        self.train_transforms = train_transforms or self._get_default_train_transforms()
        self.val_transforms = val_transforms or self._get_default_val_transforms()
        self.test_transforms = test_transforms or self._get_default_test_transforms()
        
        # Datasets
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        
        # Data info
        self.dims = None
        self.num_classes = self.data_config.num_classes
        
    def _get_default_train_transforms(self) -> transforms.Compose:
        """Get default training transforms"""
        if self.task == "classification":
            return transforms.Compose([
                transforms.ToPILImage() if not hasattr(transforms, 'ToTensor') else transforms.Lambda(lambda x: x),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:  # segmentation
            return transforms.Compose([
                transforms.ToPILImage() if not hasattr(transforms, 'ToTensor') else transforms.Lambda(lambda x: x),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    
    def _get_default_val_transforms(self) -> transforms.Compose:
        """Get default validation transforms"""
        return transforms.Compose([
            transforms.ToPILImage() if not hasattr(transforms, 'ToTensor') else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def _get_default_test_transforms(self) -> transforms.Compose:
        """Get default test transforms"""
        return self._get_default_val_transforms()
    
    def prepare_data(self) -> None:
        """Download and prepare data (called once per node)"""
        # Generate synthetic data if needed
        if self.task == "classification":
            self.datasets = create_synthetic_image_dataset(
                self.data_config,
                splits=["train", "val", "test"],
                split_ratios=self.split_ratios
            )
        elif self.task == "segmentation":
            self.datasets = create_synthetic_segmentation_dataset(
                self.data_config,
                splits=["train", "val", "test"],
                split_ratios=self.split_ratios
            )
        else:
            raise ValueError(f"Unknown task: {self.task}")
    
    def setup(self, stage: Optional[str] = None) -> None:
        """Setup datasets for each stage"""
        if not hasattr(self, 'datasets'):
            self.prepare_data()
        
        if stage == "fit" or stage is None:
            self.train_dataset = self.datasets["train"]
            self.val_dataset = self.datasets["val"]
            
            # Set transforms
            self.train_dataset.transform = self.train_transforms
            self.val_dataset.transform = self.val_transforms
            
            # Calculate dimensions
            sample_input, _ = self.train_dataset[0]
            if isinstance(sample_input, torch.Tensor):
                self.dims = tuple(sample_input.shape)
            else:
                # If transforms haven't been applied yet
                self.dims = (self.data_config.num_channels,) + self.data_config.image_size
        
        if stage == "test" or stage is None:
            self.test_dataset = self.datasets["test"]
            self.test_dataset.transform = self.test_transforms
        
        if stage == "predict" or stage is None:
            # Use test dataset for prediction
            if not hasattr(self, 'test_dataset') or self.test_dataset is None:
                self.test_dataset = self.datasets["test"]
                self.test_dataset.transform = self.test_transforms
    
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
            drop_last=True,
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
    
    def get_class_names(self) -> list:
        """Get class names for the dataset"""
        if self.task == "classification":
            return [f"class_{i}" for i in range(self.num_classes)]
        elif self.task == "segmentation":
            return ["background"] + [f"object_{i}" for i in range(1, self.num_classes)]
        else:
            return []
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """Get information about the dataset"""
        return {
            "task": self.task,
            "num_classes": self.num_classes,
            "input_dims": self.dims,
            "train_size": len(self.train_dataset) if self.train_dataset else 0,
            "val_size": len(self.val_dataset) if self.val_dataset else 0,
            "test_size": len(self.test_dataset) if self.test_dataset else 0,
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
        }
    
    def visualize_batch(self, stage: str = "train", num_samples: int = 8) -> None:
        """Visualize a batch of data"""
        import matplotlib.pyplot as plt
        
        if stage == "train":
            dataloader = self.train_dataloader()
        elif stage == "val":
            dataloader = self.val_dataloader()
        else:
            dataloader = self.test_dataloader()
        
        batch = next(iter(dataloader))
        images, targets = batch
        
        # Denormalize images for visualization
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        
        fig, axes = plt.subplots(2, num_samples // 2, figsize=(15, 6))
        axes = axes.flatten()
        
        for i in range(min(num_samples, len(images))):
            # Denormalize
            img = images[i] * std + mean
            img = torch.clamp(img, 0, 1)
            
            # Convert to numpy and transpose
            if img.shape[0] == 3:  # RGB
                img_np = img.permute(1, 2, 0).numpy()
            else:  # Grayscale
                img_np = img.squeeze().numpy()
            
            axes[i].imshow(img_np, cmap='gray' if len(img_np.shape) == 2 else None)
            
            if self.task == "classification":
                axes[i].set_title(f'Class: {targets[i].item()}')
            elif self.task == "segmentation":
                axes[i].set_title(f'Segmentation')
                # Could overlay mask here
            
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def compute_class_weights(self) -> torch.Tensor:
        """Compute class weights for imbalanced datasets"""
        if self.task != "classification" or self.train_dataset is None:
            return torch.ones(self.num_classes)
        
        # Count class frequencies
        class_counts = torch.zeros(self.num_classes)
        
        for _, target in self.train_dataset:
            if isinstance(target, torch.Tensor):
                if target.dim() == 0:  # Single class
                    class_counts[target.item()] += 1
                else:  # Multi-class or one-hot
                    if target.dim() == 1 and len(target) == self.num_classes:
                        class_counts += target
                    else:
                        class_counts[target.argmax().item()] += 1
        
        # Compute inverse frequency weights
        total_samples = class_counts.sum()
        class_weights = total_samples / (self.num_classes * class_counts + 1e-8)
        
        return class_weights
    
    def __repr__(self) -> str:
        """String representation"""
        info = self.get_dataset_info()
        return (
            f"VisionDataModule(\n"
            f"  task={info['task']},\n"
            f"  num_classes={info['num_classes']},\n"
            f"  input_dims={info['input_dims']},\n"
            f"  train_size={info['train_size']},\n"
            f"  val_size={info['val_size']},\n"
            f"  test_size={info['test_size']},\n"
            f"  batch_size={info['batch_size']}\n"
            f")"
        )


# Convenience functions for common vision tasks
def get_classification_datamodule(
    num_classes: int = 10,
    image_size: Tuple[int, int] = (64, 64),
    batch_size: int = 32,
    **kwargs
) -> VisionDataModule:
    """Get a vision datamodule for classification"""
    config = VisionDatasetConfig(
        num_classes=num_classes,
        image_size=image_size,
        **kwargs
    )
    
    return VisionDataModule(
        task="classification",
        data_config=config,
        batch_size=batch_size
    )


def get_segmentation_datamodule(
    num_classes: int = 4,
    image_size: Tuple[int, int] = (64, 64),
    batch_size: int = 16,
    **kwargs
) -> VisionDataModule:
    """Get a vision datamodule for segmentation"""
    config = VisionDatasetConfig(
        num_classes=num_classes,
        image_size=image_size,
        **kwargs
    )
    
    return VisionDataModule(
        task="segmentation",
        data_config=config,
        batch_size=batch_size
    )