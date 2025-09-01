# File: src/lmpro/data/synth_vision.py

"""
Synthetic vision data generation for classification and segmentation tasks
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List, Optional, Union
from dataclasses import dataclass
from pathlib import Path
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from PIL import Image, ImageDraw
import random


@dataclass
class VisionDatasetConfig:
    """Configuration for synthetic vision datasets"""
    num_samples: int = 1000
    image_size: Tuple[int, int] = (64, 64)
    num_channels: int = 3
    num_classes: int = 10
    noise_level: float = 0.1
    background_complexity: float = 0.3
    save_path: Optional[str] = "data/synthetic/vision"


class SyntheticImageDataset(Dataset):
    """Synthetic image dataset with configurable patterns"""
    
    def __init__(
        self,
        config: VisionDatasetConfig,
        split: str = "train",
        transform=None
    ):
        self.config = config
        self.split = split
        self.transform = transform
        
        # Generate data
        self.images, self.labels = self._generate_data()
        
    def _generate_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate synthetic image data with different patterns"""
        images = []
        labels = []
        
        np.random.seed(42 if self.split == "train" else 24)
        torch.manual_seed(42 if self.split == "train" else 24)
        
        for i in range(self.config.num_samples):
            # Random class
            label = np.random.randint(0, self.config.num_classes)
            
            # Generate image based on class
            image = self._create_class_image(label)
            
            images.append(image)
            labels.append(label)
        
        images = torch.stack(images)
        labels = torch.tensor(labels, dtype=torch.long)
        
        return images, labels
    
    def _create_class_image(self, class_id: int) -> torch.Tensor:
        """Create an image for a specific class with distinct patterns"""
        h, w = self.config.image_size
        c = self.config.num_channels
        
        # Create base image with background
        image = torch.randn(c, h, w) * self.config.background_complexity
        
        # Add class-specific patterns
        if class_id == 0:  # Circles
            self._add_circles(image)
        elif class_id == 1:  # Squares
            self._add_squares(image)
        elif class_id == 2:  # Lines
            self._add_lines(image)
        elif class_id == 3:  # Gradients
            self._add_gradients(image)
        elif class_id == 4:  # Checkerboard
            self._add_checkerboard(image)
        elif class_id == 5:  # Waves
            self._add_waves(image)
        elif class_id == 6:  # Stars
            self._add_stars(image)
        elif class_id == 7:  # Grid
            self._add_grid(image)
        elif class_id == 8:  # Noise pattern
            self._add_noise_pattern(image)
        else:  # Mixed pattern
            self._add_mixed_pattern(image)
        
        # Add noise
        noise = torch.randn_like(image) * self.config.noise_level
        image = image + noise
        
        # Normalize to [0, 1]
        image = torch.clamp(image, 0, 1)
        
        return image
    
    def _add_circles(self, image: torch.Tensor) -> None:
        """Add circular patterns"""
        h, w = image.shape[-2:]
        y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
        
        # Random circles
        for _ in range(random.randint(1, 3)):
            center_y, center_x = random.randint(h//4, 3*h//4), random.randint(w//4, 3*w//4)
            radius = random.randint(h//8, h//4)
            mask = ((y - center_y) ** 2 + (x - center_x) ** 2) <= radius ** 2
            for c in range(image.shape[0]):
                image[c][mask] = 0.8 + random.random() * 0.2
    
    def _add_squares(self, image: torch.Tensor) -> None:
        """Add square patterns"""
        h, w = image.shape[-2:]
        for _ in range(random.randint(1, 3)):
            size = random.randint(h//8, h//4)
            y1 = random.randint(0, h - size)
            x1 = random.randint(0, w - size)
            for c in range(image.shape[0]):
                image[c, y1:y1+size, x1:x1+size] = 0.8 + random.random() * 0.2
    
    def _add_lines(self, image: torch.Tensor) -> None:
        """Add line patterns"""
        h, w = image.shape[-2:]
        for _ in range(random.randint(2, 5)):
            if random.random() > 0.5:  # Horizontal
                y = random.randint(0, h-1)
                for c in range(image.shape[0]):
                    image[c, y, :] = 0.8 + random.random() * 0.2
            else:  # Vertical
                x = random.randint(0, w-1)
                for c in range(image.shape[0]):
                    image[c, :, x] = 0.8 + random.random() * 0.2
    
    def _add_gradients(self, image: torch.Tensor) -> None:
        """Add gradient patterns"""
        h, w = image.shape[-2:]
        y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
        grad = (x.float() / w + y.float() / h) / 2
        for c in range(image.shape[0]):
            image[c] = image[c] * 0.3 + grad * 0.7
    
    def _add_checkerboard(self, image: torch.Tensor) -> None:
        """Add checkerboard pattern"""
        h, w = image.shape[-2:]
        size = random.randint(4, 8)
        y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
        pattern = ((y // size) + (x // size)) % 2
        for c in range(image.shape[0]):
            image[c] = image[c] * 0.3 + pattern.float() * 0.7
    
    def _add_waves(self, image: torch.Tensor) -> None:
        """Add wave patterns"""
        h, w = image.shape[-2:]
        y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
        freq = random.uniform(0.1, 0.3)
        wave = torch.sin(x.float() * freq) * torch.cos(y.float() * freq)
        for c in range(image.shape[0]):
            image[c] = image[c] * 0.5 + (wave + 1) * 0.25
    
    def _add_stars(self, image: torch.Tensor) -> None:
        """Add star patterns"""
        h, w = image.shape[-2:]
        for _ in range(random.randint(1, 3)):
            center_y, center_x = random.randint(h//4, 3*h//4), random.randint(w//4, 3*w//4)
            for angle in np.linspace(0, 2*np.pi, 8, endpoint=False):
                end_y = center_y + int(h//6 * np.sin(angle))
                end_x = center_x + int(w//6 * np.cos(angle))
                if 0 <= end_y < h and 0 <= end_x < w:
                    for c in range(image.shape[0]):
                        image[c, end_y, end_x] = 1.0
    
    def _add_grid(self, image: torch.Tensor) -> None:
        """Add grid pattern"""
        h, w = image.shape[-2:]
        spacing = random.randint(8, 16)
        for y in range(0, h, spacing):
            for c in range(image.shape[0]):
                image[c, y, :] = 0.7
        for x in range(0, w, spacing):
            for c in range(image.shape[0]):
                image[c, :, x] = 0.7
    
    def _add_noise_pattern(self, image: torch.Tensor) -> None:
        """Add structured noise pattern"""
        noise = torch.randn_like(image) * 0.3
        image += noise
    
    def _add_mixed_pattern(self, image: torch.Tensor) -> None:
        """Add mixed patterns"""
        patterns = [self._add_circles, self._add_squares, self._add_lines]
        chosen_pattern = random.choice(patterns)
        chosen_pattern(image)
    
    def __len__(self) -> int:
        return len(self.images)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image, label = self.images[idx], self.labels[idx]
        
        if self.transform:
            # Convert to PIL for transforms
            if image.dim() == 3:
                image = image.permute(1, 2, 0)
            image = Image.fromarray((image * 255).numpy().astype(np.uint8))
            image = self.transform(image)
        
        return image, label


class SyntheticSegmentationDataset(Dataset):
    """Synthetic segmentation dataset"""
    
    def __init__(
        self,
        config: VisionDatasetConfig,
        split: str = "train",
        transform=None
    ):
        self.config = config
        self.split = split
        self.transform = transform
        
        # Generate data
        self.images, self.masks = self._generate_segmentation_data()
    
    def _generate_segmentation_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate synthetic segmentation data"""
        images = []
        masks = []
        
        np.random.seed(42 if self.split == "train" else 24)
        
        for i in range(self.config.num_samples):
            image, mask = self._create_segmentation_pair()
            images.append(image)
            masks.append(mask)
        
        images = torch.stack(images)
        masks = torch.stack(masks)
        
        return images, masks
    
    def _create_segmentation_pair(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create image and segmentation mask pair"""
        h, w = self.config.image_size
        c = self.config.num_channels
        
        # Background image
        image = torch.randn(c, h, w) * 0.1 + 0.3
        mask = torch.zeros(h, w, dtype=torch.long)
        
        # Add geometric shapes with labels
        num_objects = random.randint(1, 4)
        
        for obj_id in range(1, num_objects + 1):
            shape_type = random.choice(['circle', 'rectangle', 'triangle'])
            
            if shape_type == 'circle':
                center_y = random.randint(h//4, 3*h//4)
                center_x = random.randint(w//4, 3*w//4)
                radius = random.randint(h//10, h//6)
                
                y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
                circle_mask = ((y - center_y) ** 2 + (x - center_x) ** 2) <= radius ** 2
                
                mask[circle_mask] = obj_id
                for ch in range(c):
                    image[ch][circle_mask] = random.uniform(0.6, 1.0)
                    
            elif shape_type == 'rectangle':
                size_h = random.randint(h//8, h//4)
                size_w = random.randint(w//8, w//4)
                y1 = random.randint(0, h - size_h)
                x1 = random.randint(0, w - size_w)
                
                mask[y1:y1+size_h, x1:x1+size_w] = obj_id
                for ch in range(c):
                    image[ch, y1:y1+size_h, x1:x1+size_w] = random.uniform(0.6, 1.0)
        
        # Add noise
        noise = torch.randn_like(image) * self.config.noise_level
        image = torch.clamp(image + noise, 0, 1)
        
        return image, mask
    
    def __len__(self) -> int:
        return len(self.images)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image, mask = self.images[idx], self.masks[idx]
        
        if self.transform:
            # Apply same transform to image and mask
            seed = torch.randint(0, 2**32, (1,)).item()
            torch.manual_seed(seed)
            if image.dim() == 3:
                image = image.permute(1, 2, 0)
            image_pil = Image.fromarray((image * 255).numpy().astype(np.uint8))
            image = self.transform(image_pil)
            
            torch.manual_seed(seed)
            mask_pil = Image.fromarray(mask.numpy().astype(np.uint8))
            if hasattr(self.transform, 'transforms'):
                # Apply only geometric transforms to mask
                for t in self.transform.transforms:
                    if 'Flip' in str(type(t)) or 'Rotation' in str(type(t)):
                        mask_pil = t(mask_pil)
            mask = torch.tensor(np.array(mask_pil), dtype=torch.long)
        
        return image, mask


def create_synthetic_image_dataset(
    config: VisionDatasetConfig,
    splits: List[str] = ["train", "val", "test"],
    split_ratios: List[float] = [0.7, 0.15, 0.15]
) -> dict:
    """Create synthetic image classification datasets"""
    datasets = {}
    
    total_samples = config.num_samples
    split_sizes = [int(ratio * total_samples) for ratio in split_ratios]
    
    for split, size in zip(splits, split_sizes):
        split_config = VisionDatasetConfig(
            num_samples=size,
            image_size=config.image_size,
            num_channels=config.num_channels,
            num_classes=config.num_classes,
            noise_level=config.noise_level,
            background_complexity=config.background_complexity,
            save_path=config.save_path
        )
        
        datasets[split] = SyntheticImageDataset(split_config, split=split)
    
    return datasets


def create_synthetic_segmentation_dataset(
    config: VisionDatasetConfig,
    splits: List[str] = ["train", "val", "test"],
    split_ratios: List[float] = [0.7, 0.15, 0.15]
) -> dict:
    """Create synthetic segmentation datasets"""
    datasets = {}
    
    total_samples = config.num_samples
    split_sizes = [int(ratio * total_samples) for ratio in split_ratios]
    
    for split, size in zip(splits, split_sizes):
        split_config = VisionDatasetConfig(
            num_samples=size,
            image_size=config.image_size,
            num_channels=config.num_channels,
            num_classes=config.num_classes,
            noise_level=config.noise_level,
            background_complexity=config.background_complexity,
            save_path=config.save_path
        )
        
        datasets[split] = SyntheticSegmentationDataset(split_config, split=split)
    
    return datasets


def visualize_samples(dataset: Dataset, num_samples: int = 4, save_path: Optional[str] = None) -> None:
    """Visualize samples from the dataset"""
    fig, axes = plt.subplots(2, num_samples, figsize=(num_samples * 3, 6))
    
    for i in range(num_samples):
        image, target = dataset[i]
        
        # Convert tensor to numpy
        if isinstance(image, torch.Tensor):
            if image.dim() == 3 and image.size(0) <= 3:
                image = image.permute(1, 2, 0)
            image = image.numpy()
        
        # Display image
        axes[0, i].imshow(image.squeeze() if image.ndim == 2 else image)
        axes[0, i].set_title(f'Image {i}')
        axes[0, i].axis('off')
        
        # Display target
        if isinstance(target, torch.Tensor):
            if target.dim() == 2:  # Segmentation mask
                axes[1, i].imshow(target.numpy(), cmap='tab10')
                axes[1, i].set_title(f'Mask {i}')
            else:  # Classification label
                axes[1, i].text(0.5, 0.5, f'Class: {target.item()}', 
                              ha='center', va='center', fontsize=12)
                axes[1, i].set_title(f'Label {i}')
        
        axes[1, i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    
    plt.show()