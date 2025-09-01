# File: src/lmpro/modules/vision/segmenter.py

"""
Vision segmentation module for semantic segmentation tasks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning import LightningModule
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingLR
from torchmetrics import JaccardIndex, Dice
from typing import Dict, Any, Optional, List, Tuple, Union
import numpy as np


class UNetBlock(nn.Module):
    """Basic UNet convolutional block"""
    
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout2d(dropout)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        x = self.relu(self.bn2(self.conv2(x)))
        return x


class VisionSegmenter(LightningModule):
    """
    Lightning Module for semantic segmentation using UNet architecture
    """
    
    def __init__(
        self,
        num_classes: int,
        input_channels: int = 3,
        hidden_dims: List[int] = [64, 128, 256, 512],
        dropout: float = 0.1,
        learning_rate: float = 1e-3,
        optimizer: str = "adamw",
        scheduler: str = "onecycle",
        weight_decay: float = 1e-4,
        class_weights: Optional[torch.Tensor] = None,
        dice_weight: float = 0.5,
        **kwargs
    ):
        super().__init__()
        
        self.save_hyperparameters()
        
        self.num_classes = num_classes
        self.input_channels = input_channels
        self.learning_rate = learning_rate
        self.optimizer_name = optimizer
        self.scheduler_name = scheduler
        self.weight_decay = weight_decay
        self.dice_weight = dice_weight
        
        # Build UNet
        self.encoder = self._build_encoder(hidden_dims, dropout)
        self.decoder = self._build_decoder(hidden_dims, dropout)
        self.final_conv = nn.Conv2d(hidden_dims[0], num_classes, 1)
        
        # Loss functions
        self.ce_loss = nn.CrossEntropyLoss(weight=class_weights, ignore_index=-1)
        self.dice_loss = DiceLoss(num_classes=num_classes)
        
        # Metrics
        self.train_metrics = self._create_metrics("train")
        self.val_metrics = self._create_metrics("val")
        self.test_metrics = self._create_metrics("test")
    
    def _build_encoder(self, hidden_dims: List[int], dropout: float) -> nn.ModuleList:
        """Build encoder path"""
        encoder = nn.ModuleList()
        in_channels = self.input_channels
        
        for dim in hidden_dims:
            encoder.append(UNetBlock(in_channels, dim, dropout))
            in_channels = dim
        
        return encoder
    
    def _build_decoder(self, hidden_dims: List[int], dropout: float) -> nn.ModuleList:
        """Build decoder path"""
        decoder = nn.ModuleList()
        
        # Reverse hidden dims for decoder
        decoder_dims = hidden_dims[::-1]
        
        for i in range(len(decoder_dims) - 1):
            in_dim = decoder_dims[i] + decoder_dims[i + 1]  # Skip connection
            out_dim = decoder_dims[i + 1]
            decoder.append(UNetBlock(in_dim, out_dim, dropout))
        
        return decoder
    
    def _create_metrics(self, stage: str) -> Dict[str, Any]:
        """Create metrics for a specific stage"""
        task = "binary" if self.num_classes == 2 else "multiclass"
        
        metrics = {
            "iou": JaccardIndex(task=task, num_classes=self.num_classes, average="macro"),
            "dice": Dice(num_classes=self.num_classes, average="macro"),
        }
        
        return nn.ModuleDict(metrics)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through UNet"""
        # Encoder
        encoder_features = []
        for encoder_block in self.encoder:
            x = encoder_block(x)
            encoder_features.append(x)
            x = F.max_pool2d(x, 2)
        
        # Remove last feature (will be used as bottleneck)
        bottleneck = encoder_features.pop()
        x = bottleneck
        
        # Decoder
        for i, decoder_block in enumerate(self.decoder):
            # Upsample
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
            
            # Skip connection
            skip = encoder_features[-(i+1)]
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
            
            x = torch.cat([x, skip], dim=1)
            x = decoder_block(x)
        
        # Final convolution
        x = self.final_conv(x)
        return x
    
    def _compute_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute combined loss"""
        ce_loss = self.ce_loss(logits, targets)
        dice_loss = self.dice_loss(logits, targets)
        
        total_loss = (1 - self.dice_weight) * ce_loss + self.dice_weight * dice_loss
        return total_loss, ce_loss, dice_loss
    
    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step"""
        x, y = batch
        logits = self(x)
        
        # Resize logits to match target size if needed
        if logits.shape[-2:] != y.shape[-2:]:
            logits = F.interpolate(logits, size=y.shape[-2:], mode='bilinear', align_corners=False)
        
        total_loss, ce_loss, dice_loss = self._compute_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        
        # Update metrics
        for name, metric in self.train_metrics.items():
            metric.update(preds, y)
        
        # Log metrics
        self.log("train/loss", total_loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("train/ce_loss", ce_loss, on_step=False, on_epoch=True)
        self.log("train/dice_loss", dice_loss, on_step=False, on_epoch=True)
        self.log("train/iou", self.train_metrics["iou"], prog_bar=True, on_step=False, on_epoch=True)
        
        return total_loss
    
    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Validation step"""
        x, y = batch
        logits = self(x)
        
        if logits.shape[-2:] != y.shape[-2:]:
            logits = F.interpolate(logits, size=y.shape[-2:], mode='bilinear', align_corners=False)
        
        total_loss, ce_loss, dice_loss = self._compute_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        
        # Update metrics
        for name, metric in self.val_metrics.items():
            metric.update(preds, y)
        
        # Log metrics
        self.log("val/loss", total_loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val/ce_loss", ce_loss, on_step=False, on_epoch=True)
        self.log("val/dice_loss", dice_loss, on_step=False, on_epoch=True)
        self.log("val/iou", self.val_metrics["iou"], prog_bar=True, on_step=False, on_epoch=True)
        self.log("val/dice", self.val_metrics["dice"], on_step=False, on_epoch=True)
    
    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Test step"""
        x, y = batch
        logits = self(x)
        
        if logits.shape[-2:] != y.shape[-2:]:
            logits = F.interpolate(logits, size=y.shape[-2:], mode='bilinear', align_corners=False)
        
        total_loss, _, _ = self._compute_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        
        # Update metrics
        for name, metric in self.test_metrics.items():
            metric.update(preds, y)
        
        # Log metrics
        self.log("test/loss", total_loss, on_step=False, on_epoch=True)
        self.log("test/iou", self.test_metrics["iou"], on_step=False, on_epoch=True)
        self.log("test/dice", self.test_metrics["dice"], on_step=False, on_epoch=True)
    
    def predict_step(self, batch: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]], batch_idx: int) -> Dict[str, torch.Tensor]:
        """Prediction step"""
        if isinstance(batch, tuple):
            x, _ = batch
        else:
            x = batch
        
        logits = self(x)
        probabilities = torch.softmax(logits, dim=1)
        predictions = torch.argmax(logits, dim=1)
        
        return {
            "predictions": predictions,
            "probabilities": probabilities,
            "logits": logits
        }
    
    def configure_optimizers(self) -> Dict[str, Any]:
        """Configure optimizers and schedulers"""
        if self.optimizer_name.lower() == "adam":
            optimizer = Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.optimizer_name.lower() == "adamw":
            optimizer = AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        else:
            optimizer = AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        
        config = {"optimizer": optimizer}
        
        if self.scheduler_name.lower() == "onecycle":
            scheduler = OneCycleLR(
                optimizer,
                max_lr=self.learning_rate,
                total_steps=self.trainer.estimated_stepping_batches,
                pct_start=0.3
            )
            config["lr_scheduler"] = {"scheduler": scheduler, "interval": "step"}
        elif self.scheduler_name.lower() == "cosine":
            scheduler = CosineAnnealingLR(optimizer, T_max=self.trainer.max_epochs)
            config["lr_scheduler"] = {"scheduler": scheduler, "interval": "epoch"}
        
        return config


class DiceLoss(nn.Module):
    """Dice loss for segmentation"""
    
    def __init__(self, num_classes: int, smooth: float = 1e-6):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits, dim=1)
        
        # One-hot encode targets
        targets_one_hot = F.one_hot(targets, num_classes=self.num_classes)
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()
        
        # Compute Dice coefficient for each class
        dice_scores = []
        for class_idx in range(self.num_classes):
            pred_class = probs[:, class_idx]
            target_class = targets_one_hot[:, class_idx]
            
            intersection = (pred_class * target_class).sum(dim=[1, 2])
            union = pred_class.sum(dim=[1, 2]) + target_class.sum(dim=[1, 2])
            
            dice = (2 * intersection + self.smooth) / (union + self.smooth)
            dice_scores.append(dice)
        
        dice_scores = torch.stack(dice_scores, dim=1)
        dice_loss = 1 - dice_scores.mean()
        
        return dice_loss