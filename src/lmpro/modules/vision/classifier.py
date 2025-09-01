# File: src/lmpro/modules/vision/classifier.py

"""
Vision classifier module for image classification tasks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning import LightningModule
from torch.optim import Adam, SGD, AdamW
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingLR, ReduceLROnPlateau
from torchmetrics import Accuracy, Precision, Recall, F1Score, AUROC, ConfusionMatrix
from typing import Dict, Any, Optional, List, Tuple, Union
import numpy as np

from ...utils.metrics import compute_classification_metrics, log_confusion_matrix
from ...utils.viz import plot_training_curves


class ConvBlock(nn.Module):
    """Convolutional block with BatchNorm and activation"""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        dropout: float = 0.0,
        activation: str = "relu"
    ):
        super().__init__()
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        
        if activation == "relu":
            self.activation = nn.ReLU(inplace=True)
        elif activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "silu":
            self.activation = nn.SiLU()
        else:
            self.activation = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x


class ResidualBlock(nn.Module):
    """Residual block with skip connection"""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        dropout: float = 0.0
    ):
        super().__init__()
        
        self.conv1 = ConvBlock(in_channels, out_channels, stride=stride, dropout=dropout)
        self.conv2 = ConvBlock(out_channels, out_channels, dropout=dropout)
        
        # Skip connection
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = x + residual
        return F.relu(x)


class VisionClassifier(LightningModule):
    """
    Lightning Module for image classification
    
    Supports multiple architectures:
    - Simple CNN
    - ResNet-like architecture
    - Custom architectures
    """
    
    def __init__(
        self,
        num_classes: int,
        input_channels: int = 3,
        architecture: str = "resnet",
        hidden_dims: List[int] = [64, 128, 256, 512],
        dropout: float = 0.3,
        learning_rate: float = 1e-3,
        optimizer: str = "adamw",
        scheduler: str = "onecycle",
        weight_decay: float = 1e-4,
        class_weights: Optional[torch.Tensor] = None,
        label_smoothing: float = 0.0,
        mixup_alpha: float = 0.0,
        **kwargs
    ):
        super().__init__()
        
        # Save hyperparameters
        self.save_hyperparameters()
        
        self.num_classes = num_classes
        self.input_channels = input_channels
        self.architecture = architecture
        self.learning_rate = learning_rate
        self.optimizer_name = optimizer
        self.scheduler_name = scheduler
        self.weight_decay = weight_decay
        self.label_smoothing = label_smoothing
        self.mixup_alpha = mixup_alpha
        
        # Build model
        self.backbone = self._build_backbone(hidden_dims, dropout)
        self.classifier = self._build_classifier()
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=label_smoothing
        )
        
        # Metrics
        self.train_metrics = self._create_metrics("train")
        self.val_metrics = self._create_metrics("val")
        self.test_metrics = self._create_metrics("test")
        
        # For tracking best metrics
        self.best_val_acc = 0.0
        
    def _build_backbone(self, hidden_dims: List[int], dropout: float) -> nn.Module:
        """Build the feature extraction backbone"""
        layers = []
        in_channels = self.input_channels
        
        if self.architecture == "simple":
            # Simple CNN architecture
            for i, dim in enumerate(hidden_dims):
                layers.extend([
                    ConvBlock(in_channels, dim, dropout=dropout if i > 0 else 0.0),
                    nn.MaxPool2d(2)
                ])
                in_channels = dim
                
        elif self.architecture == "resnet":
            # ResNet-like architecture
            layers.append(ConvBlock(in_channels, hidden_dims[0]))
            in_channels = hidden_dims[0]
            
            for i, dim in enumerate(hidden_dims[1:], 1):
                stride = 2 if i > 0 else 1
                layers.extend([
                    ResidualBlock(in_channels, dim, stride=stride, dropout=dropout),
                    ResidualBlock(dim, dim, dropout=dropout)
                ])
                in_channels = dim
        
        else:
            raise ValueError(f"Unknown architecture: {self.architecture}")
        
        return nn.Sequential(*layers)
    
    def _build_classifier(self) -> nn.Module:
        """Build the classification head"""
        return nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(self.hparams.dropout),
            nn.Linear(self.hparams.hidden_dims[-1], self.num_classes)
        )
    
    def _create_metrics(self, stage: str) -> Dict[str, Any]:
        """Create metrics for a specific stage"""
        task = "binary" if self.num_classes == 2 else "multiclass"
        
        metrics = {
            "accuracy": Accuracy(task=task, num_classes=self.num_classes),
            "precision": Precision(task=task, num_classes=self.num_classes, average="macro"),
            "recall": Recall(task=task, num_classes=self.num_classes, average="macro"),
            "f1": F1Score(task=task, num_classes=self.num_classes, average="macro"),
        }
        
        if self.num_classes > 2:
            metrics["auroc"] = AUROC(task=task, num_classes=self.num_classes, average="macro")
        
        return nn.ModuleDict(metrics)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits
    
    def _mixup_data(self, x: torch.Tensor, y: torch.Tensor, alpha: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """Apply mixup augmentation"""
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1
        
        batch_size = x.size(0)
        index = torch.randperm(batch_size, device=x.device)
        
        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        
        return mixed_x, y_a, y_b, lam
    
    def _mixup_criterion(self, pred: torch.Tensor, y_a: torch.Tensor, y_b: torch.Tensor, lam: float) -> torch.Tensor:
        """Compute mixup loss"""
        return lam * self.criterion(pred, y_a) + (1 - lam) * self.criterion(pred, y_b)
    
    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step"""
        x, y = batch
        
        # Apply mixup if specified
        if self.mixup_alpha > 0.0 and self.training:
            x, y_a, y_b, lam = self._mixup_data(x, y, self.mixup_alpha)
            logits = self(x)
            loss = self._mixup_criterion(logits, y_a, y_b, lam)
        else:
            logits = self(x)
            loss = self.criterion(logits, y)
        
        # Calculate metrics (use original targets for mixup)
        if self.mixup_alpha > 0.0 and self.training:
            # For mixup, we can't easily compute meaningful accuracy during training
            self.log("train/loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        else:
            preds = torch.argmax(logits, dim=1)
            
            # Update metrics
            for name, metric in self.train_metrics.items():
                if name == "auroc" and self.num_classes == 2:
                    metric.update(torch.softmax(logits, dim=1)[:, 1], y)
                elif name == "auroc":
                    metric.update(torch.softmax(logits, dim=1), y)
                else:
                    metric.update(preds, y)
            
            # Log metrics
            self.log("train/loss", loss, prog_bar=True, on_step=False, on_epoch=True)
            self.log("train/acc", self.train_metrics["accuracy"], prog_bar=True, on_step=False, on_epoch=True)
        
        return loss
    
    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Validation step"""
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        
        # Update metrics
        for name, metric in self.val_metrics.items():
            if name == "auroc" and self.num_classes == 2:
                metric.update(torch.softmax(logits, dim=1)[:, 1], y)
            elif name == "auroc":
                metric.update(torch.softmax(logits, dim=1), y)
            else:
                metric.update(preds, y)
        
        # Log metrics
        self.log("val/loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val/acc", self.val_metrics["accuracy"], prog_bar=True, on_step=False, on_epoch=True)
        
        # Store for confusion matrix
        if not hasattr(self, 'val_predictions'):
            self.val_predictions = []
            self.val_targets = []
        
        self.val_predictions.extend(preds.cpu().tolist())
        self.val_targets.extend(y.cpu().tolist())
    
    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Test step"""
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        
        # Update metrics
        for name, metric in self.test_metrics.items():
            if name == "auroc" and self.num_classes == 2:
                metric.update(torch.softmax(logits, dim=1)[:, 1], y)
            elif name == "auroc":
                metric.update(torch.softmax(logits, dim=1), y)
            else:
                metric.update(preds, y)
        
        # Log metrics
        self.log("test/loss", loss, on_step=False, on_epoch=True)
        self.log("test/acc", self.test_metrics["accuracy"], on_step=False, on_epoch=True)
    
    def on_validation_epoch_end(self) -> None:
        """Called at the end of validation epoch"""
        # Compute and log all metrics
        for name, metric in self.val_metrics.items():
            self.log(f"val/{name}", metric.compute(), prog_bar=(name == "accuracy"))
        
        # Log confusion matrix every few epochs
        if self.current_epoch % 5 == 0 and hasattr(self, 'val_predictions'):
            preds_tensor = torch.tensor(self.val_predictions)
            targets_tensor = torch.tensor(self.val_targets)
            
            log_confusion_matrix(
                self, preds_tensor, targets_tensor, 
                num_classes=self.num_classes,
                stage="val"
            )
            
            # Clear stored predictions
            self.val_predictions = []
            self.val_targets = []
        
        # Track best accuracy
        current_acc = self.val_metrics["accuracy"].compute()
        if current_acc > self.best_val_acc:
            self.best_val_acc = current_acc
            self.log("val/best_acc", self.best_val_acc, prog_bar=True)
    
    def on_test_epoch_end(self) -> None:
        """Called at the end of test epoch"""
        # Compute and log all metrics
        for name, metric in self.test_metrics.items():
            self.log(f"test/{name}", metric.compute())
    
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
        """Configure optimizers and learning rate schedulers"""
        # Optimizer
        if self.optimizer_name.lower() == "adam":
            optimizer = Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.optimizer_name.lower() == "adamw":
            optimizer = AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.optimizer_name.lower() == "sgd":
            optimizer = SGD(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay, momentum=0.9)
        else:
            optimizer = AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        
        # Scheduler
        config = {"optimizer": optimizer}
        
        if self.scheduler_name.lower() == "onecycle":
            scheduler = OneCycleLR(
                optimizer,
                max_lr=self.learning_rate,
                total_steps=self.trainer.estimated_stepping_batches,
                pct_start=0.3,
                anneal_strategy="cos"
            )
            config["lr_scheduler"] = {
                "scheduler": scheduler,
                "interval": "step"
            }
        elif self.scheduler_name.lower() == "cosine":
            scheduler = CosineAnnealingLR(optimizer, T_max=self.trainer.max_epochs)
            config["lr_scheduler"] = {
                "scheduler": scheduler,
                "interval": "epoch"
            }
        elif self.scheduler_name.lower() == "plateau":
            scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)
            config["lr_scheduler"] = {
                "scheduler": scheduler,
                "monitor": "val/acc",
                "interval": "epoch"
            }
        
        return config
    
    def get_model_size(self) -> Dict[str, int]:
        """Get model size information"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "non_trainable_parameters": total_params - trainable_params
        }
    
    def freeze_backbone(self) -> None:
        """Freeze backbone parameters for fine-tuning"""
        for param in self.backbone.parameters():
            param.requires_grad = False
    
    def unfreeze_backbone(self) -> None:
        """Unfreeze backbone parameters"""
        for param in self.backbone.parameters():
            param.requires_grad = True