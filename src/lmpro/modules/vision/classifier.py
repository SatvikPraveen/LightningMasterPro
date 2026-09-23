# File: src/lmpro/modules/vision/classifier.py

"""
Vision classifier module for image classification tasks
"""

from typing import Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning.pytorch import LightningModule
from lightning.pytorch.utilities.types import LRSchedulerConfigType, OptimizerLRScheduler
from torch.optim import SGD, Adam, AdamW, Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR, ReduceLROnPlateau
from torchmetrics import AUROC, Accuracy, F1Score, Metric, MetricCollection, Precision, Recall

from ...utils.metrics import log_confusion_matrix


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
        activation: str = "relu",
    ):
        super().__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

        if activation == "relu":
            self.activation: nn.Module = nn.ReLU(inplace=True)
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

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, dropout: float = 0.0):
        super().__init__()

        self.conv1 = ConvBlock(in_channels, out_channels, stride=stride, dropout=dropout)
        self.conv2 = ConvBlock(out_channels, out_channels, dropout=dropout)

        # Skip connection
        if stride != 1 or in_channels != out_channels:
            self.shortcut: nn.Module = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False), nn.BatchNorm2d(out_channels)
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
        self.classifier = self._build_classifier(hidden_dims[-1], dropout)

        # Loss function
        self.criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)

        # Metrics. Every metric object is handed to ``self.log`` so Lightning
        # computes it at epoch end and resets it afterwards (no cross-epoch
        # accumulation). Training only tracks accuracy to avoid the memory cost
        # of AUROC over a whole training epoch.
        self.train_metrics = self._create_metrics("train")
        self.val_metrics = self._create_metrics("val")
        self.test_metrics = self._create_metrics("test")

        # For tracking best metrics
        self.best_val_acc = 0.0

        # For confusion matrix logging
        self.val_predictions: List[int] = []
        self.val_targets: List[int] = []

    def _build_backbone(self, hidden_dims: List[int], dropout: float) -> nn.Module:
        """Build the feature extraction backbone"""
        layers = []
        in_channels = self.input_channels

        if self.architecture == "simple":
            # Simple CNN architecture
            for i, dim in enumerate(hidden_dims):
                layers.extend([ConvBlock(in_channels, dim, dropout=dropout if i > 0 else 0.0), nn.MaxPool2d(2)])
                in_channels = dim

        elif self.architecture == "resnet":
            # ResNet-like architecture
            layers.append(ConvBlock(in_channels, hidden_dims[0]))
            in_channels = hidden_dims[0]

            for i, dim in enumerate(hidden_dims[1:], 1):
                stride = 2 if i > 0 else 1
                layers.extend(
                    [
                        ResidualBlock(in_channels, dim, stride=stride, dropout=dropout),
                        ResidualBlock(dim, dim, dropout=dropout),
                    ]
                )
                in_channels = dim

        else:
            raise ValueError(f"Unknown architecture: {self.architecture}")

        return nn.Sequential(*layers)

    def _build_classifier(self, feature_dim: int, dropout: float) -> nn.Module:
        """Build the classification head"""
        return nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, self.num_classes),
        )

    def _create_metrics(self, stage: str) -> MetricCollection:
        """Create metrics for a specific stage"""
        task: Literal["binary", "multiclass"] = "binary" if self.num_classes == 2 else "multiclass"

        metrics: Dict[str, Union[Metric, MetricCollection]] = {
            "accuracy": Accuracy(task=task, num_classes=self.num_classes),
        }

        if stage != "train":
            metrics.update(
                {
                    "precision": Precision(task=task, num_classes=self.num_classes, average="macro"),
                    "recall": Recall(task=task, num_classes=self.num_classes, average="macro"),
                    "f1": F1Score(task=task, num_classes=self.num_classes, average="macro"),
                }
            )
            if self.num_classes > 2:
                metrics["auroc"] = AUROC(task=task, num_classes=self.num_classes, average="macro")

        return MetricCollection(metrics, compute_groups=False)

    def _update_metrics(self, metrics: MetricCollection, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Update every metric in ``metrics`` and return hard predictions"""
        preds = torch.argmax(logits, dim=1)
        for name, metric in metrics.items():
            if name == "auroc" and self.num_classes == 2:
                metric.update(torch.softmax(logits, dim=1)[:, 1], y)
            elif name == "auroc":
                metric.update(torch.softmax(logits, dim=1), y)
            else:
                metric.update(preds, y)
        return preds

    def _log_metrics(self, stage: str, metrics: MetricCollection) -> None:
        """Log every metric object so Lightning computes and resets it per epoch"""
        for name, metric in metrics.items():
            key = f"{stage}/acc" if name == "accuracy" else f"{stage}/{name}"
            self.log(key, metric, prog_bar=(name == "accuracy"), on_step=False, on_epoch=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits

    def _mixup_data(
        self, x: torch.Tensor, y: torch.Tensor, alpha: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
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
            # With mixup, accuracy on mixed inputs is not meaningful
            self.log("train/loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        else:
            logits = self(x)
            loss = self.criterion(logits, y)
            self._update_metrics(self.train_metrics, logits, y)
            self.log("train/loss", loss, prog_bar=True, on_step=False, on_epoch=True)
            self._log_metrics("train", self.train_metrics)

        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Validation step"""
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = self._update_metrics(self.val_metrics, logits, y)

        self.log("val/loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self._log_metrics("val", self.val_metrics)

        # Store for confusion matrix
        self.val_predictions.extend(preds.cpu().tolist())
        self.val_targets.extend(y.cpu().tolist())

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Test step"""
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        self._update_metrics(self.test_metrics, logits, y)

        self.log("test/loss", loss, on_step=False, on_epoch=True)
        self._log_metrics("test", self.test_metrics)

    def on_validation_epoch_end(self) -> None:
        """Called at the end of validation epoch"""
        # ``compute`` is cache-safe here; Lightning resets the metric objects
        # after it has computed the epoch values that were logged in
        # ``validation_step``.
        current_acc = self.val_metrics["accuracy"].compute()
        self.log("val/accuracy", current_acc)

        # Log confusion matrix every few epochs (only when a logger is attached)
        if self.current_epoch % 5 == 0 and self.logger is not None and self.val_predictions:
            log_confusion_matrix(
                self,
                torch.tensor(self.val_predictions),
                torch.tensor(self.val_targets),
                num_classes=self.num_classes,
                stage="val",
            )

        # Clear stored predictions every epoch so they don't grow unboundedly
        self.val_predictions = []
        self.val_targets = []

        # Track best accuracy
        if current_acc > self.best_val_acc:
            self.best_val_acc = float(current_acc)
        self.log("val/best_acc", self.best_val_acc, prog_bar=True)

    def predict_step(
        self, batch: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]], batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        """Prediction step"""
        if isinstance(batch, (tuple, list)):
            x = batch[0]
        else:
            x = batch

        logits = self(x)
        probabilities = torch.softmax(logits, dim=1)
        predictions = torch.argmax(logits, dim=1)

        return {"predictions": predictions, "probabilities": probabilities, "logits": logits}

    def configure_optimizers(self) -> OptimizerLRScheduler:
        """Configure optimizers and learning rate schedulers"""
        optimizer: Optimizer
        if self.optimizer_name.lower() == "adam":
            optimizer = Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.optimizer_name.lower() == "adamw":
            optimizer = AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.optimizer_name.lower() == "sgd":
            optimizer = SGD(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay, momentum=0.9)
        else:
            optimizer = AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

        lr_scheduler: Optional[LRSchedulerConfigType] = None
        if self.scheduler_name.lower() == "onecycle":
            lr_scheduler = {
                "scheduler": OneCycleLR(
                    optimizer,
                    max_lr=self.learning_rate,
                    total_steps=int(self.trainer.estimated_stepping_batches),
                    pct_start=0.3,
                    anneal_strategy="cos",
                ),
                "interval": "step",
            }
        elif self.scheduler_name.lower() == "cosine":
            lr_scheduler = {
                "scheduler": CosineAnnealingLR(optimizer, T_max=self.trainer.max_epochs),
                "interval": "epoch",
            }
        elif self.scheduler_name.lower() == "plateau":
            lr_scheduler = {
                "scheduler": ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5),
                "monitor": "val/acc",
                "interval": "epoch",
            }

        if lr_scheduler is None:
            return {"optimizer": optimizer}
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}

    def get_model_size(self) -> Dict[str, int]:
        """Get model size information"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "non_trainable_parameters": total_params - trainable_params,
        }

    def freeze_backbone(self) -> None:
        """Freeze backbone parameters for fine-tuning"""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self) -> None:
        """Unfreeze backbone parameters"""
        for param in self.backbone.parameters():
            param.requires_grad = True
