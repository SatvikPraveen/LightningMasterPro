# File: src/lmpro/modules/vision/segmenter.py

"""
Vision segmentation module for semantic segmentation tasks
"""

from typing import Dict, List, Literal, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning.pytorch import LightningModule
from lightning.pytorch.utilities.types import LRSchedulerConfigType, OptimizerLRScheduler
from torch.optim import Adam, AdamW, Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
from torchmetrics import JaccardIndex, Metric, MetricCollection
from torchmetrics.segmentation import DiceScore

IGNORE_INDEX = -1


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
        self.ce_loss = nn.CrossEntropyLoss(weight=class_weights, ignore_index=IGNORE_INDEX)
        self.dice_loss = DiceLoss(num_classes=num_classes, ignore_index=IGNORE_INDEX)

        # Metrics (all logged as metric objects so Lightning resets them per epoch)
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

    def _create_metrics(self, stage: str) -> MetricCollection:
        """Create metrics for a specific stage"""
        task: Literal["binary", "multiclass"] = "binary" if self.num_classes == 2 else "multiclass"

        metrics: Dict[str, Union[Metric, MetricCollection]] = {
            "iou": JaccardIndex(task=task, num_classes=self.num_classes, average="macro", ignore_index=IGNORE_INDEX),
            # Predictions and targets are (B, H, W) index maps, hence input_format="index".
            # "global" aggregation keeps only per-class counts instead of per-sample lists.
            "dice": DiceScore(
                num_classes=self.num_classes,
                average="macro",
                input_format="index",
                aggregation_level="global",
            ),
        }

        return MetricCollection(metrics, compute_groups=False)

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
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)

            # Skip connection
            skip = encoder_features[-(i + 1)]
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)

            x = torch.cat([x, skip], dim=1)
            x = decoder_block(x)

        # Final convolution
        x = self.final_conv(x)
        return x

    def _compute_loss(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute combined loss"""
        ce_loss = self.ce_loss(logits, targets)
        dice_loss = self.dice_loss(logits, targets)

        total_loss = (1 - self.dice_weight) * ce_loss + self.dice_weight * dice_loss
        return total_loss, ce_loss, dice_loss

    def _update_metrics(self, metrics: MetricCollection, preds: torch.Tensor, y: torch.Tensor) -> None:
        """Update metrics, excluding pixels labelled with IGNORE_INDEX"""
        metrics["iou"].update(preds, y)

        valid = y != IGNORE_INDEX
        if bool(valid.all()):
            metrics["dice"].update(preds, y)
        else:
            # DiceScore has no ignore_index: feed only each sample's valid pixels
            # as a (1, num_valid) index tensor.
            for p, t, v in zip(preds, y, valid):
                if v.any():
                    metrics["dice"].update(p[v].unsqueeze(0), t[v].unsqueeze(0))

    def _log_metrics(self, stage: str, metrics: MetricCollection) -> None:
        for name, metric in metrics.items():
            self.log(
                f"{stage}/{name}", metric, prog_bar=(name == "iou" and stage != "test"), on_step=False, on_epoch=True
            )

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step"""
        x, y = batch
        logits = self(x)

        # Resize logits to match target size if needed
        if logits.shape[-2:] != y.shape[-2:]:
            logits = F.interpolate(logits, size=y.shape[-2:], mode="bilinear", align_corners=False)

        total_loss, ce_loss, dice_loss = self._compute_loss(logits, y)
        preds = torch.argmax(logits, dim=1)

        self._update_metrics(self.train_metrics, preds, y)

        self.log("train/loss", total_loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("train/ce_loss", ce_loss, on_step=False, on_epoch=True)
        self.log("train/dice_loss", dice_loss, on_step=False, on_epoch=True)
        self._log_metrics("train", self.train_metrics)

        return total_loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Validation step"""
        x, y = batch
        logits = self(x)

        if logits.shape[-2:] != y.shape[-2:]:
            logits = F.interpolate(logits, size=y.shape[-2:], mode="bilinear", align_corners=False)

        total_loss, ce_loss, dice_loss = self._compute_loss(logits, y)
        preds = torch.argmax(logits, dim=1)

        self._update_metrics(self.val_metrics, preds, y)

        self.log("val/loss", total_loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val/ce_loss", ce_loss, on_step=False, on_epoch=True)
        self.log("val/dice_loss", dice_loss, on_step=False, on_epoch=True)
        self._log_metrics("val", self.val_metrics)

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Test step"""
        x, y = batch
        logits = self(x)

        if logits.shape[-2:] != y.shape[-2:]:
            logits = F.interpolate(logits, size=y.shape[-2:], mode="bilinear", align_corners=False)

        total_loss, _, _ = self._compute_loss(logits, y)
        preds = torch.argmax(logits, dim=1)

        self._update_metrics(self.test_metrics, preds, y)

        self.log("test/loss", total_loss, on_step=False, on_epoch=True)
        self._log_metrics("test", self.test_metrics)

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
        """Configure optimizers and schedulers"""
        optimizer: Optimizer
        if self.optimizer_name.lower() == "adam":
            optimizer = Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.optimizer_name.lower() == "adamw":
            optimizer = AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
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
                ),
                "interval": "step",
            }
        elif self.scheduler_name.lower() == "cosine":
            lr_scheduler = {
                "scheduler": CosineAnnealingLR(optimizer, T_max=self.trainer.max_epochs),
                "interval": "epoch",
            }

        if lr_scheduler is None:
            return {"optimizer": optimizer}
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}


class DiceLoss(nn.Module):
    """Dice loss for segmentation (pixels labelled ``ignore_index`` are excluded)"""

    def __init__(self, num_classes: int, smooth: float = 1e-6, ignore_index: int = IGNORE_INDEX):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits, dim=1)  # (B, C, H, W)

        # Mask ignored pixels before one-hot encoding (F.one_hot rejects negatives)
        valid = (targets != self.ignore_index).unsqueeze(1).to(probs.dtype)  # (B, 1, H, W)
        safe_targets = targets.masked_fill(targets == self.ignore_index, 0)

        targets_one_hot = F.one_hot(safe_targets, num_classes=self.num_classes)
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).to(probs.dtype)

        probs = probs * valid
        targets_one_hot = targets_one_hot * valid

        # Dice per (sample, class)
        intersection = (probs * targets_one_hot).sum(dim=(2, 3))
        union = probs.sum(dim=(2, 3)) + targets_one_hot.sum(dim=(2, 3))
        dice_scores = (2 * intersection + self.smooth) / (union + self.smooth)

        return 1 - dice_scores.mean()
