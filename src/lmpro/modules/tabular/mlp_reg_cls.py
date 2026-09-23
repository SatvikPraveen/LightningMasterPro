# File: src/lmpro/modules/tabular/mlp_reg_cls.py

"""
MLP module for tabular regression and classification tasks
"""

from typing import Dict, List, Literal, Optional, Tuple, Union

import torch
import torch.nn as nn
from lightning.pytorch import LightningModule
from lightning.pytorch.utilities.types import LRSchedulerConfigType, OptimizerLRScheduler
from torch.optim import SGD, Adam, AdamW, Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR, ReduceLROnPlateau
from torchmetrics import (
    AUROC,
    Accuracy,
    F1Score,
    MeanAbsoluteError,
    MeanSquaredError,
    Metric,
    MetricCollection,
    Precision,
    R2Score,
    Recall,
)


class MLPBlock(nn.Module):
    """MLP block with normalization and dropout"""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        dropout: float = 0.3,
        activation: str = "relu",
        use_batch_norm: bool = True,
    ):
        super().__init__()

        self.linear = nn.Linear(input_dim, output_dim)
        self.batch_norm = nn.BatchNorm1d(output_dim) if use_batch_norm else nn.Identity()
        self.dropout = nn.Dropout(dropout)

        if activation == "relu":
            self.activation: nn.Module = nn.ReLU()
        elif activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "silu":
            self.activation = nn.SiLU()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        else:
            self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        x = self.batch_norm(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x


class MLPRegressorClassifier(LightningModule):
    """
    Lightning Module for tabular data regression and classification

    Features:
    - Multi-layer perceptron with customizable architecture
    - Supports both regression and classification
    - Batch normalization and dropout
    - Multiple activation functions
    - Learning rate scheduling
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        task: str = "classification",  # "classification" or "regression"
        hidden_dims: List[int] = [512, 256, 128],
        dropout: float = 0.3,
        activation: str = "relu",
        use_batch_norm: bool = True,
        learning_rate: float = 1e-3,
        optimizer: str = "adamw",
        scheduler: str = "onecycle",
        weight_decay: float = 1e-4,
        class_weights: Optional[torch.Tensor] = None,
        label_smoothing: float = 0.0,
    ):
        super().__init__()

        self.save_hyperparameters()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.task = task
        self.learning_rate = learning_rate
        self.optimizer_name = optimizer
        self.scheduler_name = scheduler
        self.weight_decay = weight_decay

        # Build MLP
        self.layers = self._build_mlp(hidden_dims, dropout, activation, use_batch_norm)

        # Output layer
        self.output_layer = nn.Linear(hidden_dims[-1], output_dim)

        # Loss functions
        if task == "classification":
            self.criterion: nn.Module = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
        elif task == "regression":
            self.criterion = nn.MSELoss()
        else:
            raise ValueError(f"Unknown task: {task}")

        # Metrics (every object is logged so Lightning computes/resets it per epoch)
        self.train_metrics = self._create_metrics("train")
        self.val_metrics = self._create_metrics("val")
        self.test_metrics = self._create_metrics("test")

    def _build_mlp(
        self, hidden_dims: List[int], dropout: float, activation: str, use_batch_norm: bool
    ) -> nn.ModuleList:
        """Build MLP layers"""
        layers = nn.ModuleList()

        prev_dim = self.input_dim
        for hidden_dim in hidden_dims:
            layers.append(MLPBlock(prev_dim, hidden_dim, dropout, activation, use_batch_norm))
            prev_dim = hidden_dim

        return layers

    def _create_metrics(self, stage: str) -> MetricCollection:
        """Create metrics for a specific stage"""
        metrics: Dict[str, Union[Metric, MetricCollection]]
        if self.task == "classification":
            task_type: Literal["binary", "multiclass"] = "binary" if self.output_dim == 2 else "multiclass"
            metrics = {
                "accuracy": Accuracy(task=task_type, num_classes=self.output_dim),
            }
            if stage != "train":
                metrics.update(
                    {
                        "precision": Precision(task=task_type, num_classes=self.output_dim, average="macro"),
                        "recall": Recall(task=task_type, num_classes=self.output_dim, average="macro"),
                        "f1": F1Score(task=task_type, num_classes=self.output_dim, average="macro"),
                    }
                )
                if self.output_dim > 2:
                    metrics["auroc"] = AUROC(task=task_type, num_classes=self.output_dim, average="macro")
        else:  # regression
            metrics = {
                "mse": MeanSquaredError(),
                "mae": MeanAbsoluteError(),
                "r2": R2Score(),
            }

        return MetricCollection(metrics, compute_groups=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        # Handle different input shapes
        if x.dim() > 2:
            # If input has extra dimensions (e.g., time series), flatten
            batch_size = x.shape[0]
            x = x.view(batch_size, -1)

        # Pass through MLP layers
        for layer in self.layers:
            x = layer(x)

        # Output layer
        logits = self.output_layer(x)

        if self.task == "regression" and self.output_dim == 1:
            logits = logits.squeeze(-1)  # Scalar output per sample

        return logits

    def _shared_step(self, batch: Tuple[torch.Tensor, torch.Tensor], metrics: MetricCollection) -> torch.Tensor:
        """Compute loss and update every metric in ``metrics``"""
        x, y = batch
        logits = self(x)

        if self.task == "classification":
            loss = self.criterion(logits, y)
            preds = torch.argmax(logits, dim=1)
            for name, metric in metrics.items():
                if name == "auroc" and self.output_dim > 2:
                    metric.update(torch.softmax(logits, dim=1), y)
                elif name == "auroc":
                    metric.update(torch.softmax(logits, dim=1)[:, 1], y)
                else:
                    metric.update(preds, y)
        else:  # regression
            y = y.float()
            if y.shape != logits.shape:
                y = y.reshape(logits.shape)
            loss = self.criterion(logits, y)
            for metric in metrics.values():
                metric.update(logits, y)

        return loss

    def _log_metrics(self, stage: str, metrics: MetricCollection) -> None:
        for name, metric in metrics.items():
            key = f"{stage}/acc" if name == "accuracy" else f"{stage}/{name}"
            prog_bar = stage != "test" and name in ("accuracy", "mse", "r2")
            self.log(key, metric, prog_bar=prog_bar, on_step=False, on_epoch=True)

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step"""
        loss = self._shared_step(batch, self.train_metrics)
        self.log("train/loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self._log_metrics("train", self.train_metrics)
        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Validation step"""
        loss = self._shared_step(batch, self.val_metrics)
        self.log("val/loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self._log_metrics("val", self.val_metrics)

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Test step"""
        loss = self._shared_step(batch, self.test_metrics)
        self.log("test/loss", loss, on_step=False, on_epoch=True)
        self._log_metrics("test", self.test_metrics)

    def on_validation_epoch_end(self) -> None:
        """Called at the end of validation epoch"""
        if self.task == "classification":
            # compute() is cache-safe; Lightning resets the objects logged in validation_step.
            self.log("val/accuracy", self.val_metrics["accuracy"].compute())

    def predict_step(
        self, batch: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]], batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        """Prediction step"""
        if isinstance(batch, (tuple, list)):
            x = batch[0]
        else:
            x = batch

        logits = self(x)

        if self.task == "classification":
            probabilities = torch.softmax(logits, dim=1)
            predictions = torch.argmax(logits, dim=1)

            return {"predictions": predictions, "probabilities": probabilities, "logits": logits}
        else:  # regression
            return {"predictions": logits, "logits": logits}

    def configure_optimizers(self) -> OptimizerLRScheduler:
        """Configure optimizers and schedulers"""
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
                ),
                "interval": "step",
            }
        elif self.scheduler_name.lower() == "cosine":
            lr_scheduler = {
                "scheduler": CosineAnnealingLR(optimizer, T_max=self.trainer.max_epochs),
                "interval": "epoch",
            }
        elif self.scheduler_name.lower() == "plateau":
            monitor = "val/acc" if self.task == "classification" else "val/r2"
            lr_scheduler = {
                "scheduler": ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5),
                "monitor": monitor,
                "interval": "epoch",
            }

        if lr_scheduler is None:
            return {"optimizer": optimizer}
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}

    def get_feature_importance(self, x: torch.Tensor) -> torch.Tensor:
        """Compute feature importance using gradients"""
        self.eval()
        x.requires_grad_(True)

        logits = self(x)

        if self.task == "classification":
            # Use max probability class for importance
            max_class = torch.argmax(logits, dim=1)
            selected_logits = logits[torch.arange(logits.size(0)), max_class]
        else:
            # For regression, use output directly
            selected_logits = logits.sum() if logits.dim() > 1 else logits

        # Compute gradients
        gradients = torch.autograd.grad(
            outputs=selected_logits,
            inputs=x,
            grad_outputs=torch.ones_like(selected_logits),
            create_graph=False,
            retain_graph=False,
        )[0]

        # Feature importance as absolute gradient values
        importance = torch.abs(gradients).mean(dim=0)

        return importance

    def freeze_layers(self, num_layers: int) -> None:
        """Freeze first num_layers layers"""
        for i, layer in enumerate(self.layers):
            if i < num_layers:
                for param in layer.parameters():
                    param.requires_grad = False

    def unfreeze_all_layers(self) -> None:
        """Unfreeze all layers"""
        for layer in self.layers:
            for param in layer.parameters():
                param.requires_grad = True
