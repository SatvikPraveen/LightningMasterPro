# File: src/lmpro/modules/tabular/mlp_reg_cls.py

"""
MLP module for tabular regression and classification tasks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning import LightningModule
from torch.optim import Adam, AdamW, SGD
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingLR, ReduceLROnPlateau
from torchmetrics import Accuracy, Precision, Recall, F1Score, AUROC, MeanSquaredError, MeanAbsoluteError, R2Score
from typing import Dict, Any, Optional, List, Tuple, Union
import numpy as np


class MLPBlock(nn.Module):
    """MLP block with normalization and dropout"""
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        dropout: float = 0.3,
        activation: str = "relu",
        use_batch_norm: bool = True
    ):
        super().__init__()
        
        self.linear = nn.Linear(input_dim, output_dim)
        self.batch_norm = nn.BatchNorm1d(output_dim) if use_batch_norm else nn.Identity()
        self.dropout = nn.Dropout(dropout)
        
        if activation == "relu":
            self.activation = nn.ReLU()
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
        **kwargs
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
            self.criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
        elif task == "regression":
            self.criterion = nn.MSELoss()
        else:
            raise ValueError(f"Unknown task: {task}")
        
        # Metrics
        self.train_metrics = self._create_metrics("train")
        self.val_metrics = self._create_metrics("val")
        self.test_metrics = self._create_metrics("test")
    
    def _build_mlp(
        self, 
        hidden_dims: List[int], 
        dropout: float, 
        activation: str, 
        use_batch_norm: bool
    ) -> nn.ModuleList:
        """Build MLP layers"""
        layers = nn.ModuleList()
        
        prev_dim = self.input_dim
        for hidden_dim in hidden_dims:
            layers.append(MLPBlock(
                prev_dim, hidden_dim, dropout, activation, use_batch_norm
            ))
            prev_dim = hidden_dim
        
        return layers
    
    def _create_metrics(self, stage: str) -> Dict[str, Any]:
        """Create metrics for a specific stage"""
        metrics = {}
        
        if self.task == "classification":
            task_type = "binary" if self.output_dim == 2 else "multiclass"
            metrics = {
                "accuracy": Accuracy(task=task_type, num_classes=self.output_dim),
                "precision": Precision(task=task_type, num_classes=self.output_dim, average="macro"),
                "recall": Recall(task=task_type, num_classes=self.output_dim, average="macro"),
                "f1": F1Score(task=task_type, num_classes=self.output_dim, average="macro"),
            }
            
            if self.output_dim > 2:
                metrics["auroc"] = AUROC(task=task_type, num_classes=self.output_dim, average="macro")
                
        elif self.task == "regression":
            metrics = {
                "mse": MeanSquaredError(),
                "mae": MeanAbsoluteError(),
                "r2": R2Score(),
            }
        
        return nn.ModuleDict(metrics)
    
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
        
        # For regression, we might want to apply activation
        if self.task == "regression" and self.output_dim == 1:
            logits = logits.squeeze(-1)  # Remove last dimension for scalar output
        
        return logits
    
    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step"""
        x, y = batch
        logits = self(x)
        
        # Compute loss
        if self.task == "classification":
            loss = self.criterion(logits, y)
            preds = torch.argmax(logits, dim=1)
        else:  # regression
            loss = self.criterion(logits, y.float())
            preds = logits
        
        # Update metrics
        for name, metric in self.train_metrics.items():
            if self.task == "classification":
                if name == "auroc" and self.output_dim > 2:
                    metric.update(torch.softmax(logits, dim=1), y)
                elif name == "auroc":
                    metric.update(torch.softmax(logits, dim=1)[:, 1], y)
                else:
                    metric.update(preds, y)
            else:  # regression
                metric.update(preds, y)
        
        # Log metrics
        self.log("train/loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        
        if self.task == "classification":
            self.log("train/acc", self.train_metrics["accuracy"], prog_bar=True, on_step=False, on_epoch=True)
        else:
            self.log("train/mse", self.train_metrics["mse"], prog_bar=True, on_step=False, on_epoch=True)
            self.log("train/r2", self.train_metrics["r2"], prog_bar=True, on_step=False, on_epoch=True)
        
        return loss
    
    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Validation step"""
        x, y = batch
        logits = self(x)
        
        # Compute loss
        if self.task == "classification":
            loss = self.criterion(logits, y)
            preds = torch.argmax(logits, dim=1)
        else:  # regression
            loss = self.criterion(logits, y.float())
            preds = logits
        
        # Update metrics
        for name, metric in self.val_metrics.items():
            if self.task == "classification":
                if name == "auroc" and self.output_dim > 2:
                    metric.update(torch.softmax(logits, dim=1), y)
                elif name == "auroc":
                    metric.update(torch.softmax(logits, dim=1)[:, 1], y)
                else:
                    metric.update(preds, y)
            else:  # regression
                metric.update(preds, y)
        
        # Log metrics
        self.log("val/loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        
        if self.task == "classification":
            self.log("val/acc", self.val_metrics["accuracy"], prog_bar=True, on_step=False, on_epoch=True)
        else:
            self.log("val/mse", self.val_metrics["mse"], prog_bar=True, on_step=False, on_epoch=True)
            self.log("val/r2", self.val_metrics["r2"], prog_bar=True, on_step=False, on_epoch=True)
    
    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Test step"""
        x, y = batch
        logits = self(x)
        
        # Compute loss
        if self.task == "classification":
            loss = self.criterion(logits, y)
            preds = torch.argmax(logits, dim=1)
        else:  # regression
            loss = self.criterion(logits, y.float())
            preds = logits
        
        # Update metrics
        for name, metric in self.test_metrics.items():
            if self.task == "classification":
                if name == "auroc" and self.output_dim > 2:
                    metric.update(torch.softmax(logits, dim=1), y)
                elif name == "auroc":
                    metric.update(torch.softmax(logits, dim=1)[:, 1], y)
                else:
                    metric.update(preds, y)
            else:  # regression
                metric.update(preds, y)
        
        # Log metrics
        self.log("test/loss", loss, on_step=False, on_epoch=True)
        
        if self.task == "classification":
            self.log("test/acc", self.test_metrics["accuracy"], on_step=False, on_epoch=True)
        else:
            self.log("test/mse", self.test_metrics["mse"], on_step=False, on_epoch=True)
            self.log("test/r2", self.test_metrics["r2"], on_step=False, on_epoch=True)
    
    def on_validation_epoch_end(self) -> None:
        """Called at the end of validation epoch"""
        # Compute and log all metrics
        for name, metric in self.val_metrics.items():
            computed_metric = metric.compute()
            self.log(f"val/{name}", computed_metric, prog_bar=(name in ["accuracy", "r2"]))
    
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
        
        if self.task == "classification":
            probabilities = torch.softmax(logits, dim=1)
            predictions = torch.argmax(logits, dim=1)
            
            return {
                "predictions": predictions,
                "probabilities": probabilities,
                "logits": logits
            }
        else:  # regression
            return {
                "predictions": logits,
                "logits": logits
            }
    
    def configure_optimizers(self) -> Dict[str, Any]:
        """Configure optimizers and schedulers"""
        if self.optimizer_name.lower() == "adam":
            optimizer = Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.optimizer_name.lower() == "adamw":
            optimizer = AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.optimizer_name.lower() == "sgd":
            optimizer = SGD(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay, momentum=0.9)
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
        elif self.scheduler_name.lower() == "plateau":
            monitor = "val/acc" if self.task == "classification" else "val/r2"
            scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)
            config["lr_scheduler"] = {
                "scheduler": scheduler,
                "monitor": monitor,
                "interval": "epoch"
            }
        
        return config
    
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
            retain_graph=False
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