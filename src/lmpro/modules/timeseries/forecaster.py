# File: src/lmpro/modules/timeseries/forecaster.py

"""
Time series forecasting module with multiple architectures
"""

import math
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
from lightning.pytorch import LightningModule
from lightning.pytorch.utilities.types import LRSchedulerConfigType, OptimizerLRScheduler
from torch.optim import Adam, AdamW, Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
from torchmetrics import MeanAbsoluteError, MeanSquaredError, Metric, MetricCollection, R2Score


class TimeSeriesForecaster(LightningModule):
    """
    Lightning Module for time series forecasting

    Input:  ``x`` of shape ``(batch, sequence_length, input_dim)``
    Output: forecast of shape ``(batch, prediction_horizon, output_dim)``

    Targets may be ``(batch, prediction_horizon, output_dim)`` or any shape with the
    same number of elements per sample (e.g. ``(batch, prediction_horizon)`` when
    ``output_dim == 1``); losses and metrics are computed on the flattened
    ``(batch, prediction_horizon * output_dim)`` view.

    Supports multiple architectures:
    - LSTM/GRU based
    - CNN based
    - Transformer based
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        sequence_length: int,
        prediction_horizon: int = 1,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        architecture: str = "lstm",  # lstm, gru, cnn, transformer
        learning_rate: float = 1e-3,
        optimizer: str = "adamw",
        scheduler: str = "onecycle",
        weight_decay: float = 1e-4,
    ):
        super().__init__()

        self.save_hyperparameters()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.architecture = architecture
        self.learning_rate = learning_rate
        self.optimizer_name = optimizer
        self.scheduler_name = scheduler
        self.weight_decay = weight_decay

        # Build model based on architecture
        if architecture == "lstm":
            self.encoder: nn.Module = nn.LSTM(
                input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0
            )
        elif architecture == "gru":
            self.encoder = nn.GRU(
                input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0
            )
        elif architecture == "cnn":
            self.encoder = self._build_cnn_encoder(dropout)
        elif architecture == "transformer":
            self.encoder = self._build_transformer_encoder(dropout)
        else:
            raise ValueError(f"Unknown architecture: {architecture}")

        # Output projection
        if architecture in ["lstm", "gru"]:
            self.output_projection = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, prediction_horizon * output_dim),
            )
        elif architecture == "cnn":
            self.output_projection = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(self._get_cnn_output_dim(), hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, prediction_horizon * output_dim),
            )
        elif architecture == "transformer":
            self.output_projection = nn.Sequential(
                nn.LayerNorm(hidden_dim), nn.Dropout(dropout), nn.Linear(hidden_dim, prediction_horizon * output_dim)
            )

        # Loss functions
        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()

        # Metrics (all logged as metric objects so Lightning resets them per epoch)
        self.train_metrics = self._create_metrics("train")
        self.val_metrics = self._create_metrics("val")
        self.test_metrics = self._create_metrics("test")

    def _build_cnn_encoder(self, dropout: float) -> nn.Module:
        """Build CNN encoder for time series"""
        return nn.Sequential(
            nn.Conv1d(self.input_dim, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(128, self.hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )

    def _get_cnn_output_dim(self) -> int:
        """Get CNN output dimension"""
        return self.hidden_dim

    def _build_transformer_encoder(self, dropout: float) -> nn.Module:
        """Build transformer encoder"""
        input_proj = nn.Linear(self.input_dim, self.hidden_dim)

        # Batch-first positional encoding (matches batch_first=True below)
        pos_encoding = PositionalEncoding(self.hidden_dim, dropout, max(self.sequence_length, 1))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim, nhead=8, dim_feedforward=self.hidden_dim * 2, dropout=dropout, batch_first=True
        )
        transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)

        return nn.ModuleDict(
            {"input_proj": input_proj, "pos_encoding": pos_encoding, "transformer": transformer_encoder}
        )

    def _create_metrics(self, stage: str) -> MetricCollection:
        """Create metrics for forecasting (computed on (batch, horizon*output_dim))"""
        metrics: Dict[str, Union[Metric, MetricCollection]] = {
            "mse": MeanSquaredError(),
            "mae": MeanAbsoluteError(),
            "rmse": MeanSquaredError(squared=False),
            "r2": R2Score(),
        }
        return MetricCollection(metrics, compute_groups=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass -> (batch, prediction_horizon, output_dim)"""
        batch_size = x.shape[0]

        if self.architecture in ["lstm", "gru"]:
            output, _ = self.encoder(x)  # (batch, seq_len, hidden_dim)
            last_hidden = output[:, -1, :]  # (batch, hidden_dim)
            forecast = self.output_projection(last_hidden)

        elif self.architecture == "cnn":
            x_transposed = x.transpose(1, 2)  # (batch, input_dim, seq_len)
            encoded = self.encoder(x_transposed).squeeze(-1)  # (batch, hidden_dim)
            forecast = self.output_projection(encoded)

        elif self.architecture == "transformer":
            modules = self.encoder
            assert isinstance(modules, nn.ModuleDict)
            x = modules["input_proj"](x)  # (batch, seq_len, hidden_dim)
            x = modules["pos_encoding"](x)
            encoded = modules["transformer"](x)  # (batch, seq_len, hidden_dim)
            encoded = encoded.mean(dim=1)  # (batch, hidden_dim)
            forecast = self.output_projection(encoded)

        return forecast.view(batch_size, self.prediction_horizon, self.output_dim)

    def _flatten_pair(self, forecast: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Flatten forecast and target to (batch, horizon * output_dim), validating sizes"""
        batch_size = forecast.shape[0]
        forecast_flat = forecast.reshape(batch_size, -1)
        y_flat = y.reshape(batch_size, -1).to(forecast_flat.dtype)
        if forecast_flat.shape != y_flat.shape:
            raise ValueError(
                f"Target shape {tuple(y.shape)} is incompatible with forecast shape "
                f"{tuple(forecast.shape)} (prediction_horizon={self.prediction_horizon}, "
                f"output_dim={self.output_dim})"
            )
        return forecast_flat, y_flat

    def _shared_step(self, batch: Tuple[torch.Tensor, torch.Tensor], metrics: MetricCollection) -> torch.Tensor:
        x, y = batch
        forecast, y = self._flatten_pair(self(x), y)

        mse_loss = self.mse_loss(forecast, y)
        mae_loss = self.mae_loss(forecast, y)
        loss = mse_loss + 0.1 * mae_loss  # MSE with MAE regularisation

        for metric in metrics.values():
            metric.update(forecast, y)

        return loss

    def _log_metrics(self, stage: str, metrics: MetricCollection) -> None:
        for name, metric in metrics.items():
            prog_bar = stage != "test" and name in ("mse", "mae")
            self.log(f"{stage}/{name}", metric, prog_bar=prog_bar, on_step=False, on_epoch=True)

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

    def predict_step(
        self, batch: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]], batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        """Prediction step"""
        if isinstance(batch, (tuple, list)):
            x = batch[0]
        else:
            x = batch

        return {"forecast": self(x), "input": x}

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

    def forecast_multi_step(self, x: torch.Tensor, steps: int) -> torch.Tensor:
        """
        Multi-step forecasting using recursive prediction.

        Each iteration takes the first predicted step and appends it to the
        sliding input window, so ``output_dim`` must equal ``input_dim``.
        Returns ``(batch, steps, output_dim)``.
        """
        if self.output_dim != self.input_dim:
            raise ValueError(
                "Recursive forecasting requires output_dim == input_dim "
                f"(got output_dim={self.output_dim}, input_dim={self.input_dim})"
            )

        was_training = self.training
        self.eval()

        with torch.no_grad():
            if x.dim() == 2:
                x = x.unsqueeze(0)  # Add batch dimension if needed
            forecasts = []
            current_input = x.clone()

            for _ in range(steps):
                forecast = self(current_input)  # (batch, horizon, output_dim)
                next_step = forecast[:, :1, :]  # (batch, 1, output_dim)
                forecasts.append(next_step)

                # Slide window
                current_input = torch.cat([current_input[:, 1:, :], next_step], dim=1)

            multi_step_forecast = torch.cat(forecasts, dim=1)  # (batch, steps, output_dim)

        if was_training:
            self.train()
        return multi_step_forecast


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for batch-first inputs ``(batch, seq_len, d_model)``"""

    pe: torch.Tensor

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)[:, : d_model // 2]

        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        if seq_len > self.pe.size(1):
            raise ValueError(f"Sequence length {seq_len} exceeds positional encoding max_len {self.pe.size(1)}")
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)


# Convenience functions for common forecasting tasks
def create_univariate_forecaster(
    sequence_length: int = 50, prediction_horizon: int = 1, architecture: str = "lstm", **kwargs
) -> TimeSeriesForecaster:
    """Create forecaster for univariate time series"""
    return TimeSeriesForecaster(
        input_dim=1,
        output_dim=1,
        sequence_length=sequence_length,
        prediction_horizon=prediction_horizon,
        architecture=architecture,
        **kwargs,
    )


def create_multivariate_forecaster(
    input_dim: int,
    output_dim: int,
    sequence_length: int = 50,
    prediction_horizon: int = 1,
    architecture: str = "lstm",
    **kwargs,
) -> TimeSeriesForecaster:
    """Create forecaster for multivariate time series"""
    return TimeSeriesForecaster(
        input_dim=input_dim,
        output_dim=output_dim,
        sequence_length=sequence_length,
        prediction_horizon=prediction_horizon,
        architecture=architecture,
        **kwargs,
    )
