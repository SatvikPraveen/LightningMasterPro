# File: src/lmpro/modules/timeseries/forecaster.py

"""
Time series forecasting module with multiple architectures
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning import LightningModule
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingLR
from torchmetrics import MeanSquaredError, MeanAbsoluteError, R2Score
from typing import Dict, Any, Optional, List, Tuple, Union
import numpy as np


class TimeSeriesForecaster(LightningModule):
    """
    Lightning Module for time series forecasting
    
    Supports multiple architectures:
    - LSTM/GRU based
    - CNN based
    - Transformer based
    - Hybrid architectures
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
        **kwargs
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
            self.encoder = nn.LSTM(
                input_dim, hidden_dim, num_layers,
                batch_first=True, dropout=dropout if num_layers > 1 else 0
            )
        elif architecture == "gru":
            self.encoder = nn.GRU(
                input_dim, hidden_dim, num_layers,
                batch_first=True, dropout=dropout if num_layers > 1 else 0
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
                nn.Linear(hidden_dim // 2, prediction_horizon * output_dim)
            )
        elif architecture == "cnn":
            self.output_projection = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(self._get_cnn_output_dim(), hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, prediction_horizon * output_dim)
            )
        elif architecture == "transformer":
            self.output_projection = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, prediction_horizon * output_dim)
            )
        
        # Loss functions
        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()
        
        # Metrics
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
            nn.AdaptiveAvgPool1d(1)
        )
    
    def _get_cnn_output_dim(self) -> int:
        """Get CNN output dimension"""
        return self.hidden_dim
    
    def _build_transformer_encoder(self, dropout: float) -> nn.Module:
        """Build transformer encoder"""
        # Input projection
        input_proj = nn.Linear(self.input_dim, self.hidden_dim)
        
        # Positional encoding
        pos_encoding = PositionalEncoding(self.hidden_dim, dropout, self.sequence_length)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=8,
            dim_feedforward=self.hidden_dim * 2,
            dropout=dropout,
            batch_first=True
        )
        transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)
        
        return nn.ModuleDict({
            'input_proj': input_proj,
            'pos_encoding': pos_encoding,
            'transformer': transformer_encoder
        })
    
    def _create_metrics(self, stage: str) -> Dict[str, Any]:
        """Create metrics for forecasting"""
        return nn.ModuleDict({
            "mse": MeanSquaredError(),
            "mae": MeanAbsoluteError(),
            "rmse": MeanSquaredError(squared=False),
            "r2": R2Score()
        })
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        batch_size = x.shape[0]
        
        if self.architecture in ["lstm", "gru"]:
            # RNN-based forecasting
            output, _ = self.encoder(x)  # (batch, seq_len, hidden_dim)
            
            # Use last hidden state
            last_hidden = output[:, -1, :]  # (batch, hidden_dim)
            forecast = self.output_projection(last_hidden)  # (batch, pred_horizon * output_dim)
            
        elif self.architecture == "cnn":
            # CNN-based forecasting
            x_transposed = x.transpose(1, 2)  # (batch, input_dim, seq_len)
            encoded = self.encoder(x_transposed)  # (batch, hidden_dim, 1)
            encoded = encoded.squeeze(-1)  # (batch, hidden_dim)
            forecast = self.output_projection(encoded)
            
        elif self.architecture == "transformer":
            # Transformer-based forecasting
            modules = self.encoder
            
            # Input projection
            x = modules['input_proj'](x)  # (batch, seq_len, hidden_dim)
            
            # Add positional encoding
            x = modules['pos_encoding'](x)
            
            # Transformer encoding
            encoded = modules['transformer'](x)  # (batch, seq_len, hidden_dim)
            
            # Global average pooling
            encoded = encoded.mean(dim=1)  # (batch, hidden_dim)
            forecast = self.output_projection(encoded)
        
        # Reshape to (batch, pred_horizon, output_dim)
        if self.prediction_horizon == 1:
            forecast = forecast.view(batch_size, self.output_dim)
        else:
            forecast = forecast.view(batch_size, self.prediction_horizon, self.output_dim)
            if self.output_dim == 1:
                forecast = forecast.squeeze(-1)  # (batch, pred_horizon)
        
        return forecast
    
    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step"""
        x, y = batch
        forecast = self(x)
        
        # Ensure shapes match
        if forecast.shape != y.shape:
            forecast = forecast.reshape(y.shape)
        
        # Compute losses
        mse_loss = self.mse_loss(forecast, y)
        mae_loss = self.mae_loss(forecast, y)
        
        # Combined loss (primarily MSE with MAE regularization)
        loss = mse_loss + 0.1 * mae_loss
        
        # Update metrics
        for name, metric in self.train_metrics.items():
            metric.update(forecast, y)
        
        # Log metrics
        self.log("train/loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("train/mse", self.train_metrics["mse"], prog_bar=True, on_step=False, on_epoch=True)
        self.log("train/mae", self.train_metrics["mae"], on_step=False, on_epoch=True)
        
        return loss
    
    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Validation step"""
        x, y = batch
        forecast = self(x)
        
        if forecast.shape != y.shape:
            forecast = forecast.reshape(y.shape)
        
        # Compute losses
        mse_loss = self.mse_loss(forecast, y)
        mae_loss = self.mae_loss(forecast, y)
        loss = mse_loss + 0.1 * mae_loss
        
        # Update metrics
        for name, metric in self.val_metrics.items():
            metric.update(forecast, y)
        
        # Log metrics
        self.log("val/loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val/mse", self.val_metrics["mse"], prog_bar=True, on_step=False, on_epoch=True)
        self.log("val/mae", self.val_metrics["mae"], prog_bar=True, on_step=False, on_epoch=True)
        self.log("val/rmse", self.val_metrics["rmse"], on_step=False, on_epoch=True)
        self.log("val/r2", self.val_metrics["r2"], on_step=False, on_epoch=True)
    
    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Test step"""
        x, y = batch
        forecast = self(x)
        
        if forecast.shape != y.shape:
            forecast = forecast.reshape(y.shape)
        
        # Update metrics
        for name, metric in self.test_metrics.items():
            metric.update(forecast, y)
        
        # Log metrics
        self.log("test/mse", self.test_metrics["mse"], on_step=False, on_epoch=True)
        self.log("test/mae", self.test_metrics["mae"], on_step=False, on_epoch=True)
        self.log("test/rmse", self.test_metrics["rmse"], on_step=False, on_epoch=True)
        self.log("test/r2", self.test_metrics["r2"], on_step=False, on_epoch=True)
    
    def predict_step(self, batch: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]], batch_idx: int) -> Dict[str, torch.Tensor]:
        """Prediction step"""
        if isinstance(batch, tuple):
            x, _ = batch
        else:
            x = batch
        
        forecast = self(x)
        
        return {
            "forecast": forecast,
            "input": x
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
    
    def forecast_multi_step(self, x: torch.Tensor, steps: int) -> torch.Tensor:
        """Multi-step forecasting using recursive prediction"""
        self.eval()
        
        with torch.no_grad():
            forecasts = []
            current_input = x.clone()
            
            for _ in range(steps):
                # Predict next step
                forecast = self(current_input)
                
                if forecast.dim() == 1:
                    forecast = forecast.unsqueeze(0)  # Add batch dimension if needed
                
                if self.prediction_horizon == 1:
                    next_step = forecast.unsqueeze(1)  # (batch, 1, features)
                else:
                    next_step = forecast[:, :1, :]  # Take first prediction
                
                forecasts.append(next_step)
                
                # Update input sequence (sliding window)
                current_input = torch.cat([current_input[:, 1:, :], next_step], dim=1)
            
            # Concatenate all forecasts
            multi_step_forecast = torch.cat(forecasts, dim=1)  # (batch, steps, features)
            
        return multi_step_forecast


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:x.size(0), :].transpose(0, 1)
        return self.dropout(x)


# Convenience functions for common forecasting tasks
def create_univariate_forecaster(
    sequence_length: int = 50,
    prediction_horizon: int = 1,
    architecture: str = "lstm",
    **kwargs
) -> TimeSeriesForecaster:
    """Create forecaster for univariate time series"""
    return TimeSeriesForecaster(
        input_dim=1,
        output_dim=1,
        sequence_length=sequence_length,
        prediction_horizon=prediction_horizon,
        architecture=architecture,
        **kwargs
    )


def create_multivariate_forecaster(
    input_dim: int,
    output_dim: int,
    sequence_length: int = 50,
    prediction_horizon: int = 1,
    architecture: str = "lstm",
    **kwargs
) -> TimeSeriesForecaster:
    """Create forecaster for multivariate time series"""
    return TimeSeriesForecaster(
        input_dim=input_dim,
        output_dim=output_dim,
        sequence_length=sequence_length,
        prediction_horizon=prediction_horizon,
        architecture=architecture,
        **kwargs
    )