# File: src/lmpro/data/synth_timeseries.py

"""
Synthetic time series data generation for forecasting and classification tasks
"""

import torch
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional, Union
from dataclasses import dataclass
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta


@dataclass
class TimeSeriesDatasetConfig:
    """Configuration for synthetic time series datasets"""
    num_samples: int = 1000
    sequence_length: int = 100
    prediction_horizon: int = 10
    num_features: int = 1
    noise_level: float = 0.1
    trend_strength: float = 0.1
    seasonal_strength: float = 0.3
    seasonal_periods: List[int] = None
    random_state: int = 42
    save_path: Optional[str] = "data/synthetic/timeseries"
    
    def __post_init__(self):
        if self.seasonal_periods is None:
            self.seasonal_periods = [12, 24]  # Daily and weekly patterns


class SyntheticTimeSeriesDataset(Dataset):
    """Base synthetic time series dataset"""
    
    def __init__(
        self,
        config: TimeSeriesDatasetConfig,
        task: str = "forecasting",
        split: str = "train"
    ):
        self.config = config
        self.task = task
        self.split = split
        
        # Generate time series data
        self.timeseries_data = self._generate_timeseries()
        
        # Create sequences for training
        self.sequences, self.targets = self._create_sequences()
        
        # Convert to tensors
        self.sequences_tensor = torch.tensor(self.sequences, dtype=torch.float32)
        self.targets_tensor = torch.tensor(self.targets, dtype=torch.float32)
    
    def _generate_timeseries(self) -> np.ndarray:
        """Generate synthetic time series with multiple components"""
        np.random.seed(self.config.random_state + (0 if self.split == "train" else 1))
        
        # Total length needed for sequences + predictions
        total_length = self.config.num_samples + self.config.sequence_length + self.config.prediction_horizon
        
        # Initialize time series
        timeseries = np.zeros((total_length, self.config.num_features))
        
        for feature in range(self.config.num_features):
            # Generate components for each feature
            ts = self._generate_single_timeseries(total_length, feature)
            timeseries[:, feature] = ts
        
        return timeseries
    
    def _generate_single_timeseries(self, length: int, feature_idx: int = 0) -> np.ndarray:
        """Generate a single time series with trend, seasonality, and noise"""
        t = np.arange(length)
        
        # Trend component
        if self.config.trend_strength > 0:
            trend = self.config.trend_strength * t + np.random.normal(0, 0.1, length).cumsum()
        else:
            trend = np.zeros(length)
        
        # Seasonal components
        seasonal = np.zeros(length)
        if self.config.seasonal_strength > 0:
            for period in self.config.seasonal_periods:
                amplitude = self.config.seasonal_strength * np.random.uniform(0.5, 1.5)
                phase = np.random.uniform(0, 2 * np.pi)
                seasonal += amplitude * np.sin(2 * np.pi * t / period + phase)
        
        # Cyclic patterns (longer cycles)
        cyclic = 0.2 * np.sin(2 * np.pi * t / (length / 3)) if length > 50 else 0
        
        # Random walk component
        random_walk = np.random.normal(0, 0.1, length).cumsum()
        
        # AR component (autoregressive)
        ar_series = np.zeros(length)
        ar_coeff = 0.7
        for i in range(1, length):
            ar_series[i] = ar_coeff * ar_series[i-1] + np.random.normal(0, 0.1)
        
        # Noise
        noise = np.random.normal(0, self.config.noise_level, length)
        
        # Combine components
        ts = trend + seasonal + cyclic + 0.3 * random_walk + 0.2 * ar_series + noise
        
        # Add occasional anomalies
        if np.random.random() < 0.1:  # 10% chance of anomalies
            anomaly_points = np.random.choice(length, size=max(1, length // 100), replace=False)
            anomaly_magnitude = np.random.uniform(2, 5) * np.std(ts)
            ts[anomaly_points] += np.random.choice([-1, 1], len(anomaly_points)) * anomaly_magnitude
        
        return ts
    
    def _create_sequences(self) -> Tuple[np.ndarray, np.ndarray]:
        """Create input-target sequence pairs"""
        sequences = []
        targets = []
        
        data_length = len(self.timeseries_data)
        
        for i in range(data_length - self.config.sequence_length - self.config.prediction_horizon + 1):
            # Input sequence
            seq = self.timeseries_data[i:i + self.config.sequence_length]
            
            # Target (for forecasting)
            if self.task == "forecasting":
                target = self.timeseries_data[
                    i + self.config.sequence_length:
                    i + self.config.sequence_length + self.config.prediction_horizon
                ]
            else:  # classification
                # Use statistics of the sequence for classification
                seq_mean = np.mean(seq, axis=0)
                seq_trend = seq[-1] - seq[0]
                target_score = np.sum(seq_mean) + np.sum(seq_trend)
                target = 0 if target_score < -0.5 else 1 if target_score < 0.5 else 2
                target = np.array([target])
            
            sequences.append(seq)
            targets.append(target)
        
        # Limit to num_samples
        sequences = sequences[:self.config.num_samples]
        targets = targets[:self.config.num_samples]
        
        return np.array(sequences), np.array(targets)
    
    def __len__(self) -> int:
        return len(self.sequences_tensor)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.sequences_tensor[idx], self.targets_tensor[idx]


class MultiVariateTimeSeriesDataset(Dataset):
    """Multivariate time series dataset with cross-correlations"""
    
    def __init__(
        self,
        config: TimeSeriesDatasetConfig,
        task: str = "forecasting",
        split: str = "train"
    ):
        self.config = config
        self.task = task
        self.split = split
        
        # Generate correlated multivariate time series
        self.timeseries_data = self._generate_multivariate_timeseries()
        self.sequences, self.targets = self._create_sequences()
        
        self.sequences_tensor = torch.tensor(self.sequences, dtype=torch.float32)
        self.targets_tensor = torch.tensor(self.targets, dtype=torch.float32)
    
    def _generate_multivariate_timeseries(self) -> np.ndarray:
        """Generate multivariate time series with cross-correlations"""
        np.random.seed(self.config.random_state + (0 if self.split == "train" else 1))
        
        total_length = self.config.num_samples + self.config.sequence_length + self.config.prediction_horizon
        
        # Generate independent series first
        independent_series = np.zeros((total_length, self.config.num_features))
        
        for i in range(self.config.num_features):
            independent_series[:, i] = self._generate_single_timeseries(total_length, i)
        
        # Create correlation matrix
        correlation_matrix = np.random.uniform(0.3, 0.8, (self.config.num_features, self.config.num_features))
        correlation_matrix = (correlation_matrix + correlation_matrix.T) / 2  # Make symmetric
        np.fill_diagonal(correlation_matrix, 1.0)
        
        # Apply correlations using Cholesky decomposition
        try:
            L = np.linalg.cholesky(correlation_matrix)
            correlated_series = np.zeros_like(independent_series)
            
            for t in range(total_length):
                correlated_series[t] = L @ independent_series[t]
        except np.linalg.LinAlgError:
            # Fall back to original series if correlation matrix is not positive definite
            correlated_series = independent_series
        
        return correlated_series
    
    def _generate_single_timeseries(self, length: int, feature_idx: int) -> np.ndarray:
        """Generate single time series with feature-specific characteristics"""
        t = np.arange(length)
        
        # Feature-specific parameters
        trend_coeff = self.config.trend_strength * (1 + 0.2 * feature_idx)
        seasonal_coeff = self.config.seasonal_strength * (1 + 0.1 * (feature_idx % 2))
        
        # Components
        trend = trend_coeff * t / length + np.random.normal(0, 0.05, length).cumsum() * 0.1
        
        seasonal = np.zeros(length)
        for period in self.config.seasonal_periods:
            phase_shift = feature_idx * np.pi / 4  # Different phase for each feature
            seasonal += seasonal_coeff * np.sin(2 * np.pi * t / period + phase_shift)
        
        # AR process
        ar_series = np.zeros(length)
        ar_coeff = 0.5 + 0.3 * (feature_idx % 3) / 3
        for i in range(1, length):
            ar_series[i] = ar_coeff * ar_series[i-1] + np.random.normal(0, 0.1)
        
        noise = np.random.normal(0, self.config.noise_level * (1 + 0.1 * feature_idx), length)
        
        return trend + seasonal + 0.3 * ar_series + noise
    
    def _create_sequences(self) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for multivariate data"""
        sequences = []
        targets = []
        
        data_length = len(self.timeseries_data)
        
        for i in range(data_length - self.config.sequence_length - self.config.prediction_horizon + 1):
            seq = self.timeseries_data[i:i + self.config.sequence_length]
            
            if self.task == "forecasting":
                # Predict next values for all features or just the first feature
                target = self.timeseries_data[
                    i + self.config.sequence_length:
                    i + self.config.sequence_length + self.config.prediction_horizon,
                    0  # Predict only first feature for simplicity
                ]
            else:  # classification based on multivariate patterns
                seq_stats = np.concatenate([
                    np.mean(seq, axis=0),
                    np.std(seq, axis=0),
                    seq[-1] - seq[0]  # Trend for each feature
                ])
                target_score = np.sum(seq_stats)
                target = 0 if target_score < np.percentile(seq_stats, 33) else \
                        1 if target_score < np.percentile(seq_stats, 67) else 2
                target = np.array([target])
            
            sequences.append(seq)
            targets.append(target)
        
        sequences = sequences[:self.config.num_samples]
        targets = targets[:self.config.num_samples]
        
        return np.array(sequences), np.array(targets)
    
    def __len__(self) -> int:
        return len(self.sequences_tensor)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.sequences_tensor[idx], self.targets_tensor[idx]


class AnomalyTimeSeriesDataset(Dataset):
    """Time series dataset with anomaly detection labels"""
    
    def __init__(
        self,
        config: TimeSeriesDatasetConfig,
        anomaly_ratio: float = 0.05,
        split: str = "train"
    ):
        self.config = config
        self.anomaly_ratio = anomaly_ratio
        self.split = split
        
        # Generate time series with anomalies
        self.timeseries_data, self.anomaly_labels = self._generate_anomaly_timeseries()
        self.sequences, self.targets = self._create_anomaly_sequences()
        
        self.sequences_tensor = torch.tensor(self.sequences, dtype=torch.float32)
        self.targets_tensor = torch.tensor(self.targets, dtype=torch.long)
    
    def _generate_anomaly_timeseries(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate time series with labeled anomalies"""
        np.random.seed(self.config.random_state + (0 if self.split == "train" else 1))
        
        total_length = self.config.num_samples + self.config.sequence_length
        
        # Generate normal time series
        timeseries = np.zeros((total_length, self.config.num_features))
        for i in range(self.config.num_features):
            timeseries[:, i] = self._generate_single_timeseries(total_length, i)
        
        # Add anomalies
        anomaly_labels = np.zeros(total_length, dtype=int)
        num_anomalies = int(total_length * self.anomaly_ratio)
        anomaly_indices = np.random.choice(total_length, size=num_anomalies, replace=False)
        
        for idx in anomaly_indices:
            anomaly_labels[idx] = 1
            
            # Different types of anomalies
            anomaly_type = np.random.choice(['spike', 'dip', 'level_shift', 'trend_change'])
            
            if anomaly_type == 'spike':
                magnitude = np.random.uniform(3, 6) * np.std(timeseries[max(0, idx-10):idx+1])
                timeseries[idx] += magnitude
            elif anomaly_type == 'dip':
                magnitude = np.random.uniform(3, 6) * np.std(timeseries[max(0, idx-10):idx+1])
                timeseries[idx] -= magnitude
            elif anomaly_type == 'level_shift':
                shift = np.random.uniform(2, 4) * np.std(timeseries[max(0, idx-10):idx+1])
                end_idx = min(total_length, idx + np.random.randint(5, 20))
                timeseries[idx:end_idx] += shift
                anomaly_labels[idx:end_idx] = 1
            elif anomaly_type == 'trend_change':
                trend_change = np.random.uniform(0.1, 0.3)
                end_idx = min(total_length, idx + np.random.randint(10, 30))
                trend_vals = np.arange(end_idx - idx) * trend_change
                timeseries[idx:end_idx] += trend_vals.reshape(-1, 1)
                anomaly_labels[idx:end_idx] = 1
        
        return timeseries, anomaly_labels
    
    def _generate_single_timeseries(self, length: int, feature_idx: int) -> np.ndarray:
        """Generate clean time series for anomaly detection"""
        t = np.arange(length)
        
        # Simpler, more predictable patterns for anomaly detection
        trend = self.config.trend_strength * t / length
        seasonal = self.config.seasonal_strength * np.sin(2 * np.pi * t / 24)  # Daily pattern
        noise = np.random.normal(0, self.config.noise_level, length)
        
        return trend + seasonal + noise
    
    def _create_anomaly_sequences(self) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences with anomaly labels"""
        sequences = []
        targets = []
        
        data_length = len(self.timeseries_data)
        
        for i in range(data_length - self.config.sequence_length + 1):
            seq = self.timeseries_data[i:i + self.config.sequence_length]
            # Label sequence as anomalous if it contains any anomalies
            label = int(np.any(self.anomaly_labels[i:i + self.config.sequence_length]))
            
            sequences.append(seq)
            targets.append(label)
        
        sequences = sequences[:self.config.num_samples]
        targets = targets[:self.config.num_samples]
        
        return np.array(sequences), np.array(targets)
    
    def __len__(self) -> int:
        return len(self.sequences_tensor)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.sequences_tensor[idx], self.targets_tensor[idx]


def create_synthetic_timeseries_dataset(
    config: TimeSeriesDatasetConfig,
    task: str = "forecasting",
    dataset_type: str = "univariate",
    splits: List[str] = ["train", "val", "test"],
    split_ratios: List[float] = [0.7, 0.15, 0.15]
) -> dict:
    """Create synthetic time series datasets"""
    datasets = {}
    
    total_samples = config.num_samples
    split_sizes = [int(ratio * total_samples) for ratio in split_ratios]
    
    for split, size in zip(splits, split_sizes):
        split_config = TimeSeriesDatasetConfig(
            num_samples=size,
            sequence_length=config.sequence_length,
            prediction_horizon=config.prediction_horizon,
            num_features=config.num_features,
            noise_level=config.noise_level,
            trend_strength=config.trend_strength,
            seasonal_strength=config.seasonal_strength,
            seasonal_periods=config.seasonal_periods,
            random_state=config.random_state,
            save_path=config.save_path
        )
        
        if dataset_type == "univariate":
            datasets[split] = SyntheticTimeSeriesDataset(split_config, task=task, split=split)
        elif dataset_type == "multivariate":
            datasets[split] = MultiVariateTimeSeriesDataset(split_config, task=task, split=split)
        elif dataset_type == "anomaly":
            datasets[split] = AnomalyTimeSeriesDataset(split_config, split=split)
    
    return datasets


def create_synthetic_forecasting_dataset(
    config: TimeSeriesDatasetConfig,
    dataset_type: str = "univariate",
    splits: List[str] = ["train", "val", "test"],
    split_ratios: List[float] = [0.7, 0.15, 0.15]
) -> dict:
    """Create synthetic forecasting datasets"""
    return create_synthetic_timeseries_dataset(
        config, task="forecasting", dataset_type=dataset_type, splits=splits, split_ratios=split_ratios
    )


def visualize_timeseries_data(dataset: Dataset, num_samples: int = 3, save_path: Optional[str] = None) -> None:
    """Visualize time series samples"""
    fig, axes = plt.subplots(num_samples, 1, figsize=(15, 4 * num_samples))
    if num_samples == 1:
        axes = [axes]
    
    for i in range(num_samples):
        sequence, target = dataset[i]
        
        # Plot sequence
        if sequence.dim() == 2:  # Multivariate
            for feature in range(min(3, sequence.shape[1])):  # Plot up to 3 features
                axes[i].plot(sequence[:, feature].numpy(), label=f'Feature {feature}', alpha=0.7)
        else:  # Univariate
            axes[i].plot(sequence.numpy(), label='Time Series', color='blue')
        
        # Plot target if forecasting
        if hasattr(dataset, 'task') and dataset.task == "forecasting":
            target_start = len(sequence)
            target_indices = range(target_start, target_start + len(target))
            axes[i].plot(target_indices, target.numpy(), 'r--', label='Target', linewidth=2)
        
        axes[i].set_title(f'Sample {i} - Target: {target.numpy() if target.numel() == 1 else "Forecast"}')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Time series visualization saved to {save_path}")
    
    plt.show()


def analyze_timeseries_properties(dataset: Dataset) -> Dict[str, float]:
    """Analyze statistical properties of time series dataset"""
    # Sample some sequences
    sample_size = min(100, len(dataset))
    sequences = []
    targets = []
    
    for i in range(sample_size):
        seq, target = dataset[i]
        sequences.append(seq.numpy())
        targets.append(target.numpy())
    
    sequences = np.array(sequences)
    
    analysis = {}
    
    # Basic statistics
    analysis['mean_sequence_length'] = sequences.shape[1]
    analysis['num_features'] = sequences.shape[2] if sequences.ndim == 3 else 1
    analysis['mean_value'] = np.mean(sequences)
    analysis['std_value'] = np.std(sequences)
    
    # Stationarity (simplified test)
    if sequences.ndim == 3:
        # For multivariate, test first feature
        first_feature = sequences[:, :, 0].flatten()
    else:
        first_feature = sequences.flatten()
    
    # Simple stationarity indicator (variance of rolling mean)
    window_size = min(10, len(first_feature) // 10)
    if window_size > 1:
        rolling_means = []
        for i in range(len(first_feature) - window_size + 1):
            rolling_means.append(np.mean(first_feature[i:i + window_size]))
        analysis['stationarity_indicator'] = np.std(rolling_means)
    
    # Seasonality detection (basic)
    if len(first_feature) > 24:
        autocorr_12 = np.corrcoef(first_feature[:-12], first_feature[12:])[0, 1]
        autocorr_24 = np.corrcoef(first_feature[:-24], first_feature[24:])[0, 1]
        analysis['seasonality_12'] = autocorr_12
        analysis['seasonality_24'] = autocorr_24
    
    return analysis


def print_timeseries_summary(dataset: Dataset, name: str = "Time Series Dataset") -> None:
    """Print summary of time series dataset"""
    print(f"\n{name} Summary:")
    print(f"Length: {len(dataset)}")
    
    sample_seq, sample_target = dataset[0]
    print(f"Sequence shape: {sample_seq.shape}")
    print(f"Target shape: {sample_target.shape}")
    
    analysis = analyze_timeseries_properties(dataset)
    for key, value in analysis.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")
    
    if hasattr(dataset, 'task'):
        print(f"Task: {dataset.task}")
    if hasattr(dataset, 'config'):
        print(f"Prediction horizon: {dataset.config.prediction_horizon}")
        print(f"Noise level: {dataset.config.noise_level}")
        print(f"Seasonal periods: {dataset.config.seasonal_periods}")