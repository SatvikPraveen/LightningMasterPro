# tests/test_data_synth_timeseries.py
"""Tests for synthetic time series data generation."""

import sys
from pathlib import Path
import pytest
import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lmpro.data.synth_timeseries import (
    TimeSeriesDatasetConfig,
    SyntheticTimeSeriesDataset,
    MultiVariateTimeSeriesDataset,
    AnomalyTimeSeriesDataset,
    create_synthetic_timeseries_dataset,
    create_synthetic_forecasting_dataset,
)


@pytest.fixture
def small_config():
    return TimeSeriesDatasetConfig(
        num_samples=30,
        sequence_length=20,
        prediction_horizon=5,
        num_features=1,
        noise_level=0.05,
    )


@pytest.fixture
def ts_dataset(small_config):
    return SyntheticTimeSeriesDataset(config=small_config, task="forecasting")


@pytest.fixture
def mv_config():
    return TimeSeriesDatasetConfig(
        num_samples=20,
        sequence_length=15,
        prediction_horizon=3,
        num_features=3,
    )


@pytest.fixture
def multivariate_dataset(mv_config):
    return MultiVariateTimeSeriesDataset(config=mv_config)


# ─── TimeSeriesDatasetConfig ─────────────────────────────────────────────────

class TestTimeSeriesDatasetConfig:
    def test_defaults(self):
        cfg = TimeSeriesDatasetConfig()
        assert cfg.num_samples == 1000
        assert cfg.sequence_length == 100
        assert cfg.prediction_horizon == 10
        assert cfg.num_features == 1

    def test_seasonal_periods_default(self):
        cfg = TimeSeriesDatasetConfig()
        assert isinstance(cfg.seasonal_periods, list)
        assert len(cfg.seasonal_periods) > 0

    def test_custom(self):
        cfg = TimeSeriesDatasetConfig(num_samples=50, sequence_length=30)
        assert cfg.num_samples == 50
        assert cfg.sequence_length == 30


# ─── SyntheticTimeSeriesDataset ──────────────────────────────────────────────

class TestSyntheticTimeSeriesDataset:
    def test_len(self, ts_dataset, small_config):
        assert len(ts_dataset) == small_config.num_samples

    def test_item_types(self, ts_dataset):
        x, y = ts_dataset[0]
        assert isinstance(x, torch.Tensor)
        assert isinstance(y, torch.Tensor)

    def test_input_sequence_shape(self, ts_dataset, small_config):
        x, _ = ts_dataset[0]
        assert x.shape[0] == small_config.sequence_length
        assert x.shape[1] == small_config.num_features

    def test_target_shape(self, ts_dataset, small_config):
        _, y = ts_dataset[0]
        assert y.shape[0] == small_config.prediction_horizon

    def test_sequences_have_no_nans(self, ts_dataset):
        for i in range(len(ts_dataset)):
            x, y = ts_dataset[i]
            assert not torch.isnan(x).any()
            assert not torch.isnan(y).any()

    def test_train_val_differ(self, small_config):
        train = SyntheticTimeSeriesDataset(config=small_config, split="train")
        val = SyntheticTimeSeriesDataset(config=small_config, split="val")
        x_train, _ = train[0]
        x_val, _ = val[0]
        assert not torch.allclose(x_train, x_val)

    def test_float_dtype(self, ts_dataset):
        x, y = ts_dataset[0]
        assert x.dtype == torch.float32
        assert y.dtype == torch.float32


# ─── MultiVariateTimeSeriesDataset ───────────────────────────────────────────

class TestMultiVariateTimeSeriesDataset:
    def test_len(self, multivariate_dataset, mv_config):
        assert len(multivariate_dataset) == mv_config.num_samples

    def test_multivariate_input_shape(self, multivariate_dataset, mv_config):
        x, y = multivariate_dataset[0]
        assert x.shape == (mv_config.sequence_length, mv_config.num_features)

    def test_no_nans(self, multivariate_dataset):
        for i in range(len(multivariate_dataset)):
            x, y = multivariate_dataset[i]
            assert not torch.isnan(x).any()
            assert not torch.isnan(y).any()


# ─── AnomalyTimeSeriesDataset ────────────────────────────────────────────────

class TestAnomalyTimeSeriesDataset:
    def test_creates_dataset(self):
        cfg = TimeSeriesDatasetConfig(num_samples=20, sequence_length=15)
        ds = AnomalyTimeSeriesDataset(config=cfg)
        assert len(ds) == 20

    def test_returns_anomaly_label(self):
        cfg = TimeSeriesDatasetConfig(num_samples=30, sequence_length=15)
        ds = AnomalyTimeSeriesDataset(config=cfg)
        x, label = ds[0]
        # Anomaly label should be 0 or 1
        if isinstance(label, torch.Tensor):
            assert label.item() in (0, 1)
        else:
            assert label in (0, 1)

    def test_anomaly_fraction_nonzero(self):
        cfg = TimeSeriesDatasetConfig(num_samples=100, sequence_length=15)
        ds = AnomalyTimeSeriesDataset(config=cfg)
        labels = [ds[i][1].item() if isinstance(ds[i][1], torch.Tensor) else ds[i][1]
                  for i in range(len(ds))]
        assert sum(labels) > 0, "Expected some anomalies in the dataset"


# ─── Factory Functions ───────────────────────────────────────────────────────

class TestCreateSyntheticTimeseriesDataset:
    def test_creates_dataset(self):
        cfg = TimeSeriesDatasetConfig(num_samples=10, sequence_length=15, prediction_horizon=3)
        result = create_synthetic_timeseries_dataset(config=cfg)
        assert isinstance(result, dict)
        assert "train" in result
        assert isinstance(result["train"], SyntheticTimeSeriesDataset)


class TestCreateSyntheticForecastingDataset:
    def test_creates_dataset(self):
        cfg = TimeSeriesDatasetConfig(num_samples=10, sequence_length=15, prediction_horizon=4)
        result = create_synthetic_forecasting_dataset(config=cfg)
        assert isinstance(result, dict)
        assert "train" in result

    def test_input_target_shapes_correct(self):
        cfg = TimeSeriesDatasetConfig(num_samples=10, sequence_length=20, prediction_horizon=4)
        result = create_synthetic_forecasting_dataset(config=cfg)
        ds = result["train"]
        x, y = ds[0]
        assert x.shape[0] == 20
        assert y.shape[0] == 4
