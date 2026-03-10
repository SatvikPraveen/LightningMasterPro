# tests/test_data_synth_tabular.py
"""Tests for synthetic tabular data generation."""

import sys
from pathlib import Path
import pytest
import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lmpro.data.synth_tabular import (
    TabularDatasetConfig,
    SyntheticTabularDataset,
    ComplexTabularDataset,
    create_synthetic_tabular_dataset,
    create_synthetic_regression_dataset,
    create_synthetic_classification_dataset,
)


@pytest.fixture
def small_config():
    return TabularDatasetConfig(
        num_samples=40,
        num_features=8,
        num_informative=6,
        num_classes=3,
    )


@pytest.fixture
def clf_dataset(small_config):
    return SyntheticTabularDataset(config=small_config, task="classification")


@pytest.fixture
def reg_dataset(small_config):
    return SyntheticTabularDataset(config=small_config, task="regression")


# ─── TabularDatasetConfig ────────────────────────────────────────────────────

class TestTabularDatasetConfig:
    def test_defaults(self):
        cfg = TabularDatasetConfig()
        assert cfg.num_samples == 1000
        assert cfg.num_features == 20
        assert cfg.num_classes == 3

    def test_custom(self):
        cfg = TabularDatasetConfig(num_samples=200, num_features=5)
        assert cfg.num_samples == 200
        assert cfg.num_features == 5


# ─── SyntheticTabularDataset ─────────────────────────────────────────────────

class TestSyntheticTabularDataset:
    def test_len_classification(self, clf_dataset, small_config):
        assert len(clf_dataset) == small_config.num_samples

    def test_len_regression(self, reg_dataset, small_config):
        assert len(reg_dataset) == small_config.num_samples

    def test_item_types_classification(self, clf_dataset):
        x, y = clf_dataset[0]
        assert isinstance(x, torch.Tensor)
        assert isinstance(y, torch.Tensor)

    def test_feature_shape(self, clf_dataset, small_config):
        x, _ = clf_dataset[0]
        assert x.shape[0] == small_config.num_features

    def test_classification_labels_in_range(self, clf_dataset, small_config):
        for i in range(len(clf_dataset)):
            _, y = clf_dataset[i]
            assert 0 <= y.item() < small_config.num_classes

    def test_regression_labels_float(self, reg_dataset):
        for i in range(len(reg_dataset)):
            _, y = reg_dataset[i]
            assert y.dtype == torch.float32

    def test_feature_names_accessible(self, clf_dataset, small_config):
        assert hasattr(clf_dataset, "feature_names")
        assert len(clf_dataset.feature_names) == small_config.num_features

    def test_normalization_applied(self, small_config):
        ds = SyntheticTabularDataset(config=small_config, normalize=True)
        all_x = torch.stack([ds[i][0] for i in range(len(ds))])
        # Mean should be close to 0 and std close to 1 after normalization
        mean = all_x.mean(dim=0)
        assert torch.all(torch.abs(mean) < 1.0)  # within range

    def test_no_normalization(self, small_config):
        ds = SyntheticTabularDataset(config=small_config, normalize=False)
        assert len(ds) == small_config.num_samples

    def test_invalid_task_raises(self, small_config):
        with pytest.raises((ValueError, KeyError)):
            SyntheticTabularDataset(config=small_config, task="unsupported")


# ─── All Classes Present ─────────────────────────────────────────────────────

class TestClassBalance:
    def test_all_classes_in_classification(self):
        cfg = TabularDatasetConfig(num_samples=150, num_features=20, num_classes=3)
        ds = SyntheticTabularDataset(config=cfg, task="classification")
        labels = set(ds[i][1].item() for i in range(len(ds)))
        assert len(labels) == 3


# ─── Factory Functions ───────────────────────────────────────────────────────

class TestCreateSyntheticTabularDataset:
    def test_creates_dataset(self):
        cfg = TabularDatasetConfig(num_samples=20, num_features=20, num_classes=3)
        result = create_synthetic_tabular_dataset(config=cfg)
        assert isinstance(result, dict)
        assert "train" in result
        assert isinstance(result["train"], SyntheticTabularDataset)


class TestCreateSyntheticRegressionDataset:
    def test_creates_regression_dataset(self):
        cfg = TabularDatasetConfig(num_samples=15, num_features=5)
        result = create_synthetic_regression_dataset(config=cfg)
        assert isinstance(result, dict)
        assert "train" in result
        _, y = result["train"][0]
        assert y.dtype == torch.float32


class TestCreateSyntheticClassificationDataset:
    def test_creates_classification_dataset(self):
        cfg = TabularDatasetConfig(num_samples=30, num_features=20, num_classes=3)
        result = create_synthetic_classification_dataset(config=cfg)
        assert isinstance(result, dict)
        assert "train" in result
        _, y = result["train"][0]
        assert y.dtype == torch.long
