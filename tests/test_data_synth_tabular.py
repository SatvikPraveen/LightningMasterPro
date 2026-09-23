# tests/test_data_synth_tabular.py
"""Tests for synthetic tabular data generation."""

import pytest
import torch

from lmpro.data.synth_tabular import (
    ComplexTabularDataset,
    SyntheticTabularDataset,
    TabularDatasetConfig,
    create_synthetic_classification_dataset,
    create_synthetic_regression_dataset,
    create_synthetic_tabular_dataset,
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

    @pytest.mark.parametrize("task", ["classification", "regression"])
    def test_val_and_test_splits_differ(self, small_config, task):
        train = SyntheticTabularDataset(config=small_config, task=task, split="train", normalize=False)
        val = SyntheticTabularDataset(config=small_config, task=task, split="val", normalize=False)
        test = SyntheticTabularDataset(config=small_config, task=task, split="test", normalize=False)
        assert not torch.equal(val.X_tensor, test.X_tensor)
        assert not torch.equal(train.X_tensor, val.X_tensor)
        assert not torch.equal(train.X_tensor, test.X_tensor)

    def test_same_split_is_deterministic(self, small_config):
        a = SyntheticTabularDataset(config=small_config, split="val")
        b = SyntheticTabularDataset(config=small_config, split="val")
        assert torch.equal(a.X_tensor, b.X_tensor)


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

    @pytest.mark.parametrize("dataset_type", ["simple", "complex", "time_varying"])
    def test_val_and_test_splits_differ(self, dataset_type):
        cfg = TabularDatasetConfig(num_samples=100, num_features=8, num_informative=5, num_classes=3)
        result = create_synthetic_tabular_dataset(config=cfg, dataset_type=dataset_type)
        val, test = result["val"], result["test"]
        if dataset_type == "complex":
            assert isinstance(val, ComplexTabularDataset)
        x_val = val[0][0] if dataset_type == "time_varying" else val.X_tensor
        x_test = test[0][0] if dataset_type == "time_varying" else test.X_tensor
        assert not torch.equal(x_val, x_test)


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


class TestSplitsShareGeneratingFunction:
    """Val/test must come from the same problem as train (sklearn's make_* draw a new one per seed)."""

    def test_regression_val_is_predictable_from_train(self):
        from sklearn.linear_model import LinearRegression

        from lmpro.data.synth_tabular import SyntheticTabularDataset, TabularDatasetConfig

        cfg = TabularDatasetConfig(num_samples=300, num_features=20)
        train, val = (
            SyntheticTabularDataset(cfg, task="regression", split=s, normalize=False) for s in ("train", "val")
        )
        model = LinearRegression().fit(train.X, train.y)
        assert model.score(val.X, val.y) > 0.9
        assert abs(train.y.std() - 1.0) < 0.1  # unit-scale targets

    def test_classification_val_is_predictable_from_train(self):
        from sklearn.linear_model import LogisticRegression

        from lmpro.data.synth_tabular import SyntheticTabularDataset, TabularDatasetConfig

        cfg = TabularDatasetConfig(num_samples=300, num_features=20, num_classes=3)
        train, val = (
            SyntheticTabularDataset(cfg, task="classification", split=s, normalize=False) for s in ("train", "val")
        )
        model = LogisticRegression(max_iter=2000).fit(train.X, train.y)
        assert model.score(val.X, val.y) > 1.5 / cfg.num_classes  # clearly above chance
