# tests/test_data_synth_vision.py
"""Tests for synthetic vision data generation."""

import sys
from pathlib import Path
import pytest
import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lmpro.data.synth_vision import (
    VisionDatasetConfig,
    SyntheticImageDataset,
    SyntheticSegmentationDataset,
    create_synthetic_image_dataset,
    create_synthetic_segmentation_dataset,
)


@pytest.fixture
def small_config():
    return VisionDatasetConfig(
        num_samples=20,
        image_size=(16, 16),
        num_channels=3,
        num_classes=4,
    )


@pytest.fixture
def image_dataset(small_config):
    return SyntheticImageDataset(config=small_config, split="train")


@pytest.fixture
def seg_dataset():
    cfg = VisionDatasetConfig(num_samples=10, image_size=(16, 16), num_classes=3)
    return SyntheticSegmentationDataset(config=cfg, split="train")


# ─── VisionDatasetConfig ─────────────────────────────────────────────────────

class TestVisionDatasetConfig:
    def test_defaults(self):
        cfg = VisionDatasetConfig()
        assert cfg.num_samples == 1000
        assert cfg.image_size == (64, 64)
        assert cfg.num_channels == 3
        assert cfg.num_classes == 10

    def test_custom(self):
        cfg = VisionDatasetConfig(num_samples=50, num_classes=5)
        assert cfg.num_samples == 50
        assert cfg.num_classes == 5


# ─── SyntheticImageDataset ───────────────────────────────────────────────────

class TestSyntheticImageDataset:
    def test_len(self, image_dataset, small_config):
        assert len(image_dataset) == small_config.num_samples

    def test_item_types(self, image_dataset, small_config):
        img, label = image_dataset[0]
        assert isinstance(img, torch.Tensor)
        assert isinstance(label, torch.Tensor)

    def test_image_shape(self, image_dataset, small_config):
        img, _ = image_dataset[0]
        h, w = small_config.image_size
        assert img.shape == (small_config.num_channels, h, w)

    def test_labels_in_range(self, image_dataset, small_config):
        for i in range(len(image_dataset)):
            _, label = image_dataset[i]
            assert 0 <= label.item() < small_config.num_classes

    def test_train_val_split_different(self, small_config):
        train = SyntheticImageDataset(config=small_config, split="train")
        val = SyntheticImageDataset(config=small_config, split="val")
        img_train, _ = train[0]
        img_val, _ = val[0]
        # Different splits should generally produce different data
        assert not torch.allclose(img_train, img_val)

    def test_transform_applied(self, small_config):
        called = {"n": 0}

        def my_transform(img):
            # img is a PIL Image when transform is applied
            called["n"] += 1
            return img  # pass through

        ds = SyntheticImageDataset(config=small_config, transform=my_transform)
        _ = ds[0]
        assert called["n"] == 1

    def test_class_balance(self, small_config):
        cfg = VisionDatasetConfig(num_samples=100, num_classes=4, image_size=(8, 8))
        ds = SyntheticImageDataset(config=cfg)
        labels = [ds[i][1].item() for i in range(len(ds))]
        unique = set(labels)
        # All classes should appear
        assert len(unique) == cfg.num_classes


# ─── SyntheticSegmentationDataset ───────────────────────────────────────────

class TestSyntheticSegmentationDataset:
    def test_len(self, seg_dataset):
        assert len(seg_dataset) == 10

    def test_item_types(self, seg_dataset):
        img, mask = seg_dataset[0]
        assert isinstance(img, torch.Tensor)
        assert isinstance(mask, torch.Tensor)

    def test_image_shape(self, seg_dataset):
        img, _ = seg_dataset[0]
        assert img.shape[0] == 3  # channels
        assert img.ndim == 3

    def test_mask_shape_matches_image(self, seg_dataset):
        img, mask = seg_dataset[0]
        assert mask.shape[-2:] == img.shape[-2:]

    def test_mask_values_in_range(self, seg_dataset):
        for i in range(len(seg_dataset)):
            _, mask = seg_dataset[i]
            assert mask.min() >= 0
            assert mask.max() < seg_dataset.config.num_classes  # labels 0..num_classes-1


# ─── Factory Functions ───────────────────────────────────────────────────────

class TestCreateSyntheticImageDataset:
    def test_creates_dataset(self):
        cfg = VisionDatasetConfig(num_samples=30, image_size=(8, 8), num_classes=3)
        result = create_synthetic_image_dataset(config=cfg)
        assert isinstance(result, dict)
        assert "train" in result
        assert isinstance(result["train"], SyntheticImageDataset)

    def test_returns_three_splits(self):
        cfg = VisionDatasetConfig(num_samples=30, image_size=(8, 8))
        result = create_synthetic_image_dataset(config=cfg)
        assert len(result) == 3
        assert set(result.keys()) == {"train", "val", "test"}


class TestCreateSyntheticSegmentationDataset:
    def test_creates_dataset(self):
        cfg = VisionDatasetConfig(num_samples=20, image_size=(8, 8), num_classes=3)
        result = create_synthetic_segmentation_dataset(config=cfg)
        assert isinstance(result, dict)
        assert "train" in result
        assert isinstance(result["train"], SyntheticSegmentationDataset)
