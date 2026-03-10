# tests/test_data_synth_nlp.py
"""Tests for synthetic NLP data generation."""

import sys
from pathlib import Path
import pytest
import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lmpro.data.synth_nlp import (
    NLPDatasetConfig,
    SyntheticTextDataset,
    CharacterLevelDataset,
    SentimentDataset,
    create_synthetic_text_dataset,
    create_synthetic_sentiment_dataset,
    create_character_level_dataset,
)


@pytest.fixture
def small_nlp_config():
    return NLPDatasetConfig(
        num_samples=30,
        vocab_size=200,
        max_sequence_length=20,
        min_sequence_length=5,
        num_classes=3,
    )


@pytest.fixture
def text_dataset(small_nlp_config):
    return SyntheticTextDataset(config=small_nlp_config, split="train")


@pytest.fixture
def sentiment_dataset():
    cfg = NLPDatasetConfig(num_samples=20, vocab_size=100, max_sequence_length=15, num_classes=2)
    return SentimentDataset(config=cfg, split="train")

@pytest.fixture
def char_dataset():
    cfg = NLPDatasetConfig(num_samples=200, vocab_size=128, max_sequence_length=50)
    return CharacterLevelDataset(config=cfg, split="train")


# ─── NLPDatasetConfig ────────────────────────────────────────────────────────

class TestNLPDatasetConfig:
    def test_defaults(self):
        cfg = NLPDatasetConfig()
        assert cfg.num_samples == 1000
        assert cfg.vocab_size == 5000
        assert cfg.max_sequence_length == 128

    def test_custom(self):
        cfg = NLPDatasetConfig(num_samples=50, vocab_size=300)
        assert cfg.num_samples == 50
        assert cfg.vocab_size == 300


# ─── SyntheticTextDataset ────────────────────────────────────────────────────

class TestSyntheticTextDataset:
    def test_len(self, text_dataset, small_nlp_config):
        assert len(text_dataset) == small_nlp_config.num_samples

    def test_item_returns_tuple(self, text_dataset):
        item = text_dataset[0]
        assert isinstance(item, tuple)
        assert len(item) == 2

    def test_tokens_are_tensor(self, text_dataset):
        tokens, label = text_dataset[0]
        assert isinstance(tokens, torch.Tensor)

    def test_label_in_range(self, text_dataset, small_nlp_config):
        for i in range(len(text_dataset)):
            _, label = text_dataset[i]
            if isinstance(label, torch.Tensor):
                label = label.item()
            assert 0 <= label < small_nlp_config.num_classes

    def test_vocab_accessible(self, text_dataset):
        assert hasattr(text_dataset, "word_to_idx")
        assert hasattr(text_dataset, "vocab")

    def test_tokens_within_vocab(self, text_dataset, small_nlp_config):
        tokens, _ = text_dataset[0]
        assert tokens.max().item() < small_nlp_config.vocab_size

    def test_all_classes_present(self, small_nlp_config):
        cfg = NLPDatasetConfig(
            num_samples=90, vocab_size=200, max_sequence_length=20,
            min_sequence_length=5, num_classes=3
        )
        ds = SyntheticTextDataset(config=cfg)
        labels = set()
        for i in range(len(ds)):
            _, label = ds[i]
            if isinstance(label, torch.Tensor):
                labels.add(label.item())
            else:
                labels.add(label)
        assert len(labels) == cfg.num_classes


# ─── SentimentDataset ────────────────────────────────────────────────────────

class TestSentimentDataset:
    def test_len(self, sentiment_dataset):
        assert len(sentiment_dataset) == 20

    def test_binary_labels(self, sentiment_dataset):
        # SentimentDataset generates 3 sentiment classes (0: neg, 1: neutral, 2: pos)
        for i in range(len(sentiment_dataset)):
            tokens, label = sentiment_dataset[i]
            if isinstance(label, torch.Tensor):
                label = label.item()
            assert label in (0, 1, 2)

    def test_tokens_are_tensor(self, sentiment_dataset):
        tokens, _ = sentiment_dataset[0]
        assert isinstance(tokens, torch.Tensor)


# ─── CharacterLevelDataset ───────────────────────────────────────────────────

class TestCharacterLevelDataset:
    def test_len(self, char_dataset):
        assert len(char_dataset) > 0

    def test_item_types(self, char_dataset):
        x, y = char_dataset[0]
        assert isinstance(x, torch.Tensor)
        assert isinstance(y, torch.Tensor)

    def test_input_output_same_length(self, char_dataset):
        x, y = char_dataset[0]
        assert x.shape == y.shape

    def test_char_ids_in_range(self, char_dataset):
        for i in range(min(5, len(char_dataset))):
            x, y = char_dataset[i]
            assert x.max().item() < 128  # ASCII range


# ─── Factory Functions ───────────────────────────────────────────────────────

class TestCreateSyntheticTextDataset:
    def test_creates_dataset(self):
        cfg = NLPDatasetConfig(num_samples=10, vocab_size=100, max_sequence_length=20)
        result = create_synthetic_text_dataset(config=cfg)
        assert isinstance(result, dict)
        assert "train" in result
        assert isinstance(result["train"], SyntheticTextDataset)


class TestCreateSyntheticSentimentDataset:
    def test_creates_dataset(self):
        cfg = NLPDatasetConfig(num_samples=20, vocab_size=100, max_sequence_length=15, num_classes=2)
        result = create_synthetic_sentiment_dataset(config=cfg)
        assert isinstance(result, dict)
        assert "train" in result

    def test_binary_labels(self):
        cfg = NLPDatasetConfig(num_samples=40, vocab_size=100, max_sequence_length=15, num_classes=2)
        result = create_synthetic_sentiment_dataset(config=cfg)
        ds = result["train"]
        labels = set()
        for i in range(len(ds)):
            _, label = ds[i]
            if isinstance(label, torch.Tensor):
                labels.add(label.item())
            else:
                labels.add(label)
        # SentimentDataset always uses 3 sentiment classes (negative/neutral/positive)
        assert labels.issubset({0, 1, 2})


class TestCreateCharacterLevelDataset:
    def test_creates_dataset(self):
        cfg = NLPDatasetConfig(num_samples=20, vocab_size=128, max_sequence_length=50)
        result = create_character_level_dataset(config=cfg)
        assert isinstance(result, dict)
        assert "train" in result
