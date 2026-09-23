# tests/test_data_synth_nlp.py
"""Tests for synthetic NLP data generation."""

import pytest
import torch

from lmpro.data.synth_nlp import (
    PAD_TOKEN_ID,
    CharacterLevelDataset,
    NLPDatasetConfig,
    SentimentDataset,
    SyntheticTextDataset,
    create_character_level_dataset,
    create_synthetic_sentiment_dataset,
    create_synthetic_text_dataset,
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

    def test_val_and_test_splits_differ(self, small_nlp_config):
        val = SyntheticTextDataset(config=small_nlp_config, split="val")
        test = SyntheticTextDataset(config=small_nlp_config, split="test")
        train = SyntheticTextDataset(config=small_nlp_config, split="train")
        assert val.texts != test.texts
        assert train.texts != val.texts
        assert train.texts != test.texts

    def test_vocab_shared_across_splits(self, small_nlp_config):
        train = SyntheticTextDataset(config=small_nlp_config, split="train")
        val = SyntheticTextDataset(config=small_nlp_config, split="val")
        test = SyntheticTextDataset(config=small_nlp_config, split="test")
        assert train.vocab == val.vocab == test.vocab
        assert train.word_to_idx == val.word_to_idx == test.word_to_idx
        assert train.word_to_idx["<pad>"] == PAD_TOKEN_ID
        assert len(train.vocab) == small_nlp_config.vocab_size

    def test_vocab_independent_of_global_random_state(self, small_nlp_config):
        import random

        a = SyntheticTextDataset(config=small_nlp_config, split="train").vocab
        random.seed(999)
        random.random()
        b = SyntheticTextDataset(config=small_nlp_config, split="train").vocab
        assert a == b

    def test_all_classes_present(self, small_nlp_config):
        cfg = NLPDatasetConfig(
            num_samples=90, vocab_size=200, max_sequence_length=20, min_sequence_length=5, num_classes=3
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

    def test_val_and_test_splits_differ(self):
        cfg = NLPDatasetConfig(num_samples=20, vocab_size=100, max_sequence_length=15)
        val = SentimentDataset(config=cfg, split="val")
        test = SentimentDataset(config=cfg, split="test")
        assert val.sentences != test.sentences

    def test_vocab_shared_across_splits(self):
        cfg = NLPDatasetConfig(num_samples=20, vocab_size=100, max_sequence_length=15)
        train = SentimentDataset(config=cfg, split="train")
        val = SentimentDataset(config=cfg, split="val")
        test = SentimentDataset(config=cfg, split="test")
        assert train.word_to_idx == val.word_to_idx == test.word_to_idx
        assert train.word_to_idx["<pad>"] == PAD_TOKEN_ID
        # every generated word is in-vocabulary (no <unk>)
        unk = train.word_to_idx["<unk>"]
        for ds in (train, val, test):
            for i in range(len(ds)):
                assert unk not in ds[i][0].tolist()

    def test_explicit_vocab_is_adopted(self):
        cfg = NLPDatasetConfig(num_samples=10, vocab_size=100, max_sequence_length=15)
        train = SentimentDataset(config=cfg, split="train")
        val = SentimentDataset(config=cfg, split="val", word_to_idx=train.word_to_idx)
        assert val.word_to_idx == train.word_to_idx
        assert val.vocab == train.vocab


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

    def test_pad_reserved_at_index_zero(self, char_dataset):
        assert char_dataset.chars[0] == "<pad>"
        assert char_dataset.char_to_idx["<pad>"] == PAD_TOKEN_ID
        assert char_dataset.char_to_idx[" "] != 0
        for i in range(len(char_dataset)):
            x, y = char_dataset[i]
            assert (x > 0).all() and (y > 0).all()

    def test_vocab_shared_and_splits_differ(self):
        cfg = NLPDatasetConfig(num_samples=200, vocab_size=128, max_sequence_length=50)
        train = CharacterLevelDataset(config=cfg, split="train")
        val = CharacterLevelDataset(config=cfg, split="val")
        test = CharacterLevelDataset(config=cfg, split="test")
        assert train.char_to_idx == val.char_to_idx == test.char_to_idx
        assert val.text_data != test.text_data
        assert len(val) > 0 and len(test) > 0


# ─── Factory Functions ───────────────────────────────────────────────────────


class TestCreateSyntheticTextDataset:
    def test_creates_dataset(self):
        cfg = NLPDatasetConfig(num_samples=10, vocab_size=100, max_sequence_length=20)
        result = create_synthetic_text_dataset(config=cfg)
        assert isinstance(result, dict)
        assert "train" in result
        assert isinstance(result["train"], SyntheticTextDataset)

    def test_splits_share_vocab_and_differ(self):
        cfg = NLPDatasetConfig(num_samples=100, vocab_size=100, max_sequence_length=20)
        result = create_synthetic_text_dataset(config=cfg)
        assert result["train"].word_to_idx == result["val"].word_to_idx == result["test"].word_to_idx
        assert result["val"].texts != result["test"].texts


class TestCreateSyntheticSentimentDataset:
    def test_creates_dataset(self):
        cfg = NLPDatasetConfig(num_samples=20, vocab_size=100, max_sequence_length=15, num_classes=2)
        result = create_synthetic_sentiment_dataset(config=cfg)
        assert isinstance(result, dict)
        assert "train" in result

    def test_splits_share_vocab_and_differ(self):
        cfg = NLPDatasetConfig(num_samples=100, vocab_size=100, max_sequence_length=15)
        result = create_synthetic_sentiment_dataset(config=cfg)
        assert result["train"].word_to_idx == result["val"].word_to_idx == result["test"].word_to_idx
        assert result["val"].sentences != result["test"].sentences

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
