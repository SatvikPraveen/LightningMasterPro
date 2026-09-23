# File: src/lmpro/data/synth_nlp.py

"""
Synthetic NLP data generation for text classification and language modeling
"""

import random
import string
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

# Index 0 is reserved for padding in every vocabulary built here.
PAD_TOKEN = "<pad>"
PAD_TOKEN_ID = 0

# Seed offsets so that train / val / test never share a random stream.
_SPLIT_OFFSETS = {"train": 0, "val": 1, "test": 2}
_BASE_SEED = 42
_VOCAB_SEED = 1234  # split-independent, so every split shares the same vocabulary


def split_seed(split: str, base: int = _BASE_SEED) -> int:
    """Return a distinct, deterministic seed for a dataset split"""
    offset = _SPLIT_OFFSETS.get(split)
    if offset is None:
        offset = 3 + sum(ord(ch) for ch in split) % 997
    return base + offset


@dataclass
class NLPDatasetConfig:
    """Configuration for synthetic NLP datasets"""

    num_samples: int = 1000
    vocab_size: int = 5000
    max_sequence_length: int = 128
    min_sequence_length: int = 10
    num_classes: int = 3
    noise_level: float = 0.1
    save_path: Optional[str] = "data/synthetic/nlp"


POSITIVE_WORDS = [
    "great",
    "amazing",
    "wonderful",
    "excellent",
    "fantastic",
    "awesome",
    "brilliant",
    "outstanding",
    "superb",
    "magnificent",
    "perfect",
    "love",
    "best",
    "incredible",
    "marvelous",
    "exceptional",
    "remarkable",
]

NEGATIVE_WORDS = [
    "terrible",
    "awful",
    "horrible",
    "bad",
    "worst",
    "hate",
    "disgusting",
    "disappointing",
    "dreadful",
    "appalling",
    "pathetic",
    "useless",
    "annoying",
    "frustrating",
    "shocking",
    "disastrous",
    "catastrophic",
]

NEUTRAL_WORDS = [
    "okay",
    "average",
    "normal",
    "standard",
    "typical",
    "regular",
    "common",
    "ordinary",
    "usual",
    "general",
    "basic",
    "simple",
    "plain",
    "moderate",
]

FILLER_WORDS = [
    "the",
    "a",
    "an",
    "and",
    "or",
    "but",
    "in",
    "on",
    "at",
    "to",
    "for",
    "of",
    "with",
    "by",
    "from",
    "up",
    "about",
    "into",
    "over",
    "after",
    "is",
    "was",
    "are",
    "were",
    "be",
    "been",
    "have",
    "has",
    "had",
    "do",
    "does",
    "did",
    "will",
    "would",
    "could",
    "should",
    "may",
    "might",
    "this",
    "that",
    "these",
    "those",
    "i",
    "you",
    "he",
    "she",
    "it",
    "we",
    "they",
]


def build_text_vocabulary(vocab_size: int, seed: int = _VOCAB_SEED) -> List[str]:
    """
    Build the shared synthetic vocabulary for ``SyntheticTextDataset``.

    The random filler words are drawn from a private RNG seeded with ``seed`` so
    the vocabulary is identical for every split and independent of global state.
    """
    rng = random.Random(seed)
    base = [PAD_TOKEN, "<unk>", "<sos>", "<eos>"] + POSITIVE_WORDS + NEGATIVE_WORDS + NEUTRAL_WORDS + FILLER_WORDS

    additional_words: List[str] = []
    seen = set(base)
    while len(base) + len(additional_words) < vocab_size:
        word = "".join(rng.choices(string.ascii_lowercase, k=rng.randint(3, 8)))
        if word not in seen:
            seen.add(word)
            additional_words.append(word)

    return (base + additional_words)[:vocab_size]


class SyntheticTextDataset(Dataset):
    """Synthetic text classification dataset"""

    def __init__(
        self,
        config: NLPDatasetConfig,
        split: str = "train",
        tokenizer=None,
        vocab: Optional[List[str]] = None,
    ):
        self.config = config
        self.split = split
        self.tokenizer = tokenizer

        # Vocabulary is shared across splits (either passed in or built deterministically)
        self.vocab = list(vocab) if vocab is not None else build_text_vocabulary(config.vocab_size)
        self.word_to_idx = {word: idx for idx, word in enumerate(self.vocab)}
        self.idx_to_word = {idx: word for idx, word in enumerate(self.vocab)}

        # Generate data
        self.texts, self.labels = self._generate_text_data()

        # Tokenize texts
        self.tokenized_texts = [self._tokenize_text(text) for text in self.texts]

    def _generate_text_data(self) -> Tuple[List[str], List[int]]:
        """Generate synthetic text data"""
        texts = []
        labels = []

        seed = split_seed(self.split)
        random.seed(seed)
        np.random.seed(seed)

        vocab_set = set(self.vocab)
        positive_pool = [w for w in POSITIVE_WORDS if w in vocab_set]
        negative_pool = [w for w in NEGATIVE_WORDS if w in vocab_set]
        neutral_pool = [w for w in NEUTRAL_WORDS if w in vocab_set]
        filler_pool = [w for w in FILLER_WORDS if w in vocab_set] or ["<unk>"]

        for _ in range(self.config.num_samples):
            label = random.randint(0, self.config.num_classes - 1)
            text = self._generate_text_for_class(label, positive_pool, negative_pool, neutral_pool, filler_pool)

            texts.append(text)
            labels.append(label)

        return texts, labels

    def _generate_text_for_class(
        self, class_id: int, pos_pool: List[str], neg_pool: List[str], neut_pool: List[str], filler_pool: List[str]
    ) -> str:
        """Generate text for a specific class"""
        word_count_max = max(self.config.min_sequence_length, self.config.max_sequence_length // 4)
        length = random.randint(self.config.min_sequence_length, word_count_max)

        words = []

        # Add class-specific words with higher probability
        class_word_prob = 0.3

        for _ in range(length):
            if random.random() < class_word_prob:
                if class_id == 0 and pos_pool:  # Positive
                    words.append(random.choice(pos_pool))
                elif class_id == 1 and neg_pool:  # Negative
                    words.append(random.choice(neg_pool))
                elif class_id == 2 and neut_pool:  # Neutral
                    words.append(random.choice(neut_pool))
                else:
                    words.append(random.choice(filler_pool))
            else:
                # Add filler words
                words.append(random.choice(filler_pool))

        return " ".join(words)

    def _tokenize_text(self, text: str) -> List[int]:
        """Convert text to token indices"""
        if self.tokenizer:
            return self.tokenizer.encode(text)

        words = text.lower().split()
        unk = self.word_to_idx["<unk>"]
        tokens = [self.word_to_idx.get(word, unk) for word in words]

        # Pad or truncate
        if len(tokens) < self.config.max_sequence_length:
            tokens.extend([PAD_TOKEN_ID] * (self.config.max_sequence_length - len(tokens)))
        else:
            tokens = tokens[: self.config.max_sequence_length]

        return tokens

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        tokens = torch.tensor(self.tokenized_texts[idx], dtype=torch.long)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return tokens, label


class CharacterLevelDataset(Dataset):
    """
    Character-level language modeling dataset

    The character vocabulary is fixed (derived from the text patterns) and shared
    by every split; index 0 is reserved for ``<pad>`` so real characters never
    collide with the padding / ignore index used by the language model.
    """

    PATTERNS = [
        "abcdefghijklmnopqrstuvwxyz" * 10,
        "0123456789" * 20,
        "hello world " * 50,
        "the quick brown fox jumps over the lazy dog " * 25,
        "artificial intelligence machine learning deep learning " * 20,
    ]

    def __init__(self, config: NLPDatasetConfig, split: str = "train", sequence_length: int = 100):
        self.config = config
        self.split = split
        self.sequence_length = sequence_length

        # Build (shared) character vocabulary with pad at index 0
        self.chars = self.build_char_vocab()
        self.char_to_idx = {ch: idx for idx, ch in enumerate(self.chars)}
        self.idx_to_char = {idx: ch for idx, ch in enumerate(self.chars)}

        # Generate character data
        self.text_data = self._generate_character_data()

        # Create sequences
        self.sequences = self._create_sequences()

    @classmethod
    def build_char_vocab(cls) -> List[str]:
        """Characters that can appear in any split, with ``<pad>`` at index 0"""
        charset = set()
        for pattern in cls.PATTERNS:
            charset.update(pattern)
            charset.update(pattern.upper())
        charset.add(" ")
        return [PAD_TOKEN] + sorted(charset)

    def _generate_character_data(self) -> str:
        """Generate character-level text data"""
        text = ""
        random.seed(split_seed(self.split))

        for _ in range(max(1, self.config.num_samples // 100)):
            pattern = random.choice(self.PATTERNS)
            # Add some variation
            if random.random() > 0.5:
                pattern = pattern.upper()
            text += pattern + " "

        return text[: self.config.num_samples * 10]  # Ensure sufficient length

    def _create_sequences(self) -> List[Tuple[List[int], List[int]]]:
        """Create input-target sequence pairs"""
        sequences = []

        step = max(1, self.sequence_length // 2)
        for i in range(0, len(self.text_data) - self.sequence_length - 1, step):
            input_seq = self.text_data[i : i + self.sequence_length]
            target_seq = self.text_data[i + 1 : i + self.sequence_length + 1]

            input_indices = [self.char_to_idx[ch] for ch in input_seq]
            target_indices = [self.char_to_idx[ch] for ch in target_seq]

            sequences.append((input_indices, target_indices))

        return sequences[: self.config.num_samples]

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        input_seq, target_seq = self.sequences[idx]
        return torch.tensor(input_seq, dtype=torch.long), torch.tensor(target_seq, dtype=torch.long)


class SentimentDataset(Dataset):
    """
    Synthetic sentiment analysis dataset

    The vocabulary is built from the fixed templates, so it is identical for every
    split; a ``word_to_idx`` mapping can also be passed explicitly to share it.
    """

    POSITIVE_TEMPLATES = [
        "I love this {}",
        "This {} is amazing",
        "Great {} experience",
        "Wonderful {} quality",
        "Excellent {} service",
        "Best {} ever",
    ]

    NEGATIVE_TEMPLATES = [
        "I hate this {}",
        "This {} is terrible",
        "Awful {} experience",
        "Poor {} quality",
        "Worst {} service",
        "Bad {} overall",
    ]

    NEUTRAL_TEMPLATES = [
        "This {} is okay",
        "Average {} quality",
        "Normal {} experience",
        "Standard {} service",
        "Regular {} item",
        "Typical {} product",
    ]

    OBJECTS = ["product", "service", "item", "experience", "quality", "food", "movie", "book", "place", "thing"]

    MAX_LEN = 32  # Shorter for sentiment analysis

    def __init__(
        self,
        config: NLPDatasetConfig,
        split: str = "train",
        word_to_idx: Optional[Dict[str, int]] = None,
    ):
        self.config = config
        self.split = split

        # Build / adopt shared vocabulary
        if word_to_idx is not None:
            self.word_to_idx = dict(word_to_idx)
            self.vocab = [w for w, _ in sorted(self.word_to_idx.items(), key=lambda kv: kv[1])]
        else:
            self.vocab, self.word_to_idx = self.build_sentiment_vocab()

        # Generate sentiment data
        self.sentences, self.sentiments = self._generate_sentiment_data()

        # Tokenize
        self.tokenized_sentences = [self._tokenize_sentence(sent) for sent in self.sentences]

    @classmethod
    def build_sentiment_vocab(cls) -> Tuple[List[str], Dict[str, int]]:
        """Vocabulary covering every word any template can produce"""
        all_words = set()
        for template in cls.POSITIVE_TEMPLATES + cls.NEGATIVE_TEMPLATES + cls.NEUTRAL_TEMPLATES:
            for obj in cls.OBJECTS:
                all_words.update(template.format(obj).lower().split())

        vocab = [PAD_TOKEN, "<unk>"] + sorted(all_words)
        word_to_idx = {word: idx for idx, word in enumerate(vocab)}
        return vocab, word_to_idx

    def _generate_sentiment_data(self) -> Tuple[List[str], List[int]]:
        """Generate synthetic sentiment data"""
        sentences = []
        sentiments = []

        random.seed(split_seed(self.split))

        for _ in range(self.config.num_samples):
            sentiment = random.randint(0, 2)  # 0: negative, 1: neutral, 2: positive
            obj = random.choice(self.OBJECTS)

            if sentiment == 0:
                template = random.choice(self.NEGATIVE_TEMPLATES)
            elif sentiment == 1:
                template = random.choice(self.NEUTRAL_TEMPLATES)
            else:
                template = random.choice(self.POSITIVE_TEMPLATES)

            sentences.append(template.format(obj))
            sentiments.append(sentiment)

        return sentences, sentiments

    def _tokenize_sentence(self, sentence: str) -> List[int]:
        """Tokenize sentence to indices"""
        words = sentence.lower().split()
        unk = self.word_to_idx["<unk>"]
        tokens = [self.word_to_idx.get(word, unk) for word in words]

        # Pad or truncate
        if len(tokens) < self.MAX_LEN:
            tokens.extend([PAD_TOKEN_ID] * (self.MAX_LEN - len(tokens)))
        else:
            tokens = tokens[: self.MAX_LEN]

        return tokens

    def __len__(self) -> int:
        return len(self.sentences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        tokens = torch.tensor(self.tokenized_sentences[idx], dtype=torch.long)
        sentiment = torch.tensor(self.sentiments[idx], dtype=torch.long)
        return tokens, sentiment


def _split_config(config: NLPDatasetConfig, num_samples: int, **overrides) -> NLPDatasetConfig:
    values = dict(
        num_samples=num_samples,
        vocab_size=config.vocab_size,
        max_sequence_length=config.max_sequence_length,
        min_sequence_length=config.min_sequence_length,
        num_classes=config.num_classes,
        noise_level=config.noise_level,
        save_path=config.save_path,
    )
    values.update(overrides)
    return NLPDatasetConfig(**values)


def create_synthetic_text_dataset(
    config: NLPDatasetConfig,
    splits: List[str] = ["train", "val", "test"],
    split_ratios: List[float] = [0.7, 0.15, 0.15],
) -> dict:
    """Create synthetic text classification datasets sharing one vocabulary"""
    datasets = {}

    total_samples = config.num_samples
    split_sizes = [int(ratio * total_samples) for ratio in split_ratios]

    vocab = build_text_vocabulary(config.vocab_size)
    for split, size in zip(splits, split_sizes):
        datasets[split] = SyntheticTextDataset(_split_config(config, size), split=split, vocab=vocab)

    return datasets


def create_synthetic_sentiment_dataset(
    config: NLPDatasetConfig,
    splits: List[str] = ["train", "val", "test"],
    split_ratios: List[float] = [0.7, 0.15, 0.15],
) -> dict:
    """Create synthetic sentiment analysis datasets sharing one vocabulary"""
    datasets = {}

    total_samples = config.num_samples
    split_sizes = [int(ratio * total_samples) for ratio in split_ratios]

    _, word_to_idx = SentimentDataset.build_sentiment_vocab()
    for split, size in zip(splits, split_sizes):
        # negative, neutral, positive
        split_config = _split_config(config, size, num_classes=3)
        datasets[split] = SentimentDataset(split_config, split=split, word_to_idx=word_to_idx)

    return datasets


def create_character_level_dataset(
    config: NLPDatasetConfig,
    sequence_length: int = 100,
    splits: List[str] = ["train", "val", "test"],
    split_ratios: List[float] = [0.7, 0.15, 0.15],
) -> dict:
    """Create character-level language modeling datasets"""
    datasets = {}

    total_samples = config.num_samples
    split_sizes = [int(ratio * total_samples) for ratio in split_ratios]

    for split, size in zip(splits, split_sizes):
        split_config = _split_config(config, size, max_sequence_length=sequence_length)
        datasets[split] = CharacterLevelDataset(split_config, split=split, sequence_length=sequence_length)

    return datasets


def print_dataset_stats(dataset: Dataset, name: str = "Dataset") -> None:
    """Print statistics about the dataset"""
    print(f"\n{name} Statistics:")
    print(f"Size: {len(dataset)}")

    if hasattr(dataset, "vocab"):
        print(f"Vocabulary size: {len(dataset.vocab)}")

    if hasattr(dataset, "labels"):
        label_counts = Counter(dataset.labels)
        print(f"Label distribution: {dict(label_counts)}")

    # Sample a few examples
    print("\nSample examples:")
    for i in range(min(3, len(dataset))):
        data, target = dataset[i]
        if hasattr(dataset, "idx_to_word") and isinstance(data, torch.Tensor):
            text = " ".join(
                [dataset.idx_to_word.get(idx.item(), "<unk>") for idx in data if idx.item() != PAD_TOKEN_ID]
            )
            print(f"  Example {i}: {text[:100]}... -> {target}")
        else:
            print(f"  Example {i}: {str(data)[:100]}... -> {target}")
