# File: src/lmpro/data/synth_nlp.py

"""
Synthetic NLP data generation for text classification and language modeling
"""

import torch
import numpy as np
import random
import string
from typing import List, Tuple, Dict, Optional, Union
from dataclasses import dataclass
from torch.utils.data import Dataset
from collections import Counter, defaultdict


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


class SyntheticTextDataset(Dataset):
    """Synthetic text classification dataset"""
    
    def __init__(
        self,
        config: NLPDatasetConfig,
        split: str = "train",
        tokenizer=None
    ):
        self.config = config
        self.split = split
        self.tokenizer = tokenizer
        
        # Build vocabulary
        self.vocab, self.word_to_idx, self.idx_to_word = self._build_vocabulary()
        
        # Generate data
        self.texts, self.labels = self._generate_text_data()
        
        # Tokenize texts
        self.tokenized_texts = [self._tokenize_text(text) for text in self.texts]
    
    def _build_vocabulary(self) -> Tuple[List[str], Dict[str, int], Dict[int, str]]:
        """Build a synthetic vocabulary"""
        # Common words for different classes
        positive_words = [
            "great", "amazing", "wonderful", "excellent", "fantastic", "awesome", 
            "brilliant", "outstanding", "superb", "magnificent", "perfect", "love",
            "best", "incredible", "marvelous", "exceptional", "remarkable"
        ]
        
        negative_words = [
            "terrible", "awful", "horrible", "bad", "worst", "hate", "disgusting",
            "disappointing", "dreadful", "appalling", "pathetic", "useless", 
            "annoying", "frustrating", "shocking", "disastrous", "catastrophic"
        ]
        
        neutral_words = [
            "okay", "average", "normal", "standard", "typical", "regular", "common",
            "ordinary", "usual", "general", "basic", "simple", "plain", "moderate"
        ]
        
        # Common filler words
        filler_words = [
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
            "of", "with", "by", "from", "up", "about", "into", "over", "after",
            "is", "was", "are", "were", "be", "been", "have", "has", "had", "do",
            "does", "did", "will", "would", "could", "should", "may", "might",
            "this", "that", "these", "those", "i", "you", "he", "she", "it", "we", "they"
        ]
        
        # Additional random words
        additional_words = []
        for _ in range(self.config.vocab_size - len(positive_words + negative_words + neutral_words + filler_words)):
            word = ''.join(random.choices(string.ascii_lowercase, k=random.randint(3, 8)))
            additional_words.append(word)
        
        vocab = ["<pad>", "<unk>", "<sos>", "<eos>"] + positive_words + negative_words + neutral_words + filler_words + additional_words
        vocab = vocab[:self.config.vocab_size]
        
        word_to_idx = {word: idx for idx, word in enumerate(vocab)}
        idx_to_word = {idx: word for idx, word in enumerate(vocab)}
        
        return vocab, word_to_idx, idx_to_word
    
    def _generate_text_data(self) -> Tuple[List[str], List[int]]:
        """Generate synthetic text data"""
        texts = []
        labels = []
        
        random.seed(42 if self.split == "train" else 24)
        np.random.seed(42 if self.split == "train" else 24)
        
        # Define word pools for each class
        positive_pool = [w for w in self.vocab if w in ["great", "amazing", "wonderful", "excellent", "fantastic", "awesome", "brilliant", "outstanding", "superb", "magnificent", "perfect", "love", "best", "incredible", "marvelous", "exceptional", "remarkable"]]
        negative_pool = [w for w in self.vocab if w in ["terrible", "awful", "horrible", "bad", "worst", "hate", "disgusting", "disappointing", "dreadful", "appalling", "pathetic", "useless", "annoying", "frustrating", "shocking", "disastrous", "catastrophic"]]
        neutral_pool = [w for w in self.vocab if w in ["okay", "average", "normal", "standard", "typical", "regular", "common", "ordinary", "usual", "general", "basic", "simple", "plain", "moderate"]]
        filler_pool = [w for w in self.vocab if w in ["the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by", "from", "up", "about", "into", "over", "after", "is", "was", "are", "were", "be", "been", "have", "has", "had", "do", "does", "did", "will", "would", "could", "should", "may", "might", "this", "that", "these", "those", "i", "you", "he", "she", "it", "we", "they"]]
        
        for _ in range(self.config.num_samples):
            label = random.randint(0, self.config.num_classes - 1)
            text = self._generate_text_for_class(label, positive_pool, negative_pool, neutral_pool, filler_pool)
            
            texts.append(text)
            labels.append(label)
        
        return texts, labels
    
    def _generate_text_for_class(self, class_id: int, pos_pool: List[str], neg_pool: List[str], 
                                neut_pool: List[str], filler_pool: List[str]) -> str:
        """Generate text for a specific class"""
        length = random.randint(self.config.min_sequence_length, self.config.max_sequence_length // 4)
        
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
        tokens = []
        
        for word in words:
            if word in self.word_to_idx:
                tokens.append(self.word_to_idx[word])
            else:
                tokens.append(self.word_to_idx["<unk>"])
        
        # Pad or truncate
        if len(tokens) < self.config.max_sequence_length:
            tokens.extend([self.word_to_idx["<pad>"]] * (self.config.max_sequence_length - len(tokens)))
        else:
            tokens = tokens[:self.config.max_sequence_length]
        
        return tokens
    
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        tokens = torch.tensor(self.tokenized_texts[idx], dtype=torch.long)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return tokens, label


class CharacterLevelDataset(Dataset):
    """Character-level language modeling dataset"""
    
    def __init__(
        self,
        config: NLPDatasetConfig,
        split: str = "train",
        sequence_length: int = 100
    ):
        self.config = config
        self.split = split
        self.sequence_length = sequence_length
        
        # Generate character data
        self.text_data = self._generate_character_data()
        
        # Build character vocabulary
        self.chars = sorted(list(set(self.text_data)))
        self.char_to_idx = {ch: idx for idx, ch in enumerate(self.chars)}
        self.idx_to_char = {idx: ch for idx, ch in enumerate(self.chars)}
        
        # Create sequences
        self.sequences = self._create_sequences()
    
    def _generate_character_data(self) -> str:
        """Generate character-level text data"""
        # Simple text patterns
        patterns = [
            "abcdefghijklmnopqrstuvwxyz" * 10,
            "0123456789" * 20,
            "hello world " * 50,
            "the quick brown fox jumps over the lazy dog " * 25,
            "artificial intelligence machine learning deep learning " * 20,
        ]
        
        # Mix patterns
        text = ""
        random.seed(42 if self.split == "train" else 24)
        
        for _ in range(self.config.num_samples // 100):
            pattern = random.choice(patterns)
            # Add some variation
            if random.random() > 0.5:
                pattern = pattern.upper()
            text += pattern + " "
        
        return text[:self.config.num_samples * 10]  # Ensure sufficient length
    
    def _create_sequences(self) -> List[Tuple[List[int], List[int]]]:
        """Create input-target sequence pairs"""
        sequences = []
        
        for i in range(0, len(self.text_data) - self.sequence_length - 1, self.sequence_length // 2):
            input_seq = self.text_data[i:i + self.sequence_length]
            target_seq = self.text_data[i + 1:i + self.sequence_length + 1]
            
            input_indices = [self.char_to_idx[ch] for ch in input_seq]
            target_indices = [self.char_to_idx[ch] for ch in target_seq]
            
            sequences.append((input_indices, target_indices))
        
        return sequences[:self.config.num_samples]
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        input_seq, target_seq = self.sequences[idx]
        return torch.tensor(input_seq, dtype=torch.long), torch.tensor(target_seq, dtype=torch.long)


class SentimentDataset(Dataset):
    """Synthetic sentiment analysis dataset"""
    
    def __init__(
        self,
        config: NLPDatasetConfig,
        split: str = "train"
    ):
        self.config = config
        self.split = split
        
        # Generate sentiment data
        self.sentences, self.sentiments = self._generate_sentiment_data()
        
        # Build vocabulary
        self.vocab, self.word_to_idx = self._build_sentiment_vocab()
        
        # Tokenize
        self.tokenized_sentences = [self._tokenize_sentence(sent) for sent in self.sentences]
    
    def _generate_sentiment_data(self) -> Tuple[List[str], List[int]]:
        """Generate synthetic sentiment data"""
        positive_templates = [
            "I love this {}",
            "This {} is amazing",
            "Great {} experience",
            "Wonderful {} quality",
            "Excellent {} service",
            "Best {} ever",
        ]
        
        negative_templates = [
            "I hate this {}",
            "This {} is terrible",
            "Awful {} experience", 
            "Poor {} quality",
            "Worst {} service",
            "Bad {} overall",
        ]
        
        neutral_templates = [
            "This {} is okay",
            "Average {} quality",
            "Normal {} experience",
            "Standard {} service",
            "Regular {} item",
            "Typical {} product",
        ]
        
        objects = ["product", "service", "item", "experience", "quality", "food", "movie", "book", "place", "thing"]
        
        sentences = []
        sentiments = []
        
        random.seed(42 if self.split == "train" else 24)
        
        for _ in range(self.config.num_samples):
            sentiment = random.randint(0, 2)  # 0: negative, 1: neutral, 2: positive
            obj = random.choice(objects)
            
            if sentiment == 0:
                template = random.choice(negative_templates)
            elif sentiment == 1:
                template = random.choice(neutral_templates)
            else:
                template = random.choice(positive_templates)
            
            sentence = template.format(obj)
            sentences.append(sentence)
            sentiments.append(sentiment)
        
        return sentences, sentiments
    
    def _build_sentiment_vocab(self) -> Tuple[List[str], Dict[str, int]]:
        """Build vocabulary from sentences"""
        all_words = set()
        for sentence in self.sentences:
            words = sentence.lower().split()
            all_words.update(words)
        
        vocab = ["<pad>", "<unk>"] + sorted(list(all_words))
        word_to_idx = {word: idx for idx, word in enumerate(vocab)}
        
        return vocab, word_to_idx
    
    def _tokenize_sentence(self, sentence: str) -> List[int]:
        """Tokenize sentence to indices"""
        words = sentence.lower().split()
        tokens = []
        
        for word in words:
            if word in self.word_to_idx:
                tokens.append(self.word_to_idx[word])
            else:
                tokens.append(self.word_to_idx["<unk>"])
        
        # Pad or truncate
        max_len = 32  # Shorter for sentiment analysis
        if len(tokens) < max_len:
            tokens.extend([self.word_to_idx["<pad>"]] * (max_len - len(tokens)))
        else:
            tokens = tokens[:max_len]
        
        return tokens
    
    def __len__(self) -> int:
        return len(self.sentences)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        tokens = torch.tensor(self.tokenized_sentences[idx], dtype=torch.long)
        sentiment = torch.tensor(self.sentiments[idx], dtype=torch.long)
        return tokens, sentiment


def create_synthetic_text_dataset(
    config: NLPDatasetConfig,
    splits: List[str] = ["train", "val", "test"],
    split_ratios: List[float] = [0.7, 0.15, 0.15]
) -> dict:
    """Create synthetic text classification datasets"""
    datasets = {}
    
    total_samples = config.num_samples
    split_sizes = [int(ratio * total_samples) for ratio in split_ratios]
    
    for split, size in zip(splits, split_sizes):
        split_config = NLPDatasetConfig(
            num_samples=size,
            vocab_size=config.vocab_size,
            max_sequence_length=config.max_sequence_length,
            min_sequence_length=config.min_sequence_length,
            num_classes=config.num_classes,
            noise_level=config.noise_level,
            save_path=config.save_path
        )
        
        datasets[split] = SyntheticTextDataset(split_config, split=split)
    
    return datasets


def create_synthetic_sentiment_dataset(
    config: NLPDatasetConfig,
    splits: List[str] = ["train", "val", "test"],
    split_ratios: List[float] = [0.7, 0.15, 0.15]
) -> dict:
    """Create synthetic sentiment analysis datasets"""
    datasets = {}
    
    total_samples = config.num_samples
    split_sizes = [int(ratio * total_samples) for ratio in split_ratios]
    
    for split, size in zip(splits, split_sizes):
        split_config = NLPDatasetConfig(
            num_samples=size,
            vocab_size=config.vocab_size,
            max_sequence_length=config.max_sequence_length,
            min_sequence_length=config.min_sequence_length,
            num_classes=3,  # negative, neutral, positive
            noise_level=config.noise_level,
            save_path=config.save_path
        )
        
        datasets[split] = SentimentDataset(split_config, split=split)
    
    return datasets


def create_character_level_dataset(
    config: NLPDatasetConfig,
    sequence_length: int = 100,
    splits: List[str] = ["train", "val", "test"],
    split_ratios: List[float] = [0.7, 0.15, 0.15]
) -> dict:
    """Create character-level language modeling datasets"""
    datasets = {}
    
    total_samples = config.num_samples
    split_sizes = [int(ratio * total_samples) for ratio in split_ratios]
    
    for split, size in zip(splits, split_sizes):
        split_config = NLPDatasetConfig(
            num_samples=size,
            vocab_size=config.vocab_size,
            max_sequence_length=sequence_length,
            min_sequence_length=config.min_sequence_length,
            num_classes=config.num_classes,
            noise_level=config.noise_level,
            save_path=config.save_path
        )
        
        datasets[split] = CharacterLevelDataset(split_config, split=split, sequence_length=sequence_length)
    
    return datasets


def print_dataset_stats(dataset: Dataset, name: str = "Dataset") -> None:
    """Print statistics about the dataset"""
    print(f"\n{name} Statistics:")
    print(f"Size: {len(dataset)}")
    
    if hasattr(dataset, 'vocab'):
        print(f"Vocabulary size: {len(dataset.vocab)}")
    
    if hasattr(dataset, 'labels'):
        label_counts = Counter(dataset.labels)
        print(f"Label distribution: {dict(label_counts)}")
    
    # Sample a few examples
    print(f"\nSample examples:")
    for i in range(min(3, len(dataset))):
        data, target = dataset[i]
        if hasattr(dataset, 'idx_to_word') and isinstance(data, torch.Tensor):
            text = " ".join([dataset.idx_to_word.get(idx.item(), "<unk>") for idx in data if idx.item() != 0])
            print(f"  Example {i}: {text[:100]}... -> {target}")
        else:
            print(f"  Example {i}: {str(data)[:100]}... -> {target}")