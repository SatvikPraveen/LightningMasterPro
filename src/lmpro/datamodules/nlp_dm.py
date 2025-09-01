# File: src/lmpro/datamodules/nlp_dm.py

"""
NLP DataModule for text classification and language modeling tasks
"""

import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from lightning import LightningDataModule
from typing import Optional, Dict, Any, List, Tuple
import numpy as np

from ..data.synth_nlp import (
    NLPDatasetConfig,
    SyntheticTextDataset,
    CharacterLevelDataset,
    SentimentDataset,
    create_synthetic_text_dataset,
    create_synthetic_sentiment_dataset,
    create_character_level_dataset
)
from ..utils.seed import worker_init_fn


class NLPDataModule(LightningDataModule):
    """
    Lightning DataModule for NLP tasks (classification, language modeling)
    """
    
    def __init__(
        self,
        task: str = "classification",
        data_config: Optional[NLPDatasetConfig] = None,
        batch_size: int = 32,
        num_workers: int = 4,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        max_length: Optional[int] = None,
        tokenizer: Optional[Any] = None,
        split_ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15),
        **kwargs
    ):
        super().__init__()
        
        # Save hyperparameters
        self.save_hyperparameters(logger=False)
        
        self.task = task
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.split_ratios = split_ratios
        self.tokenizer = tokenizer
        
        # Data configuration
        self.data_config = data_config or NLPDatasetConfig()
        if max_length is not None:
            self.data_config.max_sequence_length = max_length
        
        # Datasets
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        
        # Vocabulary info
        self.vocab_size = self.data_config.vocab_size
        self.pad_token_id = 0
        self.vocab = None
        self.word_to_idx = None
        self.idx_to_word = None
        
    def prepare_data(self) -> None:
        """Download and prepare data (called once per node)"""
        # Generate synthetic data based on task
        if self.task == "classification":
            self.datasets = create_synthetic_text_dataset(
                self.data_config,
                splits=["train", "val", "test"],
                split_ratios=self.split_ratios
            )
        elif self.task == "sentiment":
            self.datasets = create_synthetic_sentiment_dataset(
                self.data_config,
                splits=["train", "val", "test"],
                split_ratios=self.split_ratios
            )
        elif self.task == "language_modeling":
            self.datasets = create_character_level_dataset(
                self.data_config,
                sequence_length=self.data_config.max_sequence_length,
                splits=["train", "val", "test"],
                split_ratios=self.split_ratios
            )
        else:
            raise ValueError(f"Unknown task: {self.task}")
    
    def setup(self, stage: Optional[str] = None) -> None:
        """Setup datasets for each stage"""
        if not hasattr(self, 'datasets'):
            self.prepare_data()
        
        if stage == "fit" or stage is None:
            self.train_dataset = self.datasets["train"]
            self.val_dataset = self.datasets["val"]
            
            # Get vocabulary info from training dataset
            if hasattr(self.train_dataset, 'vocab'):
                self.vocab = self.train_dataset.vocab
                self.vocab_size = len(self.vocab)
            if hasattr(self.train_dataset, 'word_to_idx'):
                self.word_to_idx = self.train_dataset.word_to_idx
            if hasattr(self.train_dataset, 'idx_to_word'):
                self.idx_to_word = self.train_dataset.idx_to_word
            if hasattr(self.train_dataset, 'char_to_idx'):
                self.word_to_idx = self.train_dataset.char_to_idx
                self.idx_to_word = self.train_dataset.idx_to_char
                self.vocab_size = len(self.train_dataset.chars)
        
        if stage == "test" or stage is None:
            self.test_dataset = self.datasets["test"]
        
        if stage == "predict" or stage is None:
            if not hasattr(self, 'test_dataset') or self.test_dataset is None:
                self.test_dataset = self.datasets["test"]
    
    def collate_fn(self, batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Custom collate function for variable length sequences"""
        sequences, targets = zip(*batch)
        
        if self.task == "language_modeling":
            # For language modeling, sequences are already the same length
            sequences = torch.stack(sequences)
            targets = torch.stack(targets)
        else:
            # For classification tasks, pad sequences
            sequences = pad_sequence(sequences, batch_first=True, padding_value=self.pad_token_id)
            targets = torch.stack(targets)
        
        return sequences, targets
    
    def train_dataloader(self) -> DataLoader:
        """Create training dataloader"""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            collate_fn=self.collate_fn,
            worker_init_fn=worker_init_fn,
            drop_last=True,
        )
    
    def val_dataloader(self) -> DataLoader:
        """Create validation dataloader"""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            collate_fn=self.collate_fn,
            worker_init_fn=worker_init_fn,
        )
    
    def test_dataloader(self) -> DataLoader:
        """Create test dataloader"""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            collate_fn=self.collate_fn,
            worker_init_fn=worker_init_fn,
        )
    
    def predict_dataloader(self) -> DataLoader:
        """Create prediction dataloader"""
        return self.test_dataloader()
    
    def decode_sequence(self, sequence: torch.Tensor) -> str:
        """Decode a sequence of token IDs back to text"""
        if self.idx_to_word is None:
            return str(sequence.tolist())
        
        tokens = []
        for token_id in sequence:
            if token_id.item() == self.pad_token_id:
                break
            token = self.idx_to_word.get(token_id.item(), "<unk>")
            tokens.append(token)
        
        if self.task == "language_modeling":
            return "".join(tokens)  # Character-level
        else:
            return " ".join(tokens)  # Word-level
    
    def encode_text(self, text: str) -> torch.Tensor:
        """Encode text to token IDs"""
        if self.word_to_idx is None:
            raise ValueError("Vocabulary not initialized. Run setup() first.")
        
        if self.task == "language_modeling":
            # Character-level encoding
            tokens = [self.word_to_idx.get(char, self.word_to_idx.get("<unk>", 1)) for char in text]
        else:
            # Word-level encoding
            words = text.lower().split()
            tokens = [self.word_to_idx.get(word, self.word_to_idx.get("<unk>", 1)) for word in words]
        
        # Pad or truncate
        max_len = self.data_config.max_sequence_length
        if len(tokens) < max_len:
            tokens.extend([self.pad_token_id] * (max_len - len(tokens)))
        else:
            tokens = tokens[:max_len]
        
        return torch.tensor(tokens, dtype=torch.long)
    
    def get_class_names(self) -> List[str]:
        """Get class names for classification tasks"""
        if self.task == "classification":
            return [f"class_{i}" for i in range(self.data_config.num_classes)]
        elif self.task == "sentiment":
            return ["negative", "neutral", "positive"]
        else:
            return []
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """Get information about the dataset"""
        return {
            "task": self.task,
            "vocab_size": self.vocab_size,
            "max_sequence_length": self.data_config.max_sequence_length,
            "num_classes": self.data_config.num_classes if self.task != "language_modeling" else None,
            "train_size": len(self.train_dataset) if self.train_dataset else 0,
            "val_size": len(self.val_dataset) if self.val_dataset else 0,
            "test_size": len(self.test_dataset) if self.test_dataset else 0,
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
        }
    
    def visualize_batch(self, stage: str = "train", num_samples: int = 4) -> None:
        """Visualize a batch of data"""
        if stage == "train":
            dataloader = self.train_dataloader()
        elif stage == "val":
            dataloader = self.val_dataloader()
        else:
            dataloader = self.test_dataloader()
        
        batch = next(iter(dataloader))
        sequences, targets = batch
        
        print(f"\n{stage.upper()} Batch Visualization:")
        print(f"Batch shape: {sequences.shape}")
        print(f"Targets shape: {targets.shape}")
        
        for i in range(min(num_samples, len(sequences))):
            decoded_text = self.decode_sequence(sequences[i])
            
            if self.task == "language_modeling":
                target_text = self.decode_sequence(targets[i])
                print(f"\nSample {i}:")
                print(f"  Input:  {decoded_text[:100]}...")
                print(f"  Target: {target_text[:100]}...")
            else:
                class_names = self.get_class_names()
                target_name = class_names[targets[i].item()] if class_names else targets[i].item()
                print(f"\nSample {i}:")
                print(f"  Text: {decoded_text[:150]}...")
                print(f"  Label: {target_name}")
    
    def compute_class_weights(self) -> torch.Tensor:
        """Compute class weights for imbalanced classification datasets"""
        if self.task == "language_modeling" or self.train_dataset is None:
            return torch.ones(self.data_config.num_classes)
        
        # Count class frequencies
        class_counts = torch.zeros(self.data_config.num_classes)
        
        for _, target in self.train_dataset:
            if isinstance(target, torch.Tensor):
                class_counts[target.item()] += 1
        
        # Compute inverse frequency weights
        total_samples = class_counts.sum()
        class_weights = total_samples / (self.data_config.num_classes * class_counts + 1e-8)
        
        return class_weights
    
    def get_vocab_stats(self) -> Dict[str, Any]:
        """Get vocabulary statistics"""
        if self.vocab is None:
            return {}
        
        stats = {
            "vocab_size": len(self.vocab),
            "most_common_tokens": self.vocab[:20] if len(self.vocab) > 20 else self.vocab,
        }
        
        if hasattr(self.train_dataset, 'texts'):
            # Analyze text lengths
            lengths = [len(text.split()) for text in self.train_dataset.texts]
            stats.update({
                "avg_text_length": np.mean(lengths),
                "min_text_length": np.min(lengths),
                "max_text_length": np.max(lengths),
                "std_text_length": np.std(lengths),
            })
        
        return stats
    
    def create_attention_mask(self, sequences: torch.Tensor) -> torch.Tensor:
        """Create attention mask for padded sequences"""
        return (sequences != self.pad_token_id).long()
    
    def __repr__(self) -> str:
        """String representation"""
        info = self.get_dataset_info()
        return (
            f"NLPDataModule(\n"
            f"  task={info['task']},\n"
            f"  vocab_size={info['vocab_size']},\n"
            f"  max_length={info['max_sequence_length']},\n"
            f"  num_classes={info['num_classes']},\n"
            f"  train_size={info['train_size']},\n"
            f"  val_size={info['val_size']},\n"
            f"  test_size={info['test_size']},\n"
            f"  batch_size={info['batch_size']}\n"
            f")"
        )


# Convenience functions for common NLP tasks
def get_text_classification_datamodule(
    num_classes: int = 3,
    vocab_size: int = 5000,
    max_length: int = 128,
    batch_size: int = 32,
    **kwargs
) -> NLPDataModule:
    """Get an NLP datamodule for text classification"""
    config = NLPDatasetConfig(
        num_classes=num_classes,
        vocab_size=vocab_size,
        max_sequence_length=max_length,
        **kwargs
    )
    
    return NLPDataModule(
        task="classification",
        data_config=config,
        batch_size=batch_size
    )


def get_sentiment_analysis_datamodule(
    vocab_size: int = 3000,
    max_length: int = 64,
    batch_size: int = 32,
    **kwargs
) -> NLPDataModule:
    """Get an NLP datamodule for sentiment analysis"""
    config = NLPDatasetConfig(
        num_classes=3,  # negative, neutral, positive
        vocab_size=vocab_size,
        max_sequence_length=max_length,
        **kwargs
    )
    
    return NLPDataModule(
        task="sentiment",
        data_config=config,
        batch_size=batch_size
    )


def get_language_modeling_datamodule(
    sequence_length: int = 100,
    batch_size: int = 32,
    **kwargs
) -> NLPDataModule:
    """Get an NLP datamodule for character-level language modeling"""
    config = NLPDatasetConfig(
        max_sequence_length=sequence_length,
        **kwargs
    )
    
    return NLPDataModule(
        task="language_modeling",
        data_config=config,
        batch_size=batch_size
    )