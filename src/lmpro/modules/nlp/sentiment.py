# File: src/lmpro/modules/nlp/sentiment.py

"""
Sentiment classification module for text sentiment analysis
"""

from typing import Dict, List, Literal, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning.pytorch import LightningModule
from lightning.pytorch.utilities.types import LRSchedulerConfigType, OptimizerLRScheduler
from torch.optim import Adam, AdamW, Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR, ReduceLROnPlateau
from torchmetrics import AUROC, Accuracy, F1Score, Metric, MetricCollection, Precision, Recall


class SentimentClassifier(LightningModule):
    """
    Lightning Module for sentiment classification

    Supports multiple architectures:
    - LSTM/GRU based
    - CNN based
    - Attention based
    """

    def __init__(
        self,
        vocab_size: int,
        num_classes: int = 3,  # negative, neutral, positive
        embedding_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3,
        architecture: str = "lstm",  # lstm, gru, cnn, attention
        learning_rate: float = 1e-3,
        optimizer: str = "adamw",
        scheduler: str = "onecycle",
        weight_decay: float = 1e-4,
        class_weights: Optional[torch.Tensor] = None,
        max_sequence_length: int = 128,
        pad_token_id: int = 0,
    ):
        super().__init__()

        self.save_hyperparameters()

        self.vocab_size = vocab_size
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.architecture = architecture
        self.learning_rate = learning_rate
        self.optimizer_name = optimizer
        self.scheduler_name = scheduler
        self.weight_decay = weight_decay
        self.max_sequence_length = max_sequence_length
        self.pad_token_id = pad_token_id

        # Build model
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_token_id)
        self.dropout = nn.Dropout(dropout)

        if architecture == "lstm":
            self.encoder: nn.Module = nn.LSTM(
                embedding_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0
            )
            self.classifier = nn.Linear(hidden_dim, num_classes)
        elif architecture == "gru":
            self.encoder = nn.GRU(
                embedding_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0
            )
            self.classifier = nn.Linear(hidden_dim, num_classes)
        elif architecture == "cnn":
            self.encoder = self._build_cnn_encoder(dropout)
            self.classifier = nn.Linear(self._get_cnn_output_dim(), num_classes)
        elif architecture == "attention":
            self.encoder = self._build_attention_encoder(dropout)
            self.classifier = nn.Linear(hidden_dim, num_classes)
        else:
            raise ValueError(f"Unknown architecture: {architecture}")

        # Loss function
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)

        # Metrics: every object is handed to self.log so Lightning resets it per epoch.
        task: Literal["binary", "multiclass"] = "binary" if num_classes == 2 else "multiclass"
        self.train_metrics = self._create_metrics("train", task)
        self.val_metrics = self._create_metrics("val", task)
        self.test_metrics = self._create_metrics("test", task)

    def _build_cnn_encoder(self, dropout: float) -> nn.Module:
        """Build CNN encoder"""
        conv_layers = []
        kernel_sizes = [3, 4, 5]
        num_filters = self.hidden_dim // len(kernel_sizes)

        for kernel_size in kernel_sizes:
            conv = nn.Conv1d(self.embedding_dim, num_filters, kernel_size, padding=kernel_size // 2)
            conv_layers.append(conv)

        return nn.ModuleList([nn.ModuleList(conv_layers), nn.Dropout(dropout)])

    def _get_cnn_output_dim(self) -> int:
        """Get CNN output dimension"""
        kernel_sizes = [3, 4, 5]
        num_filters = self.hidden_dim // len(kernel_sizes)
        return num_filters * len(kernel_sizes)

    def _build_attention_encoder(self, dropout: float) -> nn.Module:
        """Build self-attention encoder"""
        return nn.ModuleList(
            [
                nn.MultiheadAttention(self.embedding_dim, num_heads=8, dropout=dropout, batch_first=True),
                nn.LayerNorm(self.embedding_dim),
                nn.Linear(self.embedding_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
        )

    def _create_metrics(self, stage: str, task: Literal["binary", "multiclass"]) -> MetricCollection:
        """Create metrics for a specific stage"""
        metrics: Dict[str, Union[Metric, MetricCollection]] = {
            "accuracy": Accuracy(task=task, num_classes=self.num_classes),
        }

        if stage != "train":
            metrics.update(
                {
                    "precision": Precision(task=task, num_classes=self.num_classes, average="macro"),
                    "recall": Recall(task=task, num_classes=self.num_classes, average="macro"),
                    "f1": F1Score(task=task, num_classes=self.num_classes, average="macro"),
                }
            )
            if self.num_classes > 2:
                metrics["auroc"] = AUROC(task=task, num_classes=self.num_classes, average="macro")

        return MetricCollection(metrics, compute_groups=False)

    def _build_mask(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Return a bool mask (batch, seq_len) that is True for real tokens.

        Sequences that contain no real token are given a single valid position so
        that packing, pooling and attention never see a zero-length sequence.
        """
        if attention_mask is None:
            mask = x != self.pad_token_id
        else:
            mask = attention_mask.bool()

        empty = ~mask.any(dim=1)
        if empty.any():
            mask = mask.clone()
            mask[empty, 0] = True
        return mask

    @staticmethod
    def _lengths_from_mask(mask: torch.Tensor) -> torch.Tensor:
        """Length of each sequence = index of the last real token + 1"""
        positions = torch.arange(mask.size(1), device=mask.device)
        return (mask * positions).max(dim=1).values + 1

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass"""
        mask = self._build_mask(x, attention_mask)  # (batch, seq_len) bool

        # Embedding
        embedded = self.embedding(x)  # (batch, seq_len, embed_dim)
        embedded = self.dropout(embedded)

        if self.architecture in ["lstm", "gru"]:
            # Pack padded sequences so the final hidden state is taken at each
            # sequence's real last token.
            lengths = self._lengths_from_mask(mask).cpu()
            packed = nn.utils.rnn.pack_padded_sequence(embedded, lengths, batch_first=True, enforce_sorted=False)
            _, hidden = self.encoder(packed)

            if isinstance(hidden, tuple):  # LSTM
                last_hidden = hidden[0][-1]  # (batch, hidden_dim)
            else:  # GRU
                last_hidden = hidden[-1]  # (batch, hidden_dim)

            features = self.dropout(last_hidden)

        elif self.architecture == "cnn":
            assert isinstance(self.encoder, nn.ModuleList)
            conv_layers, dropout_layer = self.encoder
            assert isinstance(conv_layers, nn.ModuleList)

            # Transpose for conv1d: (batch, embed_dim, seq_len)
            embedded = embedded.transpose(1, 2)

            conv_outputs = []
            for conv in conv_layers:
                conv_out = F.relu(conv(embedded))  # (batch, num_filters, L')
                pooled = F.max_pool1d(conv_out, kernel_size=conv_out.size(2))  # Global max pool
                conv_outputs.append(pooled.squeeze(2))  # (batch, num_filters)

            features = torch.cat(conv_outputs, dim=1)  # (batch, total_filters)
            features = dropout_layer(features)

        elif self.architecture == "attention":
            assert isinstance(self.encoder, nn.ModuleList)
            attn_layer, norm_layer, linear_layer, relu_layer, dropout_layer = self.encoder

            # key_padding_mask is True for positions that must be ignored
            attn_output, _ = attn_layer(embedded, embedded, embedded, key_padding_mask=~mask)
            attn_output = norm_layer(attn_output + embedded)  # Residual connection

            # Masked mean pooling over real tokens
            mask_f = mask.unsqueeze(-1).to(attn_output.dtype)  # (batch, seq_len, 1)
            features = (attn_output * mask_f).sum(dim=1) / mask_f.sum(dim=1)

            # Feed-forward
            features = dropout_layer(relu_layer(linear_layer(features)))

        # Classification
        logits = self.classifier(features)
        return logits

    def _update_metrics(self, metrics: MetricCollection, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        preds = torch.argmax(logits, dim=1)
        for name, metric in metrics.items():
            if name == "auroc" and self.num_classes > 2:
                metric.update(torch.softmax(logits, dim=1), y)
            elif name == "auroc":
                metric.update(torch.softmax(logits, dim=1)[:, 1], y)
            else:
                metric.update(preds, y)
        return preds

    def _log_metrics(self, stage: str, metrics: MetricCollection) -> None:
        for name, metric in metrics.items():
            key = f"{stage}/acc" if name == "accuracy" else f"{stage}/{name}"
            self.log(key, metric, prog_bar=(name == "accuracy"), on_step=False, on_epoch=True)

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step"""
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        self._update_metrics(self.train_metrics, logits, y)

        self.log("train/loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self._log_metrics("train", self.train_metrics)

        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Validation step"""
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        self._update_metrics(self.val_metrics, logits, y)

        self.log("val/loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self._log_metrics("val", self.val_metrics)

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Test step"""
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        self._update_metrics(self.test_metrics, logits, y)

        self.log("test/loss", loss, on_step=False, on_epoch=True)
        self._log_metrics("test", self.test_metrics)

    def on_validation_epoch_end(self) -> None:
        """Called at the end of validation epoch"""
        # compute() is cache-safe; Lightning resets the objects logged in validation_step.
        self.log("val/accuracy", self.val_metrics["accuracy"].compute())

    def predict_step(
        self, batch: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]], batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        """Prediction step"""
        if isinstance(batch, (tuple, list)):
            x = batch[0]
        else:
            x = batch

        logits = self(x)
        probabilities = torch.softmax(logits, dim=1)
        predictions = torch.argmax(logits, dim=1)

        return {"predictions": predictions, "probabilities": probabilities, "logits": logits}

    def configure_optimizers(self) -> OptimizerLRScheduler:
        """Configure optimizers and schedulers"""
        optimizer: Optimizer
        if self.optimizer_name.lower() == "adam":
            optimizer = Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.optimizer_name.lower() == "adamw":
            optimizer = AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        else:
            optimizer = AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

        lr_scheduler: Optional[LRSchedulerConfigType] = None
        if self.scheduler_name.lower() == "onecycle":
            lr_scheduler = {
                "scheduler": OneCycleLR(
                    optimizer,
                    max_lr=self.learning_rate,
                    total_steps=int(self.trainer.estimated_stepping_batches),
                    pct_start=0.3,
                ),
                "interval": "step",
            }
        elif self.scheduler_name.lower() == "cosine":
            lr_scheduler = {
                "scheduler": CosineAnnealingLR(optimizer, T_max=self.trainer.max_epochs),
                "interval": "epoch",
            }
        elif self.scheduler_name.lower() == "plateau":
            lr_scheduler = {
                "scheduler": ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5),
                "monitor": "val/acc",
                "interval": "epoch",
            }

        if lr_scheduler is None:
            return {"optimizer": optimizer}
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}

    def get_sentiment_labels(self) -> List[str]:
        """Get sentiment labels"""
        if self.num_classes == 2:
            return ["negative", "positive"]
        elif self.num_classes == 3:
            return ["negative", "neutral", "positive"]
        else:
            return [f"class_{i}" for i in range(self.num_classes)]
