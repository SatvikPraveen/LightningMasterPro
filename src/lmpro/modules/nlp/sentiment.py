# File: src/lmpro/modules/nlp/sentiment.py

"""
Sentiment classification module for text sentiment analysis
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning import LightningModule
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingLR, ReduceLROnPlateau
from torchmetrics import Accuracy, Precision, Recall, F1Score, AUROC, ConfusionMatrix
from typing import Dict, Any, Optional, List, Tuple, Union


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
        **kwargs
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
        
        # Build model
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.dropout = nn.Dropout(dropout)
        
        if architecture == "lstm":
            self.encoder = nn.LSTM(
                embedding_dim, hidden_dim, num_layers,
                batch_first=True, dropout=dropout if num_layers > 1 else 0
            )
            self.classifier = nn.Linear(hidden_dim, num_classes)
        elif architecture == "gru":
            self.encoder = nn.GRU(
                embedding_dim, hidden_dim, num_layers,
                batch_first=True, dropout=dropout if num_layers > 1 else 0
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
        
        # Metrics
        task = "binary" if num_classes == 2 else "multiclass"
        self.train_metrics = self._create_metrics("train", task)
        self.val_metrics = self._create_metrics("val", task)
        self.test_metrics = self._create_metrics("test", task)
        
        # For confusion matrix
        self.val_predictions = []
        self.val_targets = []
    
    def _build_cnn_encoder(self, dropout: float) -> nn.Module:
        """Build CNN encoder"""
        conv_layers = []
        kernel_sizes = [3, 4, 5]
        num_filters = self.hidden_dim // len(kernel_sizes)
        
        for kernel_size in kernel_sizes:
            conv = nn.Conv1d(
                self.embedding_dim, num_filters, 
                kernel_size, padding=kernel_size//2
            )
            conv_layers.append(conv)
        
        return nn.ModuleList([
            nn.ModuleList(conv_layers),
            nn.Dropout(dropout)
        ])
    
    def _get_cnn_output_dim(self) -> int:
        """Get CNN output dimension"""
        kernel_sizes = [3, 4, 5]
        num_filters = self.hidden_dim // len(kernel_sizes)
        return num_filters * len(kernel_sizes)
    
    def _build_attention_encoder(self, dropout: float) -> nn.Module:
        """Build self-attention encoder"""
        return nn.ModuleList([
            nn.MultiheadAttention(self.embedding_dim, num_heads=8, dropout=dropout, batch_first=True),
            nn.LayerNorm(self.embedding_dim),
            nn.Linear(self.embedding_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        ])
    
    def _create_metrics(self, stage: str, task: str) -> Dict[str, Any]:
        """Create metrics for a specific stage"""
        metrics = {
            "accuracy": Accuracy(task=task, num_classes=self.num_classes),
            "precision": Precision(task=task, num_classes=self.num_classes, average="macro"),
            "recall": Recall(task=task, num_classes=self.num_classes, average="macro"),
            "f1": F1Score(task=task, num_classes=self.num_classes, average="macro"),
        }
        
        if self.num_classes > 2:
            metrics["auroc"] = AUROC(task=task, num_classes=self.num_classes, average="macro")
        
        return nn.ModuleDict(metrics)
    
    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass"""
        # Embedding
        embedded = self.embedding(x)  # (batch, seq_len, embed_dim)
        embedded = self.dropout(embedded)
        
        if self.architecture in ["lstm", "gru"]:
            # RNN encoding
            if attention_mask is not None:
                # Pack padded sequences for efficiency
                lengths = attention_mask.sum(dim=1).cpu()
                packed = nn.utils.rnn.pack_padded_sequence(
                    embedded, lengths, batch_first=True, enforce_sorted=False
                )
                output, hidden = self.encoder(packed)
                output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
            else:
                output, hidden = self.encoder(embedded)
            
            # Use last hidden state
            if isinstance(hidden, tuple):  # LSTM
                last_hidden = hidden[0][-1]  # (batch, hidden_dim)
            else:  # GRU
                last_hidden = hidden[-1]  # (batch, hidden_dim)
            
            features = self.dropout(last_hidden)
            
        elif self.architecture == "cnn":
            # CNN encoding
            conv_layers, dropout_layer = self.encoder
            
            # Transpose for conv1d: (batch, embed_dim, seq_len)
            embedded = embedded.transpose(1, 2)
            
            conv_outputs = []
            for conv in conv_layers:
                conv_out = F.relu(conv(embedded))  # (batch, num_filters, seq_len)
                pooled = F.max_pool1d(conv_out, kernel_size=conv_out.size(2))  # Global max pool
                conv_outputs.append(pooled.squeeze(2))  # (batch, num_filters)
            
            features = torch.cat(conv_outputs, dim=1)  # (batch, total_filters)
            features = dropout_layer(features)
            
        elif self.architecture == "attention":
            # Self-attention encoding
            attn_layer, norm_layer, linear_layer, relu_layer, dropout_layer = self.encoder
            
            # Self-attention
            attn_output, _ = attn_layer(embedded, embedded, embedded, key_padding_mask=~attention_mask if attention_mask is not None else None)
            attn_output = norm_layer(attn_output + embedded)  # Residual connection
            
            # Global average pooling
            if attention_mask is not None:
                mask = attention_mask.unsqueeze(-1).float()  # (batch, seq_len, 1)
                attn_output = attn_output * mask
                features = attn_output.sum(dim=1) / mask.sum(dim=1)  # (batch, embed_dim)
            else:
                features = attn_output.mean(dim=1)  # (batch, embed_dim)
            
            # Feed-forward
            features = dropout_layer(relu_layer(linear_layer(features)))
        
        # Classification
        logits = self.classifier(features)
        return logits
    
    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step"""
        x, y = batch
        
        # Create attention mask (non-padding tokens)
        attention_mask = (x != 0).long() if hasattr(self, 'pad_token_id') else None
        
        logits = self(x, attention_mask)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        
        # Update metrics
        for name, metric in self.train_metrics.items():
            if name == "auroc" and self.num_classes > 2:
                metric.update(torch.softmax(logits, dim=1), y)
            elif name == "auroc":
                metric.update(torch.softmax(logits, dim=1)[:, 1], y)
            else:
                metric.update(preds, y)
        
        # Log metrics
        self.log("train/loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("train/acc", self.train_metrics["accuracy"], prog_bar=True, on_step=False, on_epoch=True)
        
        return loss
    
    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Validation step"""
        x, y = batch
        
        attention_mask = (x != 0).long() if hasattr(self, 'pad_token_id') else None
        logits = self(x, attention_mask)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        
        # Update metrics
        for name, metric in self.val_metrics.items():
            if name == "auroc" and self.num_classes > 2:
                metric.update(torch.softmax(logits, dim=1), y)
            elif name == "auroc":
                metric.update(torch.softmax(logits, dim=1)[:, 1], y)
            else:
                metric.update(preds, y)
        
        # Log metrics
        self.log("val/loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val/acc", self.val_metrics["accuracy"], prog_bar=True, on_step=False, on_epoch=True)
        
        # Store predictions for confusion matrix
        self.val_predictions.extend(preds.cpu().tolist())
        self.val_targets.extend(y.cpu().tolist())
    
    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Test step"""
        x, y = batch
        
        attention_mask = (x != 0).long() if hasattr(self, 'pad_token_id') else None
        logits = self(x, attention_mask)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        
        # Update metrics
        for name, metric in self.test_metrics.items():
            if name == "auroc" and self.num_classes > 2:
                metric.update(torch.softmax(logits, dim=1), y)
            elif name == "auroc":
                metric.update(torch.softmax(logits, dim=1)[:, 1], y)
            else:
                metric.update(preds, y)
        
        # Log metrics
        self.log("test/loss", loss, on_step=False, on_epoch=True)
        self.log("test/acc", self.test_metrics["accuracy"], on_step=False, on_epoch=True)
    
    def on_validation_epoch_end(self) -> None:
        """Called at the end of validation epoch"""
        # Compute and log all metrics
        for name, metric in self.val_metrics.items():
            self.log(f"val/{name}", metric.compute(), prog_bar=(name == "accuracy"))
        
        # Clear stored predictions
        self.val_predictions = []
        self.val_targets = []
    
    def predict_step(self, batch: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]], batch_idx: int) -> Dict[str, torch.Tensor]:
        """Prediction step"""
        if isinstance(batch, tuple):
            x, _ = batch
        else:
            x = batch
        
        attention_mask = (x != 0).long() if hasattr(self, 'pad_token_id') else None
        logits = self(x, attention_mask)
        probabilities = torch.softmax(logits, dim=1)
        predictions = torch.argmax(logits, dim=1)
        
        return {
            "predictions": predictions,
            "probabilities": probabilities,
            "logits": logits
        }
    
    def configure_optimizers(self) -> Dict[str, Any]:
        """Configure optimizers and schedulers"""
        if self.optimizer_name.lower() == "adam":
            optimizer = Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.optimizer_name.lower() == "adamw":
            optimizer = AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        else:
            optimizer = AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        
        config = {"optimizer": optimizer}
        
        if self.scheduler_name.lower() == "onecycle":
            scheduler = OneCycleLR(
                optimizer,
                max_lr=self.learning_rate,
                total_steps=self.trainer.estimated_stepping_batches,
                pct_start=0.3
            )
            config["lr_scheduler"] = {"scheduler": scheduler, "interval": "step"}
        elif self.scheduler_name.lower() == "cosine":
            scheduler = CosineAnnealingLR(optimizer, T_max=self.trainer.max_epochs)
            config["lr_scheduler"] = {"scheduler": scheduler, "interval": "epoch"}
        elif self.scheduler_name.lower() == "plateau":
            scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)
            config["lr_scheduler"] = {
                "scheduler": scheduler,
                "monitor": "val/acc",
                "interval": "epoch"
            }
        
        return config
    
    def get_sentiment_labels(self) -> List[str]:
        """Get sentiment labels"""
        if self.num_classes == 2:
            return ["negative", "positive"]
        elif self.num_classes == 3:
            return ["negative", "neutral", "positive"]
        else:
            return [f"class_{i}" for i in range(self.num_classes)]