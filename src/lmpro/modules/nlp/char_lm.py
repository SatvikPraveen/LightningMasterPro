# File: src/lmpro/modules/nlp/char_lm.py

"""
Character-level language model for text generation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning import LightningModule
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingLR
from typing import Dict, Any, Optional, Tuple, Union
import numpy as np


class CharacterLanguageModel(LightningModule):
    """
    Character-level language model using LSTM/GRU
    """
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3,
        rnn_type: str = "lstm",
        learning_rate: float = 1e-3,
        optimizer: str = "adamw",
        scheduler: str = "onecycle",
        weight_decay: float = 1e-4,
        gradient_clip_val: float = 1.0,
        teacher_forcing_ratio: float = 0.5,
        **kwargs
    ):
        super().__init__()
        
        self.save_hyperparameters()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.optimizer_name = optimizer
        self.scheduler_name = scheduler
        self.weight_decay = weight_decay
        self.gradient_clip_val = gradient_clip_val
        self.teacher_forcing_ratio = teacher_forcing_ratio
        
        # Model layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        if rnn_type.lower() == "lstm":
            self.rnn = nn.LSTM(
                embedding_dim, hidden_dim, num_layers,
                dropout=dropout if num_layers > 1 else 0,
                batch_first=True
            )
        elif rnn_type.lower() == "gru":
            self.rnn = nn.GRU(
                embedding_dim, hidden_dim, num_layers,
                dropout=dropout if num_layers > 1 else 0,
                batch_first=True
            )
        else:
            raise ValueError(f"Unknown RNN type: {rnn_type}")
        
        self.dropout = nn.Dropout(dropout)
        self.output_projection = nn.Linear(hidden_dim, vocab_size)
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)
        
        # For tracking metrics
        self.train_perplexity = []
        self.val_perplexity = []
    
    def forward(
        self, 
        x: torch.Tensor, 
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass"""
        # Embedding
        embedded = self.embedding(x)  # (batch, seq_len, embed_dim)
        embedded = self.dropout(embedded)
        
        # RNN
        output, hidden = self.rnn(embedded, hidden)  # (batch, seq_len, hidden_dim)
        output = self.dropout(output)
        
        # Output projection
        logits = self.output_projection(output)  # (batch, seq_len, vocab_size)
        
        return logits, hidden
    
    def init_hidden(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initialize hidden states"""
        h_0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=self.device)
        
        if isinstance(self.rnn, nn.LSTM):
            c_0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=self.device)
            return (h_0, c_0)
        else:  # GRU
            return h_0
    
    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step"""
        input_seq, target_seq = batch
        batch_size, seq_len = input_seq.shape
        
        # Forward pass
        logits, _ = self(input_seq)  # (batch, seq_len, vocab_size)
        
        # Reshape for loss computation
        logits = logits.view(-1, self.vocab_size)  # (batch*seq_len, vocab_size)
        targets = target_seq.view(-1)  # (batch*seq_len,)
        
        # Compute loss
        loss = self.criterion(logits, targets)
        
        # Compute perplexity
        perplexity = torch.exp(loss)
        
        # Log metrics
        self.log("train/loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("train/perplexity", perplexity, prog_bar=True, on_step=False, on_epoch=True)
        
        return loss
    
    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Validation step"""
        input_seq, target_seq = batch
        
        # Forward pass
        logits, _ = self(input_seq)
        
        # Reshape for loss computation
        logits = logits.view(-1, self.vocab_size)
        targets = target_seq.view(-1)
        
        # Compute loss
        loss = self.criterion(logits, targets)
        perplexity = torch.exp(loss)
        
        # Log metrics
        self.log("val/loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val/perplexity", perplexity, prog_bar=True, on_step=False, on_epoch=True)
        
        # Compute accuracy
        pred_tokens = torch.argmax(logits, dim=1)
        mask = targets != 0  # Ignore padding tokens
        if mask.sum() > 0:
            accuracy = (pred_tokens == targets)[mask].float().mean()
            self.log("val/accuracy", accuracy, on_step=False, on_epoch=True)
    
    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Test step"""
        input_seq, target_seq = batch
        
        # Forward pass
        logits, _ = self(input_seq)
        
        # Reshape for loss computation
        logits = logits.view(-1, self.vocab_size)
        targets = target_seq.view(-1)
        
        # Compute loss
        loss = self.criterion(logits, targets)
        perplexity = torch.exp(loss)
        
        # Log metrics
        self.log("test/loss", loss, on_step=False, on_epoch=True)
        self.log("test/perplexity", perplexity, on_step=False, on_epoch=True)
        
        # Compute accuracy
        pred_tokens = torch.argmax(logits, dim=1)
        mask = targets != 0
        if mask.sum() > 0:
            accuracy = (pred_tokens == targets)[mask].float().mean()
            self.log("test/accuracy", accuracy, on_step=False, on_epoch=True)
    
    def generate(
        self,
        prompt: torch.Tensor,
        max_length: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None
    ) -> torch.Tensor:
        """Generate text given a prompt"""
        self.eval()
        
        with torch.no_grad():
            batch_size = prompt.shape[0]
            generated = prompt.clone()
            hidden = self.init_hidden(batch_size)
            
            # Process prompt
            if prompt.shape[1] > 1:
                prompt_logits, hidden = self(prompt[:, :-1], hidden)
            
            # Generate tokens
            for _ in range(max_length):
                # Get last token
                last_token = generated[:, -1:] 
                
                # Forward pass
                logits, hidden = self(last_token, hidden)
                logits = logits[:, -1, :] / temperature  # (batch_size, vocab_size)
                
                # Apply top-k filtering
                if top_k is not None:
                    top_k_logits, top_k_indices = torch.topk(logits, top_k, dim=-1)
                    logits.fill_(-float('inf'))
                    logits.scatter_(-1, top_k_indices, top_k_logits)
                
                # Apply top-p (nucleus) filtering
                if top_p is not None:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                    sorted_indices_to_remove[:, 0] = False
                    
                    for i in range(batch_size):
                        indices_to_remove = sorted_indices[i][sorted_indices_to_remove[i]]
                        logits[i][indices_to_remove] = -float('inf')
                
                # Sample next token
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Append to generated sequence
                generated = torch.cat([generated, next_token], dim=1)
                
                # Check for end of sequence (optional)
                if next_token.item() == 0:  # Assuming 0 is pad/end token
                    break
        
        return generated
    
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
        
        return config
    
    def on_before_optimizer_step(self, optimizer) -> None:
        """Gradient clipping"""
        if self.gradient_clip_val > 0:
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.gradient_clip_val)
    
    def predict_step(self, batch: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]], batch_idx: int) -> Dict[str, torch.Tensor]:
        """Prediction step for generation"""
        if isinstance(batch, tuple):
            input_seq, _ = batch
        else:
            input_seq = batch
        
        # Generate continuations
        generated = self.generate(input_seq, max_length=50, temperature=0.8)
        
        return {
            "input": input_seq,
            "generated": generated,
            "continuation": generated[:, input_seq.shape[1]:]
        }