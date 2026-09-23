# File: src/lmpro/modules/nlp/char_lm.py

"""
Character-level language model for text generation
"""

from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
from lightning.pytorch import LightningModule
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
from torchmetrics import MeanMetric

PAD_TOKEN_ID = 0


class CharacterLanguageModel(LightningModule):
    """
    Character-level language model using LSTM/GRU

    Token id 0 is reserved for padding / end-of-sequence and is ignored by the loss.
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
        self.pad_token_id = PAD_TOKEN_ID

        # Model layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=self.pad_token_id)

        if rnn_type.lower() == "lstm":
            self.rnn = nn.LSTM(
                embedding_dim, hidden_dim, num_layers, dropout=dropout if num_layers > 1 else 0, batch_first=True
            )
        elif rnn_type.lower() == "gru":
            self.rnn = nn.GRU(
                embedding_dim, hidden_dim, num_layers, dropout=dropout if num_layers > 1 else 0, batch_first=True
            )
        else:
            raise ValueError(f"Unknown RNN type: {rnn_type}")

        self.dropout = nn.Dropout(dropout)
        self.output_projection = nn.Linear(hidden_dim, vocab_size)

        # Loss function
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.pad_token_id)

        # Token-weighted running means of loss / accuracy per stage. Perplexity is
        # exp(mean loss) computed once per epoch, not the mean of per-batch exp().
        self.train_loss_mean = MeanMetric()
        self.val_loss_mean = MeanMetric()
        self.test_loss_mean = MeanMetric()
        self.val_acc_mean = MeanMetric()
        self.test_acc_mean = MeanMetric()

    def forward(
        self, x: torch.Tensor, hidden: Optional[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]] = None
    ) -> Tuple[torch.Tensor, Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]]:
        """Forward pass"""
        embedded = self.embedding(x)  # (batch, seq_len, embed_dim)
        embedded = self.dropout(embedded)

        output, hidden = self.rnn(embedded, hidden)  # (batch, seq_len, hidden_dim)
        output = self.dropout(output)

        logits = self.output_projection(output)  # (batch, seq_len, vocab_size)

        return logits, hidden

    def init_hidden(self, batch_size: int) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Initialize hidden states"""
        h_0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=self.device)

        if isinstance(self.rnn, nn.LSTM):
            c_0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=self.device)
            return (h_0, c_0)
        return h_0

    def _shared_step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (loss, accuracy over non-pad tokens, number of non-pad tokens)"""
        input_seq, target_seq = batch

        logits, _ = self(input_seq)  # (batch, seq_len, vocab_size)
        logits = logits.reshape(-1, self.vocab_size)
        targets = target_seq.reshape(-1)

        loss = self.criterion(logits, targets)

        mask = targets != self.pad_token_id
        num_tokens = mask.sum()
        if num_tokens > 0:
            accuracy = (torch.argmax(logits, dim=1) == targets)[mask].float().mean()
        else:
            accuracy = torch.zeros((), device=logits.device)

        return loss, accuracy, num_tokens

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step"""
        loss, _, num_tokens = self._shared_step(batch)
        self.train_loss_mean.update(loss.detach(), weight=num_tokens)

        self.log("train/loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def on_train_epoch_end(self) -> None:
        self.log("train/perplexity", torch.exp(self.train_loss_mean.compute()), prog_bar=True)
        self.train_loss_mean.reset()

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Validation step"""
        loss, accuracy, num_tokens = self._shared_step(batch)
        self.val_loss_mean.update(loss, weight=num_tokens)
        self.val_acc_mean.update(accuracy, weight=num_tokens)

        self.log("val/loss", loss, prog_bar=True, on_step=False, on_epoch=True)

    def on_validation_epoch_end(self) -> None:
        self.log("val/perplexity", torch.exp(self.val_loss_mean.compute()), prog_bar=True)
        self.log("val/accuracy", self.val_acc_mean.compute())
        self.val_loss_mean.reset()
        self.val_acc_mean.reset()

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Test step"""
        loss, accuracy, num_tokens = self._shared_step(batch)
        self.test_loss_mean.update(loss, weight=num_tokens)
        self.test_acc_mean.update(accuracy, weight=num_tokens)

        self.log("test/loss", loss, on_step=False, on_epoch=True)

    def on_test_epoch_end(self) -> None:
        self.log("test/perplexity", torch.exp(self.test_loss_mean.compute()))
        self.log("test/accuracy", self.test_acc_mean.compute())
        self.test_loss_mean.reset()
        self.test_acc_mean.reset()

    def generate(
        self,
        prompt: torch.Tensor,
        max_length: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Generate text given a prompt of shape (batch, prompt_len).

        Generation is fully vectorised over the batch; a sequence stops
        extending (emits pad) once it has produced the pad/end token.
        """
        was_training = self.training
        self.eval()

        with torch.no_grad():
            batch_size = prompt.shape[0]
            generated = prompt.clone()
            hidden = self.init_hidden(batch_size)
            finished = torch.zeros(batch_size, dtype=torch.bool, device=prompt.device)

            # Process prompt (all tokens but the last, which is fed in the loop)
            if prompt.shape[1] > 1:
                _, hidden = self(prompt[:, :-1], hidden)

            for _ in range(max_length):
                last_token = generated[:, -1:]
                logits, hidden = self(last_token, hidden)
                logits = logits[:, -1, :] / temperature  # (batch, vocab)

                # Top-k filtering
                if top_k is not None:
                    top_k = min(top_k, logits.size(-1))
                    kth_value = torch.topk(logits, top_k, dim=-1).values[:, -1:]
                    logits = logits.masked_fill(logits < kth_value, float("-inf"))

                # Top-p (nucleus) filtering
                if top_p is not None:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_remove = cumulative_probs > top_p
                    sorted_remove[:, 1:] = sorted_remove[:, :-1].clone()
                    sorted_remove[:, 0] = False
                    remove = torch.zeros_like(sorted_remove).scatter(-1, sorted_indices, sorted_remove)
                    logits = logits.masked_fill(remove, float("-inf"))

                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)  # (batch, 1)

                # Finished sequences keep emitting pad
                next_token = next_token.masked_fill(finished.unsqueeze(1), self.pad_token_id)
                generated = torch.cat([generated, next_token], dim=1)

                finished = finished | (next_token.squeeze(1) == self.pad_token_id)
                if bool(finished.all()):
                    break

        if was_training:
            self.train()
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
                optimizer, max_lr=self.learning_rate, total_steps=self.trainer.estimated_stepping_batches, pct_start=0.3
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

    def predict_step(
        self, batch: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]], batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        """Prediction step for generation"""
        if isinstance(batch, (tuple, list)):
            input_seq = batch[0]
        else:
            input_seq = batch

        generated = self.generate(input_seq, max_length=50, temperature=0.8)

        return {"input": input_seq, "generated": generated, "continuation": generated[:, input_seq.shape[1] :]}
