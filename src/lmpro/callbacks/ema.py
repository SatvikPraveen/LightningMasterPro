# File: src/lmpro/callbacks/ema.py

"""
Exponential Moving Average callback for model weights
"""

from typing import Any, Dict, Optional

import torch
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback


class EMACallback(Callback):
    """
    Exponential Moving Average callback.

    Maintains an exponential moving average of the model's state dict during
    training and optionally swaps the EMA weights in for validation/testing.

    * Floating-point tensors are averaged; other tensors (for example
      ``num_batches_tracked`` in BatchNorm) are copied directly.
    * With ``warmup=True`` the effective decay is
      ``min(decay, (1 + n) / (10 + n))`` where ``n`` is the number of updates
      so far, so the average is not dominated by the random initial weights.
    * The EMA state is persisted through the callback's ``state_dict`` (stored
      under ``checkpoint["callbacks"]``) and is NOT reset on resume.
    """

    def __init__(
        self,
        decay: float = 0.999,
        start_epoch: int = 0,
        update_every: int = 1,
        use_ema_for_validation: bool = True,
        warmup: bool = True,
    ):
        super().__init__()
        if not 0.0 <= decay <= 1.0:
            raise ValueError("decay must be in [0, 1]")
        if update_every < 1:
            raise ValueError("update_every must be >= 1")

        self.decay = decay
        self.start_epoch = start_epoch
        self.update_every = update_every
        self.use_ema_for_validation = use_ema_for_validation
        self.warmup = warmup

        self.ema_model: Optional[Dict[str, torch.Tensor]] = None
        self.original_model_state: Optional[Dict[str, torch.Tensor]] = None
        self.update_counter = 0  # training batches seen (drives ``update_every``)
        self.num_updates = 0  # EMA updates applied (drives warm-up)

    # ------------------------------------------------------------------
    # Core
    # ------------------------------------------------------------------

    @staticmethod
    def _clone_state(pl_module: LightningModule) -> Dict[str, torch.Tensor]:
        return {k: v.detach().clone() for k, v in pl_module.state_dict().items()}

    def current_decay(self) -> float:
        """Effective decay for the next update (accounts for warm-up)."""
        if not self.warmup:
            return self.decay
        n = self.num_updates
        return min(self.decay, (1.0 + n) / (10.0 + n))

    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Initialise the EMA weights, unless they were restored from a checkpoint."""
        if self.ema_model is None:
            self.ema_model = self._clone_state(pl_module)
        else:
            self.ema_model = {k: v.to(pl_module.device) for k, v in self.ema_model.items()}

    def on_train_batch_end(
        self, trainer: Trainer, pl_module: LightningModule, outputs: Any, batch: Any, batch_idx: int
    ) -> None:
        if self.ema_model is None or trainer.current_epoch < self.start_epoch:
            return
        self.update_counter += 1
        if self.update_counter % self.update_every == 0:
            self._update_ema(pl_module)

    @torch.no_grad()
    def _update_ema(self, pl_module: LightningModule) -> None:
        decay = self.current_decay()
        for key, value in pl_module.state_dict().items():
            if key not in self.ema_model:
                self.ema_model[key] = value.detach().clone()
                continue
            ema_value = self.ema_model[key]
            if ema_value.device != value.device:
                ema_value = ema_value.to(value.device)
                self.ema_model[key] = ema_value
            if value.is_floating_point():
                ema_value.mul_(decay).add_(value.detach(), alpha=1.0 - decay)
            else:
                ema_value.copy_(value)
        self.num_updates += 1

    # ------------------------------------------------------------------
    # Weight swapping for evaluation
    # ------------------------------------------------------------------

    def _swap_in_ema(self, pl_module: LightningModule) -> None:
        if self.ema_model is None or self.original_model_state is not None:
            return
        self.original_model_state = self._clone_state(pl_module)
        pl_module.load_state_dict(self.ema_model, strict=True)

    def _swap_out_ema(self, pl_module: LightningModule) -> None:
        if self.original_model_state is None:
            return
        pl_module.load_state_dict(self.original_model_state, strict=True)
        self.original_model_state = None

    def on_validation_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if self.use_ema_for_validation:
            self._swap_in_ema(pl_module)

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if self.use_ema_for_validation:
            self._swap_out_ema(pl_module)

    def on_test_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self._swap_in_ema(pl_module)

    def on_test_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self._swap_out_ema(pl_module)

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def apply_ema_weights(self, pl_module: LightningModule) -> None:
        """Permanently load the EMA weights into ``pl_module``."""
        if self.ema_model is not None:
            pl_module.load_state_dict(self.ema_model, strict=True)

    def get_ema_model(self) -> Optional[Dict[str, torch.Tensor]]:
        return self.ema_model

    # ------------------------------------------------------------------
    # Persistence (single source of truth: the callback state dict)
    # ------------------------------------------------------------------

    def state_dict(self) -> Dict[str, Any]:
        return {
            "ema_model": None if self.ema_model is None else {k: v.cpu() for k, v in self.ema_model.items()},
            "decay": self.decay,
            "update_counter": self.update_counter,
            "num_updates": self.num_updates,
            "start_epoch": self.start_epoch,
            "update_every": self.update_every,
            "warmup": self.warmup,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        ema = state_dict.get("ema_model")
        self.ema_model = None if ema is None else {k: v.clone() for k, v in ema.items()}
        self.decay = state_dict.get("decay", self.decay)
        self.update_counter = state_dict.get("update_counter", 0)
        self.num_updates = state_dict.get("num_updates", 0)
        self.start_epoch = state_dict.get("start_epoch", self.start_epoch)
        self.update_every = state_dict.get("update_every", self.update_every)
        self.warmup = state_dict.get("warmup", self.warmup)


def create_ema_callback(
    decay: float = 0.999,
    start_epoch: int = 0,
    use_for_validation: bool = True,
    **kwargs: Any,
) -> EMACallback:
    """Create EMA callback with common settings"""
    return EMACallback(decay=decay, start_epoch=start_epoch, use_ema_for_validation=use_for_validation, **kwargs)
