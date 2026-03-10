# src/lmpro/callbacks/gradient_monitor.py
"""
Gradient Monitor Callback for LightningMasterPro.

Tracks gradient norms, detects vanishing/exploding gradients, and optionally
logs per-layer gradient statistics for debugging deep training runs.
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.utilities.rank_zero import rank_zero_warn


class GradientMonitorCallback(Callback):
    """
    Monitors gradient norms during training.

    Features:
    - Global gradient norm logging (L2 norm across all parameters)
    - Per-layer gradient norm logging (optional)
    - Vanishing gradient detection (norm < threshold)
    - Exploding gradient detection (norm > threshold)
    - Logging interval control (every N steps)

    Args:
        log_every_n_steps: Log gradients every N training steps (default: 50).
        log_per_layer: If True, log gradient norms for each named parameter.
        vanishing_threshold: Warn when the global gradient norm falls below
            this value (default: 1e-7). Set to None to disable.
        exploding_threshold: Warn when the global gradient norm exceeds this
            value (default: 1e3). Set to None to disable.
        norm_type: Type of gradient norm to compute (default: 2.0).
    """

    def __init__(
        self,
        log_every_n_steps: int = 50,
        log_per_layer: bool = False,
        vanishing_threshold: Optional[float] = 1e-7,
        exploding_threshold: Optional[float] = 1e3,
        norm_type: float = 2.0,
    ):
        super().__init__()
        self.log_every_n_steps = log_every_n_steps
        self.log_per_layer = log_per_layer
        self.vanishing_threshold = vanishing_threshold
        self.exploding_threshold = exploding_threshold
        self.norm_type = norm_type

        self._global_norm_history: List[Tuple[int, float]] = []
        self._vanishing_count = 0
        self._exploding_count = 0

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _compute_global_norm(self, model: nn.Module) -> float:
        """Compute the global L2 gradient norm over all parameters."""
        params = [p for p in model.parameters() if p.grad is not None]
        if not params:
            return 0.0
        total_norm = torch.norm(
            torch.stack([torch.norm(p.grad.detach(), self.norm_type) for p in params]),
            self.norm_type,
        )
        return total_norm.item()

    def _log_per_layer_norms(
        self, model: nn.Module, pl_module: LightningModule
    ) -> None:
        """Log gradient norm for each named parameter."""
        for name, param in model.named_parameters():
            if param.grad is None:
                continue
            layer_norm = param.grad.detach().norm(self.norm_type).item()
            safe_name = name.replace(".", "/")
            pl_module.log(
                f"grad_norm/{safe_name}",
                layer_norm,
                on_step=True,
                on_epoch=False,
                prog_bar=False,
                logger=True,
            )

    # ------------------------------------------------------------------
    # Callback hooks
    # ------------------------------------------------------------------

    def on_after_backward(
        self,
        trainer: "Trainer",
        pl_module: "LightningModule",
    ) -> None:
        step = trainer.global_step
        if step % self.log_every_n_steps != 0:
            return

        global_norm = self._compute_global_norm(pl_module)
        pl_module.log(
            "grad_norm/global",
            global_norm,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            logger=True,
        )
        self._global_norm_history.append((step, global_norm))

        # Anomaly detection
        if self.vanishing_threshold is not None and 0 < global_norm < self.vanishing_threshold:
            self._vanishing_count += 1
            rank_zero_warn(
                f"Vanishing gradient detected at step {step}: "
                f"global norm = {global_norm:.2e} (threshold {self.vanishing_threshold:.2e})"
            )

        if self.exploding_threshold is not None and global_norm > self.exploding_threshold:
            self._exploding_count += 1
            rank_zero_warn(
                f"Exploding gradient detected at step {step}: "
                f"global norm = {global_norm:.2e} (threshold {self.exploding_threshold:.2e})"
            )

        if self.log_per_layer:
            self._log_per_layer_norms(pl_module, pl_module)

    def on_train_end(
        self,
        trainer: "Trainer",
        pl_module: "LightningModule",
    ) -> None:
        if self._vanishing_count:
            rank_zero_warn(
                f"GradientMonitor: {self._vanishing_count} vanishing gradient "
                "events detected during training."
            )
        if self._exploding_count:
            rank_zero_warn(
                f"GradientMonitor: {self._exploding_count} exploding gradient "
                "events detected during training."
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_global_norm_history(self) -> List[Tuple[int, float]]:
        """Return [(step, global_norm), ...] history."""
        return list(self._global_norm_history)

    @property
    def vanishing_event_count(self) -> int:
        return self._vanishing_count

    @property
    def exploding_event_count(self) -> int:
        return self._exploding_count

    def reset_history(self) -> None:
        self._global_norm_history.clear()
        self._vanishing_count = 0
        self._exploding_count = 0


def create_gradient_monitor(
    log_every_n_steps: int = 50,
    log_per_layer: bool = False,
    vanishing_threshold: Optional[float] = 1e-7,
    exploding_threshold: Optional[float] = 1e3,
    norm_type: float = 2.0,
) -> GradientMonitorCallback:
    """Factory function for GradientMonitorCallback."""
    return GradientMonitorCallback(
        log_every_n_steps=log_every_n_steps,
        log_per_layer=log_per_layer,
        vanishing_threshold=vanishing_threshold,
        exploding_threshold=exploding_threshold,
        norm_type=norm_type,
    )
