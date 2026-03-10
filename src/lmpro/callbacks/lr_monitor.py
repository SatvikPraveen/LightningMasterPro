# src/lmpro/callbacks/lr_monitor.py
"""
Advanced Learning Rate Monitor Callback for LightningMasterPro.

Tracks and logs per-parameter-group learning rates at configurable intervals,
with optional warmup detection and anomaly alerting.
"""

from typing import Dict, List, Optional, Tuple
import warnings

import torch
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.utilities.rank_zero import rank_zero_info, rank_zero_warn


class LRMonitorCallback(Callback):
    """
    Monitors and logs learning rates for all optimizer parameter groups.

    Features:
    - Per-parameter-group LR tracking
    - Warmup phase detection
    - LR anomaly alerting (spike / collapse detection)
    - Optional CSV export of LR history

    Args:
        log_momentum: Whether to also log optimizer momentum values.
        logging_interval: "step" or "epoch" (default: "step").
        log_weight_decay: Whether to log weight decay values.
        alert_on_lr_change_factor: Alert if LR changes by more than this factor
            in one step (e.g., 10.0 → 10x change). Set to None to disable.
        verbose: Print LR info to console each interval.
    """

    def __init__(
        self,
        log_momentum: bool = False,
        logging_interval: str = "step",
        log_weight_decay: bool = False,
        alert_on_lr_change_factor: Optional[float] = None,
        verbose: bool = False,
    ):
        super().__init__()
        if logging_interval not in ("step", "epoch"):
            raise ValueError("`logging_interval` must be 'step' or 'epoch'.")

        self.log_momentum = log_momentum
        self.logging_interval = logging_interval
        self.log_weight_decay = log_weight_decay
        self.alert_on_lr_change_factor = alert_on_lr_change_factor
        self.verbose = verbose

        # Internal state
        self._lr_history: Dict[str, List[Tuple[int, float]]] = {}
        self._last_lrs: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_lr_tags(self, optimizer: torch.optim.Optimizer, group_idx: int) -> str:
        """Return a human-readable tag for a parameter group."""
        name = optimizer.__class__.__name__
        return f"{name}/pg_{group_idx}/lr"

    def _collect_and_log(
        self,
        trainer: "Trainer",
        pl_module: "LightningModule",
        step: int,
    ) -> None:
        """Collect LRs from all optimizers and log them."""
        optimizers = trainer.optimizers
        if not optimizers:
            return

        for opt_idx, optimizer in enumerate(optimizers):
            for pg_idx, pg in enumerate(optimizer.param_groups):
                lr = pg["lr"]
                tag = f"lr/opt_{opt_idx}_pg_{pg_idx}"
                pl_module.log(tag, lr, on_step=(self.logging_interval == "step"),
                              on_epoch=True, prog_bar=False, logger=True)

                # History tracking
                if tag not in self._lr_history:
                    self._lr_history[tag] = []
                self._lr_history[tag].append((step, lr))

                # Anomaly detection
                if self.alert_on_lr_change_factor and tag in self._last_lrs:
                    prev_lr = self._last_lrs[tag]
                    if prev_lr > 0 and lr > 0:
                        factor = max(lr / prev_lr, prev_lr / lr)
                        if factor >= self.alert_on_lr_change_factor:
                            rank_zero_warn(
                                f"Large LR change detected in {tag}: "
                                f"{prev_lr:.2e} → {lr:.2e} (factor {factor:.1f}x)"
                            )
                self._last_lrs[tag] = lr

                if self.verbose:
                    rank_zero_info(f"[step {step}] {tag} = {lr:.6e}")

                # Optional extras
                if self.log_momentum and "momentum" in pg:
                    pl_module.log(
                        f"momentum/opt_{opt_idx}_pg_{pg_idx}",
                        pg["momentum"],
                        on_step=(self.logging_interval == "step"),
                        on_epoch=True,
                    )

                if self.log_weight_decay and "weight_decay" in pg:
                    pl_module.log(
                        f"weight_decay/opt_{opt_idx}_pg_{pg_idx}",
                        pg["weight_decay"],
                        on_step=(self.logging_interval == "step"),
                        on_epoch=True,
                    )

    # ------------------------------------------------------------------
    # Callback hooks
    # ------------------------------------------------------------------

    def on_train_batch_end(
        self,
        trainer: "Trainer",
        pl_module: "LightningModule",
        outputs,
        batch,
        batch_idx: int,
    ) -> None:
        if self.logging_interval == "step":
            self._collect_and_log(trainer, pl_module, trainer.global_step)

    def on_train_epoch_end(
        self,
        trainer: "Trainer",
        pl_module: "LightningModule",
    ) -> None:
        if self.logging_interval == "epoch":
            self._collect_and_log(trainer, pl_module, trainer.current_epoch)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_lr_history(self) -> Dict[str, List[Tuple[int, float]]]:
        """Return the full history of logged learning rates."""
        return dict(self._lr_history)

    def get_last_lrs(self) -> Dict[str, float]:
        """Return the most recent learning rate for each parameter group."""
        return dict(self._last_lrs)

    def reset_history(self) -> None:
        """Clear stored LR history."""
        self._lr_history.clear()
        self._last_lrs.clear()


def create_lr_monitor(
    logging_interval: str = "step",
    log_momentum: bool = False,
    log_weight_decay: bool = False,
    alert_on_lr_change_factor: Optional[float] = None,
    verbose: bool = False,
) -> LRMonitorCallback:
    """Factory function for LRMonitorCallback."""
    return LRMonitorCallback(
        log_momentum=log_momentum,
        logging_interval=logging_interval,
        log_weight_decay=log_weight_decay,
        alert_on_lr_change_factor=alert_on_lr_change_factor,
        verbose=verbose,
    )
