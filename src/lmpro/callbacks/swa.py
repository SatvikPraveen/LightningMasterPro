# File: src/lmpro/callbacks/swa.py

"""
Stochastic Weight Averaging callback
"""

from typing import Any, Callable, Dict, List, Optional, Union

import torch
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.utilities.rank_zero import rank_zero_info, rank_zero_warn
from lightning.pytorch.utilities.types import LRSchedulerConfig
from torch.optim.swa_utils import SWALR

_BN_TYPES = (torch.nn.modules.batchnorm._BatchNorm,)


class SWACallback(Callback):
    """
    Stochastic Weight Averaging callback.

    From ``swa_epoch_start`` on, the model weights are averaged at the end of
    every epoch. When SWA starts, the user's LR scheduler(s) are *replaced* by a
    :class:`torch.optim.swa_utils.SWALR` (one per optimizer) that Lightning then
    steps once per epoch, mirroring Lightning's built-in
    ``StochasticWeightAveraging``. With ``annealing_strategy="constant"`` the LR
    is set to ``swa_lrs`` directly and no scheduler is installed.

    At the end of training the averaged weights are loaded into the module and
    BatchNorm running statistics are recomputed over the training data
    (``momentum=None`` cumulative average, as in
    :func:`torch.optim.swa_utils.update_bn`).

    The SWA state is persisted through the callback's ``state_dict`` and is
    NOT reset when resuming from a checkpoint.
    """

    def __init__(
        self,
        swa_lrs: Union[float, List[float]] = 1e-2,
        swa_epoch_start: Union[int, float] = 0.8,
        annealing_epochs: int = 10,
        annealing_strategy: str = "cos",
        avg_fn: Optional[Callable[[torch.Tensor, torch.Tensor, int], torch.Tensor]] = None,
        device: Optional[Union[torch.device, str]] = None,
        update_bn: bool = True,
        bn_update_batches: int = 100,
    ):
        super().__init__()
        if annealing_strategy not in ("cos", "linear", "constant"):
            raise ValueError("annealing_strategy must be 'cos', 'linear' or 'constant'")
        if isinstance(swa_epoch_start, float) and not 0.0 <= swa_epoch_start <= 1.0:
            raise ValueError("A float swa_epoch_start must be in [0, 1]")

        self.swa_lrs = list(swa_lrs) if isinstance(swa_lrs, (list, tuple)) else [swa_lrs]
        self.swa_epoch_start = swa_epoch_start
        self.annealing_epochs = annealing_epochs
        self.annealing_strategy = annealing_strategy
        self.avg_fn = avg_fn or self._default_avg_fn
        self.device = torch.device(device) if device is not None else None
        self.update_bn = update_bn
        self.bn_update_batches = bn_update_batches

        self.swa_model: Optional[Dict[str, torch.Tensor]] = None
        self.swa_n = 0
        self.original_model_state: Optional[Dict[str, torch.Tensor]] = None
        self._swa_epoch_start_absolute: Optional[int] = None
        self._scheduler_installed = False
        self._swa_schedulers: List[SWALR] = []

    @staticmethod
    def _default_avg_fn(averaged: torch.Tensor, current: torch.Tensor, num_averaged: int) -> torch.Tensor:
        """Running mean: avg + (x - avg) / (n + 1)."""
        return averaged + (current - averaged) / (num_averaged + 1)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @property
    def swa_start_epoch(self) -> Optional[int]:
        return self._swa_epoch_start_absolute

    def _resolve_start_epoch(self, trainer: Trainer) -> int:
        if isinstance(self.swa_epoch_start, float):
            max_epochs = trainer.max_epochs if trainer.max_epochs and trainer.max_epochs > 0 else 1
            return int(max_epochs * self.swa_epoch_start)
        return int(self.swa_epoch_start)

    def _clone_state(self, pl_module: LightningModule) -> Dict[str, torch.Tensor]:
        return {k: v.detach().clone().to(self.device) for k, v in pl_module.state_dict().items()}

    def _is_active(self, trainer: Trainer) -> bool:
        return self._swa_epoch_start_absolute is not None and trainer.current_epoch >= self._swa_epoch_start_absolute

    # ------------------------------------------------------------------
    # Callback hooks
    # ------------------------------------------------------------------

    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if self.device is None:
            self.device = pl_module.device
        self._swa_epoch_start_absolute = self._resolve_start_epoch(trainer)
        rank_zero_info(f"SWA will start at epoch {self._swa_epoch_start_absolute}")

        if self.swa_model is None:
            # Fresh run: seed the average with the current weights (n=0, so the
            # first real update overwrites it).
            self.swa_model = self._clone_state(pl_module)
            self.swa_n = 0
        else:
            # Restored from checkpoint: keep the running average, just relocate it.
            self.swa_model = {k: v.to(self.device) for k, v in self.swa_model.items()}

    def on_train_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if self._is_active(trainer) and not self._scheduler_installed:
            self._install_swa_schedulers(trainer)

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if self._is_active(trainer):
            self._update_swa_model(pl_module)

    @torch.no_grad()
    def _update_swa_model(self, pl_module: LightningModule) -> None:
        for key, value in pl_module.state_dict().items():
            value = value.detach().to(self.device)
            if key not in self.swa_model:
                self.swa_model[key] = value.clone()
            elif value.is_floating_point():
                self.swa_model[key] = self.avg_fn(self.swa_model[key], value, self.swa_n)
            else:
                self.swa_model[key] = value.clone()
        self.swa_n += 1
        if self.swa_n % 10 == 0:
            rank_zero_info(f"Updated SWA model (n={self.swa_n})")

    def _install_swa_schedulers(self, trainer: Trainer) -> None:
        """Replace the user's LR scheduler(s) with SWALR (or a constant LR)."""
        self._scheduler_installed = True
        self._swa_schedulers = []
        configs = trainer.lr_scheduler_configs

        if configs:
            names = ", ".join(type(cfg.scheduler).__name__ for cfg in configs)
            rank_zero_info(f"SWA: replacing LR scheduler(s) [{names}] for the SWA phase")
        configs.clear()

        for i, optimizer in enumerate(trainer.optimizers):
            swa_lr = self.swa_lrs[min(i, len(self.swa_lrs) - 1)]
            if self.annealing_strategy == "constant":
                for group in optimizer.param_groups:
                    group["lr"] = swa_lr
                continue
            scheduler = SWALR(
                optimizer,
                swa_lr=swa_lr,
                anneal_epochs=self.annealing_epochs,
                anneal_strategy="cos" if self.annealing_strategy == "cos" else "linear",
            )
            self._swa_schedulers.append(scheduler)
            configs.append(LRSchedulerConfig(scheduler, interval="epoch", frequency=1))

        rank_zero_info(f"SWA: schedulers installed with LR {self.swa_lrs} ({self.annealing_strategy})")

    def on_train_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if self.swa_model is None or self.swa_n == 0:
            return
        self.apply_swa_weights(pl_module)
        rank_zero_info(f"Applied SWA weights (averaged over {self.swa_n} epochs)")
        if self.update_bn:
            self._update_bn_statistics(trainer, pl_module)

    @torch.no_grad()
    def _update_bn_statistics(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Recompute BatchNorm running stats like ``torch.optim.swa_utils.update_bn``."""
        bn_modules = [m for m in pl_module.modules() if isinstance(m, _BN_TYPES)]
        if not bn_modules:
            return

        try:
            train_dataloader = trainer.train_dataloader
        except Exception:
            train_dataloader = None
        if train_dataloader is None:
            rank_zero_warn("SWA: no train dataloader available; skipping BN statistics update")
            return

        momenta = {}
        for module in bn_modules:
            module.reset_running_stats()
            momenta[module] = module.momentum
            module.momentum = None  # cumulative moving average

        was_training = pl_module.training
        pl_module.train()
        try:
            for batch_idx, batch in enumerate(train_dataloader):
                if batch_idx >= self.bn_update_batches:
                    break
                x = batch[0] if isinstance(batch, (list, tuple)) else batch
                if isinstance(x, torch.Tensor):
                    x = x.to(pl_module.device)
                pl_module(x)
            rank_zero_info("SWA: updated BatchNorm statistics")
        except Exception as exc:  # keep training results even if BN update fails
            rank_zero_warn(f"SWA: could not update BN statistics: {exc}")
        finally:
            for module, momentum in momenta.items():
                module.momentum = momentum
            pl_module.train(was_training)

    # ------------------------------------------------------------------
    # Weight swapping for evaluation
    # ------------------------------------------------------------------

    def _swap_in_swa(self, pl_module: LightningModule) -> None:
        if self.swa_model is None or self.swa_n == 0 or self.original_model_state is not None:
            return
        self.original_model_state = {k: v.detach().clone() for k, v in pl_module.state_dict().items()}
        pl_module.load_state_dict(self.swa_model, strict=True)

    def _swap_out_swa(self, pl_module: LightningModule) -> None:
        if self.original_model_state is None:
            return
        pl_module.load_state_dict(self.original_model_state, strict=True)
        self.original_model_state = None

    def on_validation_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if self._is_active(trainer):
            self._swap_in_swa(pl_module)

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self._swap_out_swa(pl_module)

    def on_test_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self._swap_in_swa(pl_module)

    def on_test_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self._swap_out_swa(pl_module)

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def get_swa_model(self) -> Optional[Dict[str, torch.Tensor]]:
        return self.swa_model

    def apply_swa_weights(self, pl_module: LightningModule) -> None:
        """Permanently load the averaged weights into ``pl_module``."""
        if self.swa_model is not None:
            pl_module.load_state_dict(self.swa_model, strict=True)

    # ------------------------------------------------------------------
    # Persistence (single source of truth: the callback state dict)
    # ------------------------------------------------------------------

    def state_dict(self) -> Dict[str, Any]:
        return {
            "swa_model": None if self.swa_model is None else {k: v.cpu() for k, v in self.swa_model.items()},
            "swa_n": self.swa_n,
            "swa_epoch_start_absolute": self._swa_epoch_start_absolute,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        swa = state_dict.get("swa_model")
        self.swa_model = None if swa is None else {k: v.clone() for k, v in swa.items()}
        self.swa_n = state_dict.get("swa_n", 0)
        self._swa_epoch_start_absolute = state_dict.get("swa_epoch_start_absolute")
        # Schedulers are not restored by Lightning for configs that did not exist
        # at checkpoint time, so re-install them on the next epoch start.
        self._scheduler_installed = False


def create_swa_callback(
    swa_lr: float = 1e-2,
    swa_epoch_start: Union[int, float] = 0.8,
    annealing_epochs: int = 10,
    **kwargs: Any,
) -> SWACallback:
    """Create SWA callback with common settings"""
    return SWACallback(swa_lrs=swa_lr, swa_epoch_start=swa_epoch_start, annealing_epochs=annealing_epochs, **kwargs)
