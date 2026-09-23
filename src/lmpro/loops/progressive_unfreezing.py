# src/lmpro/loops/progressive_unfreezing.py
"""
Progressive Unfreezing Loop for LightningMasterPro.

Implements gradual layer unfreezing (ULMFiT-style) as a training loop override.
Layers are frozen at the start and unfrozen in groups from the top (output side)
toward the bottom (input/embedding side) at configurable epoch intervals.
"""

import re
from typing import List, Optional, Tuple

import torch.nn as nn
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.utilities.rank_zero import rank_zero_info


class ProgressiveUnfreezingCallback(Callback):
    """
    Progressively unfreezes model layers during training.

    At a given start epoch, all parameters except those in `always_train_groups`
    are frozen. Every `unfreeze_every_n_epochs` epochs thereafter, the next
    group (deeper layers) is unfrozen.

    Layer groups are discovered automatically from the model's named children,
    or can be provided explicitly as lists of parameter name prefixes.

    Args:
        unfreeze_every_n_epochs: Unfreeze one layer group every N epochs.
        start_epoch: Epoch at which to begin progressive unfreezing (default: 0).
        layer_groups: Optional explicit list of parameter name prefix groups,
            ordered from output (first unfrozen) to input (last unfrozen).
            If None, groups are auto-detected from model's direct children.
        always_train_params: Regex patterns for parameters that are always trained
            (e.g., batch-norm layers). Default: batch-norm and bias params.
        lr_scale_factor: If set, discriminative learning rates are used: the
            parameters of group ``k`` (0 = output side, unfrozen first) are moved
            into their own optimizer parameter group with
            ``lr = base_lr * lr_scale_factor ** k`` when the group is unfrozen,
            so deeper (input-side) layers train with smaller learning rates.
            Note that LR schedulers created before the new parameter groups
            exist will not see them.
        verbose: Log unfreezing events to console.
    """

    def __init__(
        self,
        unfreeze_every_n_epochs: int = 1,
        start_epoch: int = 0,
        layer_groups: Optional[List[List[str]]] = None,
        always_train_params: Optional[List[str]] = None,
        lr_scale_factor: Optional[float] = None,
        verbose: bool = True,
    ):
        super().__init__()
        self.unfreeze_every_n_epochs = unfreeze_every_n_epochs
        self.start_epoch = start_epoch
        self.layer_groups = layer_groups
        self.always_train_params = always_train_params or [
            r".*\.bias$",
            r".*batch_norm.*",
            r".*bn\d+.*",
            r".*norm.*\.weight$",
        ]
        self.lr_scale_factor = lr_scale_factor
        self.verbose = verbose

        if lr_scale_factor is not None and lr_scale_factor <= 0:
            raise ValueError("lr_scale_factor must be > 0")

        self._resolved_groups: List[List[str]] = []
        self._unfrozen_group_idx: int = 0  # next group to unfreeze

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _always_train(self, name: str) -> bool:
        return any(re.match(pat, name) for pat in self.always_train_params)

    def _resolve_groups(self, model: nn.Module) -> List[List[str]]:
        """
        Return layer groups ordered output→input (first to be unfrozen first).
        """
        if self.layer_groups is not None:
            return self.layer_groups

        # Auto-detect: top-level children as groups, reversed so output first
        children = list(model.named_children())
        if not children:
            # Flat model — treat all params as one group
            return [[name for name, _ in model.named_parameters()]]

        groups: List[List[str]] = []
        for child_name, child_module in reversed(children):
            group = [f"{child_name}.{pname}" for pname, _ in child_module.named_parameters()]
            if group:
                groups.append(group)
        return groups

    def _set_requires_grad(
        self,
        model: nn.Module,
        param_names: List[str],
        requires_grad: bool,
    ) -> None:
        param_dict = dict(model.named_parameters())
        for name in param_names:
            if name in param_dict:
                param_dict[name].requires_grad = requires_grad

    def _freeze_all_except_always_train(self, model: nn.Module) -> None:
        for name, param in model.named_parameters():
            param.requires_grad = self._always_train(name)

    def _count_trainable(self, model: nn.Module) -> Tuple[int, int]:
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return trainable, total

    def _apply_discriminative_lr(
        self, trainer: "Trainer", pl_module: "LightningModule", param_names: List[str], group_idx: int
    ) -> None:
        """Move the group's params into a dedicated param group with a scaled LR."""
        if self.lr_scale_factor is None:
            return
        param_dict = dict(pl_module.named_parameters())
        group_params = [param_dict[n] for n in param_names if n in param_dict]
        group_ids = {id(p) for p in group_params}
        scale = self.lr_scale_factor**group_idx

        for optimizer in trainer.optimizers:
            moved = []
            base_lr = None
            for pg in optimizer.param_groups:
                keep = []
                for p in pg["params"]:
                    if id(p) in group_ids:
                        moved.append(p)
                        if base_lr is None:
                            base_lr = pg.get("initial_lr", pg["lr"])
                    else:
                        keep.append(p)
                pg["params"] = keep
            if not moved:
                continue
            optimizer.add_param_group({"params": moved, "lr": base_lr * scale})
            # Drop param groups that became empty after the move
            optimizer.param_groups[:] = [pg for pg in optimizer.param_groups if pg["params"]]
            if self.verbose:
                rank_zero_info(
                    f"ProgressiveUnfreezing: group {group_idx} lr = {base_lr * scale:.3e} "
                    f"(base {base_lr:.3e} x {self.lr_scale_factor}^{group_idx})"
                )

    # ------------------------------------------------------------------
    # Callback hooks
    # ------------------------------------------------------------------

    def on_fit_start(self, trainer: "Trainer", pl_module: "LightningModule") -> None:
        self._resolved_groups = self._resolve_groups(pl_module)
        self._unfrozen_group_idx = 0

        # Freeze all parameters initially
        self._freeze_all_except_always_train(pl_module)

        if self.verbose:
            rank_zero_info(
                f"ProgressiveUnfreezing: detected {len(self._resolved_groups)} "
                "layer groups. All frozen (except always-train params)."
            )
            trainable, total = self._count_trainable(pl_module)
            rank_zero_info(f"  Trainable params: {trainable:,} / {total:,}")

    def on_train_epoch_start(self, trainer: "Trainer", pl_module: "LightningModule") -> None:
        epoch = trainer.current_epoch

        if epoch < self.start_epoch:
            return

        epochs_since_start = epoch - self.start_epoch
        target_unfrozen = (epochs_since_start // self.unfreeze_every_n_epochs) + 1
        target_unfrozen = min(target_unfrozen, len(self._resolved_groups))

        while self._unfrozen_group_idx < target_unfrozen:
            group = self._resolved_groups[self._unfrozen_group_idx]
            self._set_requires_grad(pl_module, group, True)
            self._apply_discriminative_lr(trainer, pl_module, group, self._unfrozen_group_idx)

            if self.verbose:
                rank_zero_info(
                    f"[Epoch {epoch}] ProgressiveUnfreezing: unfroze group "
                    f"{self._unfrozen_group_idx} ({len(group)} params)"
                )

            self._unfrozen_group_idx += 1

        if self.verbose and (epochs_since_start % self.unfreeze_every_n_epochs == 0):
            trainable, total = self._count_trainable(pl_module)
            rank_zero_info(f"[Epoch {epoch}] Trainable: {trainable:,} / {total:,} params")

    def on_train_end(self, trainer: "Trainer", pl_module: "LightningModule") -> None:
        # Unfreeze all params at the end so the model is fully usable
        for param in pl_module.parameters():
            param.requires_grad = True

        if self.verbose:
            rank_zero_info("ProgressiveUnfreezing: all parameters unfrozen after training.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_resolved_groups(self) -> List[List[str]]:
        """Return the resolved layer groups (populated after on_fit_start)."""
        return list(self._resolved_groups)

    @property
    def unfrozen_group_count(self) -> int:
        """Number of groups that have been unfrozen so far."""
        return self._unfrozen_group_idx


def create_progressive_unfreezing(
    unfreeze_every_n_epochs: int = 1,
    start_epoch: int = 0,
    layer_groups: Optional[List[List[str]]] = None,
    lr_scale_factor: Optional[float] = None,
    verbose: bool = True,
) -> ProgressiveUnfreezingCallback:
    """Factory function for ProgressiveUnfreezingCallback."""
    return ProgressiveUnfreezingCallback(
        unfreeze_every_n_epochs=unfreeze_every_n_epochs,
        start_epoch=start_epoch,
        layer_groups=layer_groups,
        lr_scale_factor=lr_scale_factor,
        verbose=verbose,
    )
