# File: src/lmpro/callbacks/checkpoints.py

"""
Enhanced model checkpoint callback with additional features
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.utilities.rank_zero import rank_zero_info, rank_zero_warn

SIDECAR_SUFFIXES = ("_architecture.txt", "_hparams.yaml", "_optimizer.pth")


class EnhancedModelCheckpoint(ModelCheckpoint):
    """
    Enhanced ModelCheckpoint with additional features:
    - Save model architecture
    - Save training hyperparameters
    - Save optimizer/scheduler state as a sidecar file
    - Automatic cleanup of old checkpoints (``max_checkpoints_to_keep``)

    All other keyword arguments are forwarded to
    :class:`lightning.pytorch.callbacks.ModelCheckpoint`.
    """

    def __init__(
        self,
        dirpath: Optional[str] = None,
        filename: Optional[str] = None,
        monitor: Optional[str] = None,
        verbose: bool = False,
        save_last: Optional[bool] = None,
        save_top_k: int = 1,
        save_weights_only: bool = False,
        mode: str = "min",
        auto_insert_metric_name: bool = True,
        every_n_train_steps: Optional[int] = None,
        train_time_interval: Optional[Any] = None,
        every_n_epochs: Optional[int] = None,
        save_on_train_epoch_end: Optional[bool] = None,
        save_architecture: bool = True,
        save_hyperparameters: bool = True,
        save_optimizer_state: bool = True,
        max_checkpoints_to_keep: Optional[int] = None,
        **kwargs: Any,
    ):
        super().__init__(
            dirpath=dirpath,
            filename=filename,
            monitor=monitor,
            verbose=verbose,
            save_last=save_last,
            save_top_k=save_top_k,
            save_weights_only=save_weights_only,
            mode=mode,
            auto_insert_metric_name=auto_insert_metric_name,
            every_n_train_steps=every_n_train_steps,
            train_time_interval=train_time_interval,
            every_n_epochs=every_n_epochs,
            save_on_train_epoch_end=save_on_train_epoch_end,
            **kwargs,
        )
        if max_checkpoints_to_keep is not None and max_checkpoints_to_keep < 1:
            raise ValueError("max_checkpoints_to_keep must be >= 1 or None")

        self.save_architecture = save_architecture
        self.save_hyperparameters = save_hyperparameters
        self.save_optimizer_state = save_optimizer_state
        self.max_checkpoints_to_keep = max_checkpoints_to_keep

        # Checkpoint files written by this callback that still exist on disk.
        self.saved_checkpoints: List[str] = []

    # ------------------------------------------------------------------
    # Save / remove (kept in sync with the parent's bookkeeping)
    # ------------------------------------------------------------------

    def _save_checkpoint(self, trainer: Trainer, filepath: str) -> None:
        super()._save_checkpoint(trainer, filepath)

        if filepath not in self.saved_checkpoints:
            self.saved_checkpoints.append(filepath)

        if trainer.is_global_zero and (
            self.save_architecture or self.save_hyperparameters or self.save_optimizer_state
        ):
            self._save_additional_artifacts(trainer, filepath)

        if self.max_checkpoints_to_keep is not None:
            self._cleanup_old_checkpoints(trainer)

        rank_zero_info(f"Enhanced checkpoint saved: {filepath}")

    def _remove_checkpoint(self, trainer: Trainer, filepath: str) -> None:
        """Remove a checkpoint, its sidecar files and our own tracking entry."""
        super()._remove_checkpoint(trainer, filepath)
        if trainer.is_global_zero:
            self._remove_sidecar_files(filepath)
        if filepath in self.saved_checkpoints:
            self.saved_checkpoints.remove(filepath)

    @staticmethod
    def _sidecar_paths(filepath: str) -> List[Path]:
        base = Path(filepath)
        return [base.parent / f"{base.stem}{suffix}" for suffix in SIDECAR_SUFFIXES]

    def _remove_sidecar_files(self, filepath: str) -> None:
        for sidecar in self._sidecar_paths(filepath):
            try:
                if sidecar.exists():
                    sidecar.unlink()
            except OSError as exc:
                rank_zero_warn(f"Failed to remove {sidecar}: {exc}")

    def _forget_checkpoint(self, filepath: str) -> None:
        """Drop ``filepath`` from the parent's top-k bookkeeping."""
        if filepath in self.best_k_models:
            self.best_k_models.pop(filepath)
            if self.best_k_models:
                op = max if self.mode == "min" else min
                self.kth_best_model_path = op(self.best_k_models, key=self.best_k_models.get)
                self.kth_value = self.best_k_models[self.kth_best_model_path]
                op_best = min if self.mode == "min" else max
                self.best_model_path = op_best(self.best_k_models, key=self.best_k_models.get)
                self.best_model_score = self.best_k_models[self.best_model_path]
            else:
                self.kth_best_model_path = ""
                self.kth_value = None
                if self.best_model_path == filepath:
                    self.best_model_path = ""
                    self.best_model_score = None

    def _cleanup_old_checkpoints(self, trainer: Trainer) -> None:
        """Remove the oldest checkpoints beyond ``max_checkpoints_to_keep``."""
        # Drop entries the parent (or anyone else) already deleted.
        self.saved_checkpoints = [p for p in self.saved_checkpoints if os.path.exists(p)]
        # The "last" checkpoint is always present and does not count against the limit.
        counted = [p for p in self.saved_checkpoints if p != self.last_model_path]
        if len(counted) <= self.max_checkpoints_to_keep:
            return

        candidates = sorted(
            (p for p in counted if p != self.best_model_path),
            key=lambda p: os.path.getmtime(p) if os.path.exists(p) else 0.0,
        )
        excess = len(counted) - self.max_checkpoints_to_keep
        for path in candidates[:excess]:
            try:
                self._forget_checkpoint(path)
                self._remove_checkpoint(trainer, path)
                rank_zero_info(f"Removed old checkpoint: {path}")
            except Exception as exc:
                rank_zero_warn(f"Failed to remove checkpoint {path}: {exc}")

    # ------------------------------------------------------------------
    # Sidecar artifacts
    # ------------------------------------------------------------------

    def _save_additional_artifacts(self, trainer: Trainer, filepath: str) -> None:
        base_path = Path(filepath).parent
        base_name = Path(filepath).stem
        pl_module = trainer.lightning_module

        if self.save_architecture:
            arch_path = base_path / f"{base_name}_architecture.txt"
            try:
                total_params = sum(p.numel() for p in pl_module.parameters())
                trainable_params = sum(p.numel() for p in pl_module.parameters() if p.requires_grad)
                with open(arch_path, "w") as f:
                    f.write(str(pl_module))
                    f.write("\n\n" + "=" * 50 + "\n")
                    f.write("Model Summary:\n")
                    f.write(f"Total parameters: {total_params:,}\n")
                    f.write(f"Trainable parameters: {trainable_params:,}\n")
                    f.write(f"Non-trainable parameters: {total_params - trainable_params:,}\n")
            except Exception as exc:
                rank_zero_warn(f"Failed to save model architecture: {exc}")

        if self.save_hyperparameters:
            hparams_path = base_path / f"{base_name}_hparams.yaml"
            try:
                import yaml

                hparams = dict(pl_module.hparams)
                hparams["training_info"] = {
                    "current_epoch": trainer.current_epoch,
                    "global_step": trainer.global_step,
                    "total_epochs": trainer.max_epochs,
                }
                hparams["system_info"] = {
                    "torch_version": str(torch.__version__),
                    "cuda_available": torch.cuda.is_available(),
                    "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
                }
                with open(hparams_path, "w") as f:
                    yaml.dump(hparams, f, default_flow_style=False, indent=2)
            except Exception as exc:
                rank_zero_warn(f"Failed to save hyperparameters: {exc}")

        if self.save_optimizer_state and not self.save_weights_only:
            opt_path = base_path / f"{base_name}_optimizer.pth"
            try:
                torch.save(
                    {
                        "optimizers": {f"optimizer_{i}": o.state_dict() for i, o in enumerate(trainer.optimizers)},
                        "schedulers": {
                            f"scheduler_{i}": cfg.scheduler.state_dict()
                            for i, cfg in enumerate(trainer.lr_scheduler_configs)
                        },
                        "epoch": trainer.current_epoch,
                        "global_step": trainer.global_step,
                    },
                    opt_path,
                )
            except Exception as exc:
                rank_zero_warn(f"Failed to save optimizer state: {exc}")

    # ------------------------------------------------------------------
    # Loading / info
    # ------------------------------------------------------------------

    def load_checkpoint_with_metadata(self, checkpoint_path: str) -> Dict[str, Any]:
        """Load a checkpoint together with its sidecar metadata."""
        base_path = Path(checkpoint_path).parent
        base_name = Path(checkpoint_path).stem

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        metadata: Dict[str, Any] = {}

        arch_path = base_path / f"{base_name}_architecture.txt"
        if arch_path.exists():
            metadata["architecture"] = arch_path.read_text()

        hparams_path = base_path / f"{base_name}_hparams.yaml"
        if hparams_path.exists():
            try:
                import yaml

                with open(hparams_path, "r") as f:
                    metadata["hyperparameters"] = yaml.safe_load(f)
            except Exception as exc:
                rank_zero_warn(f"Failed to load hyperparameters: {exc}")

        opt_path = base_path / f"{base_name}_optimizer.pth"
        if opt_path.exists():
            try:
                metadata["optimizer_state"] = torch.load(opt_path, map_location="cpu", weights_only=False)
            except Exception as exc:
                rank_zero_warn(f"Failed to load optimizer state: {exc}")

        return {"checkpoint": checkpoint, "metadata": metadata}

    def get_checkpoint_info(self) -> Dict[str, Any]:
        existing = [p for p in self.saved_checkpoints if os.path.exists(p)]
        info: Dict[str, Any] = {
            "total_checkpoints": len(existing),
            "checkpoint_dir": self.dirpath,
            "best_model_path": self.best_model_path,
            "last_model_path": self.last_model_path,
            "monitor": self.monitor,
            "mode": self.mode,
        }
        if existing:
            info["oldest_checkpoint"] = min(existing, key=os.path.getmtime)
            info["newest_checkpoint"] = max(existing, key=os.path.getmtime)
        return info

    def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        super().on_train_start(trainer, pl_module)
        if self.dirpath and trainer.is_global_zero:
            Path(self.dirpath).mkdir(parents=True, exist_ok=True)
            info_path = Path(self.dirpath) / "checkpoint_info.txt"
            with open(info_path, "w") as f:
                f.write("Enhanced Model Checkpoint Info\n")
                f.write("=" * 40 + "\n\n")
                f.write(f"Monitor: {self.monitor}\n")
                f.write(f"Mode: {self.mode}\n")
                f.write(f"Save top k: {self.save_top_k}\n")
                f.write(f"Save last: {self.save_last}\n")
                f.write(f"Save weights only: {self.save_weights_only}\n")
                f.write(f"Max checkpoints to keep: {self.max_checkpoints_to_keep}\n")
                f.write(f"Save architecture: {self.save_architecture}\n")
                f.write(f"Save hyperparameters: {self.save_hyperparameters}\n")

    def on_train_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        super().on_train_end(trainer, pl_module)
        if self.dirpath and trainer.is_global_zero:
            info_path = Path(self.dirpath) / "final_checkpoint_info.yaml"
            try:
                import yaml

                with open(info_path, "w") as f:
                    yaml.dump(self.get_checkpoint_info(), f, default_flow_style=False, indent=2)
            except Exception as exc:
                rank_zero_warn(f"Failed to save final checkpoint info: {exc}")
        rank_zero_info(f"Training completed. Best checkpoint: {self.best_model_path}")


# Convenience functions for common checkpoint configurations
def get_best_checkpoint_callback(
    monitor: str = "val/loss",
    mode: str = "min",
    save_top_k: int = 1,
    dirpath: str = "checkpoints/best",
    **kwargs: Any,
) -> EnhancedModelCheckpoint:
    """
    Checkpoint callback that keeps the best model(s) by ``monitor``.

    The filename references the real metric key (``{val/loss:.4f}``) so the
    value is interpolated correctly, while the human-readable label replaces
    ``/`` with ``_`` so no sub-directories are created.
    """
    label = monitor.replace("/", "_")
    return EnhancedModelCheckpoint(
        dirpath=dirpath,
        filename=f"best-epoch={{epoch:02d}}-{label}={{{monitor}:.4f}}",
        auto_insert_metric_name=False,
        monitor=monitor,
        mode=mode,
        save_top_k=save_top_k,
        save_last=False,
        **kwargs,
    )


def get_periodic_checkpoint_callback(
    every_n_epochs: int = 5,
    save_top_k: int = -1,
    dirpath: str = "checkpoints/periodic",
    max_checkpoints_to_keep: Optional[int] = 3,
    **kwargs: Any,
) -> EnhancedModelCheckpoint:
    """
    Checkpoint callback for periodic saving.

    Without a ``monitor`` Lightning only accepts ``save_top_k`` in ``{-1, 0, 1}``,
    so every periodic checkpoint is kept by Lightning and pruned to the newest
    ``max_checkpoints_to_keep`` by this callback.
    """
    if kwargs.get("monitor") is None and save_top_k > 1:
        save_top_k = -1
    return EnhancedModelCheckpoint(
        dirpath=dirpath,
        filename="epoch-{epoch:02d}",
        auto_insert_metric_name=False,
        every_n_epochs=every_n_epochs,
        save_top_k=save_top_k,
        save_last=True,
        max_checkpoints_to_keep=max_checkpoints_to_keep,
        **kwargs,
    )


def get_last_checkpoint_callback(dirpath: str = "checkpoints/last", **kwargs: Any) -> EnhancedModelCheckpoint:
    """Checkpoint callback that only keeps the last model."""
    return EnhancedModelCheckpoint(dirpath=dirpath, filename="last", save_last=True, save_top_k=0, **kwargs)
