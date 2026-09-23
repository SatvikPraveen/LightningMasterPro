# File: src/lmpro/loops/curriculum_loop.py

"""
Curriculum learning for progressive training difficulty.

Lightning 2.x has no public ``Loop`` API, so curriculum learning is implemented
as a ``Callback`` (``CurriculumLoop``) working together with a thin dataset
wrapper (``CurriculumDataset``):

* ``CurriculumDataset`` holds per-sample difficulty scores and a difficulty
  threshold; it only exposes samples whose score is ``<= threshold``.
* ``CurriculumLoop`` computes the scores with a ``CurriculumStrategy``, and at
  the end of every epoch moves the threshold along the strategy's schedule.

Because the dataset length changes between epochs, the ``Trainer`` must be
created with ``reload_dataloaders_every_n_epochs=1`` so the train dataloader
(and its sampler) is rebuilt each epoch.

Example::

    train_ds = CurriculumDataset(base_dataset)
    curriculum = CurriculumLoop(strategy="length", dataset=train_ds)
    trainer = Trainer(max_epochs=10, reload_dataloaders_every_n_epochs=1, callbacks=[curriculum])
    trainer.fit(model, DataLoader(train_ds, batch_size=32, shuffle=True))
"""

import math
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Sized, Tuple, Union, cast

import numpy as np
import torch
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.utilities.rank_zero import rank_zero_info, rank_zero_warn
from torch.utils.data import DataLoader, Dataset


def _dataset_len(dataset: Dataset) -> int:
    """Length of a map-style dataset (``Dataset`` does not declare ``__len__``)."""
    return len(cast(Sized, dataset))


class CurriculumStrategy(ABC):
    """Abstract base class for curriculum learning strategies."""

    @abstractmethod
    def get_difficulty_scores(
        self, dataset: Dataset, model: Optional[LightningModule] = None, recompute: bool = False
    ) -> np.ndarray:
        """Return one difficulty score in ``[0, 1]`` per sample (0 = easiest)."""

    @abstractmethod
    def get_curriculum_schedule(self, total_epochs: int, dataset_size: int) -> List[Tuple[int, float]]:
        """Return the curriculum schedule as ``[(epoch, difficulty_threshold), ...]``."""


def _normalize(values: np.ndarray, constant_fill: float = 1.0) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.size and values.max() > values.min():
        return (values - values.min()) / (values.max() - values.min())
    return np.full_like(values, constant_fill, dtype=np.float64)


class LengthBasedCurriculum(CurriculumStrategy):
    """Curriculum based on input sequence/sample length (shorter = easier)."""

    def __init__(self, reverse: bool = False):
        self.reverse = reverse  # If True, start with longer samples

    def get_difficulty_scores(
        self, dataset: Dataset, model: Optional[LightningModule] = None, recompute: bool = False
    ) -> np.ndarray:
        lengths = []
        for i in range(_dataset_len(dataset)):
            sample = dataset[i]
            x = sample[0] if isinstance(sample, (tuple, list)) else sample
            if isinstance(x, torch.Tensor):
                length = x.shape[0] if x.dim() >= 1 else 1
            elif hasattr(x, "__len__"):
                length = len(x)
            else:
                length = 1
            lengths.append(length)

        scores = _normalize(np.asarray(lengths))
        return 1.0 - scores if self.reverse else scores

    def get_curriculum_schedule(self, total_epochs: int, dataset_size: int) -> List[Tuple[int, float]]:
        """Linear progression of the threshold from 0.1 to 1.0."""
        return [(epoch, 0.1 + 0.9 * epoch / max(1, total_epochs - 1)) for epoch in range(total_epochs)]


class LossBasedCurriculum(CurriculumStrategy):
    """
    Curriculum based on per-sample loss under the current model (low loss = easy).

    Scores are cached after the first model-based computation; pass
    ``recompute=True`` (or call :meth:`reset`) to compute them again with the
    current model.
    """

    def __init__(self, warmup_epochs: int = 5):
        self.warmup_epochs = warmup_epochs
        self.sample_losses: Optional[np.ndarray] = None

    def reset(self) -> None:
        self.sample_losses = None

    def get_difficulty_scores(
        self, dataset: Dataset, model: Optional[LightningModule] = None, recompute: bool = False
    ) -> np.ndarray:
        if self.sample_losses is not None and not recompute:
            return self.sample_losses

        if model is None:
            # No model yet: random scores, deliberately NOT cached so a later
            # model-based call replaces them.
            return np.random.random(_dataset_len(dataset))

        was_training = model.training
        model.eval()
        losses = []
        with torch.no_grad():
            for i in range(_dataset_len(dataset)):
                sample = dataset[i]
                if isinstance(sample, (tuple, list)) and len(sample) >= 2:
                    x, y = sample[0], sample[1]
                else:
                    x, y = sample, None
                losses.append(self._sample_loss(model, x, y))
        model.train(was_training)

        self.sample_losses = _normalize(np.asarray(losses), constant_fill=0.5)
        return self.sample_losses

    @staticmethod
    def _sample_loss(model: LightningModule, x: Any, y: Any) -> float:
        device = model.device
        if isinstance(x, torch.Tensor):
            x = x.unsqueeze(0).to(device)
        if isinstance(y, torch.Tensor):
            y = y.unsqueeze(0).to(device)

        criterion = getattr(model, "criterion", None) or getattr(model, "loss_fn", None)
        try:
            if criterion is not None and y is not None:
                loss = criterion(model(x), y)
            else:
                batch = (x, y) if y is not None else x
                out = model.training_step(batch, 0)
                loss = out["loss"] if isinstance(out, dict) else out
            return float(loss.detach().item())
        except Exception:
            return 1.0  # treat un-scorable samples as hard

    def get_curriculum_schedule(self, total_epochs: int, dataset_size: int) -> List[Tuple[int, float]]:
        """Constant 0.3 during warm-up, then exponential approach to 1.0."""
        schedule = []
        for epoch in range(total_epochs):
            if epoch < self.warmup_epochs:
                threshold = 0.3
            else:
                progress = (epoch - self.warmup_epochs) / max(1, total_epochs - self.warmup_epochs - 1)
                threshold = 0.3 + 0.7 * (1 - math.exp(-3 * progress))
            schedule.append((epoch, min(threshold, 1.0)))
        return schedule


class RandomCurriculum(CurriculumStrategy):
    """Random curriculum (baseline): gradually increase the dataset size."""

    def get_difficulty_scores(
        self, dataset: Dataset, model: Optional[LightningModule] = None, recompute: bool = False
    ) -> np.ndarray:
        return np.random.random(_dataset_len(dataset))

    def get_curriculum_schedule(self, total_epochs: int, dataset_size: int) -> List[Tuple[int, float]]:
        return [(epoch, 0.2 + 0.8 * epoch / max(1, total_epochs - 1)) for epoch in range(total_epochs)]


class CurriculumDataset(Dataset):
    """
    Dataset view exposing only samples whose difficulty is below a threshold.

    Wrap your training dataset with this class and build the train ``DataLoader``
    from it; :class:`CurriculumLoop` updates ``difficulty_scores`` and the
    ``threshold`` between epochs.
    """

    def __init__(
        self,
        dataset: Dataset,
        difficulty_scores: Optional[np.ndarray] = None,
        threshold: float = 1.0,
        min_samples: int = 1,
    ):
        self.dataset = dataset
        self.min_samples = max(1, min(min_samples, _dataset_len(dataset)))
        self.threshold = threshold
        self.difficulty_scores: Optional[np.ndarray] = None
        self._active_indices = np.arange(_dataset_len(dataset))
        if difficulty_scores is not None:
            self.set_difficulty_scores(difficulty_scores)

    def set_difficulty_scores(self, scores: np.ndarray) -> None:
        scores = np.asarray(scores, dtype=np.float64)
        if scores.shape != (_dataset_len(self.dataset),):
            raise ValueError(f"Expected {_dataset_len(self.dataset)} scores, got shape {scores.shape}")
        self.difficulty_scores = scores
        self.set_threshold(self.threshold)

    def set_threshold(self, threshold: float) -> None:
        self.threshold = float(threshold)
        if self.difficulty_scores is None:
            self._active_indices = np.arange(_dataset_len(self.dataset))
            return
        selected = np.flatnonzero(self.difficulty_scores <= self.threshold)
        if len(selected) < self.min_samples:
            selected = np.argsort(self.difficulty_scores, kind="stable")[: self.min_samples]
        self._active_indices = np.sort(selected)

    @property
    def active_indices(self) -> np.ndarray:
        return self._active_indices

    @property
    def total_size(self) -> int:
        return _dataset_len(self.dataset)

    def __len__(self) -> int:
        return int(len(self._active_indices))

    def __getitem__(self, index: int) -> Any:
        return self.dataset[int(self._active_indices[index])]


class CurriculumLoop(Callback):
    """
    Callback driving curriculum learning over a :class:`CurriculumDataset`.

    The dataset can be passed explicitly (``dataset=``) or is discovered from the
    training dataloader at ``on_fit_start``. Difficulty scores are computed once
    at fit start and, if ``recompute_difficulty`` is set, recomputed with the
    current model every ``update_frequency`` epochs. The threshold for epoch
    ``e + 1`` is applied at the end of epoch ``e`` so that the dataloader rebuilt
    by ``reload_dataloaders_every_n_epochs=1`` sees the new subset.
    """

    def __init__(
        self,
        strategy: Union[str, CurriculumStrategy] = "length",
        update_frequency: int = 5,
        min_samples_per_epoch: Optional[int] = None,
        recompute_difficulty: bool = True,
        curriculum_warmup: int = 0,
        dataset: Optional[CurriculumDataset] = None,
    ):
        super().__init__()

        if isinstance(strategy, str):
            strategies = {
                "length": LengthBasedCurriculum,
                "loss": LossBasedCurriculum,
                "random": RandomCurriculum,
            }
            if strategy not in strategies:
                raise ValueError(f"Unknown curriculum strategy: {strategy}")
            self.strategy: CurriculumStrategy = strategies[strategy]()
        else:
            self.strategy = strategy

        self.update_frequency = max(1, update_frequency)
        self.min_samples_per_epoch = min_samples_per_epoch
        self.recompute_difficulty = recompute_difficulty
        self.curriculum_warmup = curriculum_warmup
        self.dataset = dataset

        # Internal state
        self.difficulty_scores: Optional[np.ndarray] = None
        self.curriculum_schedule: Optional[List[Tuple[int, float]]] = None
        self.current_epoch = 0
        self.current_threshold: Optional[float] = None
        self.curriculum_stats: List[Dict[str, Any]] = []
        self._total_epochs: int = 0

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def wrap(self, dataset: Dataset) -> CurriculumDataset:
        """Wrap ``dataset`` in a :class:`CurriculumDataset` and register it."""
        self.dataset = dataset if isinstance(dataset, CurriculumDataset) else CurriculumDataset(dataset)
        return self.dataset

    @staticmethod
    def _find_curriculum_dataset(obj: Any) -> Optional[CurriculumDataset]:
        """Walk ``DataLoader`` / ``Subset`` wrappers looking for a CurriculumDataset."""
        seen = 0
        while obj is not None and seen < 10:
            if isinstance(obj, CurriculumDataset):
                return obj
            if isinstance(obj, (list, tuple)):
                for item in obj:
                    found = CurriculumLoop._find_curriculum_dataset(item)
                    if found is not None:
                        return found
                return None
            if isinstance(obj, dict):
                return CurriculumLoop._find_curriculum_dataset(list(obj.values()))
            obj = getattr(obj, "dataset", None)
            seen += 1
        return None

    def _discover_dataset(self, trainer: Trainer) -> Optional[CurriculumDataset]:
        if self.dataset is not None:
            return self.dataset
        try:
            source = trainer.fit_loop._data_source
            loaders = source.dataloader() if source.is_defined() else None
        except Exception:
            loaders = None
        self.dataset = self._find_curriculum_dataset(loaders)
        return self.dataset

    def _compute_scores(self, model: Optional[LightningModule], recompute: bool) -> np.ndarray:
        base = self.dataset.dataset
        try:
            scores = self.strategy.get_difficulty_scores(base, model, recompute=recompute)
        except TypeError:  # user strategy without the ``recompute`` argument
            scores = self.strategy.get_difficulty_scores(base, model)
        return np.asarray(scores, dtype=np.float64)

    def _threshold_for_epoch(self, epoch: int) -> float:
        if epoch < self.curriculum_warmup:
            return 1.0
        idx = epoch - self.curriculum_warmup
        if self.curriculum_schedule and idx < len(self.curriculum_schedule):
            return float(self.curriculum_schedule[idx][1])
        return 1.0

    def _apply_epoch(self, epoch: int) -> None:
        threshold = self._threshold_for_epoch(epoch)
        self.dataset.set_threshold(threshold)
        self.current_threshold = threshold
        self.current_epoch = epoch

        num_samples = len(self.dataset)
        total = self.dataset.total_size
        stats = {
            "epoch": epoch,
            "difficulty_threshold": threshold,
            "num_samples": num_samples,
            "total_samples": total,
            "percentage": 100.0 * num_samples / max(1, total),
        }
        self.curriculum_stats.append(stats)
        rank_zero_info(
            f"Epoch {epoch}: curriculum threshold={threshold:.3f}, "
            f"samples={num_samples}/{total} ({stats['percentage']:.1f}%)"
        )

    # ------------------------------------------------------------------
    # Callback hooks
    # ------------------------------------------------------------------

    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if self._discover_dataset(trainer) is None:
            rank_zero_warn(
                "CurriculumLoop: no CurriculumDataset found in the training data; "
                "wrap your dataset with CurriculumDataset (or pass dataset=). Curriculum disabled."
            )
            return

        # Set by the DataConnector at runtime rather than declared on ``Trainer``.
        if getattr(trainer, "reload_dataloaders_every_n_epochs", None) != 1:
            rank_zero_warn(
                "CurriculumLoop requires Trainer(reload_dataloaders_every_n_epochs=1) so the "
                "train dataloader is rebuilt with the new subset every epoch."
            )

        max_epochs = trainer.max_epochs if trainer.max_epochs and trainer.max_epochs > 0 else 10
        self._total_epochs = max_epochs
        total = self.dataset.total_size
        if self.min_samples_per_epoch is None:
            self.min_samples_per_epoch = max(1, min(total, max(32, total // 20)))
        self.dataset.min_samples = max(1, min(self.min_samples_per_epoch, total))

        self.difficulty_scores = self._compute_scores(None, recompute=False)
        self.dataset.set_difficulty_scores(self.difficulty_scores)
        self.curriculum_schedule = self.strategy.get_curriculum_schedule(max_epochs, total)
        self.curriculum_stats = []

        rank_zero_info(
            f"Curriculum learning: {type(self.strategy).__name__}, dataset size={total}, "
            f"min samples/epoch={self.dataset.min_samples}"
        )
        self._apply_epoch(trainer.current_epoch)

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if self.dataset is None or self.dataset.difficulty_scores is None:
            return
        next_epoch = trainer.current_epoch + 1
        if trainer.max_epochs is not None and trainer.max_epochs > 0 and next_epoch >= trainer.max_epochs:
            return

        if self.recompute_difficulty and next_epoch % self.update_frequency == 0:
            rank_zero_info("Recomputing curriculum difficulty scores...")
            self.difficulty_scores = self._compute_scores(pl_module, recompute=True)
            self.dataset.set_difficulty_scores(self.difficulty_scores)

        self._apply_epoch(next_epoch)

    def on_fit_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if self.dataset is not None:
            self.dataset.set_threshold(1.0)  # expose the full dataset again
        self._log_final_stats()

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def _log_final_stats(self) -> None:
        if not self.curriculum_stats:
            return
        avg = float(np.mean([s["num_samples"] for s in self.curriculum_stats]))
        total = self.curriculum_stats[0]["total_samples"]
        rank_zero_info(
            f"Curriculum summary ({type(self.strategy).__name__}): {len(self.curriculum_stats)} epochs, "
            f"avg samples/epoch {avg:.1f}/{total} ({100.0 * avg / max(1, total):.1f}%)"
        )

    def get_curriculum_stats(self) -> List[Dict[str, Any]]:
        return list(self.curriculum_stats)

    def get_current_difficulty_scores(self) -> Optional[np.ndarray]:
        return self.difficulty_scores

    def create_dataloader(self, batch_size: int, **loader_kwargs: Any) -> DataLoader:
        """Build a train DataLoader over the registered CurriculumDataset."""
        if self.dataset is None:
            raise RuntimeError("No dataset registered; call wrap(dataset) or pass dataset=")
        loader_kwargs.setdefault("shuffle", True)
        return DataLoader(self.dataset, batch_size=batch_size, **loader_kwargs)

    def plot_curriculum_progression(self, save_path: Optional[str] = None, show: bool = False):
        """Plot the curriculum progression; returns the matplotlib figure (or None)."""
        if not self.curriculum_stats:
            return None
        import matplotlib.pyplot as plt

        epochs = [s["epoch"] for s in self.curriculum_stats]
        percentages = [s["percentage"] for s in self.curriculum_stats]
        thresholds = [s["difficulty_threshold"] for s in self.curriculum_stats]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        ax1.plot(epochs, percentages, marker="o", markersize=3, linewidth=2)
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Training samples (%)")
        ax1.set_title("Curriculum learning: sample progression")
        ax1.grid(True, alpha=0.3)

        ax2.plot(epochs, thresholds, marker="s", markersize=3, linewidth=2)
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Difficulty threshold")
        ax2.set_title("Curriculum learning: difficulty threshold")
        ax2.grid(True, alpha=0.3)
        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            rank_zero_info(f"Curriculum progression plot saved to {save_path}")
        if show:
            plt.show()
        return fig


# Convenience functions
def create_length_curriculum_loop(reverse: bool = False, **kwargs: Any) -> CurriculumLoop:
    """Create curriculum callback based on input length."""
    return CurriculumLoop(strategy=LengthBasedCurriculum(reverse=reverse), **kwargs)


def create_loss_curriculum_loop(warmup_epochs: int = 5, **kwargs: Any) -> CurriculumLoop:
    """Create curriculum callback based on sample loss."""
    return CurriculumLoop(strategy=LossBasedCurriculum(warmup_epochs=warmup_epochs), **kwargs)


def create_random_curriculum_loop(**kwargs: Any) -> CurriculumLoop:
    """Create random curriculum callback (baseline)."""
    return CurriculumLoop(strategy=RandomCurriculum(), **kwargs)
