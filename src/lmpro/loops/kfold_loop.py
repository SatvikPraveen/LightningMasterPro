# File: src/lmpro/loops/kfold_loop.py

"""
K-Fold Cross-Validation driver for robust model evaluation.

Lightning 2.x removed the public ``Loop`` API, so k-fold is implemented as a
plain driver: for every fold it builds fresh train/val ``DataLoader``s from
scikit-learn ``KFold``/``StratifiedKFold`` indices, a fresh model (deep copy of a
template or the result of a factory), a fresh ``Trainer``, then calls ``fit``
(and optionally ``test``), collects the fold's metrics and finally returns
mean/std statistics across folds.
"""

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from lightning.pytorch import LightningDataModule, LightningModule, Trainer
from lightning.pytorch.utilities.rank_zero import rank_zero_info
from sklearn.model_selection import KFold, StratifiedKFold
from torch.utils.data import DataLoader, Dataset, Subset

ModelSource = Union[LightningModule, Callable[[], LightningModule]]
DataSource = Union[Dataset, DataLoader, LightningDataModule]


class KFoldLoop:
    """
    K-Fold cross-validation driver.

    Example::

        kfold = KFoldLoop(num_folds=5, stratified=True)
        summary = kfold.run(
            model,                      # template module or a zero-arg factory
            dataset,                    # Dataset, DataLoader or LightningDataModule
            trainer_kwargs={"max_epochs": 10, "accelerator": "cpu"},
        )
        summary["val_loss"]["mean"], summary["val_loss"]["std"]
    """

    def __init__(
        self,
        num_folds: int = 5,
        stratified: bool = False,
        shuffle: bool = True,
        random_state: int = 42,
        save_fold_results: bool = True,
        results_dir: str = "kfold_results",
        batch_size: Optional[int] = None,
        num_workers: Optional[int] = None,
        run_test: bool = False,
        keep_fold_models: bool = True,
    ):
        if num_folds < 2:
            raise ValueError("num_folds must be >= 2")

        self.num_folds = num_folds
        self.stratified = stratified
        self.shuffle = shuffle
        self.random_state = random_state
        self.save_fold_results = save_fold_results
        self.results_dir = Path(results_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.run_test = run_test
        self.keep_fold_models = keep_fold_models

        # Internal state
        self.current_fold = 0
        self.fold_splits: List[Tuple[np.ndarray, np.ndarray]] = []
        self.fold_results: List[Dict[str, Any]] = []
        self.fold_models: List[LightningModule] = []
        self.all_fold_metrics: List[Dict[str, float]] = []
        self.summary_statistics: Dict[str, Dict[str, float]] = {}

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(
        self,
        model: ModelSource,
        data: DataSource,
        trainer_kwargs: Optional[Dict[str, Any]] = None,
        test_dataloader: Optional[DataLoader] = None,
    ) -> Dict[str, Dict[str, float]]:
        """
        Run k-fold cross-validation.

        Args:
            model: A ``LightningModule`` used as a template (deep-copied per fold)
                or a zero-argument callable returning a fresh ``LightningModule``.
            data: The full training data as a ``Dataset``, a ``DataLoader``
                (its dataset/batch size/num_workers are reused) or a
                ``LightningDataModule`` (``setup("fit")`` is called and the
                training dataset is taken from ``train_dataloader()``).
            trainer_kwargs: Keyword arguments forwarded to every fold's ``Trainer``.
            test_dataloader: Optional held-out loader; if given (or ``run_test``
                is True and the data source provides one) ``trainer.test`` runs
                after every fold.

        Returns:
            Dict ``metric -> {"mean", "std", "min", "max", "values"}`` over folds.
        """
        trainer_kwargs = dict(trainer_kwargs or {})
        dataset, batch_size, num_workers = self._resolve_data(data)
        if test_dataloader is None and self.run_test:
            test_dataloader = self._resolve_test_dataloader(data)

        self.reset()
        self.fold_splits = self.get_fold_splits(dataset)

        if self.save_fold_results:
            self.results_dir.mkdir(parents=True, exist_ok=True)

        rank_zero_info(
            f"K-Fold CV: {self.num_folds} folds, stratified={self.stratified}, " f"dataset size={len(dataset)}"
        )

        for fold_idx, (train_indices, val_indices) in enumerate(self.fold_splits):
            self.current_fold = fold_idx
            rank_zero_info(f"Starting fold {fold_idx + 1}/{self.num_folds}")

            train_loader = DataLoader(
                Subset(dataset, train_indices.tolist()),
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
            )
            val_loader = DataLoader(
                Subset(dataset, val_indices.tolist()),
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
            )

            fold_model = self._make_fresh_model(model)
            fold_trainer = self._make_trainer(trainer_kwargs, fold_idx)
            fold_trainer.fit(fold_model, train_dataloaders=train_loader, val_dataloaders=val_loader)

            fold_metrics = self._collect_metrics(fold_trainer)
            if test_dataloader is not None:
                test_results = fold_trainer.test(fold_model, dataloaders=test_dataloader, verbose=False)
                for result in test_results:
                    fold_metrics.update({k: float(v) for k, v in result.items()})

            self.all_fold_metrics.append(fold_metrics)
            self.fold_results.append(
                {
                    "fold": fold_idx,
                    "train_size": int(len(train_indices)),
                    "val_size": int(len(val_indices)),
                    "val_indices": [int(i) for i in val_indices],
                    "metrics": fold_metrics,
                }
            )
            if self.keep_fold_models:
                self.fold_models.append(fold_model)
            if self.save_fold_results:
                self._save_fold_results(fold_metrics, fold_idx)

            rank_zero_info(f"Completed fold {fold_idx + 1}: {fold_metrics}")

        self.current_fold = self.num_folds
        self.summary_statistics = self._compute_summary_statistics()
        if self.save_fold_results:
            self._save_summary_results(self.summary_statistics)
        self._log_summary(self.summary_statistics)
        return self.summary_statistics

    def reset(self) -> None:
        """Reset all per-run state."""
        self.current_fold = 0
        self.fold_splits = []
        self.fold_results = []
        self.fold_models = []
        self.all_fold_metrics = []
        self.summary_statistics = {}

    @property
    def done(self) -> bool:
        return self.current_fold >= self.num_folds

    # ------------------------------------------------------------------
    # Splitting
    # ------------------------------------------------------------------

    def get_fold_splits(self, dataset: Dataset) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Return the list of ``(train_indices, val_indices)`` for every fold."""
        n = len(dataset)
        if self.stratified:
            labels = self._extract_labels(dataset)
            if labels is not None:
                splitter = StratifiedKFold(
                    n_splits=self.num_folds,
                    shuffle=self.shuffle,
                    random_state=self.random_state if self.shuffle else None,
                )
                return list(splitter.split(np.zeros(n), labels))
            rank_zero_info("Could not extract labels; falling back to plain KFold")

        splitter = KFold(
            n_splits=self.num_folds,
            shuffle=self.shuffle,
            random_state=self.random_state if self.shuffle else None,
        )
        return list(splitter.split(np.zeros(n)))

    @staticmethod
    def _extract_labels(dataset: Dataset) -> Optional[np.ndarray]:
        """Extract integer labels from a dataset for stratified splitting."""
        for attr in ("targets", "labels", "y"):
            values = getattr(dataset, attr, None)
            if values is not None:
                return np.asarray(values.cpu() if isinstance(values, torch.Tensor) else values)

        if isinstance(dataset, torch.utils.data.TensorDataset) and len(dataset.tensors) >= 2:
            return dataset.tensors[-1].cpu().numpy()

        labels = []
        for i in range(len(dataset)):
            sample = dataset[i]
            if not isinstance(sample, (tuple, list)) or len(sample) < 2:
                return None
            label = sample[-1]
            if isinstance(label, torch.Tensor):
                if label.numel() != 1:
                    return None
                label = label.item()
            labels.append(label)
        return np.asarray(labels)

    # ------------------------------------------------------------------
    # Per-fold construction helpers
    # ------------------------------------------------------------------

    def _resolve_data(self, data: DataSource) -> Tuple[Dataset, int, int]:
        """Return ``(dataset, batch_size, num_workers)`` from any supported source."""
        loader: Optional[DataLoader] = None
        if isinstance(data, LightningDataModule):
            data.prepare_data()
            data.setup("fit")
            loader = data.train_dataloader()
        elif isinstance(data, DataLoader):
            loader = data

        if loader is not None:
            dataset = loader.dataset
            batch_size = self.batch_size or loader.batch_size or 32
            num_workers = self.num_workers if self.num_workers is not None else loader.num_workers
        elif isinstance(data, Dataset):
            dataset = data
            batch_size = self.batch_size or 32
            num_workers = self.num_workers or 0
        else:
            raise TypeError(f"Unsupported data source: {type(data).__name__}")

        # A DataModule may hand us a Subset of a larger dataset; k-fold over it as-is.
        if len(dataset) < self.num_folds:
            raise ValueError(f"Dataset has {len(dataset)} samples, fewer than num_folds={self.num_folds}")
        return dataset, batch_size, num_workers

    @staticmethod
    def _resolve_test_dataloader(data: DataSource) -> Optional[DataLoader]:
        if isinstance(data, LightningDataModule):
            try:
                data.setup("test")
                return data.test_dataloader()
            except Exception:
                return None
        return None

    @staticmethod
    def _make_fresh_model(model: ModelSource) -> LightningModule:
        if isinstance(model, LightningModule):
            trainer_ref = model._trainer
            model._trainer = None  # never deep-copy an attached Trainer
            try:
                return deepcopy(model)
            finally:
                model._trainer = trainer_ref
        if callable(model):
            fresh = model()
            if not isinstance(fresh, LightningModule):
                raise TypeError("model factory must return a LightningModule")
            return fresh
        raise TypeError("model must be a LightningModule or a zero-argument factory")

    def _make_trainer(self, trainer_kwargs: Dict[str, Any], fold_idx: int) -> Trainer:
        kwargs = dict(trainer_kwargs)
        if self.save_fold_results and "default_root_dir" not in kwargs:
            kwargs["default_root_dir"] = str(self.results_dir / f"fold_{fold_idx}")
        return Trainer(**kwargs)

    @staticmethod
    def _collect_metrics(trainer: Trainer) -> Dict[str, float]:
        """Collect scalar metrics from the trainer after fit."""
        metrics: Dict[str, float] = {}
        for key, value in trainer.callback_metrics.items():
            if isinstance(value, torch.Tensor):
                if value.numel() == 1:
                    metrics[key] = float(value.item())
            elif isinstance(value, (int, float)):
                metrics[key] = float(value)
        return metrics

    # ------------------------------------------------------------------
    # Ensemble helper
    # ------------------------------------------------------------------

    def ensemble_predict(self, inputs: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
        """
        Average the forward outputs of all fold models on ``inputs``.

        Requires ``keep_fold_models=True`` (the default) and a completed ``run``.
        """
        if not self.fold_models:
            raise RuntimeError("No fold models available; call run() with keep_fold_models=True first")
        outputs = []
        with torch.no_grad():
            for fold_model in self.fold_models:
                fold_model.eval()
                outputs.append(fold_model(inputs.to(fold_model.device)).cpu())
        stacked = torch.stack(outputs)
        if reduction == "mean":
            return stacked.mean(dim=0)
        if reduction == "none":
            return stacked
        raise ValueError(f"Unknown reduction: {reduction}")

    # ------------------------------------------------------------------
    # Results / reporting
    # ------------------------------------------------------------------

    def _save_fold_results(self, fold_metrics: Dict[str, float], fold_idx: int) -> None:
        fold_file = self.results_dir / f"fold_{fold_idx}_results.json"
        with open(fold_file, "w") as f:
            json.dump(_to_serializable(fold_metrics), f, indent=2)

    def _compute_summary_statistics(self) -> Dict[str, Dict[str, float]]:
        if not self.all_fold_metrics:
            return {}

        metric_names: List[str] = []
        for fold_metrics in self.all_fold_metrics:
            for name in fold_metrics:
                if name not in metric_names:
                    metric_names.append(name)

        summary: Dict[str, Dict[str, float]] = {}
        for name in metric_names:
            values = [float(m[name]) for m in self.all_fold_metrics if name in m and isinstance(m[name], (int, float))]
            if values:
                summary[name] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                    "values": values,
                }
        return summary

    def _save_summary_results(self, summary_stats: Dict[str, Dict[str, float]]) -> None:
        self.results_dir.mkdir(parents=True, exist_ok=True)
        summary_file = self.results_dir / "kfold_summary.json"
        payload = {
            "metadata": {
                "num_folds": self.num_folds,
                "stratified": self.stratified,
                "shuffle": self.shuffle,
                "random_state": self.random_state,
            },
            "summary_statistics": _to_serializable(summary_stats),
            "individual_fold_results": _to_serializable(self.fold_results),
        }
        with open(summary_file, "w") as f:
            json.dump(payload, f, indent=2)
        rank_zero_info(f"K-Fold results saved to {summary_file}")

    def _log_summary(self, summary_stats: Dict[str, Dict[str, float]]) -> None:
        rank_zero_info(f"K-FOLD CROSS-VALIDATION SUMMARY ({self.num_folds} folds)")
        for name, stats in summary_stats.items():
            rank_zero_info(
                f"{name:20s}: {stats['mean']:.4f} +/- {stats['std']:.4f} "
                f"(min: {stats['min']:.4f}, max: {stats['max']:.4f})"
            )

    def get_summary_statistics(self) -> Dict[str, Dict[str, float]]:
        return self._compute_summary_statistics()

    def get_best_fold(self, metric_name: str, mode: str = "max") -> Tuple[int, Dict[str, float]]:
        """Return ``(fold_index, fold_metrics)`` of the best fold for ``metric_name``."""
        candidates = [(i, m[metric_name]) for i, m in enumerate(self.all_fold_metrics) if metric_name in m]
        if not candidates:
            return -1, {}
        pick = max if mode == "max" else min
        best_idx, _ = pick(candidates, key=lambda item: item[1])
        return best_idx, self.all_fold_metrics[best_idx]


def _to_serializable(obj: Any) -> Any:
    """Recursively convert tensors / numpy scalars into JSON-serializable values."""
    if isinstance(obj, dict):
        return {str(k): _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(v) for v in obj]
    if isinstance(obj, torch.Tensor):
        return obj.item() if obj.numel() == 1 else obj.tolist()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


# Convenience function
def create_kfold_loop(
    num_folds: int = 5,
    stratified: bool = False,
    random_state: int = 42,
    **kwargs: Any,
) -> KFoldLoop:
    """Create k-fold driver with common settings."""
    return KFoldLoop(num_folds=num_folds, stratified=stratified, random_state=random_state, **kwargs)
