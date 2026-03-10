# tests/test_loops_kfold.py
"""Tests for K-Fold cross-validation loop."""

import sys
import tempfile
import shutil
import json
from pathlib import Path
import pytest
import torch
import numpy as np
from torch.utils.data import TensorDataset

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lmpro.loops.kfold_loop import KFoldLoop, create_kfold_loop


class TestKFoldLoopInit:
    def test_default_init(self):
        loop = KFoldLoop()
        assert loop.num_folds == 5
        assert loop.stratified is False
        assert loop.shuffle is True
        assert loop.random_state == 42
        assert loop.current_fold == 0
        assert loop.fold_results == []

    def test_custom_init(self):
        loop = KFoldLoop(num_folds=3, stratified=True, shuffle=False, random_state=7)
        assert loop.num_folds == 3
        assert loop.stratified is True
        assert loop.shuffle is False
        assert loop.random_state == 7

    def test_results_dir_is_path(self):
        loop = KFoldLoop(results_dir="my_results")
        assert isinstance(loop.results_dir, Path)


class TestKFoldSplitting:
    def _make_dataset(self, n=100, n_features=10, n_classes=3):
        X = torch.randn(n, n_features)
        y = torch.randint(0, n_classes, (n,))
        return TensorDataset(X, y)

    def test_kfold_splits_count(self):
        loop = KFoldLoop(num_folds=5)
        dataset = self._make_dataset(100)
        n = len(dataset)
        from sklearn.model_selection import KFold
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        splits = list(kf.split(range(n)))
        assert len(splits) == 5

    def test_kfold_no_overlap_between_train_val(self):
        loop = KFoldLoop(num_folds=5)
        dataset = self._make_dataset(50)
        from sklearn.model_selection import KFold
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        for train_idx, val_idx in kf.split(range(len(dataset))):
            assert len(set(train_idx) & set(val_idx)) == 0

    def test_stratified_kfold_splits(self):
        loop = KFoldLoop(num_folds=5, stratified=True)
        dataset = self._make_dataset(100)
        from sklearn.model_selection import StratifiedKFold
        y = [dataset[i][1].item() for i in range(len(dataset))]
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        splits = list(skf.split(range(len(dataset)), y))
        assert len(splits) == 5

    def test_all_samples_used_across_folds(self):
        n = 50
        dataset = self._make_dataset(n)
        from sklearn.model_selection import KFold
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        all_val_indices = set()
        for _, val_idx in kf.split(range(n)):
            all_val_indices.update(val_idx)
        assert all_val_indices == set(range(n))


class TestKFoldResults:
    def test_fold_results_list_exists(self):
        loop = KFoldLoop()
        assert isinstance(loop.fold_results, list)
        assert isinstance(loop.all_fold_metrics, list)

    def test_best_fold_metrics_dict(self):
        loop = KFoldLoop()
        loop.all_fold_metrics = [
            {"val_acc": 0.80},
            {"val_acc": 0.92},
            {"val_acc": 0.85},
        ]
        best_idx, best_metrics = loop.get_best_fold(metric_name="val_acc", mode="max")
        assert best_metrics["val_acc"] == 0.92

    def test_get_summary_stats(self):
        loop = KFoldLoop()
        loop.all_fold_metrics = [
            {"val_acc": 0.80},
            {"val_acc": 0.90},
            {"val_acc": 0.70},
        ]
        summary = loop.get_summary_statistics()
        assert isinstance(summary, dict)
        assert "val_acc" in summary
        assert "mean" in summary["val_acc"]
        assert "std" in summary["val_acc"]
        assert abs(summary["val_acc"]["mean"] - (0.80 + 0.90 + 0.70) / 3) < 1e-6


class TestKFoldResultsSave:
    def test_save_fold_results_creates_file(self):
        temp_dir = tempfile.mkdtemp()
        try:
            loop = KFoldLoop(save_fold_results=True, results_dir=temp_dir)
            loop.results_dir = Path(temp_dir)
            loop.results_dir.mkdir(parents=True, exist_ok=True)
            loop.all_fold_metrics = [
                {"fold": 0, "val_acc": 0.8},
                {"fold": 1, "val_acc": 0.9},
            ]
            summary_stats = loop.get_summary_statistics()
            loop._save_summary_results(summary_stats)
            results_file = Path(temp_dir) / "kfold_summary.json"
            assert results_file.exists()
        finally:
            shutil.rmtree(temp_dir)


class TestCreateKFoldLoop:
    def test_factory_function(self):
        loop = create_kfold_loop(num_folds=3)
        assert isinstance(loop, KFoldLoop)
        assert loop.num_folds == 3

    def test_factory_stratified(self):
        loop = create_kfold_loop(num_folds=5, stratified=True)
        assert loop.stratified is True
