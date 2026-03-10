# tests/test_utils_viz.py
"""Tests for visualization utilities."""

import sys
import tempfile
import shutil
from pathlib import Path
import pytest
import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")  # Non-interactive backend for tests

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lmpro.utils.viz import (
    plot_training_curves,
    plot_predictions,
    plot_confusion_matrix,
    plot_feature_importance,
    save_plot,
)


@pytest.fixture
def temp_dir():
    d = tempfile.mkdtemp()
    yield Path(d)
    shutil.rmtree(d)


@pytest.fixture
def sample_metrics_history():
    return {
        "train_loss": [0.9, 0.7, 0.5, 0.3],
        "val_loss":   [1.0, 0.8, 0.6, 0.4],
        "train_acc":  [0.6, 0.7, 0.8, 0.9],
        "val_acc":    [0.55, 0.65, 0.75, 0.85],
    }


# ─── plot_training_curves ───────────────────────────────────────────────────

class TestPlotTrainingCurves:
    def test_runs_without_error(self, sample_metrics_history, temp_dir):
        save_path = str(temp_dir / "curves.png")
        plot_training_curves(sample_metrics_history, save_path=save_path)
        plt.close("all")

    def test_saves_file(self, sample_metrics_history, temp_dir):
        save_path = str(temp_dir / "curves.png")
        plot_training_curves(sample_metrics_history, save_path=save_path)
        assert Path(save_path).exists()
        plt.close("all")

    def test_empty_metrics_no_crash(self, temp_dir):
        plot_training_curves({})
        plt.close("all")

    def test_metrics_without_matching_pair(self, temp_dir):
        """If only train_loss present (no val_loss), should handle gracefully."""
        metrics = {"train_loss": [1.0, 0.8], "some_other": [0.5, 0.4]}
        plot_training_curves(metrics)
        plt.close("all")


# ─── plot_predictions ───────────────────────────────────────────────────────

class TestPlotPredictions:
    def test_regression_runs(self, temp_dir):
        y_true = np.linspace(0, 10, 50)
        y_pred = y_true + np.random.randn(50) * 0.5
        save_path = str(temp_dir / "regression.png")
        plot_predictions(y_true, y_pred, task="regression", save_path=save_path)
        assert Path(save_path).exists()
        plt.close("all")

    def test_classification_runs(self, temp_dir):
        np.random.seed(42)
        y_true = np.random.randint(0, 3, 60)
        y_pred_probs = np.abs(np.random.randn(60, 3))
        y_pred_probs /= y_pred_probs.sum(axis=1, keepdims=True)
        save_path = str(temp_dir / "classification.png")
        plot_predictions(y_true, y_pred_probs, task="classification", save_path=save_path)
        assert Path(save_path).exists()
        plt.close("all")

    def test_tensor_input_works(self, temp_dir):
        y_true = torch.randn(20)
        y_pred = torch.randn(20)
        plot_predictions(y_true, y_pred, task="regression")
        plt.close("all")


# ─── plot_confusion_matrix ──────────────────────────────────────────────────

class TestPlotConfusionMatrix:
    def test_runs_with_integers(self, temp_dir):
        y_true = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 2, 0, 2, 1, 1, 1, 2])
        save_path = str(temp_dir / "cm.png")
        plot_confusion_matrix(y_true, y_pred, save_path=save_path)
        assert Path(save_path).exists()
        plt.close("all")

    def test_with_class_names(self, temp_dir):
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 0, 1, 1])
        save_path = str(temp_dir / "cm_named.png")
        plot_confusion_matrix(y_true, y_pred, class_names=["cat", "dog"], save_path=save_path)
        assert Path(save_path).exists()
        plt.close("all")

    def test_normalized_false(self, temp_dir):
        y_true = np.array([0, 1, 2])
        y_pred = np.array([0, 1, 2])
        save_path = str(temp_dir / "cm_raw.png")
        plot_confusion_matrix(y_true, y_pred, normalize=False, save_path=save_path)
        assert Path(save_path).exists()
        plt.close("all")

    def test_tensor_input(self, temp_dir):
        y_true = torch.tensor([0, 1, 0, 1])
        y_pred = torch.tensor([0, 0, 1, 1])
        save_path = str(temp_dir / "cm_tensor.png")
        plot_confusion_matrix(y_true, y_pred, save_path=save_path)
        assert Path(save_path).exists()
        plt.close("all")


# ─── plot_feature_importance ────────────────────────────────────────────────

class TestPlotFeatureImportance:
    def test_runs_with_numpy(self, temp_dir):
        names = [f"feature_{i}" for i in range(15)]
        importance = np.abs(np.random.randn(15))
        save_path = str(temp_dir / "fi.png")
        plot_feature_importance(names, importance, top_k=10, save_path=save_path)
        assert Path(save_path).exists()
        plt.close("all")

    def test_runs_with_tensor(self, temp_dir):
        names = [f"f_{i}" for i in range(8)]
        importance = torch.abs(torch.randn(8))
        save_path = str(temp_dir / "fi_tensor.png")
        plot_feature_importance(names, importance, save_path=save_path)
        assert Path(save_path).exists()
        plt.close("all")

    def test_top_k_respected(self, temp_dir):
        names = [f"feature_{i}" for i in range(20)]
        importance = np.abs(np.random.randn(20))
        # Should not raise even when top_k < len(features)
        plot_feature_importance(names, importance, top_k=5)
        plt.close("all")


# ─── save_plot ──────────────────────────────────────────────────────────────

class TestSavePlot:
    def test_saves_png(self, temp_dir):
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3])
        path = str(temp_dir / "test.png")
        save_plot(fig, path)
        assert Path(path).exists()
        plt.close("all")

    def test_creates_parent_dirs(self, temp_dir):
        fig, ax = plt.subplots()
        path = str(temp_dir / "sub" / "dir" / "plot.png")
        save_plot(fig, path)
        assert Path(path).exists()
        plt.close("all")

    def test_custom_dpi(self, temp_dir):
        fig, ax = plt.subplots()
        path = str(temp_dir / "hires.png")
        save_plot(fig, path, dpi=72)
        assert Path(path).exists()
        plt.close("all")
