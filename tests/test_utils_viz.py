# tests/test_utils_viz.py
"""Tests for visualization utilities."""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend for tests

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lmpro.utils.viz import (  # noqa: E402
    create_learning_curve_dashboard,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_predictions,
    plot_training_curves,
    save_plot,
)


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


@pytest.fixture
def sample_metrics_history():
    return {
        "train_loss": [0.9, 0.7, 0.5, 0.3],
        "val_loss": [1.0, 0.8, 0.6, 0.4],
        "train_acc": [0.6, 0.7, 0.8, 0.9],
        "val_acc": [0.55, 0.65, 0.75, 0.85],
    }


@pytest.fixture
def no_show(monkeypatch):
    """Fail the test if a library function tries to display a figure."""

    def boom(*args, **kwargs):
        raise AssertionError("plt.show() must not be called unless show=True")

    monkeypatch.setattr(plt, "show", boom)
    monkeypatch.setattr(go.Figure, "show", boom)


class TestPlotTrainingCurves:
    def test_returns_figure_and_saves(self, sample_metrics_history, tmp_path, no_show):
        save_path = tmp_path / "curves.png"
        fig = plot_training_curves(sample_metrics_history, save_path=str(save_path))
        assert isinstance(fig, plt.Figure)
        assert save_path.exists()
        visible = [ax for ax in fig.axes if ax.get_visible()]
        assert len(visible) == 2  # loss + acc

    def test_show_true_calls_show(self, sample_metrics_history, monkeypatch):
        calls = []
        monkeypatch.setattr(plt, "show", lambda *a, **k: calls.append(1))
        plot_training_curves(sample_metrics_history, show=True)
        assert calls == [1]

    def test_empty_metrics_returns_none(self, no_show):
        assert plot_training_curves({}) is None

    def test_metrics_without_matching_pair(self, no_show):
        assert plot_training_curves({"train_loss": [1.0, 0.8], "some_other": [0.5, 0.4]}) is None

    def test_plotly_returns_figure_and_writes_html(self, sample_metrics_history, tmp_path, no_show):
        fig = plot_training_curves(sample_metrics_history, save_path=str(tmp_path / "curves.png"), use_plotly=True)
        assert isinstance(fig, go.Figure)
        assert (tmp_path / "curves.html").exists()
        assert len(fig.data) == 4


class TestPlotPredictions:
    def test_regression(self, tmp_path, no_show):
        y_true = np.linspace(0, 10, 50)
        y_pred = y_true + np.random.randn(50) * 0.5
        save_path = tmp_path / "regression.png"
        fig = plot_predictions(y_true, y_pred, task="regression", save_path=str(save_path))
        assert isinstance(fig, plt.Figure) and len(fig.axes) == 2
        assert save_path.exists()

    def test_classification(self, tmp_path, no_show):
        rng = np.random.default_rng(42)
        y_true = rng.integers(0, 3, 60)
        probs = np.abs(rng.standard_normal((60, 3)))
        probs /= probs.sum(axis=1, keepdims=True)
        save_path = tmp_path / "classification.png"
        fig = plot_predictions(
            y_true, probs, task="classification", class_names=["a", "b", "c"], save_path=str(save_path)
        )
        assert isinstance(fig, plt.Figure)
        assert save_path.exists()

    def test_tensor_input_works(self, no_show):
        fig = plot_predictions(torch.randn(20), torch.randn(20), task="regression")
        assert isinstance(fig, plt.Figure)

    def test_unknown_task_raises(self, no_show):
        with pytest.raises(ValueError):
            plot_predictions(np.zeros(3), np.zeros(3), task="clustering")


class TestPlotConfusionMatrix:
    def test_normalized_rows_sum_to_one(self, tmp_path, no_show):
        y_true = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 2, 0, 2, 1, 1, 1, 2])
        save_path = tmp_path / "cm.png"
        fig = plot_confusion_matrix(y_true, y_pred, save_path=str(save_path))
        assert isinstance(fig, plt.Figure)
        assert save_path.exists()
        texts = [float(t.get_text()) for t in fig.axes[0].texts]
        assert sum(texts) == pytest.approx(3.0)  # three rows, each normalised to 1

    def test_raw_counts_with_class_names(self, no_show):
        fig = plot_confusion_matrix(
            np.array([0, 1, 0, 1]), np.array([0, 0, 1, 1]), class_names=["cat", "dog"], normalize=False
        )
        labels = [t.get_text() for t in fig.axes[0].get_xticklabels()]
        assert labels == ["cat", "dog"]
        assert fig.axes[0].get_title() == "Confusion Matrix"

    def test_tensor_and_probability_input(self, no_show):
        probs = torch.tensor([[0.9, 0.1], [0.2, 0.8], [0.6, 0.4], [0.3, 0.7]])
        fig = plot_confusion_matrix(torch.tensor([0, 1, 0, 1]), probs)
        assert isinstance(fig, plt.Figure)


class TestPlotFeatureImportance:
    def test_top_k_respected(self, tmp_path, no_show):
        names = [f"feature_{i}" for i in range(15)]
        importance = np.arange(15, dtype=float)
        save_path = tmp_path / "fi.png"
        fig = plot_feature_importance(names, importance, top_k=10, save_path=str(save_path))
        assert save_path.exists()
        ax = fig.axes[0]
        assert len(ax.patches) == 10
        assert [t.get_text() for t in ax.get_yticklabels()][-1] == "feature_14"

    def test_tensor_input_and_top_k_larger_than_features(self, no_show):
        fig = plot_feature_importance([f"f_{i}" for i in range(8)], torch.abs(torch.randn(8)), top_k=20)
        assert len(fig.axes[0].patches) == 8
        assert fig.axes[0].get_title() == "Top 8 Feature Importance"


class TestSavePlot:
    def test_saves_png_and_creates_parent_dirs(self, tmp_path):
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3])
        path = tmp_path / "sub" / "dir" / "plot.png"
        save_plot(fig, str(path), dpi=72)
        assert path.exists() and path.stat().st_size > 0


class TestDashboard:
    def test_dashboard_returns_figure(self, tmp_path, no_show):
        metrics = {
            "exp1": {"train_loss": [1, 0.5], "val_loss": [1.1, 0.6], "val_acc": [0.5, 0.7], "lr": [1e-3, 5e-4]},
            "exp2": {"train_loss": [0.9, 0.4], "val_f1": [0.4, 0.6]},
        }
        fig = create_learning_curve_dashboard(metrics, save_path=str(tmp_path / "dash.html"))
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 6
        assert (tmp_path / "dash.html").exists()
