# tests/test_utils_metrics.py
"""Tests for metrics utilities."""

import sys
from pathlib import Path
import pytest
import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lmpro.utils.metrics import (
    get_metrics_dict,
    compute_classification_metrics,
    compute_regression_metrics,
    compute_calibration_metrics,
    print_metrics_report,
)


# ─── get_metrics_dict ───────────────────────────────────────────────────────

class TestGetMetricsDict:
    def test_binary_task(self):
        metrics = get_metrics_dict(task="binary", num_classes=2)
        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics
        assert "auroc" in metrics

    def test_multiclass_task(self):
        metrics = get_metrics_dict(task="multiclass", num_classes=5)
        assert "accuracy" in metrics
        assert "confusion_matrix" in metrics
        # Per-class metrics should be present
        assert "accuracy_per_class" in metrics

    def test_multilabel_task(self):
        metrics = get_metrics_dict(task="multilabel", num_labels=4)
        assert "accuracy" in metrics
        assert "auroc" in metrics

    def test_regression_task(self):
        metrics = get_metrics_dict(task="regression")
        assert "mse" in metrics
        assert "mae" in metrics
        assert "rmse" in metrics
        assert "r2" in metrics

    def test_returns_torchmetrics_objects(self):
        metrics = get_metrics_dict(task="binary", num_classes=2)
        from torchmetrics import Metric
        for v in metrics.values():
            assert isinstance(v, Metric)


# ─── compute_classification_metrics ────────────────────────────────────────

class TestClassificationMetrics:
    @pytest.fixture
    def binary_data(self):
        torch.manual_seed(42)
        n = 100
        targets = torch.randint(0, 2, (n,))
        logits = torch.randn(n, 2)
        return logits, targets

    @pytest.fixture
    def multiclass_data(self):
        torch.manual_seed(42)
        n = 120
        num_classes = 4
        targets = torch.randint(0, num_classes, (n,))
        logits = torch.randn(n, num_classes)
        return logits, targets, num_classes

    def test_binary_returns_accuracy(self, binary_data):
        logits, targets = binary_data
        results = compute_classification_metrics(logits, targets, num_classes=2)
        assert "accuracy" in results
        assert 0.0 <= results["accuracy"] <= 1.0

    def test_multiclass_per_class_metrics(self, multiclass_data):
        logits, targets, n_cls = multiclass_data
        results = compute_classification_metrics(logits, targets, num_classes=n_cls)
        assert "f1_per_class" in results
        assert len(results["f1_per_class"]) == n_cls

    def test_with_class_names(self, multiclass_data):
        logits, targets, n_cls = multiclass_data
        names = [f"class_{i}" for i in range(n_cls)]
        results = compute_classification_metrics(logits, targets, num_classes=n_cls, class_names=names)
        assert "per_class_report" in results
        assert f"class_0_accuracy" in results["per_class_report"]

    def test_without_per_class(self, binary_data):
        logits, targets = binary_data
        results = compute_classification_metrics(logits, targets, num_classes=2, return_per_class=False)
        assert "accuracy_per_class" not in results

    def test_perfect_predictions(self):
        targets = torch.tensor([0, 1, 2, 0, 1, 2])
        # Create logits that perfectly predict targets
        logits = torch.zeros(6, 3)
        for i, t in enumerate(targets):
            logits[i, t] = 10.0
        results = compute_classification_metrics(logits, targets, num_classes=3)
        assert results["accuracy"] == pytest.approx(1.0, abs=1e-4)


# ─── compute_regression_metrics ─────────────────────────────────────────────

class TestRegressionMetrics:
    @pytest.fixture
    def perfect_preds(self):
        vals = torch.randn(50)
        return vals, vals

    @pytest.fixture
    def noisy_preds(self):
        torch.manual_seed(0)
        y_true = torch.randn(50)
        y_pred = y_true + torch.randn(50) * 0.5
        return y_pred, y_true

    def test_returns_all_metrics(self, noisy_preds):
        preds, targets = noisy_preds
        results = compute_regression_metrics(preds, targets)
        for key in ("mse", "mae", "rmse", "r2"):
            assert key in results

    def test_mse_nonnegative(self, noisy_preds):
        preds, targets = noisy_preds
        results = compute_regression_metrics(preds, targets)
        assert results["mse"] >= 0

    def test_perfect_preds_mse_zero(self, perfect_preds):
        preds, targets = perfect_preds
        results = compute_regression_metrics(preds, targets)
        assert results["mse"] == pytest.approx(0.0, abs=1e-5)

    def test_perfect_preds_r2_one(self, perfect_preds):
        preds, targets = perfect_preds
        results = compute_regression_metrics(preds, targets)
        assert results["r2"] == pytest.approx(1.0, abs=1e-5)

    def test_rmse_is_sqrt_of_mse(self, noisy_preds):
        preds, targets = noisy_preds
        results = compute_regression_metrics(preds, targets)
        assert results["rmse"] == pytest.approx(results["mse"] ** 0.5, rel=1e-4)

    def test_extra_metrics_booleans(self, noisy_preds):
        preds, targets = noisy_preds
        results = compute_regression_metrics(preds, targets, return_all=True)
        assert "mean_residual" in results
        assert "std_residual" in results
        assert "max_error" in results

    def test_no_extra_metrics_when_false(self, noisy_preds):
        preds, targets = noisy_preds
        results = compute_regression_metrics(preds, targets, return_all=False)
        assert "mean_residual" not in results


# ─── compute_calibration_metrics ────────────────────────────────────────────

class TestCalibrationMetrics:
    @pytest.fixture
    def calibration_data(self):
        torch.manual_seed(42)
        n = 200
        num_classes = 3
        targets = torch.randint(0, num_classes, (n,))
        probs = torch.softmax(torch.randn(n, num_classes), dim=1)
        return probs, targets, num_classes

    def test_returns_ece(self, calibration_data):
        probs, targets, _ = calibration_data
        results = compute_calibration_metrics(probs, targets)
        assert "ece" in results
        assert 0.0 <= results["ece"] <= 1.0

    def test_returns_mce(self, calibration_data):
        probs, targets, _ = calibration_data
        results = compute_calibration_metrics(probs, targets)
        assert "mce" in results
        assert 0.0 <= results["mce"] <= 1.0

    def test_bin_counts_sum_to_n(self, calibration_data):
        probs, targets, _ = calibration_data
        results = compute_calibration_metrics(probs, targets, num_bins=10)
        assert sum(results["bin_counts"]) == len(targets)

    def test_bin_accuracies_length(self, calibration_data):
        probs, targets, _ = calibration_data
        num_bins = 5
        results = compute_calibration_metrics(probs, targets, num_bins=num_bins)
        assert len(results["bin_accuracies"]) == num_bins
        assert len(results["bin_confidences"]) == num_bins

    def test_perfect_calibration(self):
        """Perfectly calibrated model should have low ECE."""
        n = 300
        # Create data where correct class always has high confidence
        targets = torch.zeros(n, dtype=torch.long)
        probs = torch.zeros(n, 2)
        probs[:, 0] = 0.95
        probs[:, 1] = 0.05
        results = compute_calibration_metrics(probs, targets, num_bins=5)
        assert results["ece"] < 0.2


# ─── print_metrics_report ──────────────────────────────────────────────────

class TestPrintMetricsReport:
    def test_runs_without_error(self, capsys):
        metrics = {"accuracy": 0.92, "f1": 0.88, "loss": 0.31}
        print_metrics_report(metrics, title="Test Report")
        captured = capsys.readouterr()
        assert "accuracy" in captured.out
        assert "Test Report" in captured.out

    def test_tensor_single_value(self, capsys):
        metrics = {"value": torch.tensor(0.75)}
        print_metrics_report(metrics)
        captured = capsys.readouterr()
        assert "value" in captured.out
