# File: src/lmpro/utils/metrics.py

"""
Metrics utilities for comprehensive model evaluation
"""

from typing import Any, Dict, List, Literal, Optional, Union, cast

import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn.functional as F
from lightning import LightningModule
from torchmetrics import (
    AUROC,
    Accuracy,
    AveragePrecision,
    CalibrationError,
    ConfusionMatrix,
    F1Score,
    MeanAbsoluteError,
    MeanSquaredError,
    Precision,
    R2Score,
    Recall,
)
from torchmetrics.functional import confusion_matrix

# Literal aliases matching the argument types torchmetrics declares.
BinaryOrMulticlass = Literal["binary", "multiclass"]
AverageMethod = Literal["micro", "macro", "weighted", "none"]
MacroAverageMethod = Literal["macro", "weighted", "none"]
NormalizeMethod = Literal["true", "pred", "all", "none"]

_BINARY_OR_MULTICLASS = ("binary", "multiclass")
_AVERAGE_METHODS = ("micro", "macro", "weighted", "none")
_MACRO_AVERAGE_METHODS = ("macro", "weighted", "none")
_NORMALIZE_METHODS = ("true", "pred", "all", "none")


def _binary_or_multiclass(task: str) -> BinaryOrMulticlass:
    """Validate ``task`` and narrow it to the literal type torchmetrics expects"""
    if task not in _BINARY_OR_MULTICLASS:
        raise ValueError(f"Expected task to be one of {_BINARY_OR_MULTICLASS}, got {task!r}")
    return cast(BinaryOrMulticlass, task)


def _average_method(average: Optional[str]) -> Optional[AverageMethod]:
    """Validate an averaging strategy accepted by accuracy / precision / recall / F1"""
    if average is not None and average not in _AVERAGE_METHODS:
        raise ValueError(f"Expected average to be one of {_AVERAGE_METHODS} or None, got {average!r}")
    return cast(Optional[AverageMethod], average)


def _macro_average_method(average: Optional[str]) -> Optional[MacroAverageMethod]:
    """Validate an averaging strategy accepted by AUROC / average precision ('micro' is not)"""
    if average is not None and average not in _MACRO_AVERAGE_METHODS:
        raise ValueError(f"Expected average to be one of {_MACRO_AVERAGE_METHODS} or None, got {average!r}")
    return cast(Optional[MacroAverageMethod], average)


def _normalize_method(normalize: Optional[str]) -> Optional[NormalizeMethod]:
    """Validate a confusion-matrix normalisation mode"""
    if normalize is not None and normalize not in _NORMALIZE_METHODS:
        raise ValueError(f"Expected normalize to be one of {_NORMALIZE_METHODS} or None, got {normalize!r}")
    return cast(Optional[NormalizeMethod], normalize)


def get_metrics_dict(
    task: str = "multiclass",
    num_classes: Optional[int] = None,
    num_labels: Optional[int] = None,
    average: str = "macro",
) -> Dict[str, Any]:
    """
    Get a comprehensive dictionary of metrics for different tasks

    Args:
        task: Type of task ('binary', 'multiclass', 'multilabel', 'regression')
        num_classes: Number of classes for classification tasks
        num_labels: Number of labels for multilabel tasks
        average: Averaging strategy for multiclass ('macro', 'micro', 'weighted')

    Returns:
        Dictionary of initialized metric objects
    """
    metrics: Dict[str, Any] = {}

    if task in ["binary", "multiclass"]:
        cls_task = _binary_or_multiclass(task)
        avg = _average_method(average)
        macro_avg = _macro_average_method(average)
        metrics.update(
            {
                "accuracy": Accuracy(task=cls_task, num_classes=num_classes, average=avg),
                "precision": Precision(task=cls_task, num_classes=num_classes, average=avg),
                "recall": Recall(task=cls_task, num_classes=num_classes, average=avg),
                "f1": F1Score(task=cls_task, num_classes=num_classes, average=avg),
                "auroc": AUROC(task=cls_task, num_classes=num_classes, average=macro_avg),
                "avg_precision": AveragePrecision(task=cls_task, num_classes=num_classes, average=macro_avg),
                "confusion_matrix": ConfusionMatrix(task=cls_task, num_classes=num_classes),
                "calibration_error": CalibrationError(task=cls_task, num_classes=num_classes),
            }
        )

        # Add per-class metrics for multiclass
        if task == "multiclass" and num_classes:
            metrics.update(
                {
                    "accuracy_per_class": Accuracy(task="multiclass", num_classes=num_classes, average=None),
                    "precision_per_class": Precision(task="multiclass", num_classes=num_classes, average=None),
                    "recall_per_class": Recall(task="multiclass", num_classes=num_classes, average=None),
                    "f1_per_class": F1Score(task="multiclass", num_classes=num_classes, average=None),
                }
            )

    elif task == "multilabel":
        avg = _average_method(average)
        macro_avg = _macro_average_method(average)
        metrics.update(
            {
                "accuracy": Accuracy(task="multilabel", num_labels=num_labels, average=avg),
                "precision": Precision(task="multilabel", num_labels=num_labels, average=avg),
                "recall": Recall(task="multilabel", num_labels=num_labels, average=avg),
                "f1": F1Score(task="multilabel", num_labels=num_labels, average=avg),
                "auroc": AUROC(task="multilabel", num_labels=num_labels, average=macro_avg),
                "avg_precision": AveragePrecision(task="multilabel", num_labels=num_labels, average=macro_avg),
            }
        )

    elif task == "regression":
        metrics.update(
            {
                "mse": MeanSquaredError(),
                "mae": MeanAbsoluteError(),
                "rmse": MeanSquaredError(squared=False),
                "r2": R2Score(),
            }
        )

    return metrics


def compute_classification_metrics(
    preds: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    class_names: Optional[List[str]] = None,
    return_per_class: bool = True,
) -> Dict[str, Union[float, torch.Tensor]]:
    """
    Compute comprehensive classification metrics

    Args:
        preds: Predictions (logits or probabilities)
        targets: Ground truth labels
        num_classes: Number of classes
        class_names: Optional class names for reporting
        return_per_class: Whether to return per-class metrics

    Returns:
        Dictionary of computed metrics
    """
    # Convert logits to probabilities if needed
    if preds.dim() > 1 and preds.size(1) > 1:
        probs = F.softmax(preds, dim=1)  # (N, C)
        pred_labels = torch.argmax(preds, dim=1)
    else:
        probs = torch.sigmoid(preds.reshape(-1))  # (N,) positive-class probability
        pred_labels = (probs > 0.5).long()

    task: BinaryOrMulticlass = "binary" if num_classes == 2 else "multiclass"

    # Binary torchmetrics expect (N,) positive-class probabilities, not (N, 2)
    if task == "binary" and probs.dim() == 2:
        probs = probs[:, 1]

    results: Dict[str, Any] = {}

    # Basic metrics
    acc = Accuracy(task=task, num_classes=num_classes)
    prec = Precision(task=task, num_classes=num_classes, average="macro")
    rec = Recall(task=task, num_classes=num_classes, average="macro")
    f1 = F1Score(task=task, num_classes=num_classes, average="macro")

    results.update(
        {
            "accuracy": acc(pred_labels, targets).item(),
            "precision_macro": prec(pred_labels, targets).item(),
            "recall_macro": rec(pred_labels, targets).item(),
            "f1_macro": f1(pred_labels, targets).item(),
        }
    )

    # AUROC and Average Precision
    try:
        auroc = AUROC(task=task, num_classes=num_classes, average="macro")
        avg_prec = AveragePrecision(task=task, num_classes=num_classes, average="macro")

        results.update(
            {
                "auroc_macro": auroc(probs, targets).item(),
                "avg_precision_macro": avg_prec(probs, targets).item(),
            }
        )
    except Exception as e:
        print(f"Warning: Could not compute AUROC/AP: {e}")

    # Per-class metrics (binary metrics have no per-class output, so use the
    # multiclass formulation with two classes to get one value per class)
    if return_per_class:
        acc_per_class = Accuracy(task="multiclass", num_classes=num_classes, average=None)
        prec_per_class = Precision(task="multiclass", num_classes=num_classes, average=None)
        rec_per_class = Recall(task="multiclass", num_classes=num_classes, average=None)
        f1_per_class = F1Score(task="multiclass", num_classes=num_classes, average=None)

        results.update(
            {
                "accuracy_per_class": acc_per_class(pred_labels, targets),
                "precision_per_class": prec_per_class(pred_labels, targets),
                "recall_per_class": rec_per_class(pred_labels, targets),
                "f1_per_class": f1_per_class(pred_labels, targets),
            }
        )

        # Create per-class report
        if class_names:
            per_class_report = {}
            for i, name in enumerate(class_names):
                per_class_report[f"{name}_accuracy"] = results["accuracy_per_class"][i].item()
                per_class_report[f"{name}_precision"] = results["precision_per_class"][i].item()
                per_class_report[f"{name}_recall"] = results["recall_per_class"][i].item()
                per_class_report[f"{name}_f1"] = results["f1_per_class"][i].item()

            results["per_class_report"] = per_class_report

    return results


def compute_regression_metrics(preds: torch.Tensor, targets: torch.Tensor, return_all: bool = True) -> Dict[str, float]:
    """
    Compute comprehensive regression metrics

    Args:
        preds: Predictions
        targets: Ground truth values
        return_all: Whether to return all available metrics

    Returns:
        Dictionary of computed metrics
    """
    results: Dict[str, float] = {}

    # Basic metrics
    mse_metric = MeanSquaredError()
    mae_metric = MeanAbsoluteError()
    rmse_metric = MeanSquaredError(squared=False)
    r2_metric = R2Score()

    results.update(
        {
            "mse": mse_metric(preds, targets).item(),
            "mae": mae_metric(preds, targets).item(),
            "rmse": rmse_metric(preds, targets).item(),
            "r2": r2_metric(preds, targets).item(),
        }
    )

    if return_all:
        # Additional metrics
        residuals = targets - preds

        results.update(
            {
                "mean_residual": torch.mean(residuals).item(),
                "std_residual": torch.std(residuals).item(),
                "max_error": torch.max(torch.abs(residuals)).item(),
                "mean_abs_percentage_error": torch.mean(torch.abs(residuals / (targets + 1e-8))).item() * 100,
            }
        )

    return results


def log_confusion_matrix(
    module: LightningModule,
    preds: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    class_names: Optional[List[str]] = None,
    normalize: Optional[str] = "true",
    stage: str = "val",
) -> torch.Tensor:
    """
    Log confusion matrix to tensorboard/wandb

    Args:
        module: Lightning module for logging
        preds: Predictions (logits or labels)
        targets: Ground truth labels
        num_classes: Number of classes
        class_names: Optional class names
        normalize: Normalization ('true', 'pred', 'all', or None)
        stage: Stage name for logging

    Returns:
        Confusion matrix tensor
    """
    # Convert logits to labels if needed
    if preds.dim() > 1 and preds.size(1) > 1:
        pred_labels = torch.argmax(preds, dim=1)
    else:
        pred_labels = preds

    # Compute confusion matrix
    task: BinaryOrMulticlass = "binary" if num_classes == 2 else "multiclass"
    cm = confusion_matrix(
        pred_labels, targets, task=task, num_classes=num_classes, normalize=_normalize_method(normalize)
    )

    # Create visualization
    tick_labels: List[str] = class_names or [str(i) for i in range(num_classes)]
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm.cpu().numpy(),
        annot=True,
        fmt=".2f" if normalize else "d",
        cmap="Blues",
        xticklabels=tick_labels,
        yticklabels=tick_labels,
    )
    plt.title(f"Confusion Matrix - {stage.upper()}")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")

    # Log to tensorboard
    if hasattr(module.logger, "experiment"):
        module.logger.experiment.add_figure(f"{stage}/confusion_matrix", plt.gcf(), module.current_epoch)

    plt.close()

    return cm


def compute_calibration_metrics(probs: torch.Tensor, targets: torch.Tensor, num_bins: int = 10) -> Dict[str, Any]:
    """
    Compute calibration metrics (ECE, MCE, etc.)

    Args:
        probs: Predicted probabilities
        targets: Ground truth labels
        num_bins: Number of bins for calibration

    Returns:
        Dictionary of calibration metrics (``ece``/``mce`` floats plus per-bin lists)
    """
    # Expected Calibration Error
    ece_metric = CalibrationError(task="multiclass", num_classes=probs.size(1), n_bins=num_bins)
    ece = ece_metric(probs, targets).item()

    # Additional calibration analysis
    max_probs, pred_labels = torch.max(probs, dim=1)
    correct = pred_labels.eq(targets)

    # Bin predictions by confidence
    bin_boundaries = torch.linspace(0, 1, num_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    bin_accuracies: List[float] = []
    bin_confidences: List[float] = []
    bin_counts: List[int] = []

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = max_probs.gt(bin_lower.item()) & max_probs.le(bin_upper.item())
        prop_in_bin = in_bin.float().mean()

        if prop_in_bin.item() > 0:
            accuracy_in_bin = correct[in_bin].float().mean()
            avg_confidence_in_bin = max_probs[in_bin].mean()

            bin_accuracies.append(accuracy_in_bin.item())
            bin_confidences.append(avg_confidence_in_bin.item())
            bin_counts.append(int(in_bin.sum().item()))
        else:
            bin_accuracies.append(0)
            bin_confidences.append(0)
            bin_counts.append(0)

    # Maximum Calibration Error
    bin_accuracies_t = torch.tensor(bin_accuracies)
    bin_confidences_t = torch.tensor(bin_confidences)
    mce = torch.max(torch.abs(bin_accuracies_t - bin_confidences_t)).item()

    return {
        "ece": ece,
        "mce": mce,
        "bin_accuracies": bin_accuracies_t.tolist(),
        "bin_confidences": bin_confidences_t.tolist(),
        "bin_counts": bin_counts,
    }


def print_metrics_report(metrics: Dict[str, Any], title: str = "Metrics Report") -> None:
    """
    Print a formatted metrics report

    Args:
        metrics: Dictionary of metrics
        title: Title for the report
    """
    print(f"\n{'='*50}")
    print(f"{title:^50}")
    print(f"{'='*50}")

    for key, value in metrics.items():
        if isinstance(value, torch.Tensor):
            if value.numel() == 1:
                print(f"{key:<25}: {value.item():.4f}")
            else:
                print(f"{key:<25}: {value}")
        elif isinstance(value, (int, float)):
            print(f"{key:<25}: {value:.4f}")
        elif isinstance(value, dict):
            print(f"\n{key}:")
            for sub_key, sub_value in value.items():
                print(f"  {sub_key:<23}: {sub_value:.4f}")
        else:
            print(f"{key:<25}: {value}")

    print(f"{'='*50}\n")
