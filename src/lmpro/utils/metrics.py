# File: src/lmpro/utils/metrics.py

"""
Metrics utilities for comprehensive model evaluation
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Optional, Union, List, Tuple
from torchmetrics import (
    Accuracy, Precision, Recall, F1Score, AUROC, AveragePrecision,
    MeanSquaredError, MeanAbsoluteError, R2Score, 
    ConfusionMatrix, CalibrationError
)
from torchmetrics.functional import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from lightning import LightningModule


def get_metrics_dict(
    task: str = "multiclass", 
    num_classes: Optional[int] = None,
    num_labels: Optional[int] = None,
    average: str = "macro"
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
    metrics = {}
    
    if task in ["binary", "multiclass"]:
        metrics.update({
            "accuracy": Accuracy(task=task, num_classes=num_classes, average=average),
            "precision": Precision(task=task, num_classes=num_classes, average=average),
            "recall": Recall(task=task, num_classes=num_classes, average=average), 
            "f1": F1Score(task=task, num_classes=num_classes, average=average),
            "auroc": AUROC(task=task, num_classes=num_classes, average=average),
            "avg_precision": AveragePrecision(task=task, num_classes=num_classes, average=average),
            "confusion_matrix": ConfusionMatrix(task=task, num_classes=num_classes),
            "calibration_error": CalibrationError(task=task, num_classes=num_classes),
        })
        
        # Add per-class metrics for multiclass
        if task == "multiclass" and num_classes:
            metrics.update({
                "accuracy_per_class": Accuracy(task=task, num_classes=num_classes, average=None),
                "precision_per_class": Precision(task=task, num_classes=num_classes, average=None),
                "recall_per_class": Recall(task=task, num_classes=num_classes, average=None),
                "f1_per_class": F1Score(task=task, num_classes=num_classes, average=None),
            })
            
    elif task == "multilabel":
        metrics.update({
            "accuracy": Accuracy(task=task, num_labels=num_labels, average=average),
            "precision": Precision(task=task, num_labels=num_labels, average=average),
            "recall": Recall(task=task, num_labels=num_labels, average=average),
            "f1": F1Score(task=task, num_labels=num_labels, average=average),
            "auroc": AUROC(task=task, num_labels=num_labels, average=average),
            "avg_precision": AveragePrecision(task=task, num_labels=num_labels, average=average),
        })
        
    elif task == "regression":
        metrics.update({
            "mse": MeanSquaredError(),
            "mae": MeanAbsoluteError(),
            "rmse": MeanSquaredError(squared=False),
            "r2": R2Score(),
        })
    
    return metrics


def compute_classification_metrics(
    preds: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    class_names: Optional[List[str]] = None,
    return_per_class: bool = True
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
        probs = F.softmax(preds, dim=1)
        pred_labels = torch.argmax(preds, dim=1)
    else:
        probs = torch.sigmoid(preds)
        pred_labels = (probs > 0.5).long()
    
    task = "binary" if num_classes == 2 else "multiclass"
    
    results = {}
    
    # Basic metrics
    acc = Accuracy(task=task, num_classes=num_classes)
    prec = Precision(task=task, num_classes=num_classes, average="macro")
    rec = Recall(task=task, num_classes=num_classes, average="macro") 
    f1 = F1Score(task=task, num_classes=num_classes, average="macro")
    
    results.update({
        "accuracy": acc(pred_labels, targets).item(),
        "precision_macro": prec(pred_labels, targets).item(),
        "recall_macro": rec(pred_labels, targets).item(),
        "f1_macro": f1(pred_labels, targets).item(),
    })
    
    # AUROC and Average Precision
    try:
        auroc = AUROC(task=task, num_classes=num_classes, average="macro")
        avg_prec = AveragePrecision(task=task, num_classes=num_classes, average="macro")
        
        results.update({
            "auroc_macro": auroc(probs, targets).item(),
            "avg_precision_macro": avg_prec(probs, targets).item(),
        })
    except Exception as e:
        print(f"Warning: Could not compute AUROC/AP: {e}")
    
    # Per-class metrics
    if return_per_class:
        acc_per_class = Accuracy(task=task, num_classes=num_classes, average=None)
        prec_per_class = Precision(task=task, num_classes=num_classes, average=None)
        rec_per_class = Recall(task=task, num_classes=num_classes, average=None)
        f1_per_class = F1Score(task=task, num_classes=num_classes, average=None)
        
        results.update({
            "accuracy_per_class": acc_per_class(pred_labels, targets),
            "precision_per_class": prec_per_class(pred_labels, targets),
            "recall_per_class": rec_per_class(pred_labels, targets),
            "f1_per_class": f1_per_class(pred_labels, targets),
        })
        
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


def compute_regression_metrics(
    preds: torch.Tensor,
    targets: torch.Tensor,
    return_all: bool = True
) -> Dict[str, float]:
    """
    Compute comprehensive regression metrics
    
    Args:
        preds: Predictions
        targets: Ground truth values
        return_all: Whether to return all available metrics
        
    Returns:
        Dictionary of computed metrics
    """
    results = {}
    
    # Basic metrics
    mse_metric = MeanSquaredError()
    mae_metric = MeanAbsoluteError()
    rmse_metric = MeanSquaredError(squared=False)
    r2_metric = R2Score()
    
    results.update({
        "mse": mse_metric(preds, targets).item(),
        "mae": mae_metric(preds, targets).item(),
        "rmse": rmse_metric(preds, targets).item(),
        "r2": r2_metric(preds, targets).item(),
    })
    
    if return_all:
        # Additional metrics
        residuals = targets - preds
        
        results.update({
            "mean_residual": torch.mean(residuals).item(),
            "std_residual": torch.std(residuals).item(),
            "max_error": torch.max(torch.abs(residuals)).item(),
            "mean_abs_percentage_error": torch.mean(torch.abs(residuals / (targets + 1e-8))).item() * 100,
        })
    
    return results


def log_confusion_matrix(
    module: LightningModule,
    preds: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    class_names: Optional[List[str]] = None,
    normalize: str = "true",
    stage: str = "val"
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
    task = "binary" if num_classes == 2 else "multiclass"
    cm = confusion_matrix(pred_labels, targets, task=task, num_classes=num_classes, normalize=normalize)
    
    # Create visualization
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm.cpu().numpy(),
        annot=True,
        fmt='.2f' if normalize else 'd',
        cmap='Blues',
        xticklabels=class_names or list(range(num_classes)),
        yticklabels=class_names or list(range(num_classes))
    )
    plt.title(f'Confusion Matrix - {stage.upper()}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # Log to tensorboard
    if hasattr(module.logger, 'experiment'):
        module.logger.experiment.add_figure(
            f'{stage}/confusion_matrix',
            plt.gcf(),
            module.current_epoch
        )
    
    plt.close()
    
    return cm


def compute_calibration_metrics(
    probs: torch.Tensor,
    targets: torch.Tensor,
    num_bins: int = 10
) -> Dict[str, float]:
    """
    Compute calibration metrics (ECE, MCE, etc.)
    
    Args:
        probs: Predicted probabilities
        targets: Ground truth labels
        num_bins: Number of bins for calibration
        
    Returns:
        Dictionary of calibration metrics
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
    
    bin_accuracies = []
    bin_confidences = []
    bin_counts = []
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = max_probs.gt(bin_lower.item()) & max_probs.le(bin_upper.item())
        prop_in_bin = in_bin.float().mean()
        
        if prop_in_bin.item() > 0:
            accuracy_in_bin = correct[in_bin].float().mean()
            avg_confidence_in_bin = max_probs[in_bin].mean()
            
            bin_accuracies.append(accuracy_in_bin.item())
            bin_confidences.append(avg_confidence_in_bin.item())
            bin_counts.append(in_bin.sum().item())
        else:
            bin_accuracies.append(0)
            bin_confidences.append(0)
            bin_counts.append(0)
    
    # Maximum Calibration Error
    bin_accuracies = torch.tensor(bin_accuracies)
    bin_confidences = torch.tensor(bin_confidences)
    mce = torch.max(torch.abs(bin_accuracies - bin_confidences)).item()
    
    return {
        "ece": ece,
        "mce": mce,
        "bin_accuracies": bin_accuracies.tolist(),
        "bin_confidences": bin_confidences.tolist(),
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