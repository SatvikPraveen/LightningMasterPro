# File: src/lmpro/utils/viz.py

"""
Visualization utilities for training curves, predictions, and analysis.

Every plotting function returns the figure it created and never displays it
unless ``show=True`` is passed, so the functions are safe to call from
scripts, tests and headless environments.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import torch
from plotly.subplots import make_subplots


def _to_numpy(values: Union[torch.Tensor, np.ndarray, List[float]]) -> np.ndarray:
    if isinstance(values, torch.Tensor):
        return values.detach().cpu().numpy()
    return np.asarray(values)


def _pair_train_val(metrics_history: Dict[str, List[float]]) -> Dict[str, Dict[str, List[float]]]:
    train_metrics = {k: v for k, v in metrics_history.items() if k.startswith("train_")}
    val_metrics = {k: v for k, v in metrics_history.items() if k.startswith("val_")}
    groups = {}
    for train_key, train_values in train_metrics.items():
        name = train_key[len("train_") :]
        val_key = f"val_{name}"
        if val_key in val_metrics:
            groups[name] = {"train": train_values, "val": val_metrics[val_key]}
    return groups


def _finish(fig: plt.Figure, save_path: Optional[str], show: bool, label: str) -> plt.Figure:
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"{label} saved to {save_path}")
    if show:
        plt.show()
    return fig


def plot_training_curves(
    metrics_history: Dict[str, List[float]],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 8),
    use_plotly: bool = False,
    show: bool = False,
) -> Optional[Any]:
    """
    Plot training and validation curves for every ``train_<x>``/``val_<x>`` pair.

    Args:
        metrics_history: Dictionary of metric name to list of values.
        save_path: Optional path to save the plot.
        figsize: Figure size for matplotlib.
        use_plotly: Whether to use plotly for interactive plots.
        show: Display the figure (``plt.show()`` / ``fig.show()``).

    Returns:
        The matplotlib (or plotly) figure, or None if nothing could be plotted.
    """
    if use_plotly:
        return _plot_training_curves_plotly(metrics_history, save_path, show=show)

    metric_groups = _pair_train_val(metrics_history)
    n_metrics = len(metric_groups)
    if n_metrics == 0:
        print("No matching train/val metrics found")
        return None

    n_cols = min(2, n_metrics)
    n_rows = (n_metrics + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
    axes = axes.flatten()

    for idx, (metric_name, values) in enumerate(metric_groups.items()):
        ax = axes[idx]
        train_values = _to_numpy(values["train"])
        val_values = _to_numpy(values["val"])
        epochs = np.arange(1, len(train_values) + 1)
        ax.plot(epochs, train_values, "b-", label=f"Train {metric_name}", linewidth=2)
        ax.plot(np.arange(1, len(val_values) + 1), val_values, "r-", label=f"Val {metric_name}", linewidth=2)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric_name.capitalize())
        ax.set_title(f"{metric_name.capitalize()} Curves")
        ax.legend()
        ax.grid(True, alpha=0.3)

        best = np.argmin if "loss" in metric_name else np.argmax
        if len(train_values):
            i = int(best(train_values))
            ax.annotate(
                f"Best: {train_values[i]:.4f}",
                xy=(i + 1, train_values[i]),
                xytext=(10, 10),
                textcoords="offset points",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="blue", alpha=0.3),
            )
        if len(val_values):
            i = int(best(val_values))
            ax.annotate(
                f"Best: {val_values[i]:.4f}",
                xy=(i + 1, val_values[i]),
                xytext=(10, -15),
                textcoords="offset points",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="red", alpha=0.3),
            )

    for idx in range(n_metrics, len(axes)):
        axes[idx].set_visible(False)

    fig.tight_layout()
    return _finish(fig, save_path, show, "Training curves")


def _plot_training_curves_plotly(
    metrics_history: Dict[str, List[float]],
    save_path: Optional[str] = None,
    show: bool = False,
) -> Optional[go.Figure]:
    """Plot training curves using plotly for interactivity."""
    metric_groups = _pair_train_val(metrics_history)
    n_metrics = len(metric_groups)
    if n_metrics == 0:
        return None

    n_rows = (n_metrics + 1) // 2
    fig = make_subplots(
        rows=n_rows,
        cols=2,
        subplot_titles=list(metric_groups.keys()),
        vertical_spacing=0.1,
        horizontal_spacing=0.1,
    )
    for idx, (metric_name, values) in enumerate(metric_groups.items()):
        row, col = idx // 2 + 1, idx % 2 + 1
        epochs = list(range(1, len(values["train"]) + 1))
        fig.add_trace(
            go.Scatter(
                x=epochs, y=list(values["train"]), name=f"Train {metric_name}", line=dict(color="blue", width=2)
            ),
            row=row,
            col=col,
        )
        fig.add_trace(
            go.Scatter(
                x=list(range(1, len(values["val"]) + 1)),
                y=list(values["val"]),
                name=f"Val {metric_name}",
                line=dict(color="red", width=2),
            ),
            row=row,
            col=col,
        )

    fig.update_layout(title="Training Curves", height=400 * n_rows, showlegend=True)
    fig.update_xaxes(title_text="Epoch")
    fig.update_yaxes(title_text="Value")

    if save_path:
        html_path = str(Path(save_path).with_suffix(".html"))
        fig.write_html(html_path)
        print(f"Interactive training curves saved to {html_path}")
    if show:
        fig.show()
    return fig


def plot_predictions(
    y_true: Union[torch.Tensor, np.ndarray],
    y_pred: Union[torch.Tensor, np.ndarray],
    task: str = "regression",
    class_names: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8),
    show: bool = False,
) -> plt.Figure:
    """
    Plot predictions vs ground truth.

    Args:
        y_true: Ground truth values/labels.
        y_pred: Predicted values/probabilities.
        task: 'regression' or 'classification'.
        class_names: Class names for classification.
        save_path: Optional path to save plot.
        figsize: Figure size.
        show: Display the figure.
    """
    y_true = _to_numpy(y_true)
    y_pred = _to_numpy(y_pred)
    if task == "regression":
        return _plot_regression_predictions(y_true, y_pred, save_path, figsize, show)
    if task == "classification":
        return _plot_classification_predictions(y_true, y_pred, class_names, save_path, figsize, show)
    raise ValueError(f"Unknown task: {task}")


def _plot_regression_predictions(
    y_true: np.ndarray, y_pred: np.ndarray, save_path: Optional[str], figsize: Tuple[int, int], show: bool
) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    axes[0].scatter(y_true, y_pred, alpha=0.6, s=20)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    axes[0].plot([min_val, max_val], [min_val, max_val], "r--", lw=2, label="Perfect Prediction")
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r2 = 1 - np.sum((y_true - y_pred) ** 2) / ss_tot if ss_tot > 0 else float("nan")
    axes[0].set_xlabel("True Values")
    axes[0].set_ylabel("Predictions")
    axes[0].set_title(f"Predictions vs True Values (R^2 = {r2:.3f})")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    residuals = y_true - y_pred
    axes[1].scatter(y_pred, residuals, alpha=0.6, s=20)
    axes[1].axhline(y=0, color="r", linestyle="--", lw=2)
    axes[1].set_xlabel("Predictions")
    axes[1].set_ylabel("Residuals")
    axes[1].set_title("Residuals Plot")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    return _finish(fig, save_path, show, "Prediction plot")


def _plot_classification_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]],
    save_path: Optional[str],
    figsize: Tuple[int, int],
    show: bool,
) -> plt.Figure:
    from sklearn.metrics import confusion_matrix

    if y_pred.ndim > 1:
        y_pred_labels = np.argmax(y_pred, axis=1)
        confidence = np.max(y_pred, axis=1)
    else:
        y_pred_labels = y_pred
        confidence = np.ones_like(y_pred, dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    cm = confusion_matrix(y_true, y_pred_labels)
    tick_labels = class_names if class_names is not None else list(range(cm.shape[0]))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=tick_labels, yticklabels=tick_labels, ax=axes[0])
    axes[0].set_title("Confusion Matrix")
    axes[0].set_ylabel("True Label")
    axes[0].set_xlabel("Predicted Label")

    correct = y_true == y_pred_labels
    axes[1].hist(confidence[correct], bins=20, alpha=0.7, label="Correct", color="green")
    axes[1].hist(confidence[~correct], bins=20, alpha=0.7, label="Incorrect", color="red")
    axes[1].set_xlabel("Prediction Confidence")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Confidence Distribution")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    return _finish(fig, save_path, show, "Classification plot")


def plot_confusion_matrix(
    y_true: Union[torch.Tensor, np.ndarray],
    y_pred: Union[torch.Tensor, np.ndarray],
    class_names: Optional[List[str]] = None,
    normalize: bool = True,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 6),
    show: bool = False,
) -> plt.Figure:
    """Plot a (optionally row-normalised) confusion matrix."""
    from sklearn.metrics import confusion_matrix

    y_true = _to_numpy(y_true)
    y_pred = _to_numpy(y_pred)
    if y_pred.ndim > 1:
        y_pred = np.argmax(y_pred, axis=1)

    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        cm = np.divide(cm.astype(float), row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums > 0)
        fmt, title = ".2f", "Normalized Confusion Matrix"
    else:
        fmt, title = "d", "Confusion Matrix"

    fig, ax = plt.subplots(figsize=figsize)
    tick_labels = class_names if class_names is not None else list(range(cm.shape[0]))
    sns.heatmap(cm, annot=True, fmt=fmt, cmap="Blues", xticklabels=tick_labels, yticklabels=tick_labels, ax=ax)
    ax.set_title(title)
    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")
    return _finish(fig, save_path, show, "Confusion matrix")


def plot_feature_importance(
    feature_names: List[str],
    importance_values: Union[torch.Tensor, np.ndarray],
    top_k: int = 20,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6),
    show: bool = False,
) -> plt.Figure:
    """Horizontal bar chart of the ``top_k`` most important features."""
    importance_values = _to_numpy(importance_values)
    top_k = min(top_k, len(importance_values))
    indices = np.argsort(importance_values)[-top_k:]
    top_features = [feature_names[i] for i in indices]
    top_importance = importance_values[indices]

    fig, ax = plt.subplots(figsize=figsize)
    ax.barh(range(len(top_features)), top_importance)
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features)
    ax.set_xlabel("Importance")
    ax.set_title(f"Top {top_k} Feature Importance")
    ax.grid(True, alpha=0.3, axis="x")

    scale = float(np.max(top_importance)) if len(top_importance) else 1.0
    for i, v in enumerate(top_importance):
        ax.text(v + 0.01 * scale, i, f"{v:.3f}", va="center", fontsize=9)

    fig.tight_layout()
    return _finish(fig, save_path, show, "Feature importance plot")


def save_plot(fig: plt.Figure, save_path: str, dpi: int = 300) -> None:
    """Save a matplotlib figure, creating parent directories as needed."""
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    print(f"Plot saved to {save_path}")


def create_learning_curve_dashboard(
    metrics_dict: Dict[str, Dict[str, List[float]]],
    save_path: Optional[str] = None,
    show: bool = False,
) -> go.Figure:
    """
    Create an interactive plotly dashboard comparing multiple experiments.

    Args:
        metrics_dict: Dictionary of experiment_name -> metrics_history.
        save_path: Path to save the HTML dashboard.
        show: Display the figure.
    """
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=["Loss", "Accuracy", "F1 Score", "Learning Rate"],
        vertical_spacing=0.1,
        horizontal_spacing=0.1,
    )
    colors = px.colors.qualitative.Set1
    panels = [
        ("train_loss", "Train Loss", 1, 1, "solid", True),
        ("val_loss", "Val Loss", 1, 1, "dash", True),
        ("train_acc", "Train Acc", 1, 2, "solid", False),
        ("val_acc", "Val Acc", 1, 2, "dash", False),
        ("val_f1", "Val F1", 2, 1, "solid", False),
        ("lr", "LR", 2, 2, "solid", False),
    ]

    for exp_idx, (exp_name, metrics) in enumerate(metrics_dict.items()):
        color = colors[exp_idx % len(colors)]
        for key, label, row, col, dash, legend in panels:
            if key not in metrics:
                continue
            values = list(metrics[key])
            fig.add_trace(
                go.Scatter(
                    x=list(range(1, len(values) + 1)),
                    y=values,
                    name=f"{exp_name} {label}",
                    line=dict(color=color, dash=dash),
                    showlegend=legend,
                ),
                row=row,
                col=col,
            )

    fig.update_layout(title="Training Dashboard - Multiple Experiments", height=800, hovermode="x unified")
    if save_path:
        fig.write_html(save_path)
        print(f"Interactive dashboard saved to {save_path}")
    if show:
        fig.show()
    return fig
