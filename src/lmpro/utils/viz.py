# File: src/lmpro/utils/viz.py

"""
Visualization utilities for training curves, predictions, and analysis
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Union, Tuple, Any
import pandas as pd
from pathlib import Path
from lightning import LightningModule
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px


def plot_training_curves(
    metrics_history: Dict[str, List[float]],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 8),
    use_plotly: bool = False
) -> None:
    """
    Plot training and validation curves
    
    Args:
        metrics_history: Dictionary of metric name to list of values
        save_path: Optional path to save the plot
        figsize: Figure size for matplotlib
        use_plotly: Whether to use plotly for interactive plots
    """
    if use_plotly:
        _plot_training_curves_plotly(metrics_history, save_path)
        return
    
    # Filter metrics by train/val
    train_metrics = {k: v for k, v in metrics_history.items() if k.startswith('train_')}
    val_metrics = {k: v for k, v in metrics_history.items() if k.startswith('val_')}
    
    # Group metrics by type (loss, accuracy, etc.)
    metric_groups = {}
    for train_key, train_values in train_metrics.items():
        metric_name = train_key.replace('train_', '')
        val_key = f'val_{metric_name}'
        
        if val_key in val_metrics:
            metric_groups[metric_name] = {
                'train': train_values,
                'val': val_metrics[val_key]
            }
    
    # Create subplots
    n_metrics = len(metric_groups)
    if n_metrics == 0:
        print("No matching train/val metrics found")
        return
    
    n_cols = min(2, n_metrics)
    n_rows = (n_metrics + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_metrics == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes if isinstance(axes, (list, np.ndarray)) else [axes]
    else:
        axes = axes.flatten()
    
    # Plot each metric
    for idx, (metric_name, values) in enumerate(metric_groups.items()):
        ax = axes[idx]
        
        epochs = range(1, len(values['train']) + 1)
        ax.plot(epochs, values['train'], 'b-', label=f'Train {metric_name}', linewidth=2)
        ax.plot(epochs, values['val'], 'r-', label=f'Val {metric_name}', linewidth=2)
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric_name.capitalize())
        ax.set_title(f'{metric_name.capitalize()} Curves')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add best value annotations
        best_train_idx = np.argmin(values['train']) if 'loss' in metric_name else np.argmax(values['train'])
        best_val_idx = np.argmin(values['val']) if 'loss' in metric_name else np.argmax(values['val'])
        
        ax.annotate(f'Best: {values["train"][best_train_idx]:.4f}',
                   xy=(best_train_idx + 1, values['train'][best_train_idx]),
                   xytext=(10, 10), textcoords='offset points',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='blue', alpha=0.3))
        
        ax.annotate(f'Best: {values["val"][best_val_idx]:.4f}',
                   xy=(best_val_idx + 1, values['val'][best_val_idx]),
                   xytext=(10, -15), textcoords='offset points',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='red', alpha=0.3))
    
    # Hide extra subplots
    for idx in range(n_metrics, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training curves saved to {save_path}")
    
    plt.show()


def _plot_training_curves_plotly(
    metrics_history: Dict[str, List[float]],
    save_path: Optional[str] = None
) -> None:
    """Plot training curves using plotly for interactivity"""
    train_metrics = {k: v for k, v in metrics_history.items() if k.startswith('train_')}
    val_metrics = {k: v for k, v in metrics_history.items() if k.startswith('val_')}
    
    metric_groups = {}
    for train_key, train_values in train_metrics.items():
        metric_name = train_key.replace('train_', '')
        val_key = f'val_{metric_name}'
        
        if val_key in val_metrics:
            metric_groups[metric_name] = {
                'train': train_values,
                'val': val_metrics[val_key]
            }
    
    n_metrics = len(metric_groups)
    if n_metrics == 0:
        return
    
    # Create subplots
    fig = make_subplots(
        rows=(n_metrics + 1) // 2, cols=2,
        subplot_titles=list(metric_groups.keys()),
        vertical_spacing=0.1, horizontal_spacing=0.1
    )
    
    for idx, (metric_name, values) in enumerate(metric_groups.items()):
        row = (idx // 2) + 1
        col = (idx % 2) + 1
        
        epochs = list(range(1, len(values['train']) + 1))
        
        # Add training curve
        fig.add_trace(
            go.Scatter(
                x=epochs, y=values['train'],
                name=f'Train {metric_name}',
                line=dict(color='blue', width=2),
                hovertemplate=f'Epoch: %{{x}}<br>Train {metric_name}: %{{y:.4f}}<extra></extra>'
            ),
            row=row, col=col
        )
        
        # Add validation curve
        fig.add_trace(
            go.Scatter(
                x=epochs, y=values['val'],
                name=f'Val {metric_name}',
                line=dict(color='red', width=2),
                hovertemplate=f'Epoch: %{{x}}<br>Val {metric_name}: %{{y:.4f}}<extra></extra>'
            ),
            row=row, col=col
        )
    
    fig.update_layout(
        title="Training Curves",
        height=400 * ((n_metrics + 1) // 2),
        showlegend=True
    )
    
    fig.update_xaxes(title_text="Epoch")
    fig.update_yaxes(title_text="Value")
    
    if save_path:
        fig.write_html(save_path.replace('.png', '.html'))
        print(f"Interactive training curves saved to {save_path.replace('.png', '.html')}")
    
    fig.show()


def plot_predictions(
    y_true: Union[torch.Tensor, np.ndarray],
    y_pred: Union[torch.Tensor, np.ndarray],
    task: str = "regression",
    class_names: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8)
) -> None:
    """
    Plot predictions vs ground truth
    
    Args:
        y_true: Ground truth values/labels
        y_pred: Predicted values/probabilities
        task: Type of task ('regression', 'classification')
        class_names: Class names for classification
        save_path: Optional path to save plot
        figsize: Figure size
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()
    
    if task == "regression":
        _plot_regression_predictions(y_true, y_pred, save_path, figsize)
    elif task == "classification":
        _plot_classification_predictions(y_true, y_pred, class_names, save_path, figsize)


def _plot_regression_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: Optional[str],
    figsize: Tuple[int, int]
) -> None:
    """Plot regression predictions"""
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Scatter plot
    axes[0].scatter(y_true, y_pred, alpha=0.6, s=20)
    
    # Perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    
    # Calculate R²
    r2 = 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - y_true.mean()) ** 2)
    
    axes[0].set_xlabel('True Values')
    axes[0].set_ylabel('Predictions')
    axes[0].set_title(f'Predictions vs True Values (R² = {r2:.3f})')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Residuals plot
    residuals = y_true - y_pred
    axes[1].scatter(y_pred, residuals, alpha=0.6, s=20)
    axes[1].axhline(y=0, color='r', linestyle='--', lw=2)
    
    axes[1].set_xlabel('Predictions')
    axes[1].set_ylabel('Residuals')
    axes[1].set_title('Residuals Plot')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Prediction plot saved to {save_path}")
    
    plt.show()


def _plot_classification_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]],
    save_path: Optional[str],
    figsize: Tuple[int, int]
) -> None:
    """Plot classification predictions"""
    # Convert probabilities to predictions if needed
    if y_pred.ndim > 1:
        y_pred_labels = np.argmax(y_pred, axis=1)
        confidence = np.max(y_pred, axis=1)
    else:
        y_pred_labels = y_pred
        confidence = np.ones_like(y_pred)
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Confusion matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred_labels)
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=axes[0])
    axes[0].set_title('Confusion Matrix')
    axes[0].set_ylabel('True Label')
    axes[0].set_xlabel('Predicted Label')
    
    # Confidence distribution
    correct = y_true == y_pred_labels
    axes[1].hist(confidence[correct], bins=20, alpha=0.7, label='Correct', color='green')
    axes[1].hist(confidence[~correct], bins=20, alpha=0.7, label='Incorrect', color='red')
    
    axes[1].set_xlabel('Prediction Confidence')
    axes[1].set_ylabel('Count')
    axes[1].set_title('Confidence Distribution')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Classification plot saved to {save_path}")
    
    plt.show()


def plot_confusion_matrix(
    y_true: Union[torch.Tensor, np.ndarray],
    y_pred: Union[torch.Tensor, np.ndarray],
    class_names: Optional[List[str]] = None,
    normalize: bool = True,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 6)
) -> None:
    """
    Plot confusion matrix
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels or probabilities
        class_names: Class names for labeling
        normalize: Whether to normalize the confusion matrix
        save_path: Optional path to save plot
        figsize: Figure size
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()
    
    # Convert probabilities to labels if needed
    if y_pred.ndim > 1:
        y_pred = np.argmax(y_pred, axis=1)
    
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
        title = 'Normalized Confusion Matrix'
    else:
        fmt = 'd'
        title = 'Confusion Matrix'
    
    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    
    plt.show()


def plot_feature_importance(
    feature_names: List[str],
    importance_values: Union[torch.Tensor, np.ndarray],
    top_k: int = 20,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6)
) -> None:
    """
    Plot feature importance
    
    Args:
        feature_names: List of feature names
        importance_values: Feature importance values
        top_k: Number of top features to show
        save_path: Optional path to save plot
        figsize: Figure size
    """
    if isinstance(importance_values, torch.Tensor):
        importance_values = importance_values.cpu().numpy()
    
    # Get top k features
    indices = np.argsort(importance_values)[-top_k:]
    top_features = [feature_names[i] for i in indices]
    top_importance = importance_values[indices]
    
    plt.figure(figsize=figsize)
    plt.barh(range(len(top_features)), top_importance)
    plt.yticks(range(len(top_features)), top_features)
    plt.xlabel('Importance')
    plt.title(f'Top {top_k} Feature Importance')
    plt.grid(True, alpha=0.3, axis='x')
    
    # Add value labels on bars
    for i, v in enumerate(top_importance):
        plt.text(v + 0.01 * max(top_importance), i, f'{v:.3f}', 
                va='center', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Feature importance plot saved to {save_path}")
    
    plt.show()


def save_plot(fig, save_path: str, dpi: int = 300) -> None:
    """
    Save matplotlib figure with high quality
    
    Args:
        fig: Matplotlib figure
        save_path: Path to save the figure
        dpi: DPI for the saved figure
    """
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    print(f"Plot saved to {save_path}")


def create_learning_curve_dashboard(
    metrics_dict: Dict[str, Dict[str, List[float]]],
    save_path: Optional[str] = None
) -> None:
    """
    Create an interactive dashboard for multiple experiments
    
    Args:
        metrics_dict: Dictionary of experiment_name -> metrics_history
        save_path: Path to save HTML dashboard
    """
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=['Loss', 'Accuracy', 'F1 Score', 'Learning Rate'],
        vertical_spacing=0.1, horizontal_spacing=0.1
    )
    
    colors = px.colors.qualitative.Set1
    
    for exp_idx, (exp_name, metrics) in enumerate(metrics_dict.items()):
        color = colors[exp_idx % len(colors)]
        
        epochs = list(range(1, len(metrics.get('train_loss', [])) + 1))
        
        # Loss plot
        if 'train_loss' in metrics:
            fig.add_trace(
                go.Scatter(x=epochs, y=metrics['train_loss'], name=f'{exp_name} Train Loss',
                          line=dict(color=color, dash='solid')),
                row=1, col=1
            )
        if 'val_loss' in metrics:
            fig.add_trace(
                go.Scatter(x=epochs, y=metrics['val_loss'], name=f'{exp_name} Val Loss',
                          line=dict(color=color, dash='dash')),
                row=1, col=1
            )
        
        # Accuracy plot
        if 'train_acc' in metrics:
            fig.add_trace(
                go.Scatter(x=epochs, y=metrics['train_acc'], name=f'{exp_name} Train Acc',
                          line=dict(color=color, dash='solid'), showlegend=False),
                row=1, col=2
            )
        if 'val_acc' in metrics:
            fig.add_trace(
                go.Scatter(x=epochs, y=metrics['val_acc'], name=f'{exp_name} Val Acc',
                          line=dict(color=color, dash='dash'), showlegend=False),
                row=1, col=2
            )
        
        # F1 Score plot
        if 'val_f1' in metrics:
            fig.add_trace(
                go.Scatter(x=epochs, y=metrics['val_f1'], name=f'{exp_name} Val F1',
                          line=dict(color=color), showlegend=False),
                row=2, col=1
            )
        
        # Learning rate plot
        if 'lr' in metrics:
            fig.add_trace(
                go.Scatter(x=epochs, y=metrics['lr'], name=f'{exp_name} LR',
                          line=dict(color=color), showlegend=False),
                row=2, col=2
            )
    
    fig.update_layout(
        title="Training Dashboard - Multiple Experiments",
        height=800,
        hovermode='x unified'
    )
    
    if save_path:
        fig.write_html(save_path)
        print(f"Interactive dashboard saved to {save_path}")
    
    fig.show()