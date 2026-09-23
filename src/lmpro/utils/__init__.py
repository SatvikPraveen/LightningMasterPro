# File: src/lmpro/utils/__init__.py

"""
Utility functions and helpers for LightningMasterPro
"""

from .interpretability import (
    ActivationStatsHook,
    compute_saliency_map,
    feature_importance_perturbation,
    integrated_gradients,
    occlusion_sensitivity,
)
from .metrics import (
    compute_classification_metrics,
    compute_regression_metrics,
    get_metrics_dict,
    log_confusion_matrix,
)
from .seed import (
    SeedContext,
    get_random_state,
    seed_everything_deterministic,
    set_random_state,
    worker_init_fn,
)
from .viz import (
    create_learning_curve_dashboard,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_predictions,
    plot_training_curves,
    save_plot,
)

__all__ = [
    # Seed utilities
    "seed_everything_deterministic",
    "get_random_state",
    "set_random_state",
    "SeedContext",
    "worker_init_fn",
    # Metrics utilities
    "get_metrics_dict",
    "log_confusion_matrix",
    "compute_classification_metrics",
    "compute_regression_metrics",
    # Visualization utilities
    "plot_training_curves",
    "plot_predictions",
    "plot_confusion_matrix",
    "plot_feature_importance",
    "save_plot",
    "create_learning_curve_dashboard",
    # Interpretability utilities
    "compute_saliency_map",
    "integrated_gradients",
    "feature_importance_perturbation",
    "occlusion_sensitivity",
    "ActivationStatsHook",
]
