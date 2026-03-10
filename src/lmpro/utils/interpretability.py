# src/lmpro/utils/interpretability.py
"""
Model interpretability utilities for LightningMasterPro.

Provides lightweight, dependency-free tools for:
- Gradient-based feature importance (vanilla backprop saliency)
- Integrated Gradients
- Input-perturbation feature importance (tabular)
- Simple occlusion sensitivity (vision)
- Activation statistics summary
"""

from typing import Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import numpy as np


# ─── Gradient-Based Saliency ─────────────────────────────────────────────────

def compute_saliency_map(
    model: nn.Module,
    inputs: torch.Tensor,
    target_class: Optional[int] = None,
    abs_value: bool = True,
) -> torch.Tensor:
    """
    Compute a vanilla gradient saliency map for the given inputs.

    Args:
        model: The neural network model (in eval mode preferred).
        inputs: Input tensor of shape (B, ...). Requires grad will be set.
        target_class: Class index to differentiate w.r.t. If None, uses
            the argmax of the model output.
        abs_value: If True, return the absolute value of gradients.

    Returns:
        Saliency map tensor with the same shape as `inputs`.
    """
    model.eval()
    inp = inputs.clone().detach().requires_grad_(True)
    output = model(inp)

    if target_class is None:
        target_class = output.argmax(dim=-1)

    if isinstance(target_class, int):
        score = output[:, target_class].sum()
    else:
        score = output.gather(1, target_class.view(-1, 1)).sum()

    model.zero_grad()
    score.backward()

    saliency = inp.grad.detach()
    if abs_value:
        saliency = saliency.abs()
    return saliency


# ─── Integrated Gradients ────────────────────────────────────────────────────

def integrated_gradients(
    model: nn.Module,
    inputs: torch.Tensor,
    baseline: Optional[torch.Tensor] = None,
    target_class: Optional[int] = None,
    n_steps: int = 50,
) -> torch.Tensor:
    """
    Compute integrated gradients attribution for the given inputs.

    Args:
        model: The neural network model.
        inputs: Input tensor of shape (B, ...).
        baseline: Baseline tensor (same shape as inputs). Defaults to zeros.
        target_class: Class index to attribute. If None, uses argmax.
        n_steps: Number of interpolation steps (higher → more accurate).

    Returns:
        Attribution tensor with the same shape as `inputs`.
    """
    if baseline is None:
        baseline = torch.zeros_like(inputs)

    # Determine target class from full input if not given
    if target_class is None:
        with torch.no_grad():
            out = model(inputs)
        target_class = out.argmax(dim=-1)

    # Build interpolated inputs: (n_steps, B, ...)
    alphas = torch.linspace(0.0, 1.0, n_steps, device=inputs.device)
    # Expand dimensions for broadcasting
    extra_dims = (1,) * (inputs.ndim - 1)
    alphas = alphas.view(-1, *extra_dims)
    interpolated = baseline.unsqueeze(0) + alphas * (inputs - baseline).unsqueeze(0)
    interpolated = interpolated.view(-1, *inputs.shape[1:]).requires_grad_(True)

    model.eval()
    output = model(interpolated)

    if isinstance(target_class, int):
        score = output[:, target_class].sum()
    else:
        # Repeat target class for each interpolation step
        tc = target_class.repeat(n_steps)
        score = output.gather(1, tc.view(-1, 1)).sum()

    model.zero_grad()
    score.backward()

    grads = interpolated.grad.detach().view(n_steps, *inputs.shape)
    avg_grads = grads.mean(dim=0)
    attrs = (inputs - baseline) * avg_grads
    return attrs


# ─── Perturbation-Based Feature Importance (Tabular) ─────────────────────────

def feature_importance_perturbation(
    model: nn.Module,
    inputs: torch.Tensor,
    feature_names: Optional[List[str]] = None,
    n_perturbations: int = 10,
    noise_scale: float = 0.1,
    reduction: str = "mean",
) -> Dict[str, float]:
    """
    Estimate feature importance for tabular data by corrupting each feature
    and measuring the change in model confidence.

    Args:
        model: A classification model returning logits of shape (B, C).
        inputs: Input tensor of shape (B, num_features).
        feature_names: Optional list of feature names (len = num_features).
        n_perturbations: Number of noise samples per feature.
        noise_scale: Scale of Gaussian noise added to each feature.
        reduction: "mean" or "max" — how to aggregate across the batch.

    Returns:
        Dict mapping feature_name (or "feature_i") → importance score.
    """
    model.eval()
    num_features = inputs.shape[-1]
    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(num_features)]

    with torch.no_grad():
        base_logits = model(inputs)
        base_conf = base_logits.softmax(dim=-1).max(dim=-1).values  # (B,)

    importances: Dict[str, float] = {}
    for feat_idx, feat_name in enumerate(feature_names):
        delta_sum = torch.zeros(inputs.shape[0], device=inputs.device)
        for _ in range(n_perturbations):
            perturbed = inputs.clone()
            noise = torch.randn_like(perturbed[:, feat_idx]) * noise_scale
            perturbed[:, feat_idx] = perturbed[:, feat_idx] + noise

            with torch.no_grad():
                pert_logits = model(perturbed)
                pert_conf = pert_logits.softmax(dim=-1).max(dim=-1).values

            delta_sum += (base_conf - pert_conf).abs()

        avg_delta = delta_sum / n_perturbations
        if reduction == "max":
            importances[feat_name] = avg_delta.max().item()
        else:
            importances[feat_name] = avg_delta.mean().item()

    # Normalize to [0, 1]
    max_imp = max(importances.values()) if importances else 1.0
    if max_imp > 0:
        importances = {k: v / max_imp for k, v in importances.items()}

    return importances


# ─── Occlusion Sensitivity (Vision) ──────────────────────────────────────────

def occlusion_sensitivity(
    model: nn.Module,
    image: torch.Tensor,
    target_class: int,
    patch_size: int = 8,
    stride: int = 4,
    occlusion_value: float = 0.0,
) -> torch.Tensor:
    """
    Compute an occlusion sensitivity map for a single image.

    Slides an occlusion patch across the image and measures the drop in
    output confidence.

    Args:
        model: Classification model.
        image: Image tensor of shape (C, H, W) or (1, C, H, W).
        target_class: Target class index.
        patch_size: Size of the occlusion square patch.
        stride: Slide stride.
        occlusion_value: Fill value for the occluded region (default: 0.0).

    Returns:
        Sensitivity map of shape (H, W) — higher = more important region.
    """
    model.eval()

    if image.ndim == 3:
        image = image.unsqueeze(0)  # (1, C, H, W)

    _, C, H, W = image.shape

    with torch.no_grad():
        base_logits = model(image)
        base_score = base_logits.softmax(dim=-1)[0, target_class].item()

    sensitivity = torch.zeros(H, W, device=image.device)
    counts = torch.zeros(H, W, device=image.device)

    for y in range(0, H - patch_size + 1, stride):
        for x in range(0, W - patch_size + 1, stride):
            occluded = image.clone()
            occluded[:, :, y : y + patch_size, x : x + patch_size] = occlusion_value

            with torch.no_grad():
                pert_logits = model(occluded)
                pert_score = pert_logits.softmax(dim=-1)[0, target_class].item()

            drop = base_score - pert_score
            sensitivity[y : y + patch_size, x : x + patch_size] += drop
            counts[y : y + patch_size, x : x + patch_size] += 1

    # Average overlapping regions
    counts = counts.clamp(min=1)
    sensitivity = sensitivity / counts
    return sensitivity


# ─── Activation Statistics ───────────────────────────────────────────────────

class ActivationStatsHook:
    """
    Registers forward hooks on named modules to capture activation statistics.

    Usage::

        hook = ActivationStatsHook(model, layers=["layer1", "layer2"])
        _ = model(inputs)
        stats = hook.get_stats()
        hook.remove()
    """

    def __init__(
        self,
        model: nn.Module,
        layers: Optional[List[str]] = None,
    ):
        self.stats: Dict[str, Dict[str, float]] = {}
        self._hooks = []

        named_modules = dict(model.named_modules())
        if layers is None:
            layers = [n for n, m in named_modules.items()
                      if isinstance(m, (nn.Linear, nn.Conv2d, nn.LSTM, nn.GRU))]

        for layer_name in layers:
            if layer_name not in named_modules:
                continue
            module = named_modules[layer_name]

            def make_hook(name):
                def hook_fn(module, input, output):
                    act = output.detach().float()
                    self.stats[name] = {
                        "mean": act.mean().item(),
                        "std": act.std().item(),
                        "min": act.min().item(),
                        "max": act.max().item(),
                        "dead_fraction": (act == 0).float().mean().item(),
                    }
                return hook_fn

            handle = module.register_forward_hook(make_hook(layer_name))
            self._hooks.append(handle)

    def get_stats(self) -> Dict[str, Dict[str, float]]:
        """Return captured activation statistics."""
        return dict(self.stats)

    def remove(self) -> None:
        """Remove all registered hooks."""
        for handle in self._hooks:
            handle.remove()
        self._hooks.clear()
