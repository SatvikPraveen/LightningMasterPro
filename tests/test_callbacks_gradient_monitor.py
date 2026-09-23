# tests/test_callbacks_gradient_monitor.py
"""Tests for GradientMonitorCallback with a real Trainer run."""

import sys
from pathlib import Path

import lightning as L
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lmpro.callbacks.gradient_monitor import GradientMonitorCallback, create_gradient_monitor  # noqa: E402


class SimpleModel(L.LightningModule):
    def __init__(self, scale: float = 1.0):
        super().__init__()
        self.fc = nn.Linear(4, 2)
        self.scale = scale

    def forward(self, x):
        return self.fc(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        return self.scale * nn.functional.mse_loss(self(x), y)

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=1e-2)


def run(callback, model=None):
    loader = DataLoader(TensorDataset(torch.randn(8, 4), torch.randn(8, 2)), batch_size=4)
    trainer = L.Trainer(
        max_epochs=1,
        limit_train_batches=2,
        callbacks=[callback],
        accelerator="cpu",
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    trainer.fit(model or SimpleModel(), loader)
    return trainer


class TestGradientMonitorInit:
    def test_defaults(self):
        cb = GradientMonitorCallback()
        assert cb.log_every_n_steps == 50
        assert cb.log_per_layer is False
        assert cb.vanishing_threshold == 1e-7
        assert cb.exploding_threshold == 1e3
        assert cb.get_global_norm_history() == []

    def test_factory(self):
        cb = create_gradient_monitor(log_every_n_steps=5, log_per_layer=True, norm_type=1.0)
        assert isinstance(cb, GradientMonitorCallback)
        assert cb.log_every_n_steps == 5 and cb.log_per_layer is True and cb.norm_type == 1.0


class TestGradientMonitorTraining:
    def test_logs_global_norm_every_step(self):
        cb = GradientMonitorCallback(log_every_n_steps=1)
        trainer = run(cb)
        history = cb.get_global_norm_history()
        assert [step for step, _ in history] == [0, 1]
        assert all(norm > 0 for _, norm in history)
        assert cb.vanishing_event_count == 0
        assert cb.exploding_event_count == 0
        assert "grad_norm/global" in trainer.logged_metrics

    def test_per_layer_logging(self):
        cb = GradientMonitorCallback(log_every_n_steps=1, log_per_layer=True)
        trainer = run(cb)
        assert "grad_norm/fc/weight" in trainer.logged_metrics
        assert "grad_norm/fc/bias" in trainer.logged_metrics

    def test_log_interval_respected(self):
        cb = GradientMonitorCallback(log_every_n_steps=2)
        run(cb)
        assert [step for step, _ in cb.get_global_norm_history()] == [0]

    def test_exploding_gradient_detected(self):
        cb = GradientMonitorCallback(log_every_n_steps=1, exploding_threshold=1e-6, vanishing_threshold=None)
        with pytest.warns(UserWarning, match="Exploding gradient"):
            run(cb, SimpleModel(scale=1e3))
        assert cb.exploding_event_count == 2

    def test_vanishing_gradient_detected(self):
        cb = GradientMonitorCallback(log_every_n_steps=1, vanishing_threshold=1e6, exploding_threshold=None)
        with pytest.warns(UserWarning, match="Vanishing gradient"):
            run(cb)
        assert cb.vanishing_event_count == 2

    def test_reset_history(self):
        cb = GradientMonitorCallback(log_every_n_steps=1)
        run(cb)
        cb.reset_history()
        assert cb.get_global_norm_history() == []
        assert cb.vanishing_event_count == 0 and cb.exploding_event_count == 0
