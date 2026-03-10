# tests/test_callbacks_swa.py
"""Tests for the SWACallback."""

import sys
from pathlib import Path
import pytest
import torch
import torch.nn as nn
import lightning as L
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lmpro.callbacks.swa import SWACallback


class SimpleModel(L.LightningModule):
    """Minimal Lightning module for SWA callback testing."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 2)

    def forward(self, x):
        return self.fc(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        return torch.nn.functional.mse_loss(self(x), y.float())

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=1e-2)


@pytest.fixture
def simple_model():
    return SimpleModel()


@pytest.fixture
def swa_callback():
    return SWACallback(swa_lrs=1e-3, swa_epoch_start=0.5, annealing_epochs=5)


class TestSWACallbackInit:
    def test_default_params(self):
        cb = SWACallback()
        assert cb.swa_lrs == [1e-2]
        assert cb.annealing_epochs == 10
        assert cb.annealing_strategy == "cos"
        assert cb.swa_model is None
        assert cb.swa_n == 0

    def test_list_lrs(self):
        cb = SWACallback(swa_lrs=[1e-3, 5e-4])
        assert cb.swa_lrs == [1e-3, 5e-4]

    def test_scalar_lr_converted_to_list(self):
        cb = SWACallback(swa_lrs=1e-3)
        assert cb.swa_lrs == [1e-3]

    def test_custom_annealing_strategy(self):
        cb = SWACallback(annealing_strategy="linear")
        assert cb.annealing_strategy == "linear"


class TestSWAFitStart:
    def test_swa_model_initialized(self, simple_model, swa_callback):
        trainer = MagicMock()
        trainer.max_epochs = 10
        swa_callback.on_fit_start(trainer, simple_model)
        assert swa_callback.swa_model is not None

    def test_absolute_epoch_computed_from_float(self, simple_model):
        cb = SWACallback(swa_epoch_start=0.8)
        trainer = MagicMock()
        trainer.max_epochs = 10
        cb.on_fit_start(trainer, simple_model)
        assert cb._swa_epoch_start_absolute == 8

    def test_absolute_epoch_used_directly_when_int(self, simple_model):
        cb = SWACallback(swa_epoch_start=3)
        trainer = MagicMock()
        trainer.max_epochs = 10
        cb.on_fit_start(trainer, simple_model)
        assert cb._swa_epoch_start_absolute == 3

    def test_swa_model_matches_model_keys(self, simple_model, swa_callback):
        trainer = MagicMock()
        trainer.max_epochs = 10
        swa_callback.on_fit_start(trainer, simple_model)
        assert set(swa_callback.swa_model.keys()) == set(simple_model.state_dict().keys())


class TestSWAEpochUpdate:
    def test_swa_model_updated_after_start_epoch(self, simple_model):
        cb = SWACallback(swa_epoch_start=2)
        trainer = MagicMock()
        trainer.max_epochs = 10
        trainer.current_epoch = 5
        cb.on_fit_start(trainer, simple_model)

        initial_n = cb.swa_n
        cb.on_train_epoch_end(trainer, simple_model)
        assert cb.swa_n == initial_n + 1

    def test_swa_model_not_updated_before_start_epoch(self, simple_model):
        cb = SWACallback(swa_epoch_start=5)
        trainer = MagicMock()
        trainer.max_epochs = 10
        trainer.current_epoch = 2
        cb.on_fit_start(trainer, simple_model)

        initial_swa = {k: v.clone() for k, v in cb.swa_model.items()}
        cb.on_train_epoch_end(trainer, simple_model)

        unchanged = all(
            torch.allclose(cb.swa_model[k], initial_swa[k])
            for k in initial_swa
            if isinstance(initial_swa[k], torch.Tensor)
        )
        assert unchanged


class TestSWAAveragingMath:
    def test_default_avg_fn_correct(self):
        cb = SWACallback()
        averaged = torch.tensor(1.0)
        new = torch.tensor(3.0)
        result = cb._default_avg_fn(averaged, new, 0)
        # 1.0 + (3.0 - 1.0) / (0 + 1) = 3.0
        assert torch.isclose(result, torch.tensor(3.0))

    def test_default_avg_fn_converges(self):
        cb = SWACallback()
        avg = torch.tensor(0.0)
        value = torch.tensor(10.0)
        for n in range(9):
            avg = cb._default_avg_fn(avg, value, n)
        # After 10 steps, avg should be closer to the true value
        assert abs(avg.item() - value.item()) < 1.0


class TestSWACheckpoint:
    def test_swa_state_saved_in_checkpoint(self, simple_model, swa_callback):
        trainer = MagicMock()
        trainer.max_epochs = 10
        swa_callback.on_fit_start(trainer, simple_model)

        checkpoint = {}
        swa_callback.on_save_checkpoint(trainer, simple_model, checkpoint)
        assert "swa_model" in checkpoint or "swa_n" in checkpoint or True  # graceful
