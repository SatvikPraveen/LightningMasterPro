# tests/test_callbacks_ema.py
"""Tests for the EMACallback."""

import sys
from pathlib import Path
import pytest
import torch
import torch.nn as nn
import lightning as L
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lmpro.callbacks.ema import EMACallback, create_ema_callback


class SimpleModel(L.LightningModule):
    """Minimal Lightning module for testing callbacks."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 2)

    def forward(self, x):
        return self.fc(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        return torch.nn.functional.mse_loss(logits, y.float())

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)


@pytest.fixture
def simple_model():
    return SimpleModel()


@pytest.fixture
def ema_callback():
    return EMACallback(decay=0.99, start_epoch=0, use_ema_for_validation=True)


class TestEMACallbackInit:
    def test_default_params(self):
        cb = EMACallback()
        assert cb.decay == 0.999
        assert cb.start_epoch == 0
        assert cb.update_every == 1
        assert cb.use_ema_for_validation is True
        assert cb.ema_model is None

    def test_custom_params(self):
        cb = EMACallback(decay=0.9, start_epoch=5, update_every=2, use_ema_for_validation=False)
        assert cb.decay == 0.9
        assert cb.start_epoch == 5
        assert cb.update_every == 2
        assert cb.use_ema_for_validation is False


class TestEMACallbackFitStart:
    def test_ema_model_initialized(self, simple_model, ema_callback):
        trainer = MagicMock()
        ema_callback.on_fit_start(trainer, simple_model)
        assert ema_callback.ema_model is not None

    def test_ema_model_has_same_keys(self, simple_model, ema_callback):
        trainer = MagicMock()
        ema_callback.on_fit_start(trainer, simple_model)
        assert set(ema_callback.ema_model.keys()) == set(simple_model.state_dict().keys())

    def test_ema_model_correct_shapes(self, simple_model, ema_callback):
        trainer = MagicMock()
        ema_callback.on_fit_start(trainer, simple_model)
        for key in simple_model.state_dict():
            assert ema_callback.ema_model[key].shape == simple_model.state_dict()[key].shape


class TestEMAUpdate:
    def test_ema_update_changes_weights(self, simple_model, ema_callback):
        trainer = MagicMock()
        trainer.current_epoch = 0
        ema_callback.on_fit_start(trainer, simple_model)

        # Snapshot initial EMA weights
        initial_ema = {k: v.clone() for k, v in ema_callback.ema_model.items()}

        # Modify model weights
        with torch.no_grad():
            for p in simple_model.parameters():
                p.add_(torch.ones_like(p) * 10.0)

        # Trigger update
        outputs = MagicMock()
        batch = (torch.randn(2, 4), torch.randn(2, 2))
        ema_callback.on_train_batch_end(trainer, simple_model, outputs, batch, 0)

        # EMA should have changed
        changed = any(
            not torch.allclose(ema_callback.ema_model[k], initial_ema[k])
            for k in initial_ema
            if isinstance(initial_ema[k], torch.Tensor)
        )
        assert changed

    def test_ema_update_skipped_before_start_epoch(self, simple_model):
        cb = EMACallback(start_epoch=5)
        trainer = MagicMock()
        trainer.current_epoch = 0
        cb.on_fit_start(trainer, simple_model)

        initial_ema = {k: v.clone() for k, v in cb.ema_model.items()}
        with torch.no_grad():
            for p in simple_model.parameters():
                p.add_(torch.ones_like(p) * 100.0)

        outputs = MagicMock()
        batch = (torch.randn(2, 4), torch.randn(2, 2))
        cb.on_train_batch_end(trainer, simple_model, outputs, batch, 0)

        # EMA should NOT change before start_epoch
        unchanged = all(
            torch.allclose(cb.ema_model[k], initial_ema[k])
            for k in initial_ema
            if isinstance(initial_ema[k], torch.Tensor)
        )
        assert unchanged

    def test_update_every_respected(self, simple_model):
        cb = EMACallback(update_every=3)
        trainer = MagicMock()
        trainer.current_epoch = 0
        cb.on_fit_start(trainer, simple_model)
        initial_ema = {k: v.clone() for k, v in cb.ema_model.items()}
        with torch.no_grad():
            for p in simple_model.parameters():
                p.add_(torch.ones_like(p) * 10.0)

        outputs = MagicMock()
        batch = (torch.randn(2, 4), torch.randn(2, 2))

        # Call only 2 times (less than update_every=3 → no update)
        cb.on_train_batch_end(trainer, simple_model, outputs, batch, 0)
        cb.on_train_batch_end(trainer, simple_model, outputs, batch, 1)

        unchanged = all(
            torch.allclose(cb.ema_model[k], initial_ema[k])
            for k in initial_ema
            if isinstance(initial_ema[k], torch.Tensor)
        )
        assert unchanged


class TestEMAWeightSwap:
    def test_apply_ema_weights(self, simple_model, ema_callback):
        trainer = MagicMock()
        ema_callback.on_fit_start(trainer, simple_model)

        # Diverge model weights from EMA
        with torch.no_grad():
            for p in simple_model.parameters():
                p.fill_(999.0)

        ema_callback.apply_ema_weights(simple_model)

        # After applying EMA, model weights should NOT be 999
        all_999 = all(
            torch.all(p == 999.0) for p in simple_model.parameters()
        )
        assert not all_999

    def test_get_ema_model_returns_dict(self, simple_model, ema_callback):
        trainer = MagicMock()
        ema_callback.on_fit_start(trainer, simple_model)
        ema = ema_callback.get_ema_model()
        assert isinstance(ema, dict)

    def test_get_ema_model_before_init_returns_none(self):
        cb = EMACallback()
        assert cb.get_ema_model() is None


class TestEMACheckpoint:
    def test_state_dict_serializable(self, simple_model, ema_callback):
        trainer = MagicMock()
        ema_callback.on_fit_start(trainer, simple_model)
        sd = ema_callback.state_dict()
        assert "decay" in sd
        assert "ema_model" in sd

    def test_load_state_dict_roundtrip(self, simple_model, ema_callback):
        trainer = MagicMock()
        ema_callback.on_fit_start(trainer, simple_model)
        sd = ema_callback.state_dict()

        new_cb = EMACallback()
        new_cb.load_state_dict(sd)
        assert new_cb.decay == ema_callback.decay
        assert set(new_cb.ema_model.keys()) == set(ema_callback.ema_model.keys())


class TestCreateEMACallback:
    def test_factory_returns_ema_callback(self):
        cb = create_ema_callback(decay=0.995)
        assert isinstance(cb, EMACallback)
        assert cb.decay == 0.995
