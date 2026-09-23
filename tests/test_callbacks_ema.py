# tests/test_callbacks_ema.py
"""Tests for the EMACallback."""

import sys
from pathlib import Path

import lightning as L
import pytest
import torch
import torch.nn as nn
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lmpro.callbacks.ema import EMACallback, create_ema_callback  # noqa: E402


class SimpleModel(L.LightningModule):
    """Minimal Lightning module (with BatchNorm, so non-float buffers exist)."""

    def __init__(self):
        super().__init__()
        self.bn = nn.BatchNorm1d(4)
        self.fc = nn.Linear(4, 2)

    def forward(self, x):
        return self.fc(self.bn(x))

    def training_step(self, batch, batch_idx):
        x, y = batch
        return nn.functional.mse_loss(self(x), y)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        self.log("val_loss", nn.functional.mse_loss(self(x), y))

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.5)


def make_loader(n=16):
    g = torch.Generator().manual_seed(1)
    return DataLoader(TensorDataset(torch.randn(n, 4, generator=g), torch.randn(n, 2, generator=g)), batch_size=4)


def make_trainer(max_epochs, callbacks, tmp_path):
    return L.Trainer(
        max_epochs=max_epochs,
        callbacks=callbacks,
        default_root_dir=str(tmp_path),
        accelerator="cpu",
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
    )


def float_state(state):
    return {k: v for k, v in state.items() if v.is_floating_point()}


class FakeTrainer:
    """Only the attributes EMACallback reads."""

    def __init__(self, current_epoch=0):
        self.current_epoch = current_epoch


@pytest.fixture
def simple_model():
    return SimpleModel()


@pytest.fixture
def ema_callback():
    return EMACallback(decay=0.99, start_epoch=0, use_ema_for_validation=True, warmup=False)


class TestEMACallbackInit:
    def test_default_params(self):
        cb = EMACallback()
        assert cb.decay == 0.999
        assert cb.start_epoch == 0
        assert cb.update_every == 1
        assert cb.use_ema_for_validation is True
        assert cb.warmup is True
        assert cb.ema_model is None

    def test_custom_params(self):
        cb = EMACallback(decay=0.9, start_epoch=5, update_every=2, use_ema_for_validation=False)
        assert cb.decay == 0.9
        assert cb.start_epoch == 5
        assert cb.update_every == 2
        assert cb.use_ema_for_validation is False

    def test_invalid_params(self):
        with pytest.raises(ValueError):
            EMACallback(decay=1.5)
        with pytest.raises(ValueError):
            EMACallback(update_every=0)

    def test_warmup_decay_schedule(self):
        cb = EMACallback(decay=0.999)
        assert cb.current_decay() == pytest.approx(0.1)
        cb.num_updates = 10
        assert cb.current_decay() == pytest.approx(11 / 20)
        cb.num_updates = 100000
        assert cb.current_decay() == pytest.approx(0.999)


class TestEMACallbackFitStart:
    def test_ema_model_initialized_with_same_keys_and_shapes(self, simple_model, ema_callback):
        ema_callback.on_fit_start(FakeTrainer(), simple_model)
        assert set(ema_callback.ema_model) == set(simple_model.state_dict())
        for key, value in simple_model.state_dict().items():
            assert ema_callback.ema_model[key].shape == value.shape
            assert torch.equal(ema_callback.ema_model[key], value)
            assert ema_callback.ema_model[key].data_ptr() != value.data_ptr()

    def test_fit_start_does_not_reset_existing_ema(self, simple_model, ema_callback):
        ema_callback.on_fit_start(FakeTrainer(), simple_model)
        for v in ema_callback.ema_model.values():
            if v.is_floating_point():
                v.fill_(7.0)
        ema_callback.on_fit_start(FakeTrainer(), simple_model)
        assert torch.all(ema_callback.ema_model["fc.weight"] == 7.0)


class TestEMAUpdate:
    def test_ema_update_moves_towards_weights(self, simple_model, ema_callback):
        ema_callback.on_fit_start(FakeTrainer(), simple_model)
        initial = float_state({k: v.clone() for k, v in ema_callback.ema_model.items()})

        with torch.no_grad():
            for p in simple_model.parameters():
                p.add_(10.0)
        ema_callback.on_train_batch_end(FakeTrainer(), simple_model, None, None, 0)

        for key, before in initial.items():
            expected = 0.99 * before + 0.01 * simple_model.state_dict()[key]
            torch.testing.assert_close(ema_callback.ema_model[key], expected)
        assert ema_callback.num_updates == 1

    def test_non_float_buffers_copied_not_averaged(self, simple_model, ema_callback):
        ema_callback.on_fit_start(FakeTrainer(), simple_model)
        simple_model.bn.num_batches_tracked.fill_(5)
        ema_callback.on_train_batch_end(FakeTrainer(), simple_model, None, None, 0)
        tracked = ema_callback.ema_model["bn.num_batches_tracked"]
        assert tracked.dtype == torch.long
        assert tracked.item() == 5

    def test_ema_update_skipped_before_start_epoch(self, simple_model):
        cb = EMACallback(start_epoch=5)
        cb.on_fit_start(FakeTrainer(current_epoch=0), simple_model)
        initial = {k: v.clone() for k, v in cb.ema_model.items()}
        with torch.no_grad():
            for p in simple_model.parameters():
                p.add_(100.0)
        cb.on_train_batch_end(FakeTrainer(current_epoch=0), simple_model, None, None, 0)
        assert all(torch.equal(cb.ema_model[k], initial[k]) for k in initial)
        assert cb.num_updates == 0

    def test_update_every_respected(self, simple_model):
        cb = EMACallback(update_every=3)
        cb.on_fit_start(FakeTrainer(), simple_model)
        with torch.no_grad():
            for p in simple_model.parameters():
                p.add_(10.0)
        cb.on_train_batch_end(FakeTrainer(), simple_model, None, None, 0)
        cb.on_train_batch_end(FakeTrainer(), simple_model, None, None, 1)
        assert cb.num_updates == 0
        cb.on_train_batch_end(FakeTrainer(), simple_model, None, None, 2)
        assert cb.num_updates == 1


class TestEMAWeightSwap:
    def test_validation_swaps_and_restores(self, simple_model, ema_callback):
        ema_callback.on_fit_start(FakeTrainer(), simple_model)
        with torch.no_grad():
            simple_model.fc.weight.fill_(999.0)
        raw = simple_model.fc.weight.clone()

        ema_callback.on_validation_epoch_start(FakeTrainer(), simple_model)
        assert not torch.any(simple_model.fc.weight == 999.0)
        ema_callback.on_validation_epoch_end(FakeTrainer(), simple_model)
        assert torch.equal(simple_model.fc.weight, raw)
        assert ema_callback.original_model_state is None

    def test_apply_ema_weights(self, simple_model, ema_callback):
        ema_callback.on_fit_start(FakeTrainer(), simple_model)
        with torch.no_grad():
            simple_model.fc.weight.fill_(999.0)
        ema_callback.apply_ema_weights(simple_model)
        assert torch.equal(simple_model.fc.weight, ema_callback.ema_model["fc.weight"])

    def test_get_ema_model(self, simple_model, ema_callback):
        assert EMACallback().get_ema_model() is None
        ema_callback.on_fit_start(FakeTrainer(), simple_model)
        assert isinstance(ema_callback.get_ema_model(), dict)


class TestEMACheckpoint:
    def test_state_dict_roundtrip(self, simple_model, ema_callback):
        ema_callback.on_fit_start(FakeTrainer(), simple_model)
        ema_callback.on_train_batch_end(FakeTrainer(), simple_model, None, None, 0)
        sd = ema_callback.state_dict()
        assert {"ema_model", "decay", "update_counter", "num_updates"} <= set(sd)

        new_cb = EMACallback()
        new_cb.load_state_dict(sd)
        assert new_cb.decay == ema_callback.decay
        assert new_cb.num_updates == 1
        for k, v in ema_callback.ema_model.items():
            assert torch.equal(new_cb.ema_model[k], v)


class _Snapshot(L.Callback):
    """Record EMA weights and raw weights at the start of (resumed) fit."""

    def __init__(self, ema_cb):
        self.ema_cb = ema_cb
        self.ema_at_start = None
        self.raw_at_start = None

    def on_fit_start(self, trainer, pl_module):
        self.ema_at_start = {k: v.clone() for k, v in self.ema_cb.ema_model.items()}
        self.raw_at_start = {k: v.clone() for k, v in pl_module.state_dict().items()}


class TestEMAResume:
    def test_resume_keeps_ema_state(self, tmp_path):
        ema = EMACallback(decay=0.9, warmup=False)
        ckpt_cb = ModelCheckpoint(dirpath=str(tmp_path / "ckpt"), save_last=True, save_top_k=0)
        trainer = make_trainer(2, [ema, ckpt_cb], tmp_path)
        trainer.fit(SimpleModel(), make_loader(), make_loader())

        ema_saved = float_state({k: v.clone() for k, v in ema.ema_model.items()})
        assert ema.num_updates == 8

        # The checkpoint must carry the EMA state through the callback state dict
        ckpt = torch.load(ckpt_cb.last_model_path, map_location="cpu", weights_only=False)
        ema_in_ckpt = float_state(ckpt["callbacks"][ema.state_key]["ema_model"])
        for k, v in ema_saved.items():
            assert torch.equal(ema_in_ckpt[k], v)

        # Resume for one more epoch with a fresh callback and model
        ema2 = EMACallback(decay=0.9, warmup=False)
        snap = _Snapshot(ema2)
        trainer2 = make_trainer(3, [ema2, snap], tmp_path)
        trainer2.fit(SimpleModel(), make_loader(), make_loader(), ckpt_path=ckpt_cb.last_model_path)

        # EMA at resume start equals the saved average, not the restored raw weights
        ema_at_start = float_state(snap.ema_at_start)
        raw_at_start = float_state(snap.raw_at_start)
        for k, v in ema_saved.items():
            assert torch.equal(ema_at_start[k], v), k
        assert any(not torch.allclose(ema_at_start[k], raw_at_start[k]) for k in ema_saved)

        # ... and training continued from it (counter continued, weights moved on)
        assert ema2.num_updates == 12
        assert any(not torch.allclose(ema2.ema_model[k], ema_saved[k]) for k in ema_saved)


class TestCreateEMACallback:
    def test_factory_returns_ema_callback(self):
        cb = create_ema_callback(decay=0.995, use_for_validation=False)
        assert isinstance(cb, EMACallback)
        assert cb.decay == 0.995
        assert cb.use_ema_for_validation is False
