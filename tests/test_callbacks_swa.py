# tests/test_callbacks_swa.py
"""Tests for the SWACallback."""

import sys
from pathlib import Path

import lightning as L
import pytest
import torch
import torch.nn as nn
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.optim.swa_utils import SWALR
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lmpro.callbacks.swa import SWACallback, create_swa_callback  # noqa: E402


class SimpleModel(L.LightningModule):
    """Minimal module with BatchNorm and a StepLR scheduler."""

    def __init__(self, lr=1e-2):
        super().__init__()
        self.bn = nn.BatchNorm1d(4)
        self.fc = nn.Linear(4, 2)
        self.lr = lr

    def forward(self, x):
        return self.fc(self.bn(x))

    def training_step(self, batch, batch_idx):
        x, y = batch
        return nn.functional.mse_loss(self(x), y)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        self.log("val_loss", nn.functional.mse_loss(self(x), y))

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


def make_loader(n=16):
    g = torch.Generator().manual_seed(2)
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


class FakeTrainer:
    def __init__(self, max_epochs=10, current_epoch=0):
        self.max_epochs = max_epochs
        self.current_epoch = current_epoch


class TestSWACallbackInit:
    def test_default_params(self):
        cb = SWACallback()
        assert cb.swa_lrs == [1e-2]
        assert cb.annealing_epochs == 10
        assert cb.annealing_strategy == "cos"
        assert cb.swa_model is None
        assert cb.swa_n == 0
        assert cb.original_model_state is None

    def test_list_and_scalar_lrs(self):
        assert SWACallback(swa_lrs=[1e-3, 5e-4]).swa_lrs == [1e-3, 5e-4]
        assert SWACallback(swa_lrs=1e-3).swa_lrs == [1e-3]

    def test_invalid_params(self):
        with pytest.raises(ValueError):
            SWACallback(annealing_strategy="bogus")
        with pytest.raises(ValueError):
            SWACallback(swa_epoch_start=1.5)


class TestSWAFitStart:
    def test_start_epoch_from_float_and_int(self):
        cb = SWACallback(swa_epoch_start=0.8)
        cb.on_fit_start(FakeTrainer(max_epochs=10), SimpleModel())
        assert cb.swa_start_epoch == 8
        cb = SWACallback(swa_epoch_start=3)
        cb.on_fit_start(FakeTrainer(max_epochs=10), SimpleModel())
        assert cb.swa_start_epoch == 3

    def test_swa_model_initialized(self):
        model = SimpleModel()
        cb = SWACallback(swa_epoch_start=0.5)
        cb.on_fit_start(FakeTrainer(), model)
        assert set(cb.swa_model) == set(model.state_dict())
        assert cb.swa_n == 0

    def test_fit_start_does_not_reset_restored_average(self):
        model = SimpleModel()
        cb = SWACallback(swa_epoch_start=1)
        cb.on_fit_start(FakeTrainer(), model)
        cb.swa_model["fc.weight"].fill_(3.0)
        cb.swa_n = 4
        cb.on_fit_start(FakeTrainer(), model)
        assert torch.all(cb.swa_model["fc.weight"] == 3.0)
        assert cb.swa_n == 4


class TestSWAEpochUpdate:
    def test_updated_only_after_start_epoch(self):
        model = SimpleModel()
        cb = SWACallback(swa_epoch_start=2)
        cb.on_fit_start(FakeTrainer(), model)
        initial = {k: v.clone() for k, v in cb.swa_model.items()}

        with torch.no_grad():
            model.fc.weight.fill_(1.0)
        cb.on_train_epoch_end(FakeTrainer(current_epoch=1), model)
        assert cb.swa_n == 0
        assert torch.equal(cb.swa_model["fc.weight"], initial["fc.weight"])

        cb.on_train_epoch_end(FakeTrainer(current_epoch=2), model)
        assert cb.swa_n == 1
        assert torch.all(cb.swa_model["fc.weight"] == 1.0)  # first update overwrites

        with torch.no_grad():
            model.fc.weight.fill_(3.0)
        cb.on_train_epoch_end(FakeTrainer(current_epoch=3), model)
        assert cb.swa_n == 2
        assert torch.all(cb.swa_model["fc.weight"] == 2.0)  # running mean of 1 and 3


class TestSWAAveragingMath:
    def test_default_avg_fn(self):
        cb = SWACallback()
        assert cb._default_avg_fn(torch.tensor(1.0), torch.tensor(3.0), 0).item() == pytest.approx(3.0)
        assert cb._default_avg_fn(torch.tensor(1.0), torch.tensor(3.0), 1).item() == pytest.approx(2.0)

    def test_default_avg_fn_is_running_mean(self):
        cb = SWACallback()
        avg = torch.tensor(0.0)
        values = [10.0, 20.0, 30.0, 40.0]
        for n, v in enumerate(values):
            avg = cb._default_avg_fn(avg, torch.tensor(v), n)
        assert avg.item() == pytest.approx(25.0)


class TestSWACheckpoint:
    def test_state_dict_roundtrip(self):
        model = SimpleModel()
        cb = SWACallback(swa_epoch_start=0)
        cb.on_fit_start(FakeTrainer(), model)
        cb.on_train_epoch_end(FakeTrainer(current_epoch=0), model)
        sd = cb.state_dict()
        assert sd["swa_n"] == 1
        assert sd["swa_epoch_start_absolute"] == 0
        assert set(sd["swa_model"]) == set(model.state_dict())

        new_cb = SWACallback()
        new_cb.load_state_dict(sd)
        assert new_cb.swa_n == 1
        for k, v in cb.swa_model.items():
            assert torch.equal(new_cb.swa_model[k], v)


class TestSWATraining:
    def test_scheduler_replaced_and_weights_applied(self, tmp_path):
        model = SimpleModel(lr=1e-2)
        swa = SWACallback(swa_lrs=5e-3, swa_epoch_start=1, annealing_epochs=1, annealing_strategy="linear")
        trainer = make_trainer(3, [swa], tmp_path)
        trainer.fit(model, make_loader(), make_loader())

        # SWA ran for epochs 1 and 2
        assert swa.swa_n == 2
        # The StepLR scheduler was replaced by SWALR, which Lightning stepped
        assert len(trainer.lr_scheduler_configs) == 1
        assert isinstance(trainer.lr_scheduler_configs[0].scheduler, SWALR)
        assert trainer.optimizers[0].param_groups[0]["lr"] == pytest.approx(5e-3)
        # Averaged weights loaded into the model at the end
        for k, v in swa.swa_model.items():
            if v.is_floating_point() and "running" not in k:
                assert torch.equal(model.state_dict()[k], v), k
        # BN update produced finite stats and restored the BN momentum
        assert torch.isfinite(model.bn.running_mean).all()
        assert model.bn.momentum == 0.1

    def test_bn_update_uses_cumulative_average(self, tmp_path):
        model = SimpleModel()
        swa = SWACallback(swa_epoch_start=0, annealing_strategy="constant", bn_update_batches=100)
        trainer = make_trainer(1, [swa], tmp_path)
        loader = make_loader(16)
        trainer.fit(model, loader)

        # With momentum=None the running mean is the plain mean over the data
        x = torch.cat([b[0] for b in loader])
        torch.testing.assert_close(model.bn.running_mean, x.mean(0), atol=1e-5, rtol=1e-4)
        assert model.bn.num_batches_tracked.item() == 4
        # No scheduler installed for the constant strategy
        assert trainer.lr_scheduler_configs == []

    def test_resume_keeps_swa_average(self, tmp_path):
        swa = SWACallback(swa_epoch_start=0, annealing_strategy="constant", update_bn=False)
        ckpt_cb = ModelCheckpoint(dirpath=str(tmp_path / "ckpt"), save_last=True, save_top_k=0)
        trainer = make_trainer(2, [swa, ckpt_cb], tmp_path)
        trainer.fit(SimpleModel(), make_loader())
        assert swa.swa_n == 2

        ckpt = torch.load(ckpt_cb.last_model_path, map_location="cpu", weights_only=False)
        assert ckpt["callbacks"][swa.state_key]["swa_n"] == 2

        swa2 = SWACallback(swa_epoch_start=0, annealing_strategy="constant", update_bn=False)
        trainer2 = make_trainer(3, [swa2], tmp_path)
        trainer2.fit(SimpleModel(), make_loader(), ckpt_path=ckpt_cb.last_model_path)
        assert swa2.swa_n == 3  # continued from 2, not reset


class TestCreateSWACallback:
    def test_factory(self):
        cb = create_swa_callback(swa_lr=1e-3, swa_epoch_start=2, annealing_epochs=4)
        assert isinstance(cb, SWACallback)
        assert cb.swa_lrs == [1e-3]
        assert cb.swa_epoch_start == 2
        assert cb.annealing_epochs == 4
