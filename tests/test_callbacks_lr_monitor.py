# tests/test_callbacks_lr_monitor.py
"""Tests for LRMonitorCallback with a real Trainer run."""

import sys
from pathlib import Path

import lightning as L
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lmpro.callbacks.lr_monitor import LRMonitorCallback, create_lr_monitor  # noqa: E402


class SimpleModel(L.LightningModule):
    def __init__(self, gamma: float = 1.0):
        super().__init__()
        self.fc = nn.Linear(4, 2)
        self.gamma = gamma

    def forward(self, x):
        return self.fc(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        return nn.functional.mse_loss(self(x), y)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=1e-2, momentum=0.9, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=self.gamma)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}


def run(callback, model=None, max_epochs=1):
    loader = DataLoader(TensorDataset(torch.randn(8, 4), torch.randn(8, 2)), batch_size=4)
    trainer = L.Trainer(
        max_epochs=max_epochs,
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


class TestLRMonitorInit:
    def test_defaults(self):
        cb = LRMonitorCallback()
        assert cb.logging_interval == "step"
        assert cb.log_momentum is False
        assert cb.get_lr_history() == {}

    def test_invalid_interval(self):
        with pytest.raises(ValueError):
            LRMonitorCallback(logging_interval="minute")

    def test_factory(self):
        cb = create_lr_monitor(logging_interval="epoch", log_momentum=True, verbose=True)
        assert isinstance(cb, LRMonitorCallback)
        assert cb.logging_interval == "epoch" and cb.log_momentum and cb.verbose


class TestLRMonitorTraining:
    def test_step_logging_records_history(self):
        cb = LRMonitorCallback(logging_interval="step")
        trainer = run(cb)
        history = cb.get_lr_history()
        assert list(history) == ["lr/opt_0_pg_0"]
        assert len(history["lr/opt_0_pg_0"]) == 2
        assert all(lr == pytest.approx(1e-2) for _, lr in history["lr/opt_0_pg_0"])
        assert cb.get_last_lrs()["lr/opt_0_pg_0"] == pytest.approx(1e-2)
        assert "lr/opt_0_pg_0_step" in trainer.logged_metrics

    def test_epoch_logging(self):
        cb = LRMonitorCallback(logging_interval="epoch")
        run(cb, max_epochs=2)
        assert [step for step, _ in cb.get_lr_history()["lr/opt_0_pg_0"]] == [0, 1]

    def test_momentum_and_weight_decay_logged(self):
        cb = LRMonitorCallback(log_momentum=True, log_weight_decay=True)
        trainer = run(cb)
        assert trainer.logged_metrics["momentum/opt_0_pg_0_step"] == pytest.approx(0.9)
        assert trainer.logged_metrics["weight_decay/opt_0_pg_0_step"] == pytest.approx(1e-4)

    def test_lr_change_alert(self):
        cb = LRMonitorCallback(alert_on_lr_change_factor=5.0)
        with pytest.warns(UserWarning, match="Large LR change"):
            run(cb, SimpleModel(gamma=0.1))
        lrs = [lr for _, lr in cb.get_lr_history()["lr/opt_0_pg_0"]]
        assert lrs[0] == pytest.approx(1e-3)  # scheduler stepped once before the log
        assert lrs[1] == pytest.approx(1e-4)

    def test_reset_history(self):
        cb = LRMonitorCallback()
        run(cb)
        cb.reset_history()
        assert cb.get_lr_history() == {} and cb.get_last_lrs() == {}
