# tests/test_callbacks_checkpoints.py
"""Tests for the EnhancedModelCheckpoint callback."""

import sys
import tempfile
import shutil
from pathlib import Path
import pytest
import torch
import torch.nn as nn
import lightning as L
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lmpro.callbacks.checkpoints import EnhancedModelCheckpoint


class SimpleModel(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 2)
        self.save_hyperparameters()

    def forward(self, x):
        return self.fc(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        return torch.nn.functional.cross_entropy(self(x), y)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)


@pytest.fixture
def temp_dir():
    d = tempfile.mkdtemp()
    yield Path(d)
    shutil.rmtree(d)


@pytest.fixture
def ckpt_callback(temp_dir):
    return EnhancedModelCheckpoint(
        dirpath=str(temp_dir),
        monitor="val_loss",
        mode="min",
        save_top_k=2,
        filename="model-{epoch:02d}-{val_loss:.4f}",
    )


class TestEnhancedCheckpointInit:
    def test_inherits_model_checkpoint(self):
        from lightning.pytorch.callbacks import ModelCheckpoint
        cb = EnhancedModelCheckpoint(monitor="val_loss")
        assert isinstance(cb, ModelCheckpoint)

    def test_custom_flags(self):
        cb = EnhancedModelCheckpoint(
            save_architecture=False,
            save_hyperparameters=False,
            save_optimizer_state=False,
            max_checkpoints_to_keep=3,
        )
        assert cb.save_architecture is False
        assert cb.save_hyperparameters is False
        assert cb.save_optimizer_state is False
        assert cb.max_checkpoints_to_keep == 3

    def test_default_flags(self):
        cb = EnhancedModelCheckpoint()
        assert cb.save_architecture is True
        assert cb.save_hyperparameters is True
        assert cb.save_optimizer_state is True
        assert cb.max_checkpoints_to_keep is None


class TestEnhancedCheckpointTraining:
    def test_checkpoint_saves_during_training(self, temp_dir):
        model = SimpleModel()
        dm_x = torch.randn(8, 4)
        dm_y = torch.randint(0, 2, (8,))
        dataset = torch.utils.data.TensorDataset(dm_x, dm_y)
        loader = torch.utils.data.DataLoader(dataset, batch_size=4)

        cb = EnhancedModelCheckpoint(
            dirpath=str(temp_dir),
            monitor=None,  # save every epoch
            every_n_epochs=1,
        )

        trainer = L.Trainer(
            max_epochs=2,
            callbacks=[cb],
            enable_progress_bar=False,
            enable_model_summary=False,
            logger=False,
            accelerator="cpu",
        )
        trainer.fit(model, train_dataloaders=loader)

        # At least one checkpoint file should exist
        ckpts = list(temp_dir.glob("*.ckpt"))
        assert len(ckpts) >= 1

    def test_checkpoint_loadable(self, temp_dir):
        model = SimpleModel()
        dm_x = torch.randn(8, 4)
        dm_y = torch.randint(0, 2, (8,))
        dataset = torch.utils.data.TensorDataset(dm_x, dm_y)
        loader = torch.utils.data.DataLoader(dataset, batch_size=4)

        cb = EnhancedModelCheckpoint(
            dirpath=str(temp_dir),
            monitor=None,
            every_n_epochs=1,
        )

        trainer = L.Trainer(
            max_epochs=1,
            callbacks=[cb],
            enable_progress_bar=False,
            enable_model_summary=False,
            logger=False,
            accelerator="cpu",
        )
        trainer.fit(model, train_dataloaders=loader)

        ckpts = list(temp_dir.glob("*.ckpt"))
        assert len(ckpts) >= 1

        loaded_model = SimpleModel.load_from_checkpoint(str(ckpts[0]))
        assert loaded_model is not None


class TestMaxCheckpointsCleanup:
    def test_max_checkpoints_parameter_set(self, temp_dir):
        cb = EnhancedModelCheckpoint(
            dirpath=str(temp_dir),
            max_checkpoints_to_keep=2,
        )
        assert cb.max_checkpoints_to_keep == 2
