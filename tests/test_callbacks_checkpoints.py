# tests/test_callbacks_checkpoints.py
"""Tests for the EnhancedModelCheckpoint callback."""

import sys
from pathlib import Path

import lightning as L
import pytest
import torch
import torch.nn as nn
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lmpro.callbacks.checkpoints import (  # noqa: E402
    EnhancedModelCheckpoint,
    get_best_checkpoint_callback,
    get_last_checkpoint_callback,
    get_periodic_checkpoint_callback,
)


class SimpleModel(L.LightningModule):
    def __init__(self, lr: float = 1e-3):
        super().__init__()
        self.fc = nn.Linear(4, 2)
        self.save_hyperparameters()

    def forward(self, x):
        return self.fc(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        return nn.functional.cross_entropy(self(x), y)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        self.log("val/loss", nn.functional.cross_entropy(self(x), y))

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)


def make_loader(n=8):
    return DataLoader(TensorDataset(torch.randn(n, 4), torch.randint(0, 2, (n,))), batch_size=4)


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


class TestEnhancedCheckpointInit:
    def test_inherits_model_checkpoint(self):
        assert isinstance(EnhancedModelCheckpoint(monitor="val_loss"), ModelCheckpoint)

    def test_custom_flags(self):
        cb = EnhancedModelCheckpoint(
            save_architecture=False, save_hyperparameters=False, save_optimizer_state=False, max_checkpoints_to_keep=3
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

    def test_kwargs_forwarded_to_model_checkpoint(self):
        cb = EnhancedModelCheckpoint(enable_version_counter=False)
        assert cb._enable_version_counter is False

    def test_invalid_max_checkpoints(self):
        with pytest.raises(ValueError):
            EnhancedModelCheckpoint(max_checkpoints_to_keep=0)


class TestEnhancedCheckpointTraining:
    def test_checkpoint_saves_sidecars_and_is_loadable(self, tmp_path):
        cb = EnhancedModelCheckpoint(dirpath=str(tmp_path), monitor=None, every_n_epochs=1)
        trainer = make_trainer(2, [cb], tmp_path)
        trainer.fit(SimpleModel(), make_loader())

        ckpts = sorted(tmp_path.glob("*.ckpt"))
        assert len(ckpts) == 1  # save_top_k=1 with monitor=None keeps only the newest
        stem = ckpts[0].stem
        assert (tmp_path / f"{stem}_architecture.txt").exists()
        assert (tmp_path / f"{stem}_hparams.yaml").exists()
        assert (tmp_path / f"{stem}_optimizer.pth").exists()
        # Sidecars of the removed epoch-0 checkpoint are gone too
        assert not list(tmp_path.glob("epoch=0*_architecture.txt"))
        assert (tmp_path / "checkpoint_info.txt").exists()
        assert (tmp_path / "final_checkpoint_info.yaml").exists()

        loaded = SimpleModel.load_from_checkpoint(str(ckpts[0]))
        assert isinstance(loaded, SimpleModel)

        bundle = cb.load_checkpoint_with_metadata(str(ckpts[0]))
        assert "state_dict" in bundle["checkpoint"]
        assert "architecture" in bundle["metadata"]
        assert bundle["metadata"]["hyperparameters"]["lr"] == 1e-3
        assert "optimizer_0" in bundle["metadata"]["optimizer_state"]["optimizers"]

    def test_saved_checkpoints_tracks_only_existing_files(self, tmp_path):
        cb = EnhancedModelCheckpoint(dirpath=str(tmp_path), monitor=None, every_n_epochs=1)
        trainer = make_trainer(3, [cb], tmp_path)
        trainer.fit(SimpleModel(), make_loader())
        assert all(Path(p).exists() for p in cb.saved_checkpoints)
        assert len(cb.saved_checkpoints) == 1
        assert cb.get_checkpoint_info()["total_checkpoints"] == 1


class TestMaxCheckpointsCleanup:
    def test_cleanup_beyond_limit_does_not_crash(self, tmp_path):
        cb = EnhancedModelCheckpoint(
            dirpath=str(tmp_path),
            filename="epoch-{epoch:02d}",
            auto_insert_metric_name=False,
            monitor=None,
            every_n_epochs=1,
            save_top_k=-1,  # parent keeps everything ...
            max_checkpoints_to_keep=2,  # ... we prune down to two
        )
        trainer = make_trainer(5, [cb], tmp_path)
        trainer.fit(SimpleModel(), make_loader())

        ckpts = sorted(p.name for p in tmp_path.glob("*.ckpt"))
        assert ckpts == ["epoch-03.ckpt", "epoch-04.ckpt"]
        assert sorted(Path(p).name for p in cb.saved_checkpoints) == ckpts
        # Sidecars of pruned checkpoints were removed
        for stem in ("epoch-00", "epoch-01", "epoch-02"):
            assert not list(tmp_path.glob(f"{stem}_*"))
        assert (tmp_path / "epoch-04_hparams.yaml").exists()
        assert Path(cb.best_model_path).name == "epoch-04.ckpt"

    def test_cleanup_with_top_k_monitoring(self, tmp_path):
        cb = EnhancedModelCheckpoint(
            dirpath=str(tmp_path),
            filename="e{epoch:02d}",
            auto_insert_metric_name=False,
            monitor="val/loss",
            mode="min",
            save_top_k=3,
            save_last=True,
            max_checkpoints_to_keep=2,
        )
        trainer = make_trainer(5, [cb], tmp_path)
        trainer.fit(SimpleModel(), make_loader(), make_loader())

        # Parent bookkeeping stays consistent with what is on disk
        for path in cb.best_k_models:
            assert Path(path).exists()
        assert Path(cb.best_model_path).exists()
        assert Path(cb.last_model_path).exists()
        assert all(Path(p).exists() for p in cb.saved_checkpoints)
        non_last = [p for p in cb.saved_checkpoints if p != cb.last_model_path]
        assert len(non_last) == 2
        assert sorted(p.name for p in tmp_path.glob("*.ckpt")) == sorted(Path(p).name for p in cb.saved_checkpoints)


class TestFactories:
    def test_best_checkpoint_filename_uses_real_metric_key(self, tmp_path):
        cb = get_best_checkpoint_callback(monitor="val/loss", dirpath=str(tmp_path), save_top_k=1)
        name = cb.format_checkpoint_name({"epoch": 3, "val/loss": torch.tensor(0.123456)})
        assert Path(name).name == "best-epoch=03-val_loss=0.1235.ckpt"
        assert Path(name).parent == tmp_path  # no sub-directory created by '/'

        trainer = make_trainer(2, [cb], tmp_path)
        trainer.fit(SimpleModel(), make_loader(), make_loader())
        best = Path(cb.best_model_path)
        assert best.exists() and best.parent == tmp_path
        assert best.name.startswith("best-epoch=") and "val_loss=" in best.name

    def test_periodic_factory_runs_and_prunes(self, tmp_path):
        periodic = get_periodic_checkpoint_callback(every_n_epochs=1, dirpath=str(tmp_path), max_checkpoints_to_keep=2)
        assert periodic.every_n_epochs == 1 and periodic.save_top_k == -1 and periodic.save_last is True
        trainer = make_trainer(4, [periodic], tmp_path)
        trainer.fit(SimpleModel(), make_loader())
        names = sorted(p.name for p in tmp_path.glob("*.ckpt"))
        assert names == ["epoch-02.ckpt", "epoch-03.ckpt", "last.ckpt"]

    def test_last_factory(self):
        last = get_last_checkpoint_callback()
        assert last.save_last is True and last.save_top_k == 0
