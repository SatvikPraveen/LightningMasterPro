# tests/test_loops_progressive_unfreezing.py
"""Tests for ProgressiveUnfreezingCallback with a real Trainer run."""

import sys
from pathlib import Path

import lightning as L
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lmpro.loops.progressive_unfreezing import (  # noqa: E402
    ProgressiveUnfreezingCallback,
    create_progressive_unfreezing,
)


class ThreeBlockModel(L.LightningModule):
    """embed -> hidden -> head; groups are unfrozen head first."""

    def __init__(self):
        super().__init__()
        self.embed = nn.Linear(4, 8)
        self.hidden = nn.Linear(8, 8)
        self.head = nn.Linear(8, 2)
        self.trainable_per_epoch = []

    def forward(self, x):
        return self.head(torch.relu(self.hidden(torch.relu(self.embed(x)))))

    def training_step(self, batch, batch_idx):
        x, y = batch
        return nn.functional.mse_loss(self(x), y)

    def on_train_epoch_start(self):
        self.trainable_per_epoch.append(sorted(n for n, p in self.named_parameters() if p.requires_grad))

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=1e-2)


def run(callback, model, max_epochs=1):
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
    trainer.fit(model, loader)
    return trainer


class TestInit:
    def test_defaults_and_factory(self):
        cb = ProgressiveUnfreezingCallback()
        assert cb.unfreeze_every_n_epochs == 1 and cb.start_epoch == 0 and cb.lr_scale_factor is None
        cb = create_progressive_unfreezing(unfreeze_every_n_epochs=2, lr_scale_factor=0.5, verbose=False)
        assert isinstance(cb, ProgressiveUnfreezingCallback)
        assert cb.unfreeze_every_n_epochs == 2 and cb.lr_scale_factor == 0.5

    def test_invalid_scale(self):
        with pytest.raises(ValueError):
            ProgressiveUnfreezingCallback(lr_scale_factor=0)


class TestUnfreezingSchedule:
    def test_groups_resolved_output_first(self):
        cb = ProgressiveUnfreezingCallback(verbose=False)
        run(cb, ThreeBlockModel())
        groups = cb.get_resolved_groups()
        assert [g[0].split(".")[0] for g in groups] == ["head", "hidden", "embed"]

    def test_one_group_per_epoch(self):
        cb = ProgressiveUnfreezingCallback(unfreeze_every_n_epochs=1, verbose=False)
        model = ThreeBlockModel()
        run(cb, model, max_epochs=3)

        e0, e1, e2 = model.trainable_per_epoch
        # Biases are always trainable; weights are unfrozen head -> hidden -> embed
        assert "head.weight" in e0 and "hidden.weight" not in e0 and "embed.weight" not in e0
        assert "hidden.weight" in e1 and "embed.weight" not in e1
        assert "embed.weight" in e2
        assert cb.unfrozen_group_count == 3
        # Everything is trainable again after training
        assert all(p.requires_grad for p in model.parameters())

    def test_start_epoch_delays_unfreezing(self):
        cb = ProgressiveUnfreezingCallback(start_epoch=1, verbose=False)
        model = ThreeBlockModel()
        run(cb, model, max_epochs=2)
        e0, e1 = model.trainable_per_epoch
        assert "head.weight" not in e0
        assert "head.weight" in e1
        assert cb.unfrozen_group_count == 1

    def test_explicit_layer_groups(self):
        cb = ProgressiveUnfreezingCallback(
            layer_groups=[["head.weight"], ["embed.weight", "hidden.weight"]], verbose=False
        )
        model = ThreeBlockModel()
        run(cb, model, max_epochs=2)
        e0, e1 = model.trainable_per_epoch
        assert "head.weight" in e0 and "embed.weight" not in e0
        assert {"embed.weight", "hidden.weight"} <= set(e1)


class TestDiscriminativeLR:
    def test_lr_scale_factor_creates_scaled_param_groups(self):
        cb = ProgressiveUnfreezingCallback(lr_scale_factor=0.5, verbose=False)
        model = ThreeBlockModel()
        trainer = run(cb, model, max_epochs=3)

        optimizer = trainer.optimizers[0]
        lr_of = {}
        for pg in optimizer.param_groups:
            for p in pg["params"]:
                lr_of[id(p)] = pg["lr"]

        assert lr_of[id(model.head.weight)] == pytest.approx(1e-2)  # group 0: base lr
        assert lr_of[id(model.hidden.weight)] == pytest.approx(5e-3)  # group 1: base * 0.5
        assert lr_of[id(model.embed.weight)] == pytest.approx(2.5e-3)  # group 2: base * 0.25
        # Every parameter is still in exactly one param group
        all_ids = [id(p) for pg in optimizer.param_groups for p in pg["params"]]
        assert sorted(all_ids) == sorted(id(p) for p in model.parameters())

    def test_without_scale_factor_single_group(self):
        cb = ProgressiveUnfreezingCallback(verbose=False)
        trainer = run(cb, ThreeBlockModel(), max_epochs=3)
        assert len(trainer.optimizers[0].param_groups) == 1
