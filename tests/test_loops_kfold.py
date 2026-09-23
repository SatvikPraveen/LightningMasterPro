# tests/test_loops_kfold.py
"""Tests for the K-Fold cross-validation driver."""

import json
import sys
from pathlib import Path

import lightning as L
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lmpro.loops.kfold_loop import KFoldLoop, create_kfold_loop  # noqa: E402


class TinyClassifier(L.LightningModule):
    def __init__(self, in_dim: int = 4, n_classes: int = 3):
        super().__init__()
        self.fc = nn.Linear(in_dim, n_classes)

    def forward(self, x):
        return self.fc(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = nn.functional.cross_entropy(self(x), y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        self.log("val_loss", nn.functional.cross_entropy(logits, y), prog_bar=False)
        self.log("val_acc", (logits.argmax(-1) == y).float().mean())

    def test_step(self, batch, batch_idx):
        x, y = batch
        self.log("test_loss", nn.functional.cross_entropy(self(x), y))

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.1)


def make_dataset(n=30, n_features=4, n_classes=3):
    g = torch.Generator().manual_seed(0)
    return TensorDataset(torch.randn(n, n_features, generator=g), torch.randint(0, n_classes, (n,), generator=g))


TRAINER_KWARGS = dict(
    max_epochs=1,
    limit_train_batches=2,
    limit_val_batches=2,
    accelerator="cpu",
    logger=False,
    enable_checkpointing=False,
    enable_progress_bar=False,
    enable_model_summary=False,
)


class TestKFoldLoopInit:
    def test_default_init(self):
        loop = KFoldLoop()
        assert loop.num_folds == 5
        assert loop.stratified is False
        assert loop.shuffle is True
        assert loop.random_state == 42
        assert loop.current_fold == 0
        assert loop.fold_results == []
        assert loop.done is False

    def test_custom_init(self):
        loop = KFoldLoop(num_folds=3, stratified=True, shuffle=False, random_state=7)
        assert loop.num_folds == 3
        assert loop.stratified is True
        assert loop.shuffle is False
        assert loop.random_state == 7

    def test_results_dir_is_path(self):
        loop = KFoldLoop(results_dir="my_results")
        assert isinstance(loop.results_dir, Path)

    def test_too_few_folds_raises(self):
        with pytest.raises(ValueError):
            KFoldLoop(num_folds=1)


class TestKFoldSplitting:
    def test_split_count_and_disjointness(self):
        loop = KFoldLoop(num_folds=5)
        splits = loop.get_fold_splits(make_dataset(50))
        assert len(splits) == 5
        for train_idx, val_idx in splits:
            assert set(train_idx).isdisjoint(set(val_idx))
            assert len(train_idx) + len(val_idx) == 50

    def test_all_samples_used_across_folds(self):
        loop = KFoldLoop(num_folds=5)
        splits = loop.get_fold_splits(make_dataset(50))
        all_val = set()
        for _, val_idx in splits:
            all_val.update(int(i) for i in val_idx)
        assert all_val == set(range(50))

    def test_stratified_preserves_class_ratio(self):
        n = 60
        y = torch.tensor([0] * 30 + [1] * 30)
        dataset = TensorDataset(torch.randn(n, 4), y)
        loop = KFoldLoop(num_folds=3, stratified=True)
        for _, val_idx in loop.get_fold_splits(dataset):
            labels = y[torch.as_tensor(val_idx)]
            assert int((labels == 0).sum()) == int((labels == 1).sum()) == 10

    def test_splits_reproducible(self):
        ds = make_dataset(40)
        a = KFoldLoop(num_folds=4, random_state=3).get_fold_splits(ds)
        b = KFoldLoop(num_folds=4, random_state=3).get_fold_splits(ds)
        for (_, va), (_, vb) in zip(a, b):
            assert list(va) == list(vb)


class TestKFoldRun:
    def test_three_fold_run_with_template_model(self, tmp_path):
        dataset = make_dataset(30)
        loop = KFoldLoop(num_folds=3, results_dir=str(tmp_path / "kf"), batch_size=8)
        template = TinyClassifier()
        template_weights = {k: v.clone() for k, v in template.state_dict().items()}

        summary = loop.run(template, dataset, trainer_kwargs=TRAINER_KWARGS)

        assert len(loop.fold_results) == 3
        assert len(loop.all_fold_metrics) == 3
        assert loop.done

        val_sets = [set(r["val_indices"]) for r in loop.fold_results]
        assert val_sets[0].isdisjoint(val_sets[1])
        assert val_sets[0].isdisjoint(val_sets[2])
        assert val_sets[1].isdisjoint(val_sets[2])
        assert set().union(*val_sets) == set(range(30))

        assert "val_loss" in summary and "train_loss" in summary
        assert set(summary["val_loss"]) >= {"mean", "std", "min", "max", "values"}
        assert len(summary["val_loss"]["values"]) == 3

        # Template must be untouched, and fold models trained independently
        for k, v in template.state_dict().items():
            assert torch.equal(v, template_weights[k])
        assert len(loop.fold_models) == 3
        assert all(m is not template for m in loop.fold_models)

        # Results were written to disk
        assert (tmp_path / "kf" / "kfold_summary.json").exists()
        for i in range(3):
            assert (tmp_path / "kf" / f"fold_{i}_results.json").exists()
        saved = json.loads((tmp_path / "kf" / "kfold_summary.json").read_text())
        assert saved["metadata"]["num_folds"] == 3

    def test_run_with_factory_dataloader_and_test(self, tmp_path):
        dataset = make_dataset(24)
        loader = DataLoader(dataset, batch_size=6)
        test_loader = DataLoader(make_dataset(12), batch_size=6)
        loop = KFoldLoop(num_folds=3, save_fold_results=False)

        summary = loop.run(TinyClassifier, loader, trainer_kwargs=TRAINER_KWARGS, test_dataloader=test_loader)

        assert len(loop.fold_results) == 3
        assert "test_loss" in summary
        assert not (tmp_path / "kfold_results").exists()

    def test_ensemble_predict(self):
        loop = KFoldLoop(num_folds=3, save_fold_results=False)
        loop.run(TinyClassifier, make_dataset(24), trainer_kwargs=TRAINER_KWARGS)
        out = loop.ensemble_predict(torch.randn(5, 4))
        assert out.shape == (5, 3)
        stacked = loop.ensemble_predict(torch.randn(5, 4), reduction="none")
        assert stacked.shape == (3, 5, 3)

    def test_ensemble_predict_before_run_raises(self):
        with pytest.raises(RuntimeError):
            KFoldLoop(num_folds=3).ensemble_predict(torch.randn(2, 4))


class TestKFoldResults:
    def test_best_fold(self):
        loop = KFoldLoop()
        loop.all_fold_metrics = [{"val_acc": 0.80}, {"val_acc": 0.92}, {"val_acc": 0.85}]
        best_idx, best_metrics = loop.get_best_fold(metric_name="val_acc", mode="max")
        assert best_idx == 1
        assert best_metrics["val_acc"] == 0.92
        worst_idx, _ = loop.get_best_fold(metric_name="val_acc", mode="min")
        assert worst_idx == 0
        assert loop.get_best_fold("missing") == (-1, {})

    def test_get_summary_stats(self):
        loop = KFoldLoop()
        loop.all_fold_metrics = [{"val_acc": 0.80}, {"val_acc": 0.90}, {"val_acc": 0.70}]
        summary = loop.get_summary_statistics()
        assert summary["val_acc"]["mean"] == pytest.approx(0.8)
        assert summary["val_acc"]["min"] == pytest.approx(0.7)
        assert summary["val_acc"]["max"] == pytest.approx(0.9)
        assert summary["val_acc"]["values"] == [0.8, 0.9, 0.7]

    def test_save_summary_results_creates_file(self, tmp_path):
        loop = KFoldLoop(save_fold_results=True, results_dir=str(tmp_path))
        loop.all_fold_metrics = [{"val_acc": 0.8}, {"val_acc": 0.9}]
        loop._save_summary_results(loop.get_summary_statistics())
        data = json.loads((tmp_path / "kfold_summary.json").read_text())
        assert data["summary_statistics"]["val_acc"]["mean"] == pytest.approx(0.85)


class TestCreateKFoldLoop:
    def test_factory_function(self):
        loop = create_kfold_loop(num_folds=3)
        assert isinstance(loop, KFoldLoop)
        assert loop.num_folds == 3

    def test_factory_stratified(self):
        loop = create_kfold_loop(num_folds=5, stratified=True)
        assert loop.stratified is True
