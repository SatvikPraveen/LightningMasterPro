# tests/test_loops_curriculum.py
"""Tests for curriculum learning callback, dataset wrapper and strategies."""

import sys
from pathlib import Path

import lightning as L
import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lmpro.loops.curriculum_loop import (  # noqa: E402
    CurriculumDataset,
    CurriculumLoop,
    LengthBasedCurriculum,
    LossBasedCurriculum,
    RandomCurriculum,
    create_length_curriculum_loop,
    create_loss_curriculum_loop,
    create_random_curriculum_loop,
)


class VarLenDataset(torch.utils.data.Dataset):
    """Samples of varying length: sample i has length i + 1."""

    def __init__(self, n=40):
        self.items = [(torch.zeros(i + 1), torch.tensor(i % 2)) for i in range(n)]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def make_tensor_dataset(n=60, features=10):
    return TensorDataset(torch.randn(n, features), torch.randint(0, 2, (n,)))


class TinyModel(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 2)
        self.criterion = nn.CrossEntropyLoss()
        self.seen_batch_sizes = []

    def forward(self, x):
        return self.fc(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        self.seen_batch_sizes.append(x.shape[0])
        return self.criterion(self(x), y)

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.1)


# ─── Strategies ─────────────────────────────────────────────────────────────


class TestLengthBasedCurriculum:
    def test_scores_follow_length(self):
        scores = LengthBasedCurriculum().get_difficulty_scores(VarLenDataset(10))
        assert isinstance(scores, np.ndarray) and scores.shape == (10,)
        assert scores[0] == pytest.approx(0.0) and scores[-1] == pytest.approx(1.0)
        assert np.all(np.diff(scores) > 0)

    def test_equal_lengths_give_ones(self):
        scores = LengthBasedCurriculum().get_difficulty_scores(make_tensor_dataset(12))
        np.testing.assert_allclose(scores, np.ones(12))

    def test_reverse_inverts_scores(self):
        ds = VarLenDataset(10)
        normal = LengthBasedCurriculum(reverse=False).get_difficulty_scores(ds)
        reversed_ = LengthBasedCurriculum(reverse=True).get_difficulty_scores(ds)
        np.testing.assert_allclose(normal + reversed_, np.ones(10), atol=1e-6)

    def test_schedule(self):
        schedule = LengthBasedCurriculum().get_curriculum_schedule(total_epochs=10, dataset_size=100)
        thresholds = [t for _, t in schedule]
        assert len(schedule) == 10
        assert thresholds[0] == pytest.approx(0.1)
        assert thresholds[-1] == pytest.approx(1.0)
        assert all(b > a for a, b in zip(thresholds, thresholds[1:]))


class TestLossBasedCurriculum:
    def test_init(self):
        assert LossBasedCurriculum().warmup_epochs == 5
        assert LossBasedCurriculum(warmup_epochs=2).warmup_epochs == 2

    def test_random_without_model_is_not_cached(self):
        strategy = LossBasedCurriculum()
        ds = make_tensor_dataset(20)
        scores = strategy.get_difficulty_scores(ds, model=None)
        assert scores.shape == (20,)
        assert strategy.sample_losses is None

    def test_model_scores_cached_and_recomputed(self):
        strategy = LossBasedCurriculum()
        ds = make_tensor_dataset(16)
        model = TinyModel()

        first = strategy.get_difficulty_scores(ds, model)
        assert first.shape == (16,) and first.min() >= 0 and first.max() <= 1
        assert strategy.sample_losses is not None

        # Change the model: cached scores are returned unless recompute=True
        with torch.no_grad():
            model.fc.weight.mul_(10.0)
            model.fc.bias.add_(3.0)
        cached = strategy.get_difficulty_scores(ds, model)
        assert cached is first
        recomputed = strategy.get_difficulty_scores(ds, model, recompute=True)
        assert recomputed is not first
        assert not np.allclose(recomputed, first)

        strategy.reset()
        assert strategy.sample_losses is None

    def test_schedule_warmup_then_increase(self):
        schedule = LossBasedCurriculum(warmup_epochs=2).get_curriculum_schedule(8, 100)
        thresholds = [t for _, t in schedule]
        assert thresholds[0] == thresholds[1] == pytest.approx(0.3)
        assert thresholds[-1] > thresholds[2]
        assert max(thresholds) <= 1.0


class TestRandomCurriculum:
    def test_scores_in_0_1_range(self):
        scores = RandomCurriculum().get_difficulty_scores(make_tensor_dataset(30))
        assert scores.shape == (30,)
        assert scores.min() >= 0.0 and scores.max() <= 1.0

    def test_schedule_grows_to_one(self):
        schedule = RandomCurriculum().get_curriculum_schedule(5, 50)
        assert schedule[0][1] == pytest.approx(0.2)
        assert schedule[-1][1] == pytest.approx(1.0)


# ─── CurriculumDataset ──────────────────────────────────────────────────────


class TestCurriculumDataset:
    def test_full_dataset_without_scores(self):
        ds = CurriculumDataset(VarLenDataset(10))
        assert len(ds) == 10
        assert ds[3][0].shape[0] == 4

    def test_threshold_filters_samples(self):
        base = VarLenDataset(10)
        ds = CurriculumDataset(base, difficulty_scores=np.linspace(0, 1, 10), min_samples=1)
        ds.set_threshold(0.5)
        assert len(ds) == 5
        assert list(ds.active_indices) == [0, 1, 2, 3, 4]
        assert ds[4][0].shape[0] == 5  # maps to base index 4
        ds.set_threshold(1.0)
        assert len(ds) == 10

    def test_min_samples_enforced_with_easiest(self):
        ds = CurriculumDataset(VarLenDataset(10), difficulty_scores=np.linspace(0, 1, 10), min_samples=4)
        ds.set_threshold(0.0)
        assert len(ds) == 4
        assert list(ds.active_indices) == [0, 1, 2, 3]

    def test_bad_scores_shape_raises(self):
        with pytest.raises(ValueError):
            CurriculumDataset(VarLenDataset(10), difficulty_scores=np.zeros(3))


# ─── CurriculumLoop (callback) ──────────────────────────────────────────────


class TestCurriculumLoopInit:
    def test_default_init_with_string(self):
        loop = CurriculumLoop(strategy="length")
        assert isinstance(loop, L.Callback)
        assert isinstance(loop.strategy, LengthBasedCurriculum)
        assert loop.difficulty_scores is None
        assert loop.current_epoch == 0
        assert loop.curriculum_stats == []

    def test_init_with_strategy_object(self):
        strategy = LengthBasedCurriculum()
        assert CurriculumLoop(strategy=strategy).strategy is strategy

    def test_options(self):
        loop = CurriculumLoop(strategy="loss", update_frequency=2, curriculum_warmup=3)
        assert isinstance(loop.strategy, LossBasedCurriculum)
        assert loop.update_frequency == 2
        assert loop.curriculum_warmup == 3

    def test_invalid_strategy_raises(self):
        with pytest.raises(ValueError):
            CurriculumLoop(strategy="nonexistent")

    def test_wrap_registers_dataset(self):
        loop = CurriculumLoop(strategy="random")
        wrapped = loop.wrap(VarLenDataset(5))
        assert isinstance(wrapped, CurriculumDataset)
        assert loop.dataset is wrapped
        loader = loop.create_dataloader(batch_size=2)
        assert loader.dataset is wrapped


def _run_curriculum(loop, dataset, max_epochs=2, batch_size=4):
    model = TinyModel()
    trainer = L.Trainer(
        max_epochs=max_epochs,
        reload_dataloaders_every_n_epochs=1,
        callbacks=[loop],
        accelerator="cpu",
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    trainer.fit(model, DataLoader(dataset, batch_size=batch_size, shuffle=True))
    return model, trainer


class TestCurriculumLoopTraining:
    def test_threshold_advances_over_two_epochs(self):
        base = make_tensor_dataset(40, features=10)
        # Explicit scores so the subset sizes are predictable: 0.0 .. 1.0
        loop = CurriculumLoop(strategy="random", min_samples_per_epoch=4, recompute_difficulty=False)
        loop.strategy.get_difficulty_scores = lambda ds, model=None, recompute=False: np.linspace(0, 1, 40)
        curriculum_ds = loop.wrap(base)

        model, trainer = _run_curriculum(loop, curriculum_ds, max_epochs=2, batch_size=4)

        stats = loop.get_curriculum_stats()
        assert [s["epoch"] for s in stats] == [0, 1]
        # RandomCurriculum schedule with 2 epochs: 0.2 then 1.0
        assert stats[0]["difficulty_threshold"] == pytest.approx(0.2)
        assert stats[1]["difficulty_threshold"] == pytest.approx(1.0)
        assert stats[1]["difficulty_threshold"] > stats[0]["difficulty_threshold"]
        assert loop.current_threshold == pytest.approx(1.0)
        assert stats[0]["num_samples"] == 8  # scores <= 0.2 -> 8 of 40
        assert stats[1]["num_samples"] == 40

        # Epoch 0 saw 8 samples (2 batches of 4), epoch 1 saw 40 (10 batches)
        assert sum(model.seen_batch_sizes[:2]) == 8
        assert sum(model.seen_batch_sizes[2:]) == 40
        assert len(model.seen_batch_sizes) == 12

        # Full dataset exposed again after fit
        assert len(curriculum_ds) == 40

    def test_dataset_discovered_from_dataloader(self):
        loop = CurriculumLoop(strategy="length", min_samples_per_epoch=2, recompute_difficulty=False)
        curriculum_ds = CurriculumDataset(make_tensor_dataset(20))
        _run_curriculum(loop, curriculum_ds, max_epochs=2)
        assert loop.dataset is curriculum_ds
        assert loop.difficulty_scores is not None and loop.difficulty_scores.shape == (20,)
        assert len(loop.get_curriculum_stats()) == 2

    def test_loss_curriculum_recomputes_with_model(self):
        strategy = LossBasedCurriculum(warmup_epochs=1)
        loop = CurriculumLoop(strategy=strategy, update_frequency=1, min_samples_per_epoch=2)
        curriculum_ds = loop.wrap(make_tensor_dataset(16))
        assert strategy.sample_losses is None
        _run_curriculum(loop, curriculum_ds, max_epochs=2)
        # After epoch 0 the scores were recomputed with the trained model
        assert strategy.sample_losses is not None
        assert loop.difficulty_scores is strategy.sample_losses

    def test_no_curriculum_dataset_warns_and_disables(self):
        loop = CurriculumLoop(strategy="random")
        with pytest.warns(UserWarning, match="no CurriculumDataset"):
            _run_curriculum(loop, make_tensor_dataset(8), max_epochs=1)
        assert loop.get_curriculum_stats() == []


class TestCurriculumFactory:
    def test_factories(self):
        assert isinstance(create_length_curriculum_loop().strategy, LengthBasedCurriculum)
        loss_loop = create_loss_curriculum_loop(warmup_epochs=3)
        assert isinstance(loss_loop.strategy, LossBasedCurriculum)
        assert loss_loop.strategy.warmup_epochs == 3
        assert isinstance(create_random_curriculum_loop().strategy, RandomCurriculum)
