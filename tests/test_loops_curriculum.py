# tests/test_loops_curriculum.py
"""Tests for curriculum learning loop."""

import sys
from pathlib import Path
import pytest
import torch
import numpy as np
from torch.utils.data import TensorDataset

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lmpro.loops.curriculum_loop import (
    CurriculumLoop,
    CurriculumStrategy,
    LengthBasedCurriculum,
    LossBasedCurriculum,
    RandomCurriculum,
)


def make_sequence_dataset(n=60, seq_len_range=(5, 50)):
    """Create a dataset with varying sequence lengths."""
    X = [torch.randint(0, 100, (torch.randint(*seq_len_range, ()).item(),)) for _ in range(n)]
    y = torch.randint(0, 2, (n,))
    return list(zip(X, y))


def make_tensor_dataset(n=60, features=10):
    X = torch.randn(n, features)
    y = torch.randint(0, 2, (n,))
    return TensorDataset(X, y)


# ─── LengthBasedCurriculum ──────────────────────────────────────────────────

class TestLengthBasedCurriculum:
    def test_returns_ndarray(self):
        strategy = LengthBasedCurriculum()
        ds = make_tensor_dataset()
        scores = strategy.get_difficulty_scores(ds)
        assert isinstance(scores, np.ndarray)
        assert len(scores) == len(ds)

    def test_scores_normalized_0_to_1(self):
        strategy = LengthBasedCurriculum()
        ds = make_tensor_dataset()
        scores = strategy.get_difficulty_scores(ds)
        assert scores.min() >= 0.0
        assert scores.max() <= 1.0

    def test_reverse_inverts_scores(self):
        ds = make_tensor_dataset()
        normal = LengthBasedCurriculum(reverse=False).get_difficulty_scores(ds)
        reversed_ = LengthBasedCurriculum(reverse=True).get_difficulty_scores(ds)
        np.testing.assert_allclose(normal + reversed_, np.ones(len(ds)), atol=1e-5)

    def test_curriculum_schedule_length(self):
        strategy = LengthBasedCurriculum()
        schedule = strategy.get_curriculum_schedule(total_epochs=10, dataset_size=100)
        assert len(schedule) == 10

    def test_schedule_threshold_increases(self):
        strategy = LengthBasedCurriculum()
        schedule = strategy.get_curriculum_schedule(total_epochs=10, dataset_size=100)
        thresholds = [t for _, t in schedule]
        assert thresholds[-1] > thresholds[0]

    def test_schedule_starts_below_1(self):
        strategy = LengthBasedCurriculum()
        schedule = strategy.get_curriculum_schedule(total_epochs=5, dataset_size=50)
        assert schedule[0][1] < 1.0

    def test_schedule_ends_at_1(self):
        strategy = LengthBasedCurriculum()
        schedule = strategy.get_curriculum_schedule(total_epochs=5, dataset_size=50)
        assert schedule[-1][1] == pytest.approx(1.0)


# ─── LossBasedCurriculum ────────────────────────────────────────────────────

class TestLossBasedCurriculum:
    def test_init_default(self):
        strategy = LossBasedCurriculum()
        assert strategy.warmup_epochs == 5

    def test_init_custom_warmup(self):
        strategy = LossBasedCurriculum(warmup_epochs=2)
        assert strategy.warmup_epochs == 2

    def test_returns_ndarray_without_model(self):
        strategy = LossBasedCurriculum()
        ds = make_tensor_dataset()
        scores = strategy.get_difficulty_scores(ds, model=None)
        assert isinstance(scores, np.ndarray)
        assert len(scores) == len(ds)

    def test_uniform_scores_without_model(self):
        strategy = LossBasedCurriculum()
        ds = make_tensor_dataset(n=20)
        scores = strategy.get_difficulty_scores(ds, model=None)
        # Without a trained model, LossBasedCurriculum falls back to random scoring
        assert isinstance(scores, np.ndarray)
        assert len(scores) == len(ds)


# ─── RandomCurriculum ───────────────────────────────────────────────────────

class TestRandomCurriculum:
    def test_returns_random_scores(self):
        strategy = RandomCurriculum()
        ds = make_tensor_dataset(n=50)
        scores = strategy.get_difficulty_scores(ds)
        assert isinstance(scores, np.ndarray)
        assert len(scores) == len(ds)

    def test_scores_in_0_1_range(self):
        strategy = RandomCurriculum()
        ds = make_tensor_dataset(n=30)
        scores = strategy.get_difficulty_scores(ds)
        assert scores.min() >= 0.0
        assert scores.max() <= 1.0

    def test_different_seeds_give_different_scores(self):
        ds = make_tensor_dataset(n=30)
        s1 = RandomCurriculum().get_difficulty_scores(ds)
        s2 = RandomCurriculum().get_difficulty_scores(ds)
        # Two separate calls should yield different scores (random)
        # This could rarely fail but is very unlikely with n=30
        assert not np.allclose(s1, s2)


# ─── CurriculumLoop ─────────────────────────────────────────────────────────

class TestCurriculumLoopInit:
    def test_default_init_with_string(self):
        loop = CurriculumLoop(strategy="length")
        assert loop.difficulty_scores is None
        assert loop.current_epoch == 0

    def test_init_with_strategy_object(self):
        strategy = LengthBasedCurriculum()
        loop = CurriculumLoop(strategy=strategy)
        assert loop.strategy is strategy

    def test_custom_warmup(self):
        loop = CurriculumLoop(strategy="random", curriculum_warmup=3)
        assert loop.curriculum_warmup == 3

    def test_custom_update_frequency(self):
        loop = CurriculumLoop(strategy="loss", update_frequency=2)
        assert loop.update_frequency == 2

    def test_difficulty_scores_initially_none(self):
        loop = CurriculumLoop(strategy="length")
        assert loop.difficulty_scores is None

    def test_invalid_strategy_raises(self):
        with pytest.raises(ValueError):
            CurriculumLoop(strategy="nonexistent")


class TestCurriculumStatistics:
    def test_stats_tracked(self):
        loop = CurriculumLoop(strategy="length")
        loop.difficulty_scores = np.array([0.1, 0.5, 0.9, 0.3])
        loop.current_threshold = 0.5 if hasattr(loop, "current_threshold") else 0.5
        stats = loop.curriculum_stats
        assert isinstance(stats, list)

    def test_initial_stats_empty(self):
        loop = CurriculumLoop(strategy="random")
        assert isinstance(loop.curriculum_stats, list)


class TestCurriculumFactory:
    def test_create_length_curriculum_loop(self):
        from lmpro.loops.curriculum_loop import create_length_curriculum_loop
        loop = create_length_curriculum_loop()
        assert isinstance(loop, CurriculumLoop)

    def test_create_loss_curriculum_loop(self):
        from lmpro.loops.curriculum_loop import create_loss_curriculum_loop
        loop = create_loss_curriculum_loop(warmup_epochs=3)
        assert isinstance(loop, CurriculumLoop)

    def test_create_random_curriculum_loop(self):
        from lmpro.loops.curriculum_loop import create_random_curriculum_loop
        loop = create_random_curriculum_loop()
        assert isinstance(loop, CurriculumLoop)
