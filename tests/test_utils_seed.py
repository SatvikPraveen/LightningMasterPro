# tests/test_utils_seed.py
"""Tests for seed/determinism utilities."""

import os
import random
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lmpro.utils import worker_init_fn as exported_worker_init_fn  # noqa: E402
from lmpro.utils.seed import (  # noqa: E402
    SeedContext,
    get_random_state,
    seed_everything_deterministic,
    set_random_state,
    validate_reproducibility,
    worker_init_fn,
)


@pytest.fixture(autouse=True)
def restore_determinism_flags():
    flags = (
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
        torch.backends.cudnn.deterministic,
        torch.backends.cudnn.benchmark,
    )
    env = {k: os.environ.get(k) for k in ("CUBLAS_WORKSPACE_CONFIG", "PL_GLOBAL_SEED", "PL_SEED_WORKERS")}
    yield
    torch.use_deterministic_algorithms(flags[0], warn_only=flags[1])
    torch.backends.cudnn.deterministic = flags[2]
    torch.backends.cudnn.benchmark = flags[3]
    for k, v in env.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


class TestSeedEverythingDeterministic:
    def test_returns_seed(self):
        assert seed_everything_deterministic(seed=123) == 123

    def test_torch_numpy_python_reproducibility(self):
        seed_everything_deterministic(seed=42)
        a = (torch.randn(5), np.random.randn(5), [random.random() for _ in range(5)])
        seed_everything_deterministic(seed=42)
        b = (torch.randn(5), np.random.randn(5), [random.random() for _ in range(5)])
        assert torch.equal(a[0], b[0])
        np.testing.assert_array_equal(a[1], b[1])
        assert a[2] == b[2]

    def test_different_seeds_differ(self):
        seed_everything_deterministic(seed=1)
        a = torch.randn(5)
        seed_everything_deterministic(seed=2)
        assert not torch.allclose(a, torch.randn(5))

    def test_delegates_to_lightning_seed_everything(self):
        seed_everything_deterministic(seed=77, workers=True)
        assert os.environ["PL_GLOBAL_SEED"] == "77"
        assert os.environ["PL_SEED_WORKERS"] == "1"
        seed_everything_deterministic(seed=78, workers=False)
        assert os.environ["PL_GLOBAL_SEED"] == "78"
        assert os.environ["PL_SEED_WORKERS"] == "0"

    def test_deterministic_flags_only_when_asked(self):
        torch.use_deterministic_algorithms(False)
        torch.backends.cudnn.benchmark = True
        os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)

        seed_everything_deterministic(seed=5, use_deterministic_algorithms=False)
        assert torch.are_deterministic_algorithms_enabled() is False
        assert torch.backends.cudnn.benchmark is True
        assert "CUBLAS_WORKSPACE_CONFIG" not in os.environ

        seed_everything_deterministic(seed=5, use_deterministic_algorithms=True)
        assert torch.are_deterministic_algorithms_enabled() is True
        assert torch.is_deterministic_algorithms_warn_only_enabled() is True
        assert torch.backends.cudnn.deterministic is True
        assert torch.backends.cudnn.benchmark is False
        assert "CUBLAS_WORKSPACE_CONFIG" in os.environ

    def test_never_sets_cuda_launch_blocking(self):
        os.environ.pop("CUDA_LAUNCH_BLOCKING", None)
        seed_everything_deterministic(seed=9, use_deterministic_algorithms=True)
        assert "CUDA_LAUNCH_BLOCKING" not in os.environ


class TestGetSetRandomState:
    def test_get_state_returns_dict(self):
        state = get_random_state()
        assert {"torch_random", "numpy_random", "python_random"} <= set(state)

    def test_roundtrip_all_generators(self):
        seed_everything_deterministic(42)
        state = get_random_state()
        first = (torch.randn(10), np.random.randn(10), [random.random() for _ in range(5)])
        _ = torch.randn(100), np.random.randn(100), [random.random() for _ in range(50)]
        set_random_state(state)
        second = (torch.randn(10), np.random.randn(10), [random.random() for _ in range(5)])
        assert torch.equal(first[0], second[0])
        np.testing.assert_array_equal(first[1], second[1])
        assert first[2] == second[2]


class TestSeedContext:
    def test_context_restores_rng_state(self):
        seed_everything_deterministic(42)
        before = torch.randn(5)
        seed_everything_deterministic(42)
        with SeedContext(seed=999):
            _ = torch.randn(100)
        assert torch.equal(before, torch.randn(5))

    def test_context_seeds_inside(self):
        with SeedContext(seed=1):
            a = torch.randn(5)
        with SeedContext(seed=1):
            b = torch.randn(5)
        with SeedContext(seed=2):
            c = torch.randn(5)
        assert torch.equal(a, b)
        assert not torch.allclose(a, c)

    def test_context_exception_still_restores(self):
        seed_everything_deterministic(42)
        expected = torch.randn(3)
        seed_everything_deterministic(42)
        with pytest.raises(ValueError):
            with SeedContext(seed=555):
                _ = torch.randn(10)
                raise ValueError("Intentional error")
        assert torch.equal(torch.randn(3), expected)

    def test_context_restores_flags_and_env(self):
        torch.use_deterministic_algorithms(False)
        torch.backends.cudnn.benchmark = True
        os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
        os.environ["PL_GLOBAL_SEED"] = "12345"

        with SeedContext(seed=3, deterministic=True):
            assert torch.are_deterministic_algorithms_enabled() is True
            assert torch.backends.cudnn.benchmark is False
            assert "CUBLAS_WORKSPACE_CONFIG" in os.environ
            assert os.environ["PL_GLOBAL_SEED"] == "3"

        assert torch.are_deterministic_algorithms_enabled() is False
        assert torch.backends.cudnn.benchmark is True
        assert "CUBLAS_WORKSPACE_CONFIG" not in os.environ
        assert os.environ["PL_GLOBAL_SEED"] == "12345"


class TestWorkerInitFn:
    def test_exported_from_utils(self):
        assert exported_worker_init_fn is worker_init_fn

    def test_reproducible_per_worker(self):
        worker_init_fn(0, seed=10)
        a = torch.randn(3)
        worker_init_fn(0, seed=10)
        assert torch.equal(a, torch.randn(3))

    def test_different_workers_differ(self):
        worker_init_fn(0, seed=10)
        a = torch.randn(3)
        worker_init_fn(1, seed=10)
        assert not torch.allclose(a, torch.randn(3))

    def test_default_seed_uses_torch_initial_seed(self):
        torch.manual_seed(7)
        worker_init_fn(2)
        a = (torch.randn(2), np.random.rand(2), random.random())
        torch.manual_seed(7)
        worker_init_fn(2)
        b = (torch.randn(2), np.random.rand(2), random.random())
        assert torch.equal(a[0], b[0])
        np.testing.assert_array_equal(a[1], b[1])
        assert a[2] == b[2]


class TestValidateReproducibility:
    def test_deterministic_model_passes(self):
        model = torch.nn.Linear(4, 2)
        loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.randn(8, 4)), batch_size=4)
        assert validate_reproducibility(model, loader) is True

    def test_random_model_fails(self):
        class Noisy(torch.nn.Module):
            def forward(self, x):
                return x + torch.randn_like(x)

        loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.randn(8, 4)), batch_size=4)
        # Same seed inside SeedContext each iteration -> identical noise -> passes
        assert validate_reproducibility(Noisy(), loader) is True
