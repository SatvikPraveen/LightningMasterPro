# File: src/lmpro/utils/seed.py

"""
Seed utilities for reproducible experiments
"""

import os
import random
from typing import Any, Dict, Optional

import numpy as np
import torch
from lightning.pytorch import seed_everything
from lightning.pytorch.utilities.rank_zero import rank_zero_info

_DETERMINISM_ENV_VARS = ("CUBLAS_WORKSPACE_CONFIG", "PL_GLOBAL_SEED", "PL_SEED_WORKERS")


def seed_everything_deterministic(
    seed: int = 42,
    workers: bool = True,
    use_deterministic_algorithms: bool = True,
    warn_only: bool = True,
) -> int:
    """
    Seed Python, NumPy and PyTorch via :func:`lightning.pytorch.seed_everything`
    and optionally enable PyTorch's deterministic mode.

    Args:
        seed: Random seed.
        workers: Forwarded to ``seed_everything``; when True, DataLoader workers
            are seeded through Lightning's ``pl_worker_init_function``.
        use_deterministic_algorithms: If True, call
            ``torch.use_deterministic_algorithms(True)``, disable cuDNN
            benchmarking / enable cuDNN determinism and set
            ``CUBLAS_WORKSPACE_CONFIG`` (required by CUDA for deterministic
            matmuls). Nothing is changed when False.
        warn_only: Only warn (instead of raising) on non-deterministic ops.

    Returns:
        The seed that was set.
    """
    seed = seed_everything(seed, workers=workers, verbose=False)

    if use_deterministic_algorithms:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=warn_only)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    rank_zero_info(f"Global seed set to {seed} (deterministic algorithms: {use_deterministic_algorithms})")
    return seed


def get_random_state() -> Dict[str, Any]:
    """Snapshot the state of all random number generators."""
    state = {
        "python_random": random.getstate(),
        "numpy_random": np.random.get_state(),
        "torch_random": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda_random"] = torch.cuda.get_rng_state_all()
    return state


def set_random_state(state: Dict[str, Any]) -> None:
    """Restore RNG states captured with :func:`get_random_state`."""
    if "python_random" in state:
        random.setstate(state["python_random"])
    if "numpy_random" in state:
        np.random.set_state(state["numpy_random"])
    if "torch_random" in state:
        torch.set_rng_state(state["torch_random"])
    if "torch_cuda_random" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda_random"])


class SeedContext:
    """
    Context manager for a temporary seed.

    On exit it restores everything it changed: RNG states, the deterministic
    algorithms flag, cuDNN flags and the environment variables touched by
    :func:`seed_everything_deterministic`.
    """

    def __init__(self, seed: int, deterministic: bool = True):
        self.seed = seed
        self.deterministic = deterministic
        self._state: Optional[Dict[str, Any]] = None
        self._flags: Optional[Dict[str, Any]] = None
        self._env: Dict[str, Optional[str]] = {}

    def __enter__(self) -> "SeedContext":
        self._state = get_random_state()
        self._flags = {
            "deterministic": torch.are_deterministic_algorithms_enabled(),
            "warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
        }
        self._env = {name: os.environ.get(name) for name in _DETERMINISM_ENV_VARS}

        seed_everything_deterministic(
            self.seed, workers=False, use_deterministic_algorithms=self.deterministic, warn_only=True
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._state is not None:
            set_random_state(self._state)
        if self._flags is not None:
            torch.use_deterministic_algorithms(self._flags["deterministic"], warn_only=self._flags["warn_only"])
            torch.backends.cudnn.deterministic = self._flags["cudnn_deterministic"]
            torch.backends.cudnn.benchmark = self._flags["cudnn_benchmark"]
        for name, value in self._env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def worker_init_fn(worker_id: int, seed: Optional[int] = None) -> None:
    """
    DataLoader ``worker_init_fn`` giving each worker a distinct, reproducible seed.

    Args:
        worker_id: ID of the worker process.
        seed: Base seed; defaults to the worker's ``torch.initial_seed()``
            (which the DataLoader already derives from the main process seed).
    """
    if seed is None:
        seed = torch.initial_seed() % 2**32
    worker_seed = (seed + worker_id) % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def validate_reproducibility(model, dataloader, num_iterations: int = 3) -> bool:
    """Return True if ``model`` gives identical outputs on one batch under the same seed."""
    model.eval()
    test_batch = next(iter(dataloader))
    test_input = test_batch[0] if isinstance(test_batch, (list, tuple)) else test_batch

    outputs = []
    for _ in range(num_iterations):
        with SeedContext(42), torch.no_grad():
            outputs.append(model(test_input).clone())

    for i in range(1, num_iterations):
        if not torch.allclose(outputs[0], outputs[i], rtol=1e-6, atol=1e-8):
            rank_zero_info(f"Reproducibility check failed at iteration {i}")
            return False
    rank_zero_info("Reproducibility check passed")
    return True
