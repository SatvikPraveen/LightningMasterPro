# File: src/lmpro/utils/seed.py

"""
Seed utilities for reproducible experiments
"""

import os
import random
import numpy as np
import torch
from typing import Optional, Dict, Any
from lightning.pytorch.utilities.rank_zero import rank_zero_info


def seed_everything_deterministic(
    seed: int = 42,
    workers: bool = True,
    use_deterministic_algorithms: bool = True,
    warn_only: bool = True
) -> int:
    """
    Enhanced seed_everything with full deterministic mode
    
    Args:
        seed: Random seed
        workers: If True, sets PYTHONHASHSEED for worker processes  
        use_deterministic_algorithms: Use deterministic algorithms when available
        warn_only: Only warn about non-deterministic operations instead of failing
        
    Returns:
        The seed that was set
    """
    # Set Python random seed
    random.seed(seed)
    
    # Set NumPy random seed
    np.random.seed(seed)
    
    # Set PyTorch random seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Set environment variables for determinism
    if workers:
        os.environ["PYTHONHASHSEED"] = str(seed)
    
    # Configure PyTorch for deterministic operations
    if use_deterministic_algorithms:
        torch.use_deterministic_algorithms(True, warn_only=warn_only)
        
        # Set additional environment variables for deterministic behavior
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
        
        # Configure cudnn for deterministic behavior
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        # Additional settings for full determinism
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
        os.environ["PYTHONHASHSEED"] = str(seed)
    
    rank_zero_info(f"Global seed set to {seed} with deterministic mode: {use_deterministic_algorithms}")
    
    return seed


def get_random_state() -> Dict[str, Any]:
    """
    Get the current random state from all random number generators
    
    Returns:
        Dictionary containing the current random states
    """
    state = {
        "python_random": random.getstate(),
        "numpy_random": np.random.get_state(),
        "torch_random": torch.get_rng_state(),
    }
    
    if torch.cuda.is_available():
        state["torch_cuda_random"] = torch.cuda.get_rng_state_all()
    
    return state


def set_random_state(state: Dict[str, Any]) -> None:
    """
    Set the random state for all random number generators
    
    Args:
        state: Dictionary containing random states from get_random_state()
    """
    if "python_random" in state:
        random.setstate(state["python_random"])
    
    if "numpy_random" in state:
        np.random.set_state(state["numpy_random"])
    
    if "torch_random" in state:
        torch.set_rng_state(state["torch_random"])
    
    if "torch_cuda_random" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda_random"])


class SeedContext:
    """Context manager for temporary seed changes"""
    
    def __init__(self, seed: int, deterministic: bool = True):
        self.seed = seed
        self.deterministic = deterministic
        self.original_state = None
        self.original_deterministic = None
        
    def __enter__(self):
        # Save original state
        self.original_state = get_random_state()
        self.original_deterministic = torch.are_deterministic_algorithms_enabled()
        
        # Set new seed
        seed_everything_deterministic(
            self.seed, 
            use_deterministic_algorithms=self.deterministic,
            warn_only=True
        )
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore original state
        set_random_state(self.original_state)
        
        # Restore deterministic setting
        if self.original_deterministic != torch.are_deterministic_algorithms_enabled():
            torch.use_deterministic_algorithms(self.original_deterministic, warn_only=True)


def worker_init_fn(worker_id: int, seed: Optional[int] = None) -> None:
    """
    Worker initialization function for DataLoader
    Ensures each worker has a different but reproducible seed
    
    Args:
        worker_id: ID of the worker process
        seed: Base seed (if None, uses current torch seed)
    """
    if seed is None:
        seed = torch.initial_seed() % 2**32
    
    worker_seed = seed + worker_id
    
    # Set seeds for this worker
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)
    
    
def validate_reproducibility(model, dataloader, num_iterations: int = 3) -> bool:
    """
    Validate that model produces identical outputs given same inputs
    
    Args:
        model: PyTorch model to test
        dataloader: DataLoader to get test batches
        num_iterations: Number of iterations to test
        
    Returns:
        True if outputs are identical across iterations
    """
    model.eval()
    
    # Get a test batch
    test_batch = next(iter(dataloader))
    if isinstance(test_batch, (list, tuple)):
        test_input = test_batch[0]
    else:
        test_input = test_batch
    
    outputs = []
    
    # Run multiple iterations with same seed
    for i in range(num_iterations):
        with SeedContext(42):
            with torch.no_grad():
                output = model(test_input)
                outputs.append(output.clone())
    
    # Check if all outputs are identical
    for i in range(1, num_iterations):
        if not torch.allclose(outputs[0], outputs[i], rtol=1e-6, atol=1e-8):
            rank_zero_info(f"Reproducibility check failed at iteration {i}")
            return False
    
    rank_zero_info("Reproducibility check passed!")
    return True