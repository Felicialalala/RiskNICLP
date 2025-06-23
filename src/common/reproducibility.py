"""
Reproducibility utilities for RiskNICLP-5G

Ensures deterministic behavior across:
- PyTorch operations
- NumPy random operations  
- Python random module
- CUDA operations (if available)
- DGL graph operations
"""

import os
import random
import numpy as np
import torch
from typing import Dict, Any, Optional, Tuple
import warnings


def set_seed(seed: int, deterministic: bool = True, warn_cuda: bool = True) -> None:
    """
    Set random seeds for reproducible experiments
    
    Args:
        seed: Random seed value
        deterministic: Whether to enable deterministic algorithms (may impact performance)
        warn_cuda: Whether to warn about CUDA non-deterministic operations
    """
    
    # Python random
    random.seed(seed)
    
    # NumPy random
    np.random.seed(seed)
    
    # PyTorch random
    torch.manual_seed(seed)
    
    # CUDA random (if available)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # for multi-GPU
        
        if deterministic:
            # Enable deterministic algorithms
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            
            # Set environment variable for additional determinism
            os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
            
            # Use deterministic algorithms where available
            try:
                torch.use_deterministic_algorithms(True)
            except AttributeError:
                # Fallback for older PyTorch versions
                warnings.warn("torch.use_deterministic_algorithms not available. "
                            "Using legacy deterministic settings.")
                
        elif warn_cuda:
            warnings.warn(
                "CUDA is available but deterministic mode is disabled. "
                "Results may not be fully reproducible."
            )
    
    # DGL random (if available)
    try:
        import dgl
        dgl.random.seed(seed)
    except ImportError:
        pass
    
    # Set environment variables for additional reproducibility
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    print(f"🔧 Random seed set to {seed} (deterministic={deterministic})")


def get_random_states() -> Dict[str, Any]:
    """
    Get current random states for all random number generators
    
    Returns:
        Dictionary containing all random states
    """
    states = {
        'python_random': random.getstate(),
        'numpy_random': np.random.get_state(),
        'torch_random': torch.get_rng_state(),
    }
    
    # CUDA random states
    if torch.cuda.is_available():
        states['torch_cuda_random'] = torch.cuda.get_rng_state()
        if torch.cuda.device_count() > 1:
            states['torch_cuda_random_all'] = torch.cuda.get_rng_state_all()
    
    # DGL random state
    try:
        import dgl
        states['dgl_random'] = dgl.random.get_seed()
    except ImportError:
        pass
    
    return states


def restore_random_states(states: Dict[str, Any]) -> None:
    """
    Restore random states from saved state dictionary
    
    Args:
        states: Dictionary containing random states from get_random_states()
    """
    
    # Python random
    if 'python_random' in states:
        random.setstate(states['python_random'])
    
    # NumPy random
    if 'numpy_random' in states:
        np.random.set_state(states['numpy_random'])
    
    # PyTorch random
    if 'torch_random' in states:
        torch.set_rng_state(states['torch_random'])
    
    # CUDA random states
    if torch.cuda.is_available():
        if 'torch_cuda_random' in states:
            torch.cuda.set_rng_state(states['torch_cuda_random'])
        if 'torch_cuda_random_all' in states:
            torch.cuda.set_rng_state_all(states['torch_cuda_random_all'])
    
    # DGL random state
    try:
        import dgl
        if 'dgl_random' in states:
            dgl.random.seed(states['dgl_random'])
    except ImportError:
        pass
    
    print("🔧 Random states restored")


def create_reproducible_dataloader(
    dataset,
    batch_size: int,
    shuffle: bool = True,
    seed: Optional[int] = None,
    **kwargs
) -> torch.utils.data.DataLoader:
    """
    Create a DataLoader with reproducible behavior
    
    Args:
        dataset: PyTorch dataset
        batch_size: Batch size
        shuffle: Whether to shuffle data
        seed: Random seed for shuffling
        **kwargs: Additional DataLoader arguments
    """
    
    # Create reproducible generator
    if seed is not None and shuffle:
        generator = torch.Generator()
        generator.manual_seed(seed)
        kwargs['generator'] = generator
    
    # Ensure deterministic worker initialization
    if 'num_workers' in kwargs and kwargs['num_workers'] > 0:
        def worker_init_fn(worker_id):
            worker_seed = seed + worker_id if seed is not None else None
            if worker_seed is not None:
                np.random.seed(worker_seed)
                random.seed(worker_seed)
        
        kwargs['worker_init_fn'] = worker_init_fn
    
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        **kwargs
    )


class ReproducibilityContext:
    """
    Context manager for temporary reproducibility settings
    
    Example:
        with ReproducibilityContext(42, deterministic=True):
            # Reproducible operations
            model = create_model()
            result = train_model(model)
        # Original random states restored
    """
    
    def __init__(self, seed: int, deterministic: bool = True):
        self.seed = seed
        self.deterministic = deterministic
        self.original_states = None
        self.original_deterministic = None
        self.original_benchmark = None
        
    def __enter__(self):
        # Save original states
        self.original_states = get_random_states()
        
        # Save original CUDA settings
        if torch.cuda.is_available():
            self.original_deterministic = torch.backends.cudnn.deterministic
            self.original_benchmark = torch.backends.cudnn.benchmark
        
        # Set reproducible state
        set_seed(self.seed, self.deterministic, warn_cuda=False)
        
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore original states
        if self.original_states:
            restore_random_states(self.original_states)
        
        # Restore original CUDA settings
        if torch.cuda.is_available():
            if self.original_deterministic is not None:
                torch.backends.cudnn.deterministic = self.original_deterministic
            if self.original_benchmark is not None:
                torch.backends.cudnn.benchmark = self.original_benchmark


def verify_reproducibility(
    func,
    *args,
    seed: int = 42,
    n_runs: int = 3,
    tolerance: float = 1e-6,
    **kwargs
) -> Tuple[bool, list]:
    """
    Verify that a function produces reproducible results
    
    Args:
        func: Function to test
        *args: Arguments for function
        seed: Random seed to use
        n_runs: Number of runs to test
        tolerance: Tolerance for numerical differences
        **kwargs: Keyword arguments for function
    
    Returns:
        Tuple of (is_reproducible, list_of_results)
    """
    
    results = []
    
    for i in range(n_runs):
        with ReproducibilityContext(seed, deterministic=True):
            result = func(*args, **kwargs)
            results.append(result)
    
    # Check if all results are identical (within tolerance)
    is_reproducible = True
    
    if isinstance(results[0], torch.Tensor):
        for i in range(1, len(results)):
            if not torch.allclose(results[0], results[i], atol=tolerance):
                is_reproducible = False
                break
    elif isinstance(results[0], np.ndarray):
        for i in range(1, len(results)):
            if not np.allclose(results[0], results[i], atol=tolerance):
                is_reproducible = False
                break
    else:
        # For other types, use direct comparison
        for i in range(1, len(results)):
            if results[0] != results[i]:
                is_reproducible = False
                break
    
    return is_reproducible, results


def configure_reproducible_environment():
    """
    Configure environment variables for maximum reproducibility
    """
    
    # Python hash randomization
    os.environ['PYTHONHASHSEED'] = '0'
    
    # CUDA deterministic operations
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    
    # Disable CUDA kernel caching for consistent behavior
    os.environ['CUDA_CACHE_DISABLE'] = '1'
    
    # Disable MKL threading for NumPy operations
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['OMP_NUM_THREADS'] = '1'
    
    print("🔧 Environment configured for reproducibility") 