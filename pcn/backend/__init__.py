"""
PCN Backend Module - Optimized JAX functions for simulation and learning.

This module contains the low-level computational functions that implement
predictive coding dynamics. Functions are JIT-compiled for speed.

The main entry point is `run_batch`, which performs a complete training/inference
step in a single JIT-compiled function with lax.fori_loop for efficiency.

For testing and advanced usage, lower-level functions are also available.
"""

from .simulation import (
    # Main entry points
    run_batch,
    # Activation functions
    ACTIVATIONS,
)
from .backprop_simulation import run_backprop_batch

__all__ = [
    # Main entry points
    'run_batch',
    'run_backprop_batch',
    # Activation functions
    'ACTIVATIONS',
]
