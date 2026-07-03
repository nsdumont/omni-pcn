"""
NetworkState and NetworkParams - JAX pytrees for dynamic state and learnable parameters.

These are registered as JAX pytrees at module import time so they can be used
with JAX transformations like jit, grad, and vmap.
"""

from typing import NamedTuple, List
import jax.numpy as jnp
from jax import tree_util


class NetworkState(NamedTuple):
    """
    Mutable state during inference.

    Attributes:
        values: List of (batch, dim) arrays, one per layer - current activations
        errors: List of (batch, dim) arrays, one per Predict connection - prediction errors
        precisions: List of (batch, dim) arrays, one per Predict connection - precisions
        clamped: List of (batch,) bool arrays, one per layer - whether values are fixed
    """
    values: List[jnp.ndarray]
    errors: List[jnp.ndarray]
    precisions: List[jnp.ndarray]
    clamped: List[jnp.ndarray]


class NetworkParams(NamedTuple):
    """
    Learnable parameters.

    Attributes:
        predict_weights: List of (post_dim, pre_dim) arrays for Predict connections
        predict_biases: List of (post_dim,) arrays for Predict connections
        project_weights: List of weight arrays for Project connections
        project_biases: List of (post_dim,) bias arrays for Project connections (zeros(1) dummy when use_bias=False)
        modulate_weights: List of weight arrays for Modulate connections
        modulate_biases: List of (post_dim,) bias arrays for Modulate connections (zeros(1) dummy when use_bias=False)
        precision_weights: List of (1, pre_dim) or (post_dim, pre_dim) arrays per Predict connection
        precision_biases: List of (1,) or (post_dim,) arrays per Predict connection
    """
    predict_weights: List[jnp.ndarray]
    predict_biases: List[jnp.ndarray]
    project_weights: List[jnp.ndarray]
    project_biases: List[jnp.ndarray]
    modulate_weights: List[jnp.ndarray]
    modulate_biases: List[jnp.ndarray]
    precision_weights: List[jnp.ndarray]
    precision_biases: List[jnp.ndarray]


# Register as JAX pytrees at module import time
# This allows JAX transformations (jit, grad, vmap) to work with these types
tree_util.register_pytree_node(
    NetworkState,
    lambda s: ((s.values, s.errors, s.precisions, s.clamped), None),
    lambda _, children: NetworkState(*children)
)

tree_util.register_pytree_node(
    NetworkParams,
    lambda p: ((p.predict_weights, p.predict_biases,
                p.project_weights, p.project_biases,
                p.modulate_weights, p.modulate_biases,
                p.precision_weights, p.precision_biases), None),
    lambda _, children: NetworkParams(*children)
)
