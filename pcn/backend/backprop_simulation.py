"""
Backprop baseline backend.

Standard forward pass through predict connections + autodiff weight updates
via optax. No PCN dynamics, errors, or precisions.
"""

from typing import Dict, Tuple
import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from ..core.structure import NetworkStructure
from .simulation import ACTIVATIONS


def _forward(values_list, predict_conns, predict_weights, activation_types, clamped_layer_indices):
    """Run a forward pass through predict connections, updating unclamped layers.

    Activations are applied to each post layer after its value is computed
    (unlike PCN where activations are applied at read time).
    """
    for i, conn in enumerate(predict_conns):
        if conn.post_idx in clamped_layer_indices:
            continue
        pre_val = values_list[conn.pre_idx]
        act_fn = ACTIVATIONS[activation_types[conn.pre_idx]]
        pre_act = act_fn(pre_val)
        raw = conn.prediction(pre_act, predict_weights[i])
        post_act_fn = ACTIVATIONS[activation_types[conn.post_idx]]
        values_list[conn.post_idx] = post_act_fn(raw)
    return values_list


@eqx.filter_jit
def run_backprop_batch(
    sample: Dict[str, jnp.ndarray],
    predict_weights: tuple,
    structure: NetworkStructure,
    data_map: tuple,
    objective_fn,
    optimizer: optax.GradientTransformation,
    opt_state: optax.OptState,
    learning: bool = True,
) -> Tuple[tuple, optax.OptState, tuple, jnp.ndarray]:
    """
    Run a single backprop batch: forward pass, compute loss, update weights.

    Args:
        sample: Dict of data arrays.
        predict_weights: Tuple of weight arrays for predict connections.
        structure: Static NetworkStructure.
        data_map: Tuple of (layer_idx, sample_key) pairs (static).
        objective_fn: Loss function with signature
            (values: tuple, sample: dict) -> scalar.
        optimizer: An optax optimizer (static).
        opt_state: Current optimizer state.
        learning: If True, compute gradients and update weights.

    Returns:
        predict_weights: Updated (or unchanged) weight tuple.
        opt_state: Updated (or unchanged) optimizer state.
        values: Final layer values as a tuple.
        loss: Scalar loss value.
    """
    layer_dims = structure.layer_dims
    predict_conns = structure.predict_conns
    activation_types = tuple(layer.activation_type for layer in structure.layers)

    first_key = list(sample.keys())[0]
    batch_size = sample[first_key].shape[0]

    # Initialize and clamp
    values_list = [jnp.zeros((batch_size, dim)) for dim in layer_dims]
    clamped_layer_indices = frozenset(layer_idx for layer_idx, _ in data_map)
    for layer_idx, sample_key in data_map:
        values_list[layer_idx] = sample[sample_key]

    # Forward pass (for the returned values)
    values_list = _forward(
        values_list, predict_conns, predict_weights,
        activation_types, clamped_layer_indices
    )
    values = tuple(values_list)

    # Loss and gradient computation
    def loss_fn(pw):
        vl = [jnp.zeros((batch_size, dim)) for dim in layer_dims]
        for layer_idx, sample_key in data_map:
            vl[layer_idx] = sample[sample_key]
        vl = _forward(vl, predict_conns, pw, activation_types, clamped_layer_indices)
        return objective_fn(tuple(vl), sample)

    loss = loss_fn(predict_weights)

    def do_update(pw, st):
        grads = jax.grad(loss_fn)(pw)
        updates, new_st = optimizer.update(grads, st, pw)
        new_pw = optax.apply_updates(pw, updates)
        return new_pw, new_st

    def skip_update(pw, st):
        return pw, st

    predict_weights, opt_state = jax.lax.cond(
        learning, do_update, skip_update, predict_weights, opt_state
    )

    return predict_weights, opt_state, values, loss
