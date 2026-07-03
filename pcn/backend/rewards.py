"""
Reward function factories for three-factor Hebbian learning.

Reward functions follow the same ``(inputs, fn)`` signature as
``GradientDescent.loss_fn``:

    reward_fn = ((inputs,), fn)

where ``inputs`` is a NodeRef, a sample dict key string, or a tuple
mixing both; ``fn`` receives the resolved arrays as positional args and
returns a ``(batch,)`` or scalar reward.
"""

import jax.numpy as jnp


def make_mse_reward(output_node, target_key: str = "label"):
    """Factory: returns an (inputs, fn) tuple for a negative-MSE reward.

    Args:
        output_node: NodeRef of the output layer's value (e.g. l_out.value).
        target_key: Sample dict key for target values.

    Example::

        rule = pcn.ThreeFactorHebbian(
            learning_rate=1e-4,
            reward_fn=make_mse_reward(l_out.value, 'label'),
        )
    """
    def neg_mse(pred, target):
        return -jnp.mean((pred - target) ** 2, axis=-1)
    return ((output_node, target_key), neg_mse)


def make_accuracy_reward(output_node, target_key: str = "label"):
    """Factory: returns an (inputs, fn) tuple for a one-hot accuracy reward.

    Args:
        output_node: NodeRef of the output layer's value.
        target_key: Sample dict key for one-hot target values.
    """
    def accuracy(pred, target):
        return (jnp.argmax(pred, axis=-1)
                == jnp.argmax(target, axis=-1)).astype(jnp.float32)
    return ((output_node, target_key), accuracy)


def make_total_energy_reward(error_nodes):
    """Factory: returns an (inputs, fn) tuple for negative total energy.

    Args:
        error_nodes: Tuple of error NodeRefs (e.g. (pc1.error, pc2.error)).
    """
    def neg_energy(*errs):
        total = sum(jnp.sum(e ** 2, axis=-1) for e in errs)
        return -total
    return (tuple(error_nodes), neg_energy)
