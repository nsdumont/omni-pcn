"""
Pytest fixtures for PCN tests.
"""

import pytest
import jax
import jax.numpy as jnp


@pytest.fixture
def rng_key():
    """Provide a JAX random key."""
    return jax.random.PRNGKey(42)


@pytest.fixture
def simple_network():
    """Create a simple 3-layer network for testing."""
    import pcn

    net = pcn.PCNetwork(seed=0)
    with net:
        l1 = pcn.Layer(dim=16, activation=pcn.Direct(),  label="input")
        l2 = pcn.Layer(dim=8, activation=pcn.Relu(), label="hidden")
        l3 = pcn.Layer(dim=4, activation=pcn.Softmax(), label="output")

        pcn.Predict(l2, l1)
        pcn.Predict(l3, l2)

    net.build()
    return net, (l1, l2, l3)


@pytest.fixture
def complex_network():
    """Create a network with all connection types for testing."""
    import pcn

    net = pcn.PCNetwork(seed=42)
    with net:
        l1 = pcn.Layer(dim=32, activation=pcn.Direct(), label="input")
        l2 = pcn.Layer(dim=16, activation=pcn.Relu(), label="hidden1")
        l3 = pcn.Layer(dim=8, activation=pcn.Tanh(), label="hidden2")
        l4 = pcn.Layer(dim=4, activation=pcn.Softmax(), label="output")

        # Standard PC connections
        pcn.Predict(l2, l1)
        p2 = pcn.Predict(l3, l2)
        pcn.Predict(l4, l3)

        # Lateral shortcut (Project)
        pcn.Project(
            l4.value, l2.value,
            update_rule=pcn.Hebbian(learning_rate=1e-4)
        )

        # Modulate connection (uses predict connection's error)
        pcn.Modulate(
            l3.value, p2.error,
            update_rule=pcn.Hebbian(learning_rate=1e-4)
        )

    net.build()
    return net, (l1, l2, l3, l4)


@pytest.fixture
def simple_dataloader():
    """Create a simple dataloader yielding batched data.

    Returns a factory function that creates fresh iterables each time.
    The returned dataloader can be used multiple times (for multiple epochs).
    """
    def dataloader_factory(n_batches=5, batch_size=8, input_dim=16, output_dim=4):
        """Returns a list of batches that can be iterated multiple times."""
        key = jax.random.PRNGKey(123)
        batches = []
        for i in range(n_batches):
            key, k1, k2 = jax.random.split(key, 3)
            batches.append({
                'input': jax.random.normal(k1, (batch_size, input_dim)),
                'output': jax.nn.softmax(jax.random.normal(k2, (batch_size, output_dim)), axis=-1)
            })
        return batches
    return dataloader_factory


@pytest.fixture
def fixed_weight_network():
    """Create a network with fixed weights for testing."""
    import pcn

    # Weight shape is (post_dim, pre_dim) = (16, 8) for hidden->input prediction
    fixed_W = jnp.eye(16, 8) * 0.5  # Fixed weight matrix

    net = pcn.PCNetwork(seed=0)
    with net:
        l1 = pcn.Layer(dim=16, label="input")
        l2 = pcn.Layer(dim=8, activation=pcn.Relu(), label="hidden")
        l3 = pcn.Layer(dim=4, label="output")

        pcn.Predict(l2, l1, init_weight=fixed_W)
        pcn.Predict(l3, l2)

    net.build()
    return net, fixed_W
