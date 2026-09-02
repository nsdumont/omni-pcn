"""BatchStandardize activation: batch-axis semantics, guards, and registry."""
import numpy as np
import jax.numpy as jnp
import pytest

import pcn
from pcn.core.activations import ACTIVATIONS, activation_from_name


def test_standardizes_across_batch():
    x = jnp.asarray(np.random.default_rng(0).normal(2.0, 3.0, (16, 8)).astype(np.float32))
    y = pcn.BatchStandardize.fn(x)
    assert float(jnp.abs(y.mean(axis=0)).max()) < 1e-5
    assert float(jnp.abs(y.std(axis=0) - 1).max()) < 5e-3  # eps=1e-3 bounds the gain


def test_identity_guards():
    x = jnp.asarray(np.random.default_rng(1).normal(size=(1, 8)).astype(np.float32))
    assert bool(jnp.all(pcn.BatchStandardize.fn(x) == x))          # batch of 1
    assert bool(jnp.all(pcn.BatchStandardize.fn(x[0]) == x[0]))    # 1-D input


def test_constant_dim_bounded():
    x = jnp.ones((8, 4), jnp.float32)
    y = pcn.BatchStandardize.fn(x)
    assert bool(jnp.all(jnp.isfinite(y))) and float(jnp.abs(y).max()) == 0.0


def test_gradient_finite_at_zero_variance():
    """d/dx of the standardization must be finite for a constant (e.g. zero-init) batch."""
    import jax
    g = jax.grad(lambda x: jnp.sum(pcn.BatchStandardize.fn(x) ** 2))(jnp.zeros((8, 4)))
    assert bool(jnp.all(jnp.isfinite(g)))


def test_registry():
    assert ACTIVATIONS[pcn.BatchStandardize.type_id] is pcn.BatchStandardize.fn
    assert isinstance(activation_from_name('batchstd'), pcn.BatchStandardize)
    assert isinstance(activation_from_name('batch_standardize'), pcn.BatchStandardize)


def test_conn_consumes_standardized_pre():
    """A Predict conn whose pre uses BatchStandardize must see f(v) = standardized batch."""
    import optax
    net = pcn.PCNetwork()
    with net:
        a = pcn.Layer(dim=6, activation=pcn.BatchStandardize())
        b = pcn.Layer(dim=6)
        pcn.Predict(a, b, init_weight=np.eye(6, dtype=np.float32),
                    learn_weights=False, init_precision=1.0)
    net.build()
    sim = pcn.Simulation(net)
    va = np.random.default_rng(0).normal(3.0, 2.0, (8, 6)).astype(np.float32)
    res = sim.test([{'a': va, 'b': np.zeros((8, 6), np.float32)}],
                   data_map={a: 'a', b: 'b'}, iterations_per_sample=1,
                   values_optimizer=optax.sgd(0.0), return_logs=True,
                   log_every=1, verbose=False)
    e = np.asarray(res['errors'][0])
    f = -(e[:, -1] if e.ndim == 3 else e)   # e = v_b - f(v_a) = -f(v_a)
    assert np.abs(f.mean(axis=0)).max() < 1e-4
    assert np.abs(f.std(axis=0) - 1).max() < 5e-3
