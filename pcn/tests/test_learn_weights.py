"""Tests for ``Predict(learn_weights=...)`` — decoupling weight init from freezing.

Legacy behaviour: a Predict with ``init_weight`` is frozen; without it, learned.
New: ``learn_weights`` overrides this — ``True`` learns from the init, ``False``
freezes even a random init. Enables seeding a readout with e.g. ``Memory.C(0)``
and refining it.
"""

import numpy as np
import optax
import pytest

import pcn
from pcn import PCNetwork, Layer, Predict, Simulation


def _train_identity(init_weight=None, learn_weights=None, epochs=120, seed=1):
    """Fit b = a (identity) with a single linear Predict; return (W0, W_learned)."""
    net = PCNetwork(seed=seed)
    kw = {}
    if init_weight is not None:
        kw['init_weight'] = init_weight
    if learn_weights is not None:
        kw['learn_weights'] = learn_weights
    with net:
        a = Layer(4, activation=pcn.Direct(), label='a')
        b = Layer(4, activation=pcn.Direct(), label='b')
        Predict(a, b, use_bias=False, learn_precision=False, label='w', **kw)
    net.build()
    sim = Simulation(net)
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((512, 4)).astype(np.float32)
    tb = [{'a': X[i:i + 64], 'b': X[i:i + 64]} for i in range(0, 512, 64)]
    sim.train(tb, data_map={a: 'a', b: 'b'}, epochs=epochs, iterations_per_sample=0,
              learning_iterations_per_sample=4, feedforward_init=True,
              params_optimizer=optax.adam(3e-2), values_optimizer=optax.sgd(0.5),
              verbose=False)
    idx = [c.label for c in net._predict_conns].index('w')
    return np.array(net.params.predict_weights[idx])


def test_init_weight_frozen_by_default():
    """Legacy: init_weight alone freezes the weights (unchanged after training)."""
    W0 = (0.2 * np.eye(4)).astype(np.float32)
    W = _train_identity(init_weight=W0)
    np.testing.assert_allclose(W, W0, atol=1e-6)


def test_init_weight_learns_when_enabled():
    """learn_weights=True initializes from init_weight AND keeps learning it."""
    W0 = (0.2 * np.eye(4)).astype(np.float32)
    W = _train_identity(init_weight=W0, learn_weights=True)
    # started at 0.2*I, target is I -> diagonal should climb toward 1.
    assert np.mean(np.diag(W)) > 0.9
    np.testing.assert_allclose(W, np.eye(4), atol=0.15)


def test_no_init_still_learns_by_default():
    """Default (no init, learn_weights=None) learns as before."""
    W = _train_identity()
    np.testing.assert_allclose(W, np.eye(4), atol=0.15)


def test_learn_weights_false_freezes_random_init():
    """learn_weights=False freezes even a random (un-provided) init."""
    net = PCNetwork(seed=3)
    with net:
        a = Layer(4, activation=pcn.Direct(), label='a')
        b = Layer(4, activation=pcn.Direct(), label='b')
        Predict(a, b, use_bias=False, learn_precision=False,
                learn_weights=False, label='w')
    net.build()
    idx = [c.label for c in net._predict_conns].index('w')
    W_before = np.array(net.params.predict_weights[idx]).copy()
    sim = Simulation(net)
    rng = np.random.default_rng(3)
    X = rng.standard_normal((256, 4)).astype(np.float32)
    tb = [{'a': X[i:i + 64], 'b': X[i:i + 64]} for i in range(0, 256, 64)]
    sim.train(tb, data_map={a: 'a', b: 'b'}, epochs=40, iterations_per_sample=0,
              learning_iterations_per_sample=4, feedforward_init=True,
              params_optimizer=optax.adam(3e-2), values_optimizer=optax.sgd(0.5),
              verbose=False)
    W_after = np.array(net.params.predict_weights[idx])
    np.testing.assert_allclose(W_after, W_before, atol=1e-6)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
