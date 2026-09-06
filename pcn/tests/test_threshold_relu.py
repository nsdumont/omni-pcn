"""
Tests for ThresholdRelu — per-neuron subtractive-threshold ReLU.

f(x)_i = max(x_i - theta_i, 0). Thresholds are carried on the LayerSpec
(``activation_thresholds``) and baked into the backend's per-layer activation
closure, mirroring Softmax temperature / NWTA num_winners.
"""

import pytest
import jax
import jax.numpy as jnp

import pcn
from pcn.core.activations import (
    ThresholdRelu, Relu, ACTIVATIONS, activation_from_name,
)


class TestThresholdReluClass:
    def test_apply_per_neuron(self):
        act = ThresholdRelu((0.5, -0.5, 0.0))
        x = jnp.array([[1.0, -1.0, -0.2], [0.4, -0.4, 0.2]])
        expected = jnp.maximum(x - jnp.array([0.5, -0.5, 0.0]), 0)
        assert jnp.allclose(act.apply(x), expected)

    def test_apply_scalar_broadcast(self):
        act = ThresholdRelu(0.3)
        assert act.thresholds == (0.3,)
        x = jnp.array([[1.0, 0.1, -2.0]])
        assert jnp.allclose(act.apply(x), jnp.maximum(x - 0.3, 0))

    def test_zero_threshold_is_relu(self):
        act = ThresholdRelu()
        x = jnp.array([[1.5, -0.5, 0.0]])
        assert jnp.allclose(act.apply(x), Relu.fn(x))

    def test_hashable_and_eq(self):
        a = ThresholdRelu((0.1, 0.2))
        b = ThresholdRelu((0.1, 0.2))
        c = ThresholdRelu((0.1, 0.3))
        assert hash(a) == hash(b) and a == b
        assert a != c

    def test_registry_and_static_fallback(self):
        assert isinstance(activation_from_name('threshold_relu'), ThresholdRelu)
        # Static fn (non-layer slots) is plain ReLU.
        x = jnp.array([-1.0, 2.0])
        assert jnp.allclose(ACTIVATIONS[ThresholdRelu.type_id](x),
                            jnp.maximum(x, 0))


class TestThresholdReluSpec:
    def test_thresholds_on_layerspec(self):
        net = pcn.PCNetwork(seed=0)
        with net:
            l1 = pcn.Layer(dim=4, activation=pcn.Direct(), label="a")
            l2 = pcn.Layer(dim=3, activation=pcn.ThresholdRelu((0.1, 0.2, 0.3)),
                           label="b")
            pcn.Predict(l2, l1)
        net.build()
        assert net.structure.layers[0].activation_thresholds == ()
        assert net.structure.layers[1].activation_thresholds == (0.1, 0.2, 0.3)

    def test_bad_threshold_length_raises(self):
        net = pcn.PCNetwork(seed=0)
        with net:
            l1 = pcn.Layer(dim=4, activation=pcn.Direct(), label="a")
            l2 = pcn.Layer(dim=8, activation=pcn.ThresholdRelu((0.1, 0.2, 0.3)),
                           label="b")
            pcn.Predict(l2, l1)
        with pytest.raises(ValueError, match="thresholds length"):
            net.build()


def _build_net(activation, W):
    net = pcn.PCNetwork(seed=0)
    net.config(use_bias=False, learn_precision_weights=False,
               learn_precision_bias=False)
    with net:
        l1 = pcn.Layer(dim=16, activation=pcn.Direct(), label="post")
        l2 = pcn.Layer(dim=8, activation=activation, label="pre")
        pcn.Predict(l2, l1, init_weight=W)
    net.build()
    return net


class TestThresholdReluBackend:
    def test_shift_invariance_vs_relu(self, rng_key):
        """A ThresholdRelu(theta) net clamped with x equals a Relu net clamped
        with x - theta: identical predictions, hence identical errors."""
        k1, k2, k3 = jax.random.split(rng_key, 3)
        W = jax.random.normal(k1, (16, 8)) * 0.3
        theta = jnp.linspace(-0.5, 0.5, 8)
        x_post = jax.random.normal(k2, (4, 16))
        x_pre = jax.random.normal(k3, (4, 8))

        from pcn.backend.simulation import run_batch

        net_thr = _build_net(pcn.ThresholdRelu(tuple(float(t) for t in theta)),
                             jnp.array(W))
        net_relu = _build_net(pcn.Relu(), jnp.array(W))

        data_map = ((0, 'post'), (1, 'pre'))
        sample_thr = {'post': x_post, 'pre': x_pre}
        sample_relu = {'post': x_post, 'pre': x_pre - theta}

        _, _, _, _, err_thr, _, _, _ = run_batch(
            sample_thr, net_thr.params, net_thr.structure,
            data_map, 5, 1, learning=False)
        _, _, _, _, err_relu, _, _, _ = run_batch(
            sample_relu, net_relu.params, net_relu.structure,
            data_map, 5, 1, learning=False)

        assert jnp.allclose(err_thr[0], err_relu[0], atol=1e-6)

    def test_train_smoke(self, rng_key):
        """Training a small ThresholdRelu net runs and updates weights."""
        import optax

        k1, k2 = jax.random.split(rng_key)
        net = pcn.PCNetwork(seed=1)
        net.config(use_bias=True, learn_precision_weights=False,
                   learn_precision_bias=False)
        with net:
            l_in = pcn.Layer(dim=6, activation=pcn.ThresholdRelu(
                (0.0, 0.1, 0.2, 0.3, 0.4, 0.5)), label="in")
            l_out = pcn.Layer(dim=3, activation=pcn.Direct(), label="out")
            pcn.Predict(l_in, l_out)
        net.build()

        w_before = jnp.array(net.params.predict_weights[0])
        batches = [{'in': jax.random.normal(k1, (8, 6)),
                    'out': jax.random.normal(k2, (8, 3))}]
        sim = pcn.Simulation(net)
        sim.train(batches, data_map={l_in: 'in', l_out: 'out'}, epochs=2,
                  iterations_per_sample=0, learning_iterations_per_sample=4,
                  verbose=False, params_optimizer=optax.adam(1e-2),
                  values_optimizer=optax.sgd(0.1))
        w_after = net.params.predict_weights[0]
        assert w_after.shape == w_before.shape
        assert jnp.all(jnp.isfinite(w_after))
        assert not jnp.allclose(w_after, w_before)
