"""Tests for PredictConv convolutional connections."""

import pytest
import jax
import jax.numpy as jnp
import numpy as np

import pcn
from pcn.backend.simulation import run_batch


class TestPredictConvCreation:
    """Test PredictConv construction and weight initialization."""

    def test_predict_conv_creation(self):
        """Build network with PredictConv, verify weight shape (C_out, C_in, kH, kW)."""
        net = pcn.PCNetwork(seed=42)
        with net:
            l1 = pcn.Layer(dim=1 * 8 * 8, label="input")
            l2 = pcn.Layer(dim=4 * 6 * 6, label="hidden")
            pcn.PredictConv(l1, l2, kernel_size=3, input_shape=(8, 8))
        net.build()

        W = net.params.predict_weights[0]
        assert W.shape == (4, 1, 3, 3)

    def test_predict_conv_dim_mismatch(self):
        """Wrong pre.dim raises ValueError."""
        net = pcn.PCNetwork(seed=42)
        with net:
            l1 = pcn.Layer(dim=100, label="input")
            l2 = pcn.Layer(dim=4 * 6 * 6, label="hidden")
            with pytest.raises(ValueError, match="pre.dim"):
                pcn.PredictConv(l1, l2, kernel_size=3, input_shape=(8, 8))

    def test_predict_conv_post_dim_mismatch(self):
        """Wrong post.dim raises ValueError."""
        net = pcn.PCNetwork(seed=42)
        with net:
            l1 = pcn.Layer(dim=1 * 8 * 8, label="input")
            l2 = pcn.Layer(dim=100, label="hidden")
            with pytest.raises(ValueError, match="post.dim"):
                pcn.PredictConv(l1, l2, kernel_size=3, input_shape=(8, 8))


class TestPredictConvForward:
    """Test forward pass with convolutional connections."""

    def test_predict_conv_forward_shape(self):
        """Run inference, verify output flat dim is correct."""
        net = pcn.PCNetwork(seed=42)
        with net:
            l1 = pcn.Layer(dim=1 * 8 * 8, label="input")
            l2 = pcn.Layer(dim=4 * 6 * 6, label="output")
            pcn.PredictConv(l1, l2, kernel_size=3, input_shape=(8, 8))
        net.build()

        batch_size = 4
        sample = {'image': jnp.ones((batch_size, 1 * 8 * 8))}
        data_map = ((0, 'image'),)

        _, _, _, values_log, errors_log, _, _, energies = run_batch(
            sample, net.params,
            net.structure,
            data_map, 1, 1,
            learning=False, n_learning_iterations=0,
        )

        assert values_log[1][-1].shape == (batch_size, 4 * 6 * 6)

    def test_predict_conv_same_padding(self):
        """SAME padding preserves spatial dims."""
        net = pcn.PCNetwork(seed=42)
        with net:
            l1 = pcn.Layer(dim=1 * 8 * 8, label="input")
            l2 = pcn.Layer(dim=2 * 8 * 8, label="output")
            pcn.PredictConv(l1, l2, kernel_size=3, input_shape=(8, 8),
                            padding='SAME')
        net.build()

        W = net.params.predict_weights[0]
        assert W.shape == (2, 1, 3, 3)

        batch_size = 2
        sample = {'image': jnp.ones((batch_size, 1 * 8 * 8))}
        data_map = ((0, 'image'),)

        _, _, _, values_log, _, _, _, _ = run_batch(
            sample, net.params,
            net.structure,
            data_map, 1, 1,
            learning=False, n_learning_iterations=0,
        )

        assert values_log[1][-1].shape == (batch_size, 2 * 8 * 8)

    def test_predict_conv_int_padding(self):
        """Int padding=1 with kernel=3 preserves spatial dims (like SAME with stride=1)."""
        net = pcn.PCNetwork(seed=42)
        with net:
            l1 = pcn.Layer(dim=1 * 8 * 8, label="input")
            # padding=1, kernel=3, stride=1: H_out = (8+2-3)//1+1 = 8
            l2 = pcn.Layer(dim=2 * 8 * 8, label="output")
            pcn.PredictConv(l1, l2, kernel_size=3, input_shape=(8, 8),
                            padding=1)
        net.build()

        W = net.params.predict_weights[0]
        assert W.shape == (2, 1, 3, 3)


class TestMixedNetwork:
    """Test networks combining dense Predict and PredictConv."""

    def test_mixed_dense_conv_network(self):
        """Network with both Predict and PredictConv builds and trains."""
        net = pcn.PCNetwork(seed=42)
        with net:
            l1 = pcn.Layer(dim=1 * 8 * 8, label="input")
            l2 = pcn.Layer(dim=4 * 6 * 6, label="conv_out")
            l3 = pcn.Layer(dim=10, label="dense_out")

            pcn.PredictConv(l1, l2, kernel_size=3, input_shape=(8, 8))
            pcn.Predict(l2, l3)
        net.build()

        # Verify weight shapes
        assert net.params.predict_weights[0].shape == (4, 1, 3, 3)  # conv
        assert net.params.predict_weights[1].shape == (10, 4 * 6 * 6)  # dense

        batch_size = 4
        sample = {
            'image': jnp.ones((batch_size, 1 * 8 * 8)),
            'label': jnp.zeros((batch_size, 10)),
        }
        data_map = ((0, 'image'), (2, 'label'))

        new_params, _, _, values_log, _, _, _, _ = run_batch(
            sample, net.params,
            net.structure,
            data_map, 10, 10,
            learning=True,
        )

        assert values_log[1][-1].shape == (batch_size, 4 * 6 * 6)


class TestConvLearning:
    """Test that conv weights update during learning."""

    def test_conv_weights_change_during_learning(self):
        """Train a few batches, verify conv kernel changes."""
        net = pcn.PCNetwork(seed=42)
        with net:
            l1 = pcn.Layer(dim=1 * 8 * 8, label="input")
            l2 = pcn.Layer(dim=2 * 6 * 6, label="output")
            pcn.PredictConv(l1, l2, kernel_size=3, input_shape=(8, 8))
        net.build()

        W_before = net.params.predict_weights[0].copy()

        batch_size = 4
        rng = np.random.default_rng(0)
        sample = {
            'image': jnp.array(rng.standard_normal((batch_size, 1 * 8 * 8)), dtype=jnp.float32),
            'target': jnp.array(rng.standard_normal((batch_size, 2 * 6 * 6)), dtype=jnp.float32),
        }
        data_map = ((0, 'image'), (1, 'target'))

        new_params, _, _, _, _, _, _, _ = run_batch(
            sample, net.params,
            net.structure,
            data_map, 20, 20,
            learning=True,
        )

        W_after = new_params.predict_weights[0]
        assert not jnp.allclose(W_before, W_after), "Conv weights should change during training"


class TestPredictConvPool:
    """Test fused conv + spatial pool connections (conv-maxpool / conv-avgpool)."""

    def _manual(self, x, W, pool, b=None, k_pad=1, pw=2):
        xr = x.reshape(x.shape[0], W.shape[1], 8, 8)
        y = jax.lax.conv_general_dilated(
            xr, W, window_strides=(1, 1), padding=[(k_pad, k_pad)] * 2,
            dimension_numbers=('NCHW', 'OIHW', 'NCHW'))
        if b is not None:
            y = y + b[None, :, None, None]
        if pool == 'avg':
            y = jax.lax.reduce_window(y, 0., jax.lax.add, (1, 1, pw, pw),
                                      (1, 1, pw, pw), 'VALID') / (pw * pw)
        else:
            y = jax.lax.reduce_window(y, -jnp.inf, jax.lax.max, (1, 1, pw, pw),
                                      (1, 1, pw, pw), 'VALID')
        return np.asarray(y).reshape(x.shape[0], -1)

    @pytest.mark.parametrize("pool", ["avg", "max"])
    def test_shape_and_forward(self, pool):
        """3ch 8x8 --conv3(SAME)--> 4ch 8x8 --pool2--> 4ch 4x4; forward == manual."""
        net = pcn.PCNetwork(seed=0)
        net.config(use_bias=False, learn_precision_weights=False,
                   learn_precision_bias=False)
        with net:
            l_in = pcn.Layer(dim=3 * 8 * 8, activation=pcn.Direct(), label="in")
            l_out = pcn.Layer(dim=4 * 4 * 4, activation=pcn.Direct(), label="out")
            pcn.PredictConvPool(l_in, l_out, kernel_size=3, input_shape=(8, 8),
                                pool=pool, pool_size=2, stride=1, padding=1)
        net.build()
        conn = net.structure.predict_conns[0]
        W = np.asarray(net.params.predict_weights[0])
        assert W.shape == (4, 3, 3, 3)
        assert conn.out_channels == 4 and conn.output_spatial == (4, 4)
        x = np.random.default_rng(1).standard_normal((2, 3 * 8 * 8)).astype('float32')
        got = np.asarray(conn.apply(jnp.asarray(x), jnp.asarray(W)))
        assert got.shape == (2, 4 * 4 * 4)
        np.testing.assert_allclose(got, self._manual(x, W, pool), atol=1e-5)

    def test_pool_adjoint_routing(self):
        """Energy grad wrt pre value: avg routes to all inputs, max only to argmax."""
        def build(pool):
            net = pcn.PCNetwork(seed=0)
            net.config(use_bias=False, learn_precision_weights=False,
                       learn_precision_bias=False)
            with net:
                a = pcn.Layer(dim=1 * 4 * 4, activation=pcn.Direct(), label="a")
                b = pcn.Layer(dim=1 * 2 * 2, activation=pcn.Direct(), label="b")
                pcn.PredictConvPool(a, b, kernel_size=1, input_shape=(4, 4),
                                    pool=pool, pool_size=2, stride=1, padding=0)
            net.build()
            return net.structure.predict_conns[0], jnp.asarray(net.params.predict_weights[0])

        def nz(pool):
            conn, W = build(pool)
            pre = jnp.asarray(np.random.default_rng(2).standard_normal((1, 16)).astype('float32'))
            g = jax.grad(lambda p: 0.5 * jnp.sum(conn.apply(p, W) ** 2))(pre)
            return int((np.abs(np.asarray(g)) > 1e-9).sum())

        assert nz('avg') == 16          # uniform upsample: every input contributes
        assert nz('max') == 4           # one argmax per 2x2 window

    @pytest.mark.parametrize("pool", ["avg", "max"])
    def test_weights_learn(self, pool):
        net = pcn.PCNetwork(seed=1)
        net.config(use_bias=True, learn_precision_weights=False,
                   learn_precision_bias=False)
        with net:
            l_in = pcn.Layer(dim=2 * 8 * 8, activation=pcn.LeakyRelu(), label="in")
            l_out = pcn.Layer(dim=4 * 4 * 4, activation=pcn.Direct(), label="out")
            pcn.PredictConvPool(l_in, l_out, kernel_size=3, input_shape=(8, 8),
                                pool=pool, pool_size=2, stride=1, padding=1)
        net.build()
        W0 = np.asarray(net.params.predict_weights[0]).copy()
        rng = np.random.default_rng(0)
        sample = {'image': jnp.asarray(rng.standard_normal((4, 2 * 8 * 8)), dtype=jnp.float32),
                  'target': jnp.asarray(rng.standard_normal((4, 4 * 4 * 4)), dtype=jnp.float32)}
        new_params, *_ = run_batch(sample, net.params, net.structure,
                                   ((0, 'image'), (1, 'target')), 20, 20, learning=True)
        assert not np.allclose(W0, np.asarray(new_params.predict_weights[0]))

    def test_pool_window_too_large_raises(self):
        net = pcn.PCNetwork(seed=0)
        with net:
            a = pcn.Layer(dim=1 * 2 * 2, label="a")
            b = pcn.Layer(dim=1 * 1 * 1, label="b")
            with pytest.raises(ValueError, match="pool window"):
                pcn.PredictConvPool(a, b, kernel_size=1, input_shape=(2, 2),
                                    pool='max', pool_size=3)
