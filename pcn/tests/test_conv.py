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
