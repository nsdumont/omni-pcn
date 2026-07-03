"""
Test that built network graph matches specification.
"""

import pytest
import jax.numpy as jnp


class TestNetworkBuild:
    """Test network build process."""

    def test_build(self):
        """Test build() returns self for chaining."""
        import pcn

        net = pcn.PCNetwork()
        with net:
            pcn.Layer(dim=10)
            pcn.Layer(dim=5)
        result = net.build() # or can do net.build();

        assert result is net
        assert net.structure is not None
        assert isinstance(net.structure, pcn.NetworkStructure)

        assert net.params is not None
        assert isinstance(net.params, pcn.NetworkParams)


class TestLayerStructure:
    """Test layer structure matches specification."""

    def test_layer_count(self, simple_network):
        """Test correct number of layers."""
        net, _ = simple_network
        assert len(net.structure.layers) == 3

    def test_layer_dims(self, simple_network):
        """Test layer dimensions match."""
        net, _ = simple_network
        assert net.structure.layer_dims == (16, 8, 4)

    def test_layer_specs(self, simple_network):
        """Test LayerSpec fields match."""
        net, (l1, l2, l3) = simple_network

        assert net.structure.layers[0].dim == 16
        assert net.structure.layers[0].label == "input"
        assert net.structure.layers[0].activation_type == 0  # Direct

        assert net.structure.layers[1].dim == 8
        assert net.structure.layers[1].label == "hidden"
        assert net.structure.layers[1].activation_type == 1  # Relu

        assert net.structure.layers[2].dim == 4
        assert net.structure.layers[2].label == "output"
        assert net.structure.layers[2].activation_type == 2  # Softmax

    def test_label_mappings(self, simple_network):
        """Test label to index mappings."""
        net, _ = simple_network

        assert net.label_to_idx['input'] == 0
        assert net.label_to_idx['hidden'] == 1
        assert net.label_to_idx['output'] == 2

        assert net.idx_to_label[0] == 'input'
        assert net.idx_to_label[1] == 'hidden'
        assert net.idx_to_label[2] == 'output'

    def test_node_mappings(self, simple_network):
        """Test node to index mappings."""
        net, _ = simple_network

        # Layer value nodes
        assert net.node_to_idx['input-value'] == (0, 'value')
        assert net.node_to_idx['hidden-value'] == (1, 'value')
        assert net.node_to_idx['output-value'] == (2, 'value')

        # Error/precision nodes are on predict connections, not layers
        assert 'input-error' not in net.node_to_idx
        assert 'output-precision' not in net.node_to_idx


class TestPredictConnectionStructure:
    """Test Predict connection structure."""

    def test_predict_count(self, simple_network):
        """Test correct number of Predict connections."""
        net, _ = simple_network
        assert len(net.structure.predict_conns) == 2

    def test_predict_specs(self, simple_network):
        """Test PredictConnSpec fields match."""
        net, _ = simple_network

        # First connection: hidden -> input
        conn0 = net.structure.predict_conns[0]
        assert conn0.pre_idx == (1,)   # hidden
        assert conn0.post_idx == 0  # input
        assert conn0.has_fixed_weights is False

        # Second connection: output -> hidden
        conn1 = net.structure.predict_conns[1]
        assert conn1.pre_idx == (2,)   # output
        assert conn1.post_idx == 1  # hidden

    def test_predict_weight_shapes(self, simple_network):
        """Test Predict weight matrices have correct shapes."""
        net, _ = simple_network

        # hidden (8) predicts input (16): W shape is (post_dim, pre_dim) = (16, 8)
        W0 = net.params.predict_weights[0]
        assert W0.shape == (16, 8)

        # output (4) predicts hidden (8): W shape is (8, 4)
        W1 = net.params.predict_weights[1]
        assert W1.shape == (8, 4)


class TestProjectConnectionStructure:
    """Test Project connection structure."""

    def test_project_count(self, complex_network):
        """Test correct number of Project connections."""
        net, _ = complex_network
        assert len(net.structure.project_conns) == 1

    def test_project_specs(self, complex_network):
        """Test ProjectConnSpec fields match."""
        net, (l1, l2, l3, l4) = complex_network

        conn = net.structure.project_conns[0]
        assert conn.pre_idx == (3,)      # output
        assert conn.pre_node_type == 0  # value
        assert conn.post_idx == 1     # hidden1
        assert conn.post_node_type == 0  # value
        assert conn.learning_rule_type == 0  # Hebbian
        assert conn.learning_rate == 1e-4

    def test_project_weight_shapes(self, complex_network):
        """Test Project weight matrices have correct shapes."""
        net, _ = complex_network

        # output (4) -> hidden1 (16): W shape is (16, 4)
        W = net.params.project_weights[0]
        assert W.shape == (16, 4)


class TestModulateConnectionStructure:
    """Test Modulate connection structure."""

    def test_modulate_count(self, complex_network):
        """Test correct number of Modulate connections."""
        net, _ = complex_network
        assert len(net.structure.modulate_conns) == 1

    def test_modulate_specs(self, complex_network):
        """Test ModulateConnSpec fields match."""
        net, _ = complex_network

        conn = net.structure.modulate_conns[0]
        assert conn.pre_idx == (2,)      # hidden2
        assert conn.pre_node_type == 0  # value
        assert conn.post_idx == 1     # hidden1
        assert conn.post_node_type == 1  # error
        assert conn.learning_rule_type == 0  # Hebbian

    def test_modulate_weight_shapes(self, complex_network):
        """Test Modulate weight matrices have correct shapes."""
        net, _ = complex_network

        # hidden2 (8) -> hidden1 (16): W shape is (16, 8)
        W = net.params.modulate_weights[0]
        assert W.shape == (16, 8)

    def test_modulate_weight_initialization(self, complex_network):
        """Test Modulate weights initialized near 0 (with use_bias=True the bias
        anchors identity-modulation at 1, so W starts near zero)."""
        net, _ = complex_network

        W = net.params.modulate_weights[0]
        b = net.params.modulate_biases[0]
        # Default use_bias=True (from network config): W near 0, bias near 1
        assert jnp.abs(W.mean()) < 0.1
        assert jnp.abs(b.mean() - 1.0) < 1e-6


class TestPrecisionInitialization:
    """Test precision parameter initialization."""

    def test_precision_param_count(self, simple_network):
        """Test one precision_weights/bias per predict connection."""
        net, _ = simple_network
        assert len(net.params.precision_biases) == 2   # 2 predict connections
        assert len(net.params.precision_weights) == 2  # 2 predict connections

    def test_precision_bias_values(self, simple_network):
        """Precision biases are initialized so that the starting precision
        equals ``init_precision`` (default 1.0) under the chosen
        parameterization. Default is softplus, so bias = log(e - 1)."""
        import numpy as np
        net, _ = simple_network
        expected = float(np.log(np.expm1(1.0)))  # inverse softplus of 1.0
        for pb in net.params.precision_biases:
            assert jnp.allclose(pb, jnp.full_like(pb, expected))



class TestFixedWeights:
    """Test fixed weight handling."""

    def test_fixed_weight_stored(self, fixed_weight_network):
        """Test fixed weight is stored correctly."""
        net, fixed_W = fixed_weight_network

        W = net.params.predict_weights[0]
        assert jnp.allclose(W, fixed_W)

    def test_fixed_weight_flag(self, fixed_weight_network):
        """Test has_fixed_weights flag is True."""
        net, _ = fixed_weight_network

        assert net.structure.predict_conns[0].has_fixed_weights is True
        assert net.structure.predict_conns[1].has_fixed_weights is False


class TestWeightInitialization:
    """Test weight initialization (Xavier)."""

    def test_xavier_initialization(self):
        """Test weights are Xavier initialized."""
        import pcn

        net = pcn.PCNetwork(seed=0)
        with net:
            l1 = pcn.Layer(dim=100, activation=pcn.Tanh())
            l2 = pcn.Layer(dim=100, activation=pcn.Tanh())
            pcn.Predict(l2, l1)
        net.build()

        W = net.params.predict_weights[0]
        # Xavier stddev = sqrt(2 / (fan_in + fan_out)) = sqrt(2/200) ≈ 0.1
        expected_std = jnp.sqrt(1.0 / 200)

        # Check std is approximately correct (within 50% tolerance)
        actual_std = W.std()
        assert abs(actual_std - expected_std) / expected_std < 0.5

    def test_different_seeds_different_weights(self):
        """Test different seeds produce different weights."""
        import pcn

        def make_net(seed):
            net = pcn.PCNetwork(seed=seed)
            with net:
                l1 = pcn.Layer(dim=10)
                l2 = pcn.Layer(dim=5)
                pcn.Predict(l2, l1)
            net.build()
            return net

        net1 = make_net(0)
        net2 = make_net(42)

        W1 = net1.params.predict_weights[0]
        W2 = net2.params.predict_weights[0]

        assert not jnp.allclose(W1, W2)


class TestComplexNetworkStructure:
    """Test complex network with all connection types."""

    def test_all_connection_types(self, complex_network):
        """Test network has all connection types."""
        net, _ = complex_network

        assert len(net.structure.predict_conns) == 3
        assert len(net.structure.project_conns) == 1
        assert len(net.structure.modulate_conns) == 1

    def test_all_weight_lists(self, complex_network):
        """Test all weight lists have correct lengths."""
        net, _ = complex_network

        assert len(net.params.predict_weights) == 3
        assert len(net.params.project_weights) == 1
        assert len(net.params.modulate_weights) == 1
        assert len(net.params.precision_biases) == 3   # 3 predict connections
        assert len(net.params.precision_weights) == 3  # 3 predict connections

    def test_complex_layer_dims(self, complex_network):
        """Test complex network layer dimensions."""
        net, _ = complex_network
        assert net.structure.layer_dims == (32, 16, 8, 4)

    def test_complex_labels(self, complex_network):
        """Test complex network labels."""
        net, _ = complex_network
        assert net['input'] == 0
        assert net['hidden1'] == 1
        assert net['hidden2'] == 2
        assert net['output'] == 3

if __name__ == "__main__":
    tester = TestNetworkBuild()
    tester.test_build()
