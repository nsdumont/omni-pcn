"""
Tests for the Skip connection class.
"""

import pytest
import jax.numpy as jnp
import pcn


class TestSkipCreation:
    """Test Skip class construction and validation."""

    def test_basic_skip(self):
        """Skip with delay=2 creates 2 auxiliary layers and 3 Project conns."""
        net = pcn.PCNetwork(seed=0)
        with net:
            l1 = pcn.Layer(dim=16, activation=pcn.Direct(), label="pre")
            l2 = pcn.Layer(dim=8, activation=pcn.Relu(), label="hidden")
            l3 = pcn.Layer(dim=16, activation=pcn.Relu(), label="post")
            pcn.Predict(l2, l1)
            pcn.Predict(l3, l2)
            skip = pcn.Skip(l1, l3, delay=2, skip_scale=0.1)

        assert len(skip.auxiliary_layers) == 2
        assert len(skip.project_conns) == 3

    def test_delay_zero(self):
        """Skip with delay=0 creates no auxiliary layers, one Project conn."""
        net = pcn.PCNetwork(seed=0)
        with net:
            l1 = pcn.Layer(dim=8, activation=pcn.Direct(), label="pre")
            l2 = pcn.Layer(dim=8, activation=pcn.Relu(), label="post")
            pcn.Predict(l2, l1)
            skip = pcn.Skip(l1, l2, delay=0, skip_scale=0.5)

        assert len(skip.auxiliary_layers) == 0
        assert len(skip.project_conns) == 1

    def test_delay_one(self):
        """Skip with delay=1 creates 1 auxiliary layer and 2 Project conns."""
        net = pcn.PCNetwork(seed=0)
        with net:
            l1 = pcn.Layer(dim=10, activation=pcn.Direct(), label="pre")
            l2 = pcn.Layer(dim=10, activation=pcn.Relu(), label="post")
            pcn.Predict(l2, l1)
            skip = pcn.Skip(l1, l2, delay=1, skip_scale=1.0)

        assert len(skip.auxiliary_layers) == 1
        assert len(skip.project_conns) == 2

    def test_dim_mismatch_raises(self):
        """Skip raises ValueError when pre.dim != post.dim."""
        net = pcn.PCNetwork(seed=0)
        with net:
            l1 = pcn.Layer(dim=16, label="pre")
            l2 = pcn.Layer(dim=8, label="post")
            with pytest.raises(ValueError, match="pre.dim == post.dim"):
                pcn.Skip(l1, l2, delay=1, skip_scale=0.1)

    def test_outside_context_raises(self):
        """Skip outside a with-net block raises RuntimeError."""
        net = pcn.PCNetwork(seed=0)
        with net:
            l1 = pcn.Layer(dim=8, label="pre")
            l2 = pcn.Layer(dim=8, label="post")
        # Context exited
        with pytest.raises(RuntimeError, match="No PCNetwork context"):
            pcn.Skip(l1, l2, delay=1)


class TestSkipRegistration:
    """Test that Skip components are properly registered with the network."""

    def test_layers_registered(self):
        """Auxiliary layers are added to the network's layer list."""
        net = pcn.PCNetwork(seed=0)
        with net:
            l1 = pcn.Layer(dim=8, activation=pcn.Direct(), label="pre")
            l2 = pcn.Layer(dim=8, activation=pcn.Relu(), label="post")
            pcn.Predict(l2, l1)
            skip = pcn.Skip(l1, l2, delay=2, skip_scale=0.1)

        # 2 original layers + 2 auxiliary = 4
        assert len(net._layers) == 4
        for aux in skip.auxiliary_layers:
            assert aux in net._layers

    def test_projects_registered(self):
        """Project connections are added to the network's project list."""
        net = pcn.PCNetwork(seed=0)
        with net:
            l1 = pcn.Layer(dim=8, activation=pcn.Direct(), label="pre")
            l2 = pcn.Layer(dim=8, activation=pcn.Relu(), label="post")
            pcn.Predict(l2, l1)
            skip = pcn.Skip(l1, l2, delay=2, skip_scale=0.1)

        assert len(net._project_conns) == 3
        for proj in skip.project_conns:
            assert proj in net._project_conns

    def test_auxiliary_layers_use_direct_activation(self):
        """All auxiliary layers use Direct (identity) activation."""
        net = pcn.PCNetwork(seed=0)
        with net:
            l1 = pcn.Layer(dim=8, activation=pcn.Direct(), label="pre")
            l2 = pcn.Layer(dim=8, activation=pcn.Relu(), label="post")
            pcn.Predict(l2, l1)
            skip = pcn.Skip(l1, l2, delay=3, skip_scale=0.1)

        for aux in skip.auxiliary_layers:
            assert isinstance(aux.activation, pcn.Direct)

    def test_auxiliary_layers_have_correct_dim(self):
        """All auxiliary layers have the same dim as pre and post."""
        dim = 12
        net = pcn.PCNetwork(seed=0)
        with net:
            l1 = pcn.Layer(dim=dim, activation=pcn.Direct(), label="pre")
            l2 = pcn.Layer(dim=dim, activation=pcn.Relu(), label="post")
            pcn.Predict(l2, l1)
            skip = pcn.Skip(l1, l2, delay=2, skip_scale=0.1)

        for aux in skip.auxiliary_layers:
            assert aux.dim == dim

    def test_project_conns_use_no_learning(self):
        """All Project connections use NoLearning rule."""
        net = pcn.PCNetwork(seed=0)
        with net:
            l1 = pcn.Layer(dim=8, activation=pcn.Direct(), label="pre")
            l2 = pcn.Layer(dim=8, activation=pcn.Relu(), label="post")
            pcn.Predict(l2, l1)
            skip = pcn.Skip(l1, l2, delay=2, skip_scale=0.5)

        for proj in skip.project_conns:
            assert isinstance(proj.update_rule, pcn.NoLearning)


class TestSkipBuild:
    """Test that networks with Skip connections build and initialize correctly."""

    def test_network_builds(self):
        """Network with a Skip connection builds without error."""
        net = pcn.PCNetwork(seed=0)
        with net:
            l1 = pcn.Layer(dim=16, activation=pcn.Direct(), label="input")
            l2 = pcn.Layer(dim=8, activation=pcn.Relu(), label="hidden")
            l3 = pcn.Layer(dim=16, activation=pcn.Relu(), label="output")
            pcn.Predict(l2, l1)
            pcn.Predict(l3, l2)
            pcn.Skip(l1, l3, delay=2, skip_scale=0.1)
        net.build()

        assert net.structure is not None
        assert net.params is not None

    def test_project_weights_are_scaled_identity(self):
        """Built project weights equal skip_scale * I."""
        dim = 8
        scale = 0.25
        net = pcn.PCNetwork(seed=0)
        with net:
            l1 = pcn.Layer(dim=dim, activation=pcn.Direct(), label="pre")
            l2 = pcn.Layer(dim=dim, activation=pcn.Relu(), label="post")
            pcn.Predict(l2, l1)
            pcn.Skip(l1, l2, delay=1, skip_scale=scale)
        net.build()

        expected = scale * jnp.eye(dim)
        for W in net.params.project_weights:
            assert jnp.allclose(W, expected), f"Expected {expected}, got {W}"

    def test_structure_layer_count(self):
        """Structure has correct number of layers (original + auxiliary)."""
        net = pcn.PCNetwork(seed=0)
        with net:
            l1 = pcn.Layer(dim=8, activation=pcn.Direct(), label="input")
            l2 = pcn.Layer(dim=4, activation=pcn.Relu(), label="hidden")
            l3 = pcn.Layer(dim=8, activation=pcn.Relu(), label="output")
            pcn.Predict(l2, l1)
            pcn.Predict(l3, l2)
            pcn.Skip(l1, l3, delay=3, skip_scale=0.1)
        net.build()

        # 3 original + 3 auxiliary = 6
        assert len(net.structure.layers) == 6

    def test_structure_project_count(self):
        """Structure has correct number of project connections."""
        net = pcn.PCNetwork(seed=0)
        with net:
            l1 = pcn.Layer(dim=8, activation=pcn.Direct(), label="input")
            l2 = pcn.Layer(dim=4, activation=pcn.Relu(), label="hidden")
            l3 = pcn.Layer(dim=8, activation=pcn.Relu(), label="output")
            pcn.Predict(l2, l1)
            pcn.Predict(l3, l2)
            pcn.Skip(l1, l3, delay=3, skip_scale=0.1)
        net.build()

        # delay=3 -> 4 project connections
        assert len(net.structure.project_conns) == 4


class TestSkipEquivalence:
    """Test that Skip produces exactly the same network as manual construction."""

    def test_equivalent_to_manual(self):
        """Skip produces the same structure as manually creating layers + projects."""
        dim = 8
        delay = 2
        scale = 0.1

        # Manual construction
        net_manual = pcn.PCNetwork(seed=0)
        with net_manual:
            l1m = pcn.Layer(dim=dim, activation=pcn.Direct(), label="input")
            l2m = pcn.Layer(dim=4, activation=pcn.Relu(), label="hidden")
            l3m = pcn.Layer(dim=dim, activation=pcn.Relu(), label="output")
            pcn.Predict(l2m, l1m)
            pcn.Predict(l3m, l2m)

            l_aux = [l1m]
            for i in range(delay):
                l_aux.append(pcn.Layer(dim=dim, activation=pcn.Direct()))
            l_aux.append(l3m)
            for i in range(delay + 1):
                pcn.Project(
                    l_aux[i].value, l_aux[i + 1].value,
                    update_rule=pcn.NoLearning(),
                    init_weight=scale * jnp.eye(l_aux[i].dim),
                )
        net_manual.build()

        # Skip construction
        net_skip = pcn.PCNetwork(seed=0)
        with net_skip:
            l1s = pcn.Layer(dim=dim, activation=pcn.Direct(), label="input")
            l2s = pcn.Layer(dim=4, activation=pcn.Relu(), label="hidden")
            l3s = pcn.Layer(dim=dim, activation=pcn.Relu(), label="output")
            pcn.Predict(l2s, l1s)
            pcn.Predict(l3s, l2s)
            pcn.Skip(l1s, l3s, delay=delay, skip_scale=scale)
        net_skip.build()

        # Same number of layers and connections
        assert len(net_skip.structure.layers) == len(net_manual.structure.layers)
        assert len(net_skip.structure.project_conns) == len(net_manual.structure.project_conns)

        # Same layer dims
        assert net_skip.structure.layer_dims == net_manual.structure.layer_dims

        # Same project weights
        for W_skip, W_manual in zip(net_skip.params.project_weights,
                                     net_manual.params.project_weights):
            assert jnp.allclose(W_skip, W_manual)
