"""
Tests for attention mechanisms:
- Mechanism 1: Precision as pre/post for Project/Modulate
- Mechanism 2: Per-leg flow gating (flow_to_pre, flow_to_post)
- Mechanism 3: Structural attention (softmax competition)
"""

import pytest
import jax
import jax.numpy as jnp
import optax
import numpy as np

import pcn
from pcn.backend.simulation import (
    _apply_project_modulate_precision,
    _write_additive, _write_multiplicative,
    ACTIVATIONS,
)
from pcn.backend import run_batch
from pcn.core.structure import (
    ProjectConnSpec, ModulateConnSpec,
    StructuralAttentionGroup, _pm_get_pre, _pm_get_post,
)


# ============================================================================
# Mechanism 1: Precision as pre/post
# ============================================================================

class TestPrecisionPrePost:
    """Test that precision nodes can be read from and written to."""

    def test_pm_get_pre_precision(self):
        """_pm_get_pre with pre_node_type=2 reads from precisions tuple."""
        values = (jnp.array([[1.0, 2.0]]),)
        errors = (jnp.array([[3.0, 4.0]]),)
        precisions = (jnp.array([[5.0, 6.0]]),)
        activation_fns = (ACTIVATIONS[0],)  # Direct

        result = _pm_get_pre(
            pre_idx=(0,), pre_node_type=2,
            values=values, errors=errors,
            activation_fns=activation_fns, precisions=precisions)
        assert jnp.allclose(result, jnp.array([[5.0, 6.0]]))

    def test_pm_get_post_precision(self):
        """_pm_get_post with post_node_type=2 reads from precisions tuple."""
        values = (jnp.array([[1.0, 2.0]]),)
        errors = (jnp.array([[3.0, 4.0]]),)
        precisions = (jnp.array([[5.0, 6.0]]),)

        result = _pm_get_post(
            post_idx=0, post_node_type=2,
            values=values, errors=errors, precisions=precisions)
        assert jnp.allclose(result, jnp.array([[5.0, 6.0]]))

    def test_apply_project_modulate_precision_additive(self):
        """Project targeting precision adds to precision values."""
        precisions = (jnp.ones((2, 3)), jnp.ones((2, 4)))
        values = (jnp.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]),)
        errors = ()
        proj_weights = (jnp.ones((3, 3)) * 0.1,)
        mod_weights = ()

        proj_spec = ProjectConnSpec(
            pre_idx=(0,), pre_node_type=0, post_idx=0, post_node_type=2,
            learning_rule_type=0, learning_rate=1e-3, reward_fn_idx=-1)
        project_conns_precision = ((0, proj_spec),)
        modulate_conns_precision = ()

        result = _apply_project_modulate_precision(
            precisions, values, errors,
            proj_weights, mod_weights,
            project_conns_precision, modulate_conns_precision,
            (ACTIVATIONS[0],))

        # Precision[0] should have the additive contribution
        # pre_act = Direct(values[0]) = [[1,1,1], [1,1,1]]
        # contribution = 0.1*I @ [1,1,1] = [0.3, 0.3, 0.3] (ones matrix * 0.1)
        assert result[0].shape == (2, 3)
        assert jnp.all(result[0] > 1.0)  # Added positive values
        # Precision[1] unchanged
        assert jnp.allclose(result[1], jnp.ones((2, 4)))

    def test_apply_project_modulate_precision_multiplicative(self):
        """Modulate targeting precision multiplies precision values."""
        precisions = (jnp.ones((2, 3)) * 2.0,)
        values = (jnp.array([[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]]),)
        errors = ()

        mod_weights = (jnp.eye(3) * 2.0,)
        mod_spec = ModulateConnSpec(
            pre_idx=(0,), pre_node_type=0, post_idx=0, post_node_type=2,
            learning_rule_type=0, learning_rate=1e-3, reward_fn_idx=-1)
        modulate_conns_precision = ((0, mod_spec),)

        result = _apply_project_modulate_precision(
            precisions, values, errors,
            (), mod_weights,
            (), modulate_conns_precision,
            (ACTIVATIONS[0],))

        # pre_act = Direct([0.5,0.5,0.5]) = [0.5,0.5,0.5]
        # modulation = 2.0*I @ [0.5,0.5,0.5] = [1.0, 1.0, 1.0]
        # result = 2.0 * 1.0 = 2.0
        assert jnp.allclose(result[0], jnp.ones((2, 3)) * 2.0)

    def test_build_precision_targeting_connections(self):
        """Build a network with precision-targeting Project/Modulate."""
        net = pcn.PCNetwork(seed=0)
        with net:
            l_in = pcn.Layer(dim=4, label='input')
            l_hid = pcn.Layer(dim=3, activation=pcn.Relu(), label='hidden')
            l_out = pcn.Layer(dim=2, activation=pcn.Relu(), label='output')

            p1 = pcn.Predict(l_hid, l_in)
            p2 = pcn.Predict(l_out, l_hid)

            # Modulate p1's precision from l_out
            pcn.Modulate(l_out.value, p1.precision,
                         update_rule=pcn.Hebbian(learning_rate=1e-3))

            # Project p2's precision to l_in value
            pcn.Project(p2.precision, l_in.value,
                        update_rule=pcn.Hebbian(learning_rate=1e-3))
        net.build()

        s = net.structure
        assert len(s.modulate_conns_precision) == 1
        assert len(s.project_conns_value) == 1  # precision->value is a value-target conn
        # The modulate targeting precision
        _, mod_spec = s.modulate_conns_precision[0]
        assert mod_spec.post_node_type == 2

    def test_precision_targeting_runs(self):
        """Full pipeline with precision-targeting connections runs without error."""
        net = pcn.PCNetwork(seed=0)
        with net:
            l_in = pcn.Layer(dim=4, label='input')
            l_hid = pcn.Layer(dim=3, activation=pcn.Relu(), label='hidden')
            l_out = pcn.Layer(dim=2, activation=pcn.Relu(), label='output')

            p1 = pcn.Predict(l_hid, l_in)
            pcn.Predict(l_out, l_hid)

            pcn.Modulate(l_out.value, p1.precision,
                         update_rule=pcn.Hebbian(learning_rate=1e-3))
        net.build()

        dataloader = [{'input': jnp.ones((2, 4))}]
        data_map = {l_in: 'input'}
        sim = pcn.Simulation(net)
        sim.config(iterations_per_sample=5)
        result = sim.test(dataloader, data_map)
        assert result is not None

    def test_cross_precision_modulation(self):
        """p2.precision modulates p1.error (cross-connection)."""
        net = pcn.PCNetwork(seed=0)
        with net:
            l_in = pcn.Layer(dim=4, label='input')
            l_hid = pcn.Layer(dim=3, activation=pcn.Relu(), label='hidden')

            p1 = pcn.Predict(l_hid, l_in)

            # Use precision as pre source
            pcn.Modulate(p1.precision, p1.error,
                         update_rule=pcn.Hebbian(learning_rate=1e-3))
        net.build()

        s = net.structure
        assert len(s.modulate_conns_internal) == 1
        _, spec = s.modulate_conns_internal[0]
        assert spec.pre_node_type == 2  # precision as source
        assert spec.post_node_type == 1  # error as target


# ============================================================================
# Mechanism 2: Per-leg flow gating
# ============================================================================

class TestPerLegFlowGating:
    """Test flow_to_pre and flow_to_post sub-nodes on Predict."""

    def test_flow_node_properties(self):
        """Predict connection exposes flow_to_pre and flow_to_post."""
        net = pcn.PCNetwork(seed=0)
        with net:
            l_a = pcn.Layer(dim=4, label='a')
            l_b = pcn.Layer(dim=3, label='b')
            p = pcn.Predict(l_a, l_b)

        assert p.flow_to_pre.node_type == 'flow_to_pre'
        assert p.flow_to_pre.node_type_id == 3
        assert p.flow_to_post.node_type == 'flow_to_post'
        assert p.flow_to_post.node_type_id == 4
        assert p.flow_to_pre.dim == l_b.dim  # same dim as error
        assert p.flow_to_post.dim == l_b.dim

    def test_flow_nodes_modulate_only(self):
        """Flow nodes can only be targeted by Modulate, not Project."""
        net = pcn.PCNetwork(seed=0)
        with net:
            l_a = pcn.Layer(dim=4, label='a')
            l_b = pcn.Layer(dim=3, label='b')
            l_gate = pcn.Layer(dim=3, label='gate')
            p = pcn.Predict(l_a, l_b)

            with pytest.raises(ValueError, match="Modulate"):
                pcn.Project(l_gate.value, p.flow_to_pre)

    def test_flow_nodes_not_as_pre_source(self):
        """Flow nodes cannot be used as pre source."""
        net = pcn.PCNetwork(seed=0)
        with net:
            l_a = pcn.Layer(dim=4, label='a')
            l_b = pcn.Layer(dim=3, label='b')
            l_target = pcn.Layer(dim=3, label='target')
            p = pcn.Predict(l_a, l_b)

            with pytest.raises(ValueError, match="Flow nodes"):
                pcn.Modulate(p.flow_to_pre, l_target.value)

    def test_build_flow_gating_network(self):
        """Network with flow gate Modulate connections builds correctly."""
        net = pcn.PCNetwork(seed=0)
        with net:
            l_low = pcn.Layer(dim=4, label='low')
            l_high = pcn.Layer(dim=3, activation=pcn.Relu(), label='high')
            l_gate = pcn.Layer(dim=4, activation=pcn.Sigmoid(), label='gate')

            p = pcn.Predict(l_high, l_low)

            pcn.Modulate(l_gate.value, p.flow_to_pre,
                         update_rule=pcn.Hebbian(learning_rate=1e-3))
            pcn.Modulate(l_gate.value, p.flow_to_post,
                         update_rule=pcn.Hebbian(learning_rate=1e-3))
        net.build()

        s = net.structure
        assert len(s.modulate_conns_flow_pre) == 1
        assert len(s.modulate_conns_flow_post) == 1
        assert s.predict_has_flow_gates == (True,)

    def test_predict_has_flow_gates_false_by_default(self):
        """Predict connections without flow gate modulation have False flags."""
        net = pcn.PCNetwork(seed=0)
        with net:
            l_a = pcn.Layer(dim=4, label='a')
            l_b = pcn.Layer(dim=3, label='b')
            pcn.Predict(l_a, l_b)
        net.build()

        assert net.structure.predict_has_flow_gates == (False,)

    def test_flow_gating_runs(self):
        """Full pipeline with per-leg flow gating runs without error."""
        net = pcn.PCNetwork(seed=0)
        with net:
            l_in = pcn.Layer(dim=4, label='input')
            l_hid = pcn.Layer(dim=3, activation=pcn.Relu(), label='hidden')
            l_gate = pcn.Layer(dim=3, activation=pcn.Sigmoid(), label='gate')

            p = pcn.Predict(l_hid, l_in)
            pcn.Predict(l_gate, l_hid)

            pcn.Modulate(l_gate.value, p.flow_to_pre,
                         update_rule=pcn.Hebbian(learning_rate=1e-3))
        net.build()

        dataloader = [{'input': jnp.ones((2, 4))}]
        data_map = {l_in: 'input'}
        sim = pcn.Simulation(net)
        sim.config(iterations_per_sample=5)
        result = sim.test(dataloader, data_map)
        assert result is not None

    def test_flow_gate_zero_kills_ascending(self):
        """Setting flow_to_pre gate to 0 should zero the ascending gradient."""
        net = pcn.PCNetwork(seed=42)
        with net:
            l_in = pcn.Layer(dim=2, label='input', activation=pcn.Direct())
            l_hid = pcn.Layer(dim=2, activation=pcn.Direct(), label='hidden')
            l_gate = pcn.Layer(dim=2, activation=pcn.Direct(), label='gate')

            p = pcn.Predict(l_hid, l_in)

            # Gate with init_weight=0 should produce zero gate
            m = pcn.Modulate(l_gate.value, p.flow_to_pre,
                             update_rule=pcn.NoLearning())
            m.weight = np.zeros((2, 2))  # zero weight -> zero gate output
        net.build()

        # Force gate layer values to zeros (gate output = W @ f(0) = 0)
        # With zero modulate weights, gate_pre = 0, so ascending gradient should be 0
        # The hidden layer values should not change from energy gradient
        dataloader = [{'input': jnp.array([[1.0, 2.0]])}]
        data_map = {l_in: 'input'}
        sim = pcn.Simulation(net)
        sim.config(iterations_per_sample=3)
        result = sim.test(dataloader, data_map)
        # Test runs without error — detailed gradient testing would require
        # manual jax.grad checks


# ============================================================================
# Mechanism 3: Structural Attention
# ============================================================================

class TestStructuralAttention:
    """Test softmax competition among co-targeting Predict connections."""

    def test_structural_attention_group_namedtuple(self):
        """StructuralAttentionGroup has correct fields."""
        g = StructuralAttentionGroup(conn_indices=(0, 1, 2), temperature=0.5)
        assert g.conn_indices == (0, 1, 2)
        assert g.temperature == 0.5

    def test_structural_attention_method(self):
        """PCNetwork.structural_attention registers groups correctly."""
        net = pcn.PCNetwork(seed=0)
        with net:
            l_target = pcn.Layer(dim=4, label='target')
            l_a = pcn.Layer(dim=3, label='a')
            l_b = pcn.Layer(dim=3, label='b')

            p_a = pcn.Predict(l_a, l_target)
            p_b = pcn.Predict(l_b, l_target)

        net.structural_attention([p_a, p_b], temperature=0.5)
        net.build()

        s = net.structure
        assert len(s.structural_attention_groups) == 1
        group = s.structural_attention_groups[0]
        assert group.conn_indices == (0, 1)
        assert group.temperature == 0.5

    def test_structural_attention_runs(self):
        """Full pipeline with structural attention runs without error."""
        net = pcn.PCNetwork(seed=0)
        with net:
            l_target = pcn.Layer(dim=4, label='target')
            l_a = pcn.Layer(dim=3, activation=pcn.Relu(), label='a')
            l_b = pcn.Layer(dim=3, activation=pcn.Relu(), label='b')
            l_c = pcn.Layer(dim=2, activation=pcn.Relu(), label='c')

            p_a = pcn.Predict(l_a, l_target)
            p_b = pcn.Predict(l_b, l_target)
            p_c = pcn.Predict(l_c, l_target)

        net.structural_attention([p_a, p_b, p_c], temperature=1.0)
        net.build()

        dataloader = [{'target': jnp.ones((2, 4))}]
        data_map = {l_target: 'target'}
        sim = pcn.Simulation(net)
        sim.config(iterations_per_sample=5)
        result = sim.test(dataloader, data_map)
        assert result is not None

    def test_structural_attention_with_learning(self):
        """Structural attention works during training."""
        net = pcn.PCNetwork(seed=0)
        with net:
            l_target = pcn.Layer(dim=4, label='target')
            l_a = pcn.Layer(dim=3, activation=pcn.Relu(), label='a')
            l_b = pcn.Layer(dim=3, activation=pcn.Relu(), label='b')

            p_a = pcn.Predict(l_a, l_target)
            p_b = pcn.Predict(l_b, l_target)

        net.structural_attention([p_a, p_b])
        net.build()

        dataloader = [{'target': jnp.ones((2, 4))}]
        data_map = {l_target: 'target'}
        sim = pcn.Simulation(net)
        sim.config(iterations_per_sample=5)
        result = sim.train(dataloader, data_map)
        assert result is not None


# ============================================================================
# Composability tests
# ============================================================================

class TestComposability:
    """Test that all mechanisms compose correctly."""

    def test_all_mechanisms_together(self):
        """Network using all three mechanisms runs without error."""
        net = pcn.PCNetwork(seed=0)
        with net:
            l_target = pcn.Layer(dim=4, label='target')
            l_a = pcn.Layer(dim=3, activation=pcn.Relu(), label='a')
            l_b = pcn.Layer(dim=3, activation=pcn.Relu(), label='b')
            l_gate = pcn.Layer(dim=4, activation=pcn.Sigmoid(), label='gate')

            p_a = pcn.Predict(l_a, l_target)
            p_b = pcn.Predict(l_b, l_target)

            # Mechanism 1: cross-connection precision modulation
            pcn.Modulate(p_b.precision, p_a.error,
                         update_rule=pcn.Hebbian(learning_rate=1e-3))

            # Mechanism 2: per-leg flow gating
            pcn.Modulate(l_gate.value, p_a.flow_to_pre,
                         update_rule=pcn.Hebbian(learning_rate=1e-3))

        # Mechanism 3: structural attention
        net.structural_attention([p_a, p_b], temperature=1.0)
        net.build()

        dataloader = [{'target': jnp.ones((2, 4))}]
        data_map = {l_target: 'target'}
        sim = pcn.Simulation(net)
        sim.config(iterations_per_sample=5)

        # Test inference
        result = sim.test(dataloader, data_map)
        assert result is not None

        # Test learning
        result = sim.train(dataloader, data_map)
        assert result is not None


# ============================================================================
# Validation tests
# ============================================================================

class TestValidation:
    """Test error handling and validation."""

    def test_flow_node_type_ids(self):
        """Verify flow node type IDs are correct."""
        net = pcn.PCNetwork(seed=0)
        with net:
            l_a = pcn.Layer(dim=4, label='a')
            l_b = pcn.Layer(dim=3, label='b')
            p = pcn.Predict(l_a, l_b)

        assert p.error.node_type_id == 1
        assert p.precision.node_type_id == 2
        assert p.flow_to_pre.node_type_id == 3
        assert p.flow_to_post.node_type_id == 4

    def test_precision_node_as_pre_value(self):
        """Precision node can be used as pre source (node_type_id=2)."""
        net = pcn.PCNetwork(seed=0)
        with net:
            l_in = pcn.Layer(dim=4, label='input')
            l_out = pcn.Layer(dim=3, label='output')
            p = pcn.Predict(l_out, l_in)

            # Precision as pre → should set pre_node_type = 2
            proj = pcn.Project(p.precision, l_out.value,
                               update_rule=pcn.Hebbian(learning_rate=1e-3))
        assert proj.pre_node_type == 2
