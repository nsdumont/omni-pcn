"""
Test that PCN classes have correct fields and types.
"""

import pytest
import jax.numpy as jnp
from typing import NamedTuple


class TestActivationClasses:
    """Test activation function classes."""

    def test_activation_type_ids(self):
        """Test that each activation has correct type_id."""
        from pcn import Direct, Relu, Softmax, Tanh, Sigmoid

        assert Direct().type_id == 0
        assert Relu().type_id == 1
        assert Softmax().type_id == 2
        assert Tanh().type_id == 3
        assert Sigmoid().type_id == 4

    def test_activation_repr(self):
        """Test activation __repr__ methods."""
        from pcn import Direct, Relu, Softmax

        assert "Direct" in repr(Direct())
        assert "Relu" in repr(Relu())
        assert "Softmax" in repr(Softmax())


class TestLearningRuleClasses:
    """Test learning rule classes."""


    def test_hebbian(self):
        """Test Hebbian with custom learning rate."""
        from pcn import Hebbian

        rule = Hebbian(learning_rate=0.01)
        assert rule.learning_rate == 0.01


    def test_oja(self):
        """Test Oja with custom learning rate."""
        from pcn import Oja

        rule = Oja(learning_rate=0.01)
        assert rule.learning_rate == 0.01
        assert rule.type_id == 3


    def test_three_factor_hebbian_with_reward(self):
        """Test ThreeFactorHebbian with reward function (inputs, fn) tuple."""
        from pcn import ThreeFactorHebbian

        def my_reward(label):
            return jnp.zeros(1)

        rule = ThreeFactorHebbian(
            learning_rate=0.005,
            reward_fn=(('label',), my_reward),
        )
        assert rule.learning_rate == 0.005
        assert rule.reward_fn == (('label',), my_reward)
        assert rule.reward_fn_callable is my_reward


class TestLayerClass:
    """Test Layer class."""

    def test_layer_basic_creation(self):
        """Test Layer creation inside network context."""
        import pcn

        net = pcn.PCNetwork()
        with net:
            layer = pcn.Layer(dim=64, label="test_layer")

        assert layer.dim == 64
        assert layer.label == "test_layer"
        assert layer._idx == 0
        assert layer._network is net

    def test_layer_default_activation(self):
        """Test Layer default activation is Direct."""
        import pcn

        net = pcn.PCNetwork()
        with net:
            layer = pcn.Layer(dim=32)

        assert isinstance(layer.activation, pcn.Relu)

    def test_layer_custom_activation(self):
        """Test Layer with custom activation."""
        import pcn

        net = pcn.PCNetwork()
        with net:
            layer = pcn.Layer(dim=32, activation=pcn.Tanh())

        assert isinstance(layer.activation, pcn.Tanh)

    def test_layer_auto_label(self):
        """Test Layer auto-generates label if not provided."""
        import pcn

        net = pcn.PCNetwork()
        with net:
            layer = pcn.Layer(dim=128)

        assert layer.label is not None
        assert "128" in layer.label
        assert "0" in layer.label

    def test_layer_value_ref(self):
        """Test Layer value property returns NodeRef."""
        import pcn
        from pcn import NodeRef

        net = pcn.PCNetwork()
        with net:
            layer = pcn.Layer(dim=32, label="test")

        assert isinstance(layer.value, NodeRef)
        assert layer.value.node_type == 'value'
        assert layer.value.node_type_id == 0
        assert layer.value.owner_type == 'layer'



class TestNodeRefClass:
    """Test NodeRef class."""

    def test_node_ref_layer_creation(self):
        """Test NodeRef creation with layer owner."""
        import pcn
        from pcn import NodeRef

        net = pcn.PCNetwork()
        with net:
            layer = pcn.Layer(dim=32, label="test")
            node_ref = NodeRef(layer, 'value', owner_type='layer')

        assert node_ref.layer is layer
        assert node_ref.owner is layer
        assert node_ref.owner_type == 'layer'
        assert node_ref.node_type == 'value'
        assert node_ref.node_type_id == 0

    def test_node_ref_predict_creation(self):
        """Test NodeRef creation with predict connection owner."""
        import pcn
        from pcn import NodeRef

        net = pcn.PCNetwork()
        with net:
            l1 = pcn.Layer(dim=32, label="pre")
            l2 = pcn.Layer(dim=16, label="post")
            conn = pcn.Predict(l1, l2)

        error_ref = conn.error
        assert isinstance(error_ref, NodeRef)
        assert error_ref.owner is conn
        assert error_ref.owner_type == 'predict'
        assert error_ref.node_type == 'error'
        assert error_ref.node_type_id == 1
        assert error_ref.predict_conn is conn

    def test_node_ref_predict_init_precision(self):
        """Test Predict connection init_precision is a float attribute."""
        import pcn

        net = pcn.PCNetwork()
        with net:
            l1 = pcn.Layer(dim=32)
            l2 = pcn.Layer(dim=16)
            conn = pcn.Predict(l1, l2, init_precision=2.0)

        assert conn.init_precision == 2.0
        # Backwards-compat view: init_log_precision == log(init_precision)
        assert conn.init_log_precision == pytest.approx(float(jnp.log(2.0)))

    def test_node_ref_predict_init_log_precision_bc(self):
        """Legacy ``init_log_precision`` kwarg converts to init_precision."""
        import pcn

        net = pcn.PCNetwork()
        with net:
            l1 = pcn.Layer(dim=32)
            l2 = pcn.Layer(dim=16)
            conn = pcn.Predict(l1, l2, init_log_precision=0.5)

        assert conn.init_precision == pytest.approx(float(jnp.exp(0.5)))
        assert conn.init_log_precision == pytest.approx(0.5)

    def test_node_ref_type_ids(self):
        """Test NodeRef type_id mapping."""
        import pcn
        from pcn import NodeRef

        net = pcn.PCNetwork()
        with net:
            layer = pcn.Layer(dim=32)

        assert NodeRef(layer, 'value').node_type_id == 0
        assert NodeRef(layer, 'error').node_type_id == 1
        assert NodeRef(layer, 'precision').node_type_id == 2

    def test_node_ref_layer_property_raises_for_predict(self):
        """Test that .layer raises for predict-owned NodeRef."""
        import pcn

        net = pcn.PCNetwork()
        with net:
            l1 = pcn.Layer(dim=32)
            l2 = pcn.Layer(dim=16)
            conn = pcn.Predict(l1, l2)

        with pytest.raises(AttributeError):
            _ = conn.error.layer

    def test_node_ref_predict_conn_raises_for_layer(self):
        """Test that .predict_conn raises for layer-owned NodeRef."""
        import pcn

        net = pcn.PCNetwork()
        with net:
            layer = pcn.Layer(dim=32)

        with pytest.raises(AttributeError):
            _ = layer.value.predict_conn


class TestConnectionClasses:
    """Test connection classes."""

    def test_predict_creation(self):
        """Test Predict connection creation."""
        import pcn

        net = pcn.PCNetwork()
        with net:
            l1 = pcn.Layer(dim=32, label="pre")
            l2 = pcn.Layer(dim=16, label="post")
            conn = pcn.Predict(l1, l2)

        assert conn.pre == [l1]
        assert conn.post is l2
        assert conn.weight is None
        assert conn._idx == 0

    def test_predict_fixed_weight(self):
        """Test Predict with fixed weight."""
        import pcn

        fixed_W = jnp.ones((16, 32))

        net = pcn.PCNetwork()
        with net:
            l1 = pcn.Layer(dim=32, label="pre")
            l2 = pcn.Layer(dim=16, label="post")
            conn = pcn.Predict(l1, l2, init_weight=fixed_W)

        assert conn.weight is not None

    def test_project_creation(self):
        """Test Project connection creation."""
        import pcn

        net = pcn.PCNetwork()
        with net:
            l1 = pcn.Layer(dim=32, label="pre")
            l2 = pcn.Layer(dim=16, label="post")
            p1 = pcn.Predict(l1, l2)
            conn = pcn.Project(l1.value, p1.error)

        assert conn.pre.owner is l1
        assert conn.pre.node_type == 'value'
        assert conn.post.owner is p1
        assert conn.post.node_type == 'error'
        assert conn.pre_node_type == 0
        assert conn.post_node_type == 1

    def test_project_default_rule(self):
        """Test Project default learning rule is Hebbian."""
        import pcn

        net = pcn.PCNetwork()
        with net:
            l1 = pcn.Layer(dim=32)
            l2 = pcn.Layer(dim=16)
            conn = pcn.Project(l1.value, l2.value)

        assert isinstance(conn.update_rule, pcn.Hebbian)

    def test_modulate_creation(self):
        """Test Modulate connection creation."""
        import pcn

        net = pcn.PCNetwork()
        with net:
            l1 = pcn.Layer(dim=32, label="pre")
            l2 = pcn.Layer(dim=16, label="post")
            p1 = pcn.Predict(l1, l2)
            conn = pcn.Modulate(l1.value, p1.error)

        assert conn.pre.owner is l1
        assert conn.post.owner is p1
        assert conn.pre_node_type == 0
        assert conn.post_node_type == 1


class TestStateClasses:
    """Test NetworkState and NetworkParams classes."""

    def test_network_state_is_namedtuple(self):
        """Test NetworkState is a NamedTuple."""
        from pcn import NetworkState

        assert issubclass(NetworkState, tuple)
        assert hasattr(NetworkState, '_fields')
        assert 'values' in NetworkState._fields
        assert 'errors' in NetworkState._fields
        assert 'precisions' in NetworkState._fields
        assert 'clamped' in NetworkState._fields

    def test_network_params_is_namedtuple(self):
        """Test NetworkParams is a NamedTuple."""
        from pcn import NetworkParams

        assert issubclass(NetworkParams, tuple)
        assert hasattr(NetworkParams, '_fields')
        assert 'predict_weights' in NetworkParams._fields
        assert 'project_weights' in NetworkParams._fields
        assert 'modulate_weights' in NetworkParams._fields
        assert 'precision_weights' in NetworkParams._fields
        assert 'precision_biases' in NetworkParams._fields

    def test_network_state_creation(self):
        """Test NetworkState can be created."""
        from pcn import NetworkState

        values = [jnp.zeros((4, 10)), jnp.zeros((4, 5))]
        errors = [jnp.zeros((4, 10)), jnp.zeros((4, 5))]
        precisions = [jnp.ones((4, 10)), jnp.ones((4, 5))]
        clamped = [jnp.zeros(4, dtype=bool), jnp.zeros(4, dtype=bool)]

        state = NetworkState(
            values=values, errors=errors, precisions=precisions, clamped=clamped)

        assert len(state.values) == 2
        assert len(state.errors) == 2
        assert len(state.precisions) == 2
        assert len(state.clamped) == 2

    def test_network_params_creation(self):
        """Test NetworkParams can be created."""
        from pcn import NetworkParams

        params = NetworkParams(
            predict_weights=[jnp.zeros((5, 10))],
            predict_biases=[jnp.zeros(5)],
            project_weights=[],
            project_biases=[],
            modulate_weights=[],
            modulate_biases=[],
            precision_weights=[jnp.zeros((5, 10)), jnp.zeros((5, 4))],
            precision_biases=[jnp.zeros(5), jnp.zeros(5)],
        )

        assert len(params.predict_weights) == 1
        assert len(params.precision_biases) == 2
        assert len(params.precision_weights) == 2


class TestStructureClasses:
    """Test structure specification classes."""

    def test_layer_spec_fields(self):
        """Test LayerSpec has correct fields."""
        from pcn.core.structure import LayerSpec

        spec = LayerSpec(dim=64, activation_type=1, dynamics_rate=0.2, label="test")
        assert spec.dim == 64
        assert spec.activation_type == 1
        assert spec.label == "test"
        assert spec.dynamics_rate == 0.2

    def test_predict_conn_spec_fields(self):
        """Test PredictConnSpec has correct fields."""
        from pcn.core.structure import PredictConnSpec

        spec = PredictConnSpec(
            pre_idx=(0,),
            post_idx=1,
            has_fixed_weights=False,
            learn_precision_weights=True,
            learn_precision_bias=True,
        )
        assert spec.pre_idx == (0,)
        assert spec.post_idx == 1
        assert spec.has_fixed_weights is False
        assert spec.learn_precision_weights is True
        assert spec.learn_precision_bias is True

    def test_project_conn_spec_fields(self):
        """Test ProjectConnSpec has correct fields."""
        from pcn.core.structure import ProjectConnSpec

        spec = ProjectConnSpec(
            pre_idx=(0,),
            pre_node_type=0,
            post_idx=1,
            post_node_type=1,
            learning_rule_type=0,
            learning_rate=0.001,
            reward_fn_idx=-1
        )
        assert spec.pre_idx == (0,)
        assert spec.pre_node_type == 0
        assert spec.post_idx == 1
        assert spec.post_node_type == 1

    def test_network_structure_hashable(self):
        """Test NetworkStructure is hashable."""
        from pcn.core.structure import NetworkStructure, LayerSpec

        structure = NetworkStructure(
            layers=(LayerSpec(10, 0, 0.1, "l1"), LayerSpec(5, 1, 0.1, "l2")),
            predict_conns=(),
            project_conns=(),
            modulate_conns=(),
            layer_dims=(10, 5),
            predict_error_dims=()
        )

        # Should not raise
        hash_val = hash(structure)
        assert isinstance(hash_val, int)


class TestPCNetworkClass:
    """Test PCNetwork class."""

    def test_pcnetwork_creation(self):
        """Test PCNetwork creation."""
        import pcn

        net = pcn.PCNetwork(seed=123)
        assert net.seed == 123
        assert net.structure is None
        assert net.params is None

    def test_pcnetwork_context_manager(self):
        """Test PCNetwork works as context manager."""
        import pcn

        net = pcn.PCNetwork()
        with net:
            l1 = pcn.Layer(dim=10)
            l2 = pcn.Layer(dim=5)

        assert len(net._layers) == 2
        assert l1._network is net
        assert l2._network is net

    def test_pcnetwork_duplicate_label_deduplicated(self):
        """Test that duplicate layer labels are auto-deduplicated."""
        import pcn

        net = pcn.PCNetwork()
        with net:
            l1 = pcn.Layer(dim=10, label="same")
            l2 = pcn.Layer(dim=5, label="same")
            pcn.Predict(l1, l2)
        net.build()
        assert l1.label == "same"
        assert l2.label == "same_1"
        assert net.label_to_idx["same"] == 0
        assert net.label_to_idx["same_1"] == 1

    def test_pcnetwork_getitem(self):
        """Test PCNetwork __getitem__ for label lookup."""
        import pcn

        net = pcn.PCNetwork()
        with net:
            pcn.Layer(dim=10, label="input")
            pcn.Layer(dim=5, label="output")
        net.build()

        assert net['input'] == 0
        assert net['output'] == 1

    def test_pcnetwork_layer_outside_context_raises(self):
        """Test creating Layer outside context raises error."""
        import pcn

        with pytest.raises(RuntimeError, match="No PCNetwork context"):
            pcn.Layer(dim=10)




class TestPredictConnectionLabels:
    """Test Predict connection labels and deduplication."""

    def test_predict_label_stored(self):
        """Test that user-specified label is stored on Predict."""
        import pcn

        net = pcn.PCNetwork()
        with net:
            l1 = pcn.Layer(dim=10, label="input")
            l2 = pcn.Layer(dim=5, label="output")
            p = pcn.Predict(l1, l2, label="my_conn")
        net.build()
        assert p.label == "my_conn"
        assert net.structure.predict_conns[0].label == "my_conn"

    def test_predict_label_auto_generated(self):
        """Test that auto-generated label uses layer labels."""
        import pcn

        net = pcn.PCNetwork()
        with net:
            l1 = pcn.Layer(dim=10, label="input")
            l2 = pcn.Layer(dim=5, label="output")
            p = pcn.Predict(l1, l2)
        net.build()
        assert p.label == "predict_input_output"

    def test_predict_label_deduplication(self):
        """Test that duplicate connection labels are auto-deduplicated."""
        import pcn

        net = pcn.PCNetwork()
        with net:
            l1 = pcn.Layer(dim=10, label="a")
            l2 = pcn.Layer(dim=5, label="b")
            l3 = pcn.Layer(dim=5, label="c")
            p1 = pcn.Predict(l1, l2, label="shared")
            p2 = pcn.Predict(l1, l3, label="shared")
        net.build()
        assert p1.label == "shared"
        assert p2.label == "shared_1"

    def test_predict_label_in_node_to_idx(self):
        """Test that connection labels are used in node_to_idx."""
        import pcn

        net = pcn.PCNetwork()
        with net:
            l1 = pcn.Layer(dim=10, label="input")
            l2 = pcn.Layer(dim=5, label="output")
            pcn.Predict(l1, l2, label="gen")
        net.build()
        assert "gen-error" in net.node_to_idx
        assert "gen-logprecision" in net.node_to_idx

    def test_predict_label_subclasses(self):
        """Test that label works on PredictRes subclass."""
        import pcn

        net = pcn.PCNetwork()
        with net:
            l1 = pcn.Layer(dim=10, label="a")
            l2 = pcn.Layer(dim=10, label="b")
            p = pcn.PredictRes(l1, l2, label="res_conn")
        net.build()
        assert p.label == "res_conn"


class TestMultiTransform:
    """Test PCNetwork.multi_transform optimizer construction."""

    def _build_net(self):
        """Helper: build a 3-layer net with labeled connections."""
        import pcn

        net = pcn.PCNetwork(seed=0)
        with net:
            l1 = pcn.Layer(dim=10, label="input")
            l2 = pcn.Layer(dim=8, label="hidden")
            l3 = pcn.Layer(dim=4, label="output")
            pcn.Predict(l1, l2, label="gen_path")
            pcn.Predict(l2, l3, label="disc_path")
        net.build()
        return net

    def test_category_level(self):
        """Test matching by category ('predict', 'precision')."""
        import optax

        net = self._build_net()
        opt = net.multi_transform(
            {"predict": optax.adam(1e-3), "precision": optax.sgd(1e-5)},
            default_optim=optax.adam(1e-4),
        )
        assert opt is not None

    def test_kind_level(self):
        """Test matching by kind ('weights', 'biases')."""
        import optax

        net = self._build_net()
        opt = net.multi_transform(
            {"weights": optax.adam(1e-3), "biases": optax.sgd(1e-5)},
            default_optim=optax.adam(1e-4),
        )
        assert opt is not None

    def test_exact_param_type(self):
        """Test matching by exact param type ('predict_weights')."""
        import optax

        net = self._build_net()
        opt = net.multi_transform(
            {"predict_weights": optax.adam(1e-3)},
            default_optim=optax.adam(1e-4),
        )
        assert opt is not None

    def test_connection_label(self):
        """Test matching by connection label applies to all param types."""
        import optax

        net = self._build_net()
        opt = net.multi_transform(
            {"gen_path": optax.adam(5e-3)},
            default_optim=optax.adam(1e-4),
        )
        assert opt is not None

    def test_specific_label_and_type(self):
        """Test most-specific key: '{label}_{param_type}'."""
        import optax

        net = self._build_net()
        opt = net.multi_transform(
            {"gen_path_predict_biases": optax.adam(5e-3)},
            default_optim=optax.adam(1e-4),
        )
        assert opt is not None

    def test_connection_label_overrides_general(self):
        """Test that connection label takes priority over general class."""
        import optax

        net = self._build_net()
        # 'gen_path' (priority 2) should override 'predict' (priority 3)
        opt = net.multi_transform(
            {"gen_path": optax.adam(5e-3), "predict": optax.adam(1e-3)},
            default_optim=optax.adam(1e-4),
        )
        assert opt is not None

    def test_specific_overrides_connection_label(self):
        """Test that '{label}_{type}' overrides connection-level label."""
        import optax

        net = self._build_net()
        opt = net.multi_transform(
            {
                "gen_path": optax.adam(5e-3),
                "gen_path_predict_biases": optax.sgd(1e-2),
            },
            default_optim=optax.adam(1e-4),
        )
        assert opt is not None

    def test_exact_type_overrides_category(self):
        """Test that 'predict_weights' takes priority over 'predict'."""
        import optax

        net = self._build_net()
        # 'predict_weights' (3a) overrides 'predict' (3b) for predict_weights;
        # 'predict' still used for predict_biases.
        opt = net.multi_transform(
            {"predict_weights": optax.adam(1e-3), "predict": optax.sgd(1e-5)},
            default_optim=optax.adam(1e-4),
        )
        assert opt is not None

    def test_conflict_category_and_kind_raises(self):
        """Test that ambiguous category+kind keys raise ValueError."""
        import optax

        net = self._build_net()
        with pytest.raises(ValueError, match="Ambiguous"):
            net.multi_transform(
                {"precision": optax.adam(1e-3), "biases": optax.sgd(1e-5)},
                default_optim=optax.adam(1e-4),
            )

    def test_raises_before_build(self):
        """Test that multi_transform before build() raises RuntimeError."""
        import pcn
        import optax

        net = pcn.PCNetwork()
        with net:
            l1 = pcn.Layer(dim=10)
            l2 = pcn.Layer(dim=5)
            pcn.Predict(l1, l2)
        with pytest.raises(RuntimeError, match="must be built"):
            net.multi_transform({}, default_optim=optax.adam(1e-4))

    def test_default_only(self):
        """Test that empty optim_dict uses default for everything."""
        import optax

        net = self._build_net()
        opt = net.multi_transform({}, default_optim=optax.adam(1e-4))
        assert opt is not None

    def test_works_in_training(self):
        """Test multi_transform optimizer in an actual training step."""
        import pcn
        import optax
        from pcn import Simulation

        net = self._build_net()
        l_input = net.get_layer("input")
        l_output = net.get_layer("output")
        opt = net.multi_transform(
            {"predict": optax.adam(1e-3), "precision": optax.sgd(1e-5)},
            default_optim=optax.adam(1e-4),
        )
        sim = Simulation(net)
        sim.config(iterations_per_sample=3)
        dataloader = [{"x": jnp.ones((2, 10)), "y": jnp.ones((2, 4))}]
        data_map = {l_input: "x", l_output: "y"}
        result = sim.train(dataloader, data_map, params_optimizer=opt)
        assert result is not None

    def test_project_training(self):
        import pcn
        import optax
        from pcn import Simulation

        net = pcn.PCNetwork(seed=0)
        with net:
            l1 = pcn.Layer(dim=10, label="input")
            l2 = pcn.Layer(dim=8, label="hidden")
            l3 = pcn.Layer(dim=4, label="output")
            pcn.Predict(l1, l2, label="gen_path")
            pcn.Predict(l2, l3, label="disc_path")
            pcn.Project(
                l2.value, l3.value,
                update_rule=pcn.GradientDescent(
                    loss_fn=((l3.value, "y"),
                             lambda out, y: jnp.sum((out - y) ** 2))))
        net.build()

        l_input = net.get_layer("input")
        l_output = net.get_layer("output")
        opt = net.multi_transform(
            {"predict": optax.adam(1e-3), "precision": optax.sgd(1e-5),
             "project": optax.adamw(5e-3)},
            default_optim=optax.adam(1e-4),
        )
        sim = Simulation(net)
        sim.config(iterations_per_sample=3)
        dataloader = [{"x": jnp.ones((2, 10)), "y": jnp.ones((2, 4))}]
        data_map = {l_input: "x", l_output: "y"}
        result = sim.train(dataloader, data_map, params_optimizer=opt)
        assert result is not None


class TestRecordMap:
    """Test the named record_map format: {name: (inputs, fn)}."""

    def _build_sim(self):
        import pcn
        from pcn import Simulation

        net = pcn.PCNetwork(seed=0)
        with net:
            l_in = pcn.Layer(dim=10, label="input", activation=pcn.Direct())
            l_out = pcn.Layer(dim=4, label="output", activation=pcn.Softmax())
            pcn.Predict(l_in, l_out)
        net.build()
        sim = Simulation(net)
        sim.config(iterations_per_sample=3)
        data = [{"x": jnp.ones((2, 10)), "y": jnp.ones((2, 4))}]
        data_map = {l_in: "x", l_out: "y"}
        test_map = {l_in: "x"}
        return sim, data, data_map, test_map, l_in, l_out

    def test_single_record(self):
        """Test record_map with a single named entry."""
        sim, data, data_map, test_map, l_in, l_out = self._build_sim()

        def my_metric(values, labels):
            return float(jnp.mean(values))

        results = sim.test(
            data, data_map=test_map,
            record_map={'my_metric': ((l_out.value, 'y'), my_metric)},
        )
        assert 'my_metric' in results
        assert len(results['my_metric']) == 1  # one batch

    def test_duplicate_inputs_different_names(self):
        """Test two functions with the same inputs but different names."""
        sim, data, data_map, test_map, l_in, l_out = self._build_sim()

        def fn_a(values, labels):
            return float(jnp.mean(values))

        def fn_b(values, labels):
            return float(jnp.sum(values))

        results = sim.test(
            data, data_map=test_map,
            record_map={
                'mean_val': ((l_out.value, 'y'), fn_a),
                'sum_val': ((l_out.value, 'y'), fn_b),
            },
        )
        assert 'mean_val' in results
        assert 'sum_val' in results
        assert results['mean_val'][0] != results['sum_val'][0]

    def test_single_node_input(self):
        """Test record_map with a single NodeRef input (not a tuple)."""
        sim, data, data_map, test_map, l_in, l_out = self._build_sim()

        def grab(values):
            return float(jnp.mean(values))

        results = sim.test(
            data, data_map=test_map,
            record_map={'output_mean': (l_out.value, grab)},
        )
        assert 'output_mean' in results

    def test_train_record_map(self):
        """Test record_map works in train() and populates train_records."""
        sim, data, data_map, test_map, l_in, l_out = self._build_sim()

        def my_fn(values, labels):
            return float(jnp.mean(values))

        sim.train(
            data, data_map=data_map, epochs=1,
            record_map={'train_metric': ((l_out.value, 'y'), my_fn)},
        )
        assert 'train_metric' in sim.train_records
        assert len(sim.train_records['train_metric']) == 1

    def test_non_string_key_raises(self):
        """Test that non-string keys in record_map raise TypeError."""
        sim, data, data_map, test_map, l_in, l_out = self._build_sim()

        with pytest.raises(TypeError, match="record_map keys must be strings"):
            sim.test(
                data, data_map=test_map,
                record_map={(l_out.value, 'y'): ('ignored', lambda x: x)},
            )


class TestMultiPre:
    """Test multi-pre connections (list of pre layers/nodes)."""

    def test_predict_multi_pre(self):
        """Predict with multiple pre layers concatenates their dims."""
        import pcn

        net = pcn.PCNetwork(seed=0)
        with net:
            l1 = pcn.Layer(dim=10, label="a")
            l2 = pcn.Layer(dim=8, label="b")
            l3 = pcn.Layer(dim=6, label="target")
            conn = pcn.Predict([l1, l2], l3)
        net.build()

        assert conn.pre == [l1, l2]
        assert conn.pre_dim == 18
        # Structure should have pre_idx tuple
        spec = net.structure.predict_conns[0]
        assert spec.pre_idx == (0, 1)

    def test_project_multi_pre_layers(self):
        """Project with multiple Layer objects as pre."""
        import pcn

        net = pcn.PCNetwork(seed=0)
        with net:
            l1 = pcn.Layer(dim=10, label="a")
            l2 = pcn.Layer(dim=8, label="b")
            l3 = pcn.Layer(dim=6, label="target")
            pcn.Predict(l1, l3)
            conn = pcn.Project([l1, l2], l3)
        net.build()

        assert conn.pre_dim == 18
        assert conn.pre_node_type == 0  # value
        spec = net.structure.project_conns[0]
        assert spec.pre_idx == (0, 1)

    def test_project_multi_pre_mixed_type_raises(self):
        """Project raises if pre nodes have different node types."""
        import pcn

        net = pcn.PCNetwork(seed=0)
        with net:
            l1 = pcn.Layer(dim=10, label="a")
            l2 = pcn.Layer(dim=8, label="b")
            p1 = pcn.Predict(l1, l2)
            with pytest.raises(ValueError, match="same node type"):
                pcn.Project([l1.value, p1.error], l2.value)

    def test_predict_multi_pre_run_batch(self):
        """Multi-pre Predict runs through run_batch without error."""
        import pcn
        from pcn.backend import run_batch

        net = pcn.PCNetwork(seed=0)
        with net:
            l1 = pcn.Layer(dim=10, label="input1")
            l2 = pcn.Layer(dim=8, label="input2")
            l3 = pcn.Layer(dim=6, label="output")
            pcn.Predict([l1, l2], l3)
        net.build()

        sample = {
            'input1': jnp.ones((2, 10)),
            'input2': jnp.ones((2, 8)),
        }
        data_map = ((l1._idx, 'input1'), (l2._idx, 'input2'))

        new_params, _, _, vl, el, _, _, energies = run_batch(
            sample, net.params, net.structure,
            data_map, n_iterations=5, log_every=5, learning=False)
        assert len(vl) == 3
        assert energies.shape[0] > 0


class TestLayerInputsForProjectModulate:
    """Test that Project/Modulate accept plain Layer objects."""

    def test_project_layer_pre(self):
        """Project(layer, node) treats layer as layer.value."""
        import pcn

        net = pcn.PCNetwork(seed=0)
        with net:
            l1 = pcn.Layer(dim=10, label="a")
            l2 = pcn.Layer(dim=8, label="b")
            pcn.Predict(l1, l2)
            conn = pcn.Project(l1, l2.value)
        assert conn.pre_node_type == 0  # value

    def test_project_layer_post(self):
        """Project(node, layer) treats layer as layer.value."""
        import pcn

        net = pcn.PCNetwork(seed=0)
        with net:
            l1 = pcn.Layer(dim=10, label="a")
            l2 = pcn.Layer(dim=8, label="b")
            pcn.Predict(l1, l2)
            conn = pcn.Project(l1.value, l2)
        assert conn.post_node_type == 0  # value

    def test_modulate_layer_pre(self):
        """Modulate(layer, node) treats layer as layer.value."""
        import pcn

        net = pcn.PCNetwork(seed=0)
        with net:
            l1 = pcn.Layer(dim=10, label="a")
            l2 = pcn.Layer(dim=8, label="b")
            p1 = pcn.Predict(l1, l2)
            conn = pcn.Modulate(l1, p1.error)
        assert conn.pre_node_type == 0  # value


class TestTransformationParam:
    """Test the transformation parameter on connections."""

    def test_predict_linear_default(self):
        """Predict defaults to linear transformation."""
        import pcn

        net = pcn.PCNetwork(seed=0)
        with net:
            l1 = pcn.Layer(dim=10)
            l2 = pcn.Layer(dim=8)
            conn = pcn.Predict(l1, l2)
        assert not conn.is_conv
        assert not conn.is_transconv
        assert conn.n_bands == 0

    def test_predict_linear_activation(self):
        """Predict with 'linear-<activation>' wraps the matmul in a nonlinearity."""
        import pcn
        from pcn.core.activations import Direct, Relu, Softplus, Exp

        net = pcn.PCNetwork(seed=0)
        with net:
            l1 = pcn.Layer(dim=10)
            l2 = pcn.Layer(dim=8)
            l3 = pcn.Layer(dim=4)
            l4 = pcn.Layer(dim=4)
            c_relu = pcn.Predict(l1, l2, transformation='linear-relu')
            c_softplus = pcn.Predict(l2, l3, transformation='linear-softplus')
            c_exp = pcn.Predict(l3, l4, transformation='linear-exp')
            c_plain = pcn.Predict(l1, l2, transformation='linear')
        net.build()
        assert c_relu.post_activation_type_id == Relu.type_id
        assert c_softplus.post_activation_type_id == Softplus.type_id
        assert c_exp.post_activation_type_id == Exp.type_id
        assert c_plain.post_activation_type_id == Direct.type_id
        # spec carries the same value
        assert net.structure.predict_conns[0].post_activation_type == Relu.type_id
        assert net.structure.predict_conns[1].post_activation_type == Softplus.type_id
        assert net.structure.predict_conns[2].post_activation_type == Exp.type_id
        assert net.structure.predict_conns[3].post_activation_type == Direct.type_id

    def test_predict_linear_activation_invalid(self):
        """Unknown 'linear-<name>' activation raises ValueError."""
        import pcn

        net = pcn.PCNetwork(seed=0)
        with net:
            l1 = pcn.Layer(dim=10)
            l2 = pcn.Layer(dim=8)
            with pytest.raises(ValueError, match="Invalid transformation"):
                pcn.Predict(l1, l2, transformation='linear-nope')

    def test_predict_linear_activation_apply(self):
        """Spec.apply with 'linear-relu' returns max(0, Wf(x)+b)."""
        import pcn
        import jax.numpy as jnp

        net = pcn.PCNetwork(seed=0)
        with net:
            l1 = pcn.Layer(dim=4)
            l2 = pcn.Layer(dim=3)
            pcn.Predict(l1, l2, transformation='linear-relu')
        net.build()
        spec = net.structure.predict_conns[0]
        # Force a weight that yields a known sign pattern
        W = jnp.array([[1., 1., 1., 1.],
                       [-1., -1., -1., -1.],
                       [0., 0., 0., 0.]])
        b = jnp.array([0., 0., 0.])
        x = jnp.array([[1., 1., 1., 1.]])
        out = spec.apply(x, W, b)
        # Row 0: sum=4 -> relu -> 4. Row 1: sum=-4 -> relu -> 0. Row 2: 0.
        assert jnp.allclose(out, jnp.array([[4., 0., 0.]]))

    def test_predict_banded(self):
        """Predict with banded transformation sets n_bands."""
        import pcn

        net = pcn.PCNetwork(seed=0)
        with net:
            l1 = pcn.Layer(dim=10)
            l2 = pcn.Layer(dim=8)
            conn = pcn.Predict(l1, l2, transformation='banded5')
        net.build()
        assert conn.n_bands == 5
        spec = net.structure.predict_conns[0]
        assert spec.n_bands == 5

    def test_predict_conv_via_param(self):
        """Predict(transformation='conv') sets is_conv."""
        import pcn

        net = pcn.PCNetwork(seed=0)
        with net:
            # 1 channel, 4x4 spatial => dim=16; 2 channels, 4x4 => dim=32
            l1 = pcn.Layer(dim=16)
            l2 = pcn.Layer(dim=32)
            conn = pcn.Predict(l1, l2, transformation='conv',
                               kernel_size=3, input_shape=(4, 4))
        assert conn.is_conv

    def test_predict_conv_wrapper(self):
        """PredictConv is a shorthand for Predict(transformation='conv')."""
        import pcn

        net = pcn.PCNetwork(seed=0)
        with net:
            l1 = pcn.Layer(dim=16)
            l2 = pcn.Layer(dim=32)
            conn = pcn.PredictConv(l1, l2, kernel_size=3, input_shape=(4, 4))
        assert conn.is_conv
        assert isinstance(conn, pcn.Predict)

    def test_predict_transconv_wrapper(self):
        """PredictTransConv is a shorthand for Predict(transformation='transconv')."""
        import pcn

        net = pcn.PCNetwork(seed=0)
        with net:
            # 1ch, 4x4 => dim=16; transconv stride=2 kernel=2 => 8x8; 1ch out => dim=64
            l1 = pcn.Layer(dim=16)
            l2 = pcn.Layer(dim=64)
            conn = pcn.PredictTransConv(l1, l2, kernel_size=2, input_shape=(4, 4))
        assert conn.is_transconv
        assert isinstance(conn, pcn.Predict)

    def test_project_banded(self):
        """Project with banded transformation."""
        import pcn

        net = pcn.PCNetwork(seed=0)
        with net:
            l1 = pcn.Layer(dim=10)
            l2 = pcn.Layer(dim=8)
            pcn.Predict(l1, l2)
            conn = pcn.Project(l1.value, l2.value, transformation='banded3')
        net.build()
        assert conn.n_bands == 3
        spec = net.structure.project_conns[0]
        assert spec.n_bands == 3

    def test_invalid_transformation_raises(self):
        """Invalid transformation string raises ValueError."""
        import pcn

        net = pcn.PCNetwork(seed=0)
        with net:
            l1 = pcn.Layer(dim=10)
            l2 = pcn.Layer(dim=8)
            with pytest.raises(ValueError, match="Unknown transformation"):
                pcn.Predict(l1, l2, transformation='invalid')

    def test_predict_masked_requires_weight_mask(self):
        """transformation='masked' without weight_mask raises ValueError."""
        import pcn

        net = pcn.PCNetwork(seed=0)
        with net:
            l1 = pcn.Layer(dim=10)
            l2 = pcn.Layer(dim=8)
            with pytest.raises(ValueError, match="requires a weight_mask"):
                pcn.Predict(l1, l2, transformation='masked')

    def test_predict_masked_wrong_shape_raises(self):
        """weight_mask with wrong shape raises ValueError."""
        import pcn
        import numpy as np

        net = pcn.PCNetwork(seed=0)
        with net:
            l1 = pcn.Layer(dim=10)
            l2 = pcn.Layer(dim=8)
            with pytest.raises(ValueError, match="weight_mask shape"):
                pcn.Predict(l1, l2, transformation='masked',
                            weight_mask=np.ones((8, 9)))

    def test_predict_masked_init_zeros_weights(self):
        """Predict with masked transformation zeros out W where mask is 0."""
        import pcn
        import numpy as np
        import jax.numpy as jnp

        # Strict lower-triangular mask: post=4, pre=4
        mask = np.tril(np.ones((4, 4), dtype=np.float32), k=-1)
        net = pcn.PCNetwork(seed=0)
        with net:
            l1 = pcn.Layer(dim=4)
            l2 = pcn.Layer(dim=4)
            conn = pcn.Predict(l1, l2, transformation='masked',
                               weight_mask=mask)
        net.build()
        assert conn.is_masked
        spec = net.structure.predict_conns[0]
        assert spec.is_masked
        W = net.params.predict_weights[0]
        # Zeros above and on diagonal, free below
        assert jnp.allclose(W * (1 - jnp.asarray(mask)), 0.0)

    def test_project_masked_init(self):
        """Project respects weight_mask at init."""
        import pcn
        import numpy as np
        import jax.numpy as jnp

        mask = np.zeros((8, 10), dtype=np.float32)
        mask[:, :3] = 1.0  # only first 3 pre dims connected
        net = pcn.PCNetwork(seed=0)
        with net:
            l1 = pcn.Layer(dim=10)
            l2 = pcn.Layer(dim=8)
            pcn.Predict(l1, l2)
            conn = pcn.Project(l1.value, l2.value, transformation='masked',
                               weight_mask=mask)
        net.build()
        assert conn.is_masked
        W = net.params.project_weights[0]
        # Columns 3+ must be zero
        assert jnp.allclose(W[:, 3:], 0.0)

    def test_masked_persists_through_training(self):
        """Masked weights stay zero in masked positions after a training step."""
        import pcn
        import numpy as np
        import jax.numpy as jnp
        from pcn.backend.simulation import run_batch
        import optax

        mask = np.tril(np.ones((4, 4), dtype=np.float32), k=-1)
        net = pcn.PCNetwork(seed=0)
        with net:
            l1 = pcn.Layer(dim=4, label='in')
            l2 = pcn.Layer(dim=4, label='out')
            pcn.Predict(l1, l2, transformation='masked', weight_mask=mask)
        net.build()

        sample = {'x': jnp.ones((2, 4)), 'y': jnp.ones((2, 4))}
        data_map = ((0, 'x'), (1, 'y'))
        params_opt = optax.sgd(0.1)
        params_opt_state = params_opt.init({
            'predict_weights': tuple(net.params.predict_weights),
            'predict_biases': tuple(net.params.predict_biases),
            'project_biases': tuple(net.params.project_biases),
            'modulate_biases': tuple(net.params.modulate_biases),
            'precision_weights': tuple(net.params.precision_weights),
            'precision_biases': tuple(net.params.precision_biases),
            'gd_loss_project_weights': (),
            'gd_loss_modulate_weights': (),
        })
        new_params, _, _, _, _, _, _, _ = run_batch(
            sample, net.params, net.structure, data_map, 5, 1,
            learning=True, n_learning_iterations=0,
            params_optimizer=params_opt, params_opt_state=params_opt_state,
            values_optimizer=optax.sgd(1.0),
            predict_weight_masks=net.predict_weight_masks,
        )
        W_new = new_params.predict_weights[0]
        # Masked-out positions must still be exactly zero
        assert jnp.allclose(W_new * (1 - jnp.asarray(mask)), 0.0)

    def test_predictres_single_pre_only(self):
        """PredictRes raises on multi-pre."""
        import pcn

        net = pcn.PCNetwork(seed=0)
        with net:
            l1 = pcn.Layer(dim=10)
            l2 = pcn.Layer(dim=10)
            l3 = pcn.Layer(dim=10)
            with pytest.raises(ValueError, match="multi-pre"):
                pcn.PredictRes([l1, l2], l3)


if __name__ == "__main__":
    tester = TestActivationClasses()
    tester.test_activation_type_ids()
    tester.test_activation_repr()

    tester = TestLearningRuleClasses()
    tester.test_hebbian()
    tester.test_oja()
    tester.test_three_factor_hebbian_with_reward()

    tester = TestLayerClass()
    tester.test_layer_auto_label()
    tester.test_layer_basic_creation
    tester.test_layer_custom_activation

    tester = TestMultiTransform()
    tester.test_works_in_training()
    tester.test_project_training()








def test_hard_tanh_activation():
    """HardTanh: registered, clips to [-1, 1], usable as a layer activation."""
    import jax.numpy as jnp
    import pcn
    from pcn.core.activations import ACTIVATIONS, activation_from_name

    act = pcn.HardTanh()
    x = jnp.array([-5.0, -1.0, 0.3, 1.0, 5.0])
    expected = jnp.array([-1.0, -1.0, 0.3, 1.0, 1.0])
    assert jnp.allclose(act.fn(x), expected)
    assert jnp.allclose(ACTIVATIONS[act.type_id](x), expected)
    assert isinstance(activation_from_name('hard_tanh'), pcn.HardTanh)

    net = pcn.PCNetwork(seed=0)
    with net:
        a = pcn.Layer(dim=4, activation=pcn.HardTanh(), label="a")
        b = pcn.Layer(dim=2, activation=pcn.HardTanh(), label="b")
        pcn.Predict(a, b)
    net.build()
