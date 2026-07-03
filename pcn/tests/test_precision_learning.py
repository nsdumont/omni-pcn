"""Tests for learnable precision per Predict connection."""

import jax
import jax.numpy as jnp
import pytest

import pcn
from pcn.core.activations import (
    Direct, Softplus, Exp, Tanh, Leaky, MemoryActivation, Relu,
    Stochastic, StochasticActivation, gaussian_noise,
)


@pytest.fixture
def precision_network():
    """Create a simple network for precision learning tests."""
    net = pcn.PCNetwork(seed=0)
    with net:
        l1 = pcn.Layer(dim=16, label="input")
        l2 = pcn.Layer(dim=8, activation=pcn.Relu(), label="hidden")
        l3 = pcn.Layer(dim=4, label="output")

        p1 = pcn.Predict(l2, l1)
        p2 = pcn.Predict(l3, l2)

    net.build()
    return net, (l1, l2, l3), (p1, p2)


@pytest.fixture
def sample_data():
    """Create a sample batch."""
    key = jax.random.PRNGKey(99)
    k1, k2 = jax.random.split(key)
    return {
        'input': jax.random.normal(k1, (4, 16)),
        'output': jax.random.normal(k2, (4, 4)),
    }


def test_precision_updates_during_learning(precision_network, sample_data):
    """Precisions should change after training with learn_precision=True."""
    net, (l1, _, l3), _ = precision_network

    sim = pcn.Simulation(net)
    initial_precision_biases = tuple(jnp.array(p) for p in sim.params.precision_biases)
    initial_precision_weights = tuple(jnp.array(p) for p in sim.params.precision_weights)

    sim.train(
        [sample_data],
        data_map={l1: 'input', l3: 'output'},
        epochs=1,
        iterations_per_sample=20,
    )

    final_precision_biases = tuple(jnp.array(p) for p in sim.params.precision_biases)
    final_precision_weights = tuple(jnp.array(p) for p in sim.params.precision_weights)
    biases_changed = any(
        not jnp.allclose(i, f) for i, f in zip(initial_precision_biases, final_precision_biases)
    )
    weights_changed = any(
        not jnp.allclose(i, f) for i, f in zip(initial_precision_weights, final_precision_weights)
    )
    assert biases_changed, "Precision biases should change during learning"
    assert weights_changed, "Precision weights should change during learning"


def test_precision_fixed_when_disabled(sample_data):
    """Precisions should not change when learn_precision=False."""
    net = pcn.PCNetwork(seed=0)
    with net:
        l1 = pcn.Layer(dim=16, label="input")
        l2 = pcn.Layer(dim=8, activation=pcn.Relu(), label="hidden")
        l3 = pcn.Layer(dim=4, label="output")

        pcn.Predict(l2, l1, learn_precision=False)
        pcn.Predict(l3, l2, learn_precision=False)

    net.build()

    sim = pcn.Simulation(net)
    initial_precision_biases = tuple(jnp.array(p) for p in sim.params.precision_biases)
    initial_precision_weights = tuple(jnp.array(p) for p in sim.params.precision_weights)

    sim.train(
        [sample_data],
        data_map={l1: 'input', l3: 'output'},
        epochs=1,
        iterations_per_sample=20,
    )

    final_precision_biases = tuple(jnp.array(p) for p in sim.params.precision_biases)
    final_precision_weights = tuple(jnp.array(p) for p in sim.params.precision_weights)
    biases_same = all(
        jnp.allclose(i, f) for i, f in zip(initial_precision_biases, final_precision_biases)
    )
    weights_same = all(
        jnp.allclose(i, f) for i, f in zip(initial_precision_weights, final_precision_weights)
    )
    assert biases_same, "Precision biases should NOT change when learn_precision=False"
    assert weights_same, "Precision weights should NOT change when learn_precision=False"



def test_learn_precision_bias_only(sample_data):
    """learn_precision_bias=True, learn_precision_weights=False: only bias should change."""
    net = pcn.PCNetwork(seed=0)
    with net:
        l1 = pcn.Layer(dim=16, label="input")
        l2 = pcn.Layer(dim=8, activation=pcn.Relu(), label="hidden")
        l3 = pcn.Layer(dim=4, label="output")
        pcn.Predict(l2, l1, learn_precision_weights=False, learn_precision_bias=True)
        pcn.Predict(l3, l2, learn_precision_weights=False, learn_precision_bias=True)
    net.build()

    sim = pcn.Simulation(net)
    initial_biases = tuple(jnp.array(p) for p in sim.params.precision_biases)
    initial_weights = tuple(jnp.array(p) for p in sim.params.precision_weights)

    sim.train([sample_data], data_map={l1: 'input', l3: 'output'},
              epochs=1, iterations_per_sample=20)

    final_biases = tuple(jnp.array(p) for p in sim.params.precision_biases)
    final_weights = tuple(jnp.array(p) for p in sim.params.precision_weights)

    assert any(not jnp.allclose(i, f) for i, f in zip(initial_biases, final_biases)), \
        "Precision biases should change when learn_precision_bias=True"
    assert all(jnp.allclose(i, f) for i, f in zip(initial_weights, final_weights)), \
        "Precision weights should NOT change when learn_precision_weights=False"



def test_learn_precision_config_override(sample_data):
    """net.config(learn_precision=False) should override the default."""
    net = pcn.PCNetwork(seed=0)
    net.config(learn_precision=False)
    with net:
        l1 = pcn.Layer(dim=16, label="input")
        l2 = pcn.Layer(dim=8, label="hidden")
        l3 = pcn.Layer(dim=4, label="output")
        p1 = pcn.Predict(l2, l1)
        p2 = pcn.Predict(l3, l2)
    net.build()

    assert p1.learn_precision_weights is False
    assert p1.learn_precision_bias is False
    assert p2.learn_precision_weights is False
    assert p2.learn_precision_bias is False

    sim = pcn.Simulation(net)
    initial_precision_biases = tuple(jnp.array(p) for p in sim.params.precision_biases)
    initial_precision_weights = tuple(jnp.array(p) for p in sim.params.precision_weights)

    sim.train(
        [sample_data],
        data_map={l1: 'input', l3: 'output'},
        epochs=1,
        iterations_per_sample=20,
    )

    final_precision_biases = tuple(jnp.array(p) for p in sim.params.precision_biases)
    final_precision_weights = tuple(jnp.array(p) for p in sim.params.precision_weights)
    biases_same = all(
        jnp.allclose(i, f) for i, f in zip(initial_precision_biases, final_precision_biases)
    )
    weights_same = all(
        jnp.allclose(i, f) for i, f in zip(initial_precision_weights, final_precision_weights)
    )
    assert biases_same, "Precision biases should NOT change when config sets learn_precision=False"
    assert weights_same, "Precision weights should NOT change when config sets learn_precision=False"


def test_precision_parameterization_softplus(sample_data):
    """Default softplus parameterization should produce positive precision and train stably."""
    net = pcn.PCNetwork(seed=0)
    with net:
        l1 = pcn.Layer(dim=16, label="input")
        l2 = pcn.Layer(dim=8, activation=pcn.Relu(), label="hidden")
        l3 = pcn.Layer(dim=4, label="output")
        p1 = pcn.Predict(l2, l1)
        p2 = pcn.Predict(l3, l2)
    net.build()

    # Default should be softplus
    assert p1.precision_param_type == Softplus.type_id
    assert p2.precision_param_type == Softplus.type_id

    sim = pcn.Simulation(net)
    sim.train(
        [sample_data],
        data_map={l1: 'input', l3: 'output'},
        epochs=1,
        iterations_per_sample=20,
    )

    # Precision biases should have changed
    assert any(
        not jnp.allclose(jnp.zeros_like(p), p)
        for p in sim.params.precision_biases
    ), "Precision biases should change with softplus param"

    # Energies should be finite
    assert all(jnp.isfinite(e[-1]) for e in sim.train_energies), \
        "All energies should be finite with softplus parameterization"


def test_precision_parameterization_exp(sample_data):
    """Explicit 'exp' parameterization should still work."""
    net = pcn.PCNetwork(seed=0)
    with net:
        l1 = pcn.Layer(dim=16, label="input")
        l2 = pcn.Layer(dim=8, activation=pcn.Relu(), label="hidden")
        l3 = pcn.Layer(dim=4, label="output")
        p1 = pcn.Predict(l2, l1, precision_parameterization='exp')
        p2 = pcn.Predict(l3, l2, precision_parameterization='exp')
    net.build()

    assert p1.precision_param_type == Exp.type_id
    assert p2.precision_param_type == Exp.type_id

    sim = pcn.Simulation(net)
    sim.train(
        [sample_data],
        data_map={l1: 'input', l3: 'output'},
        epochs=1,
        iterations_per_sample=20,
    )

    assert all(jnp.isfinite(e[-1]) for e in sim.train_energies), \
        "All energies should be finite with exp parameterization"


def test_precision_parameterization_config(sample_data):
    """net.config(precision_parameterization='exp') should propagate."""
    net = pcn.PCNetwork(seed=0)
    net.config(precision_parameterization='exp')
    with net:
        l1 = pcn.Layer(dim=16, label="input")
        l2 = pcn.Layer(dim=8, label="hidden")
        l3 = pcn.Layer(dim=4, label="output")
        p1 = pcn.Predict(l2, l1)
        p2 = pcn.Predict(l3, l2)
    net.build()

    assert p1.precision_param_type == Exp.type_id
    assert p2.precision_param_type == Exp.type_id


def test_precision_parameterization_per_connection(sample_data):
    """Per-connection override should take priority over config default."""
    net = pcn.PCNetwork(seed=0)
    with net:
        l1 = pcn.Layer(dim=16, label="input")
        l2 = pcn.Layer(dim=8, label="hidden")
        l3 = pcn.Layer(dim=4, label="output")
        p1 = pcn.Predict(l2, l1, precision_parameterization='exp')
        p2 = pcn.Predict(l3, l2)  # should get default (softplus)
    net.build()

    assert p1.precision_param_type == Exp.type_id, "Explicit exp should be Exp.type_id"
    assert p2.precision_param_type == Softplus.type_id, "Default should be softplus"


def test_precision_parameterization_linear(sample_data):
    """Linear parameterization: precision = identity (Direct activation)."""
    net = pcn.PCNetwork(seed=0)
    with net:
        l1 = pcn.Layer(dim=16, label="input")
        l2 = pcn.Layer(dim=8, activation=pcn.Relu(), label="hidden")
        l3 = pcn.Layer(dim=4, label="output")
        p1 = pcn.Predict(l2, l1, precision_parameterization='linear')
        p2 = pcn.Predict(l3, l2, precision_parameterization='linear')
    net.build()

    assert p1.precision_param_type == Direct.type_id
    assert p2.precision_param_type == Direct.type_id

    sim = pcn.Simulation(net)
    sim.train(
        [sample_data],
        data_map={l1: 'input', l3: 'output'},
        epochs=1,
        iterations_per_sample=20,
    )

    assert all(jnp.isfinite(e[-1]) for e in sim.train_energies), \
        "All energies should be finite with linear parameterization"


# ----------------------------------------------------------------------------
# Unified activation API: precision_activation, error_activation, NodeRef.activation
# ----------------------------------------------------------------------------

def _build_simple_net(error_activation=None, precision_activation=None):
    """Helper: 3-layer net with one knob on the input Predict conn."""
    net = pcn.PCNetwork(seed=0)
    kwargs = {}
    if error_activation is not None:
        kwargs['error_activation'] = error_activation
    if precision_activation is not None:
        kwargs['precision_activation'] = precision_activation
    with net:
        l1 = pcn.Layer(dim=16, label="input")
        l2 = pcn.Layer(dim=8, activation=pcn.Relu(), label="hidden")
        l3 = pcn.Layer(dim=4, label="output")
        p1 = pcn.Predict(l2, l1, **kwargs)
        p2 = pcn.Predict(l3, l2)
    net.build()
    return net, (l1, l2, l3), (p1, p2)


def test_precision_activation_kwarg_string():
    """precision_activation accepts a string, overrides default."""
    net, _, (p1, p2) = _build_simple_net(precision_activation='exp')
    assert p1.precision_activation_type == Exp.type_id
    assert p1.precision_param_type == Exp.type_id  # back-compat alias
    assert p2.precision_activation_type == Softplus.type_id  # default


def test_precision_activation_kwarg_instance():
    """precision_activation accepts an Activation instance."""
    net = pcn.PCNetwork(seed=0)
    with net:
        l1 = pcn.Layer(dim=8)
        l2 = pcn.Layer(dim=4)
        p = pcn.Predict(l2, l1, precision_activation=Exp())
    net.build()
    assert p.precision_activation_type == Exp.type_id
    assert isinstance(p.precision_activation, Exp)


def test_precision_activation_wins_over_legacy_alias():
    """If both kwargs are given, precision_activation takes priority."""
    net = pcn.PCNetwork(seed=0)
    with net:
        l1 = pcn.Layer(dim=8)
        l2 = pcn.Layer(dim=4)
        p = pcn.Predict(
            l2, l1,
            precision_activation='softplus',
            precision_parameterization='exp',
        )
    net.build()
    assert p.precision_activation_type == Softplus.type_id


def test_error_activation_default_is_direct():
    """Default error_activation is Direct (identity), preserving PC semantics."""
    net, _, (p1, p2) = _build_simple_net()
    assert p1.error_activation_type == Direct.type_id
    assert p2.error_activation_type == Direct.type_id
    assert isinstance(p1.error_activation, Direct)


def test_error_activation_kwarg_string():
    """error_activation='tanh' is stored on conn and propagated to spec."""
    net, _, (p1, p2) = _build_simple_net(error_activation='tanh')
    assert p1.error_activation_type == Tanh.type_id
    assert p2.error_activation_type == Direct.type_id

    spec = net.structure.predict_conns[0]
    assert spec.error_activation_type == Tanh.type_id
    assert net.structure.predict_conns[1].error_activation_type == Direct.type_id


def test_error_activation_kwarg_instance():
    """error_activation accepts an Activation instance."""
    net = pcn.PCNetwork(seed=0)
    with net:
        l1 = pcn.Layer(dim=8)
        l2 = pcn.Layer(dim=4)
        p = pcn.Predict(l2, l1, error_activation=Tanh())
    net.build()
    assert p.error_activation_type == Tanh.type_id
    assert isinstance(p.error_activation, Tanh)


def test_node_ref_activation_accessor():
    """NodeRef.activation returns the right slot's activation."""
    net = pcn.PCNetwork(seed=0)
    with net:
        l1 = pcn.Layer(dim=8, activation=pcn.Relu(), label="in")
        l2 = pcn.Layer(dim=4, label="out")
        p = pcn.Predict(l2, l1, error_activation='tanh',
                        precision_activation='exp')
    net.build()

    assert isinstance(l1.value.activation, pcn.Relu)
    assert isinstance(p.error.activation, Tanh)
    assert isinstance(p.precision.activation, Exp)

    # Flow nodes have no activation
    with pytest.raises(AttributeError):
        _ = p.flow_to_pre.activation


def _build_two_net_pair(error_activation):
    """Build two equivalent 3-layer nets, one identity-error, one with error_activation."""
    def _make(ea):
        net = pcn.PCNetwork(seed=0)
        with net:
            l1 = pcn.Layer(dim=16, label="input")
            l2 = pcn.Layer(dim=8, activation=pcn.Relu(), label="hidden")
            l3 = pcn.Layer(dim=4, label="output")
            kwargs = {} if ea is None else {'error_activation': ea}
            pcn.Predict(l2, l1, **kwargs)
            pcn.Predict(l3, l2, **kwargs)
        net.build()
        return net, (l1, l2, l3)
    return _make(None), _make(error_activation)


def test_error_activation_changes_runtime_error(sample_data):
    """A non-identity error_activation produces different logged errors than the default."""
    (net_id, layers_id), (net_tanh, layers_tanh) = _build_two_net_pair('tanh')

    sim_id = pcn.Simulation(net_id)
    sim_tanh = pcn.Simulation(net_tanh)

    res_id = sim_id.test(
        [sample_data],
        data_map={layers_id[0]: 'input', layers_id[-1]: 'output'},
        iterations_per_sample=5,
    )
    res_tanh = sim_tanh.test(
        [sample_data],
        data_map={layers_tanh[0]: 'input', layers_tanh[-1]: 'output'},
        iterations_per_sample=5,
    )

    # errors_log is per-batch -> per-Predict-conn -> shape (n_logged, batch, dim)
    e_id = res_id['errors'][0][1][-1]      # batch 0, conn 1 (top), last log step
    e_tanh = res_tanh['errors'][0][1][-1]
    assert not jnp.allclose(e_id, e_tanh), \
        "error_activation='tanh' should produce different errors than identity"

    # With tanh, errors are bounded in [-1, 1]
    assert jnp.all(jnp.abs(e_tanh) <= 1.0 + 1e-5)


def test_error_activation_direct_matches_default(sample_data):
    """Explicit error_activation='linear' must reproduce the default (identity) errors."""
    (net_def, layers_def), (net_dir, layers_dir) = _build_two_net_pair('linear')

    sim_def = pcn.Simulation(net_def)
    sim_dir = pcn.Simulation(net_dir)

    res_def = sim_def.test(
        [sample_data],
        data_map={layers_def[0]: 'input', layers_def[-1]: 'output'},
        iterations_per_sample=5,
    )
    res_dir = sim_dir.test(
        [sample_data],
        data_map={layers_dir[0]: 'input', layers_dir[-1]: 'output'},
        iterations_per_sample=5,
    )

    for conn_def, conn_dir in zip(res_def['errors'][0], res_dir['errors'][0]):
        assert jnp.allclose(conn_def, conn_dir, atol=1e-6)


# ----------------------------------------------------------------------------
# Memory-aware activations (Leaky)
# ----------------------------------------------------------------------------

def test_leaky_apply_first_step_uses_zero_prev():
    """At t=0 the backend supplies prev=0, so Leaky reduces to (1-leak)*base.fn(x)."""
    act = Leaky(Relu(), leak=0.3)
    x = jnp.array([[1.0, -2.0, 3.0]])
    prev = jnp.zeros_like(x)
    y = act.apply(x, prev)
    expected = 0.7 * jnp.array([[1.0, 0.0, 3.0]])
    assert jnp.allclose(y, expected, atol=1e-6)


def test_leaky_apply_blends_prev():
    """With prev != 0, output is convex combination."""
    act = Leaky(Relu(), leak=0.3)
    x = jnp.array([[1.0, -2.0, 3.0]])
    prev = jnp.array([[2.0, 4.0, 6.0]])
    y = act.apply(x, prev)
    expected = 0.3 * prev + 0.7 * jnp.array([[1.0, 0.0, 3.0]])
    assert jnp.allclose(y, expected, atol=1e-6)


def test_leaky_apply_stops_gradient_through_prev():
    """Gradients must not flow back through the prev argument (memory term)."""
    act = Leaky(Relu(), leak=0.5)

    def loss(x, prev):
        return jnp.sum(act.apply(x, prev))

    x = jnp.array([[1.0, 2.0]])
    prev = jnp.array([[10.0, 20.0]])
    g_x, g_prev = jax.grad(loss, argnums=(0, 1))(x, prev)
    # base = Relu, x > 0 -> derivative is 0.5 (1 - leak)
    assert jnp.allclose(g_x, 0.5 * jnp.ones_like(x))
    # prev must have zero gradient because of stop_gradient
    assert jnp.allclose(g_prev, jnp.zeros_like(prev))


def test_leaky_hashability_and_equality():
    """Leaky instances are hashable (needed for JIT static-arg caching)."""
    a = Leaky(Relu(), leak=0.3)
    b = Leaky(Relu(), leak=0.3)
    c = Leaky(Relu(), leak=0.4)
    assert hash(a) == hash(b)
    assert a == b
    assert a != c
    assert hash(a) != hash(c)


def test_leaky_rejects_nested_memory():
    with pytest.raises(ValueError):
        Leaky(Leaky(Relu()))


def test_leaky_rejects_bad_leak():
    with pytest.raises(ValueError):
        Leaky(Relu(), leak=-0.1)
    with pytest.raises(ValueError):
        Leaky(Relu(), leak=1.5)


def test_predict_accepts_memory_error_activation():
    """A Predict conn accepts Leaky as error_activation."""
    net = pcn.PCNetwork(seed=0)
    with net:
        l1 = pcn.Layer(dim=8)
        l2 = pcn.Layer(dim=4)
        p = pcn.Predict(l2, l1, error_activation=Leaky(Tanh(), leak=0.2))
    net.build()
    assert isinstance(p.error_activation, Leaky)
    assert p.error_activation.has_memory
    assert p.error_activation.leak == 0.2
    # Cached on network for backend dispatch
    assert net.predict_error_activations[0] is p.error_activation
    # NodeRef accessor returns the same instance
    assert p.error.activation is p.error_activation


def test_memory_error_activation_runs_end_to_end(sample_data):
    """Inference with a memory error_activation runs and produces finite, bounded errors."""
    net = pcn.PCNetwork(seed=0)
    with net:
        l1 = pcn.Layer(dim=16, label="input")
        l2 = pcn.Layer(dim=8, activation=pcn.Relu(), label="hidden")
        l3 = pcn.Layer(dim=4, label="output")
        pcn.Predict(l2, l1, error_activation=Leaky(Tanh(), leak=0.5))
        pcn.Predict(l3, l2, error_activation=Leaky(Tanh(), leak=0.5))
    net.build()

    sim = pcn.Simulation(net)
    res = sim.test(
        [sample_data],
        data_map={net._layers[0]: 'input', net._layers[-1]: 'output'},
        iterations_per_sample=5,
    )
    # All logged errors should be finite and bounded by tanh range (1.0)
    for conn_errs in res['errors'][0]:
        assert jnp.all(jnp.isfinite(conn_errs))
        assert jnp.all(jnp.abs(conn_errs) <= 1.0 + 1e-5)


def test_memory_leak_zero_matches_base(sample_data):
    """Leaky(base, leak=0) should reproduce the base activation's behavior."""
    def _build(error_act):
        net = pcn.PCNetwork(seed=0)
        with net:
            l1 = pcn.Layer(dim=16, label="input")
            l2 = pcn.Layer(dim=8, activation=pcn.Relu(), label="hidden")
            l3 = pcn.Layer(dim=4, label="output")
            pcn.Predict(l2, l1, error_activation=error_act)
            pcn.Predict(l3, l2, error_activation=error_act)
        net.build()
        return net

    net_base = _build(Tanh())
    net_leaky0 = _build(Leaky(Tanh(), leak=0.0))

    sim_base = pcn.Simulation(net_base)
    sim_leaky = pcn.Simulation(net_leaky0)
    res_base = sim_base.test(
        [sample_data],
        data_map={net_base._layers[0]: 'input', net_base._layers[-1]: 'output'},
        iterations_per_sample=5,
    )
    res_leaky = sim_leaky.test(
        [sample_data],
        data_map={net_leaky0._layers[0]: 'input', net_leaky0._layers[-1]: 'output'},
        iterations_per_sample=5,
    )
    for e_base, e_leaky in zip(res_base['errors'][0], res_leaky['errors'][0]):
        assert jnp.allclose(e_base, e_leaky, atol=1e-5)


def test_memory_leak_nonzero_differs_from_base(sample_data):
    """Leaky with leak>0 should produce different errors than the bare base."""
    def _build(error_act):
        net = pcn.PCNetwork(seed=0)
        with net:
            l1 = pcn.Layer(dim=16, label="input")
            l2 = pcn.Layer(dim=8, activation=pcn.Relu(), label="hidden")
            l3 = pcn.Layer(dim=4, label="output")
            pcn.Predict(l2, l1, error_activation=error_act)
            pcn.Predict(l3, l2, error_activation=error_act)
        net.build()
        return net

    net_base = _build(Tanh())
    net_leaky = _build(Leaky(Tanh(), leak=0.5))

    sim_base = pcn.Simulation(net_base)
    sim_leaky = pcn.Simulation(net_leaky)
    res_base = sim_base.test(
        [sample_data],
        data_map={net_base._layers[0]: 'input', net_base._layers[-1]: 'output'},
        iterations_per_sample=10,
    )
    res_leaky = sim_leaky.test(
        [sample_data],
        data_map={net_leaky._layers[0]: 'input', net_leaky._layers[-1]: 'output'},
        iterations_per_sample=10,
    )
    # At least one conn should differ
    differs = any(
        not jnp.allclose(eb, el, atol=1e-4)
        for eb, el in zip(res_base['errors'][0], res_leaky['errors'][0])
    )
    assert differs, "Leaky with leak=0.5 should yield different errors than bare Tanh"


def test_memory_error_activation_trains_stably(sample_data):
    """A memory error_activation trains for one epoch without NaNs."""
    net = pcn.PCNetwork(seed=0)
    with net:
        l1 = pcn.Layer(dim=16, label="input")
        l2 = pcn.Layer(dim=8, activation=pcn.Relu(), label="hidden")
        l3 = pcn.Layer(dim=4, label="output")
        pcn.Predict(l2, l1, error_activation=Leaky(Tanh(), leak=0.3))
        pcn.Predict(l3, l2, error_activation=Leaky(Tanh(), leak=0.3))
    net.build()

    sim = pcn.Simulation(net)
    sim.train(
        [sample_data],
        data_map={net._layers[0]: 'input', net._layers[-1]: 'output'},
        epochs=1,
        iterations_per_sample=10,
    )
    for e in sim.train_energies:
        assert jnp.all(jnp.isfinite(e))


# ----------------------------------------------------------------------------
# Memory-aware precision activations
# ----------------------------------------------------------------------------

def test_predict_accepts_memory_precision_activation():
    """A Predict conn accepts Leaky as precision_activation."""
    net = pcn.PCNetwork(seed=0)
    with net:
        l1 = pcn.Layer(dim=8)
        l2 = pcn.Layer(dim=4)
        p = pcn.Predict(l2, l1, precision_activation=Leaky(Softplus(), leak=0.2))
    net.build()
    assert isinstance(p.precision_activation, Leaky)
    assert p.precision_activation.has_memory
    assert p.precision_activation.leak == 0.2
    # Type-id mirrors the base so init still uses softplus weight-init metadata
    assert p.precision_activation_type == Softplus.type_id
    assert net.predict_precision_activations[0] is p.precision_activation
    assert p.precision.activation is p.precision_activation


def test_memory_precision_activation_runs_end_to_end(sample_data):
    """Inference with a memory precision_activation produces finite, positive precisions."""
    net = pcn.PCNetwork(seed=0)
    with net:
        l1 = pcn.Layer(dim=16, label="input")
        l2 = pcn.Layer(dim=8, activation=pcn.Relu(), label="hidden")
        l3 = pcn.Layer(dim=4, label="output")
        pcn.Predict(l2, l1, precision_activation=Leaky(Softplus(), leak=0.4))
        pcn.Predict(l3, l2, precision_activation=Leaky(Softplus(), leak=0.4))
    net.build()

    sim = pcn.Simulation(net)
    res = sim.test(
        [sample_data],
        data_map={net._layers[0]: 'input', net._layers[-1]: 'output'},
        iterations_per_sample=5,
    )
    for conn_precs in res['precisions'][0]:
        assert jnp.all(jnp.isfinite(conn_precs))
        # Softplus output is positive; leaky-mix of positives stays >= 0
        assert jnp.all(conn_precs >= -1e-6)


def test_memory_precision_leak_zero_matches_base(sample_data):
    """Leaky(Softplus, leak=0) on precision should match the bare Softplus path."""
    def _build(prec_act):
        net = pcn.PCNetwork(seed=0)
        with net:
            l1 = pcn.Layer(dim=16, label="input")
            l2 = pcn.Layer(dim=8, activation=pcn.Relu(), label="hidden")
            l3 = pcn.Layer(dim=4, label="output")
            pcn.Predict(l2, l1, precision_activation=prec_act)
            pcn.Predict(l3, l2, precision_activation=prec_act)
        net.build()
        return net

    net_base = _build(Softplus())
    net_leaky0 = _build(Leaky(Softplus(), leak=0.0))

    sim_base = pcn.Simulation(net_base)
    sim_leaky = pcn.Simulation(net_leaky0)
    res_base = sim_base.test(
        [sample_data],
        data_map={net_base._layers[0]: 'input', net_base._layers[-1]: 'output'},
        iterations_per_sample=5,
    )
    res_leaky = sim_leaky.test(
        [sample_data],
        data_map={net_leaky0._layers[0]: 'input', net_leaky0._layers[-1]: 'output'},
        iterations_per_sample=5,
    )
    for p_base, p_leaky in zip(res_base['precisions'][0], res_leaky['precisions'][0]):
        assert jnp.allclose(p_base, p_leaky, atol=1e-5)


def test_memory_precision_leak_nonzero_differs_from_base(sample_data):
    """Leaky precision with leak>0 should diverge from the base Softplus over iterations."""
    def _build(prec_act):
        net = pcn.PCNetwork(seed=0)
        with net:
            l1 = pcn.Layer(dim=16, label="input")
            l2 = pcn.Layer(dim=8, activation=pcn.Relu(), label="hidden")
            l3 = pcn.Layer(dim=4, label="output")
            pcn.Predict(l2, l1, precision_activation=prec_act)
            pcn.Predict(l3, l2, precision_activation=prec_act)
        net.build()
        return net

    net_base = _build(Softplus())
    net_leaky = _build(Leaky(Softplus(), leak=0.5))

    sim_base = pcn.Simulation(net_base)
    sim_leaky = pcn.Simulation(net_leaky)
    res_base = sim_base.test(
        [sample_data],
        data_map={net_base._layers[0]: 'input', net_base._layers[-1]: 'output'},
        iterations_per_sample=10,
    )
    res_leaky = sim_leaky.test(
        [sample_data],
        data_map={net_leaky._layers[0]: 'input', net_leaky._layers[-1]: 'output'},
        iterations_per_sample=10,
    )
    differs = any(
        not jnp.allclose(pb, pl, atol=1e-4)
        for pb, pl in zip(res_base['precisions'][0], res_leaky['precisions'][0])
    )
    assert differs, "Leaky precision with leak=0.5 should yield different precisions than bare Softplus"


def test_memory_precision_activation_trains_stably(sample_data):
    """A memory precision_activation trains for one epoch without NaNs."""
    net = pcn.PCNetwork(seed=0)
    with net:
        l1 = pcn.Layer(dim=16, label="input")
        l2 = pcn.Layer(dim=8, activation=pcn.Relu(), label="hidden")
        l3 = pcn.Layer(dim=4, label="output")
        pcn.Predict(l2, l1, precision_activation=Leaky(Softplus(), leak=0.3))
        pcn.Predict(l3, l2, precision_activation=Leaky(Softplus(), leak=0.3))
    net.build()

    sim = pcn.Simulation(net)
    sim.train(
        [sample_data],
        data_map={net._layers[0]: 'input', net._layers[-1]: 'output'},
        epochs=1,
        iterations_per_sample=10,
    )
    for e in sim.train_energies:
        assert jnp.all(jnp.isfinite(e))


def test_memory_both_slots_compose(sample_data):
    """Memory on both error and precision slots run together without issue."""
    net = pcn.PCNetwork(seed=0)
    with net:
        l1 = pcn.Layer(dim=16, label="input")
        l2 = pcn.Layer(dim=8, activation=pcn.Relu(), label="hidden")
        l3 = pcn.Layer(dim=4, label="output")
        pcn.Predict(l2, l1,
                    error_activation=Leaky(Tanh(), leak=0.3),
                    precision_activation=Leaky(Softplus(), leak=0.3))
        pcn.Predict(l3, l2,
                    error_activation=Leaky(Tanh(), leak=0.3),
                    precision_activation=Leaky(Softplus(), leak=0.3))
    net.build()

    sim = pcn.Simulation(net)
    res = sim.test(
        [sample_data],
        data_map={net._layers[0]: 'input', net._layers[-1]: 'output'},
        iterations_per_sample=5,
    )
    for conn_errs in res['errors'][0]:
        assert jnp.all(jnp.isfinite(conn_errs))
    for conn_precs in res['precisions'][0]:
        assert jnp.all(jnp.isfinite(conn_precs))


# ----------------------------------------------------------------------------
# Stochastic (noise-injecting) activations
# ----------------------------------------------------------------------------

def test_stochastic_defaults_to_unit_gaussian():
    """Default Stochastic wraps Direct with the gaussian_noise sampler."""
    act = Stochastic()
    assert isinstance(act, StochasticActivation)
    assert act.needs_key
    assert not act.has_memory
    assert isinstance(act.base, Direct)
    assert act.noise_fn is gaussian_noise
    assert act.noise_params == ()  # sigma defaults inside gaussian_noise
    assert act.type_id == Direct().type_id


def test_stochastic_apply_no_key_is_deterministic_base():
    """With key=None (key-free backend paths) the output is the noise-free base."""
    act = Stochastic(Relu(), sigma=5.0)
    x = jnp.array([[1.0, -2.0, 3.0]])
    y = act.apply(x, key=None)
    assert jnp.allclose(y, jnp.array([[1.0, 0.0, 3.0]]))


def test_stochastic_apply_with_key_adds_noise():
    """A key produces a noisy output that differs from the base and across keys."""
    act = Stochastic(Tanh(), sigma=1.0)
    x = jnp.zeros((4, 8))
    base = jnp.tanh(x)
    y1 = act.apply(x, key=jax.random.PRNGKey(0))
    y2 = act.apply(x, key=jax.random.PRNGKey(1))
    assert not jnp.allclose(y1, base)
    assert not jnp.allclose(y1, y2)
    # Empirically, mean noise ~ 0 and std ~ sigma for the default Gaussian.
    big = Stochastic(Direct(), sigma=2.0).apply(
        jnp.zeros((2048, 4)), key=jax.random.PRNGKey(7))
    assert abs(float(jnp.mean(big))) < 0.1
    assert abs(float(jnp.std(big)) - 2.0) < 0.15


def test_stochastic_custom_noise_fn_and_params():
    """A user-supplied noise_fn and its kwargs are honoured."""
    def laplace(key, shape, scale=1.0):
        return scale * jax.random.laplace(key, shape)

    act = Stochastic(Direct(), noise_fn=laplace, scale=3.0)
    assert act.noise_fn is laplace
    assert dict(act.noise_params) == {'scale': 3.0}
    samples = act.apply(jnp.zeros((4096, 2)), key=jax.random.PRNGKey(3))
    # Laplace(scale=b) has std = b*sqrt(2).
    assert abs(float(jnp.std(samples)) - 3.0 * (2 ** 0.5)) < 0.3


def test_stochastic_noise_is_additive_constant_for_grad():
    """Noise is independent of x, so d/dx(base.fn(x)+noise) == base.fn'(x)."""
    act = Stochastic(Relu(), sigma=10.0)
    key = jax.random.PRNGKey(0)

    def loss(x):
        return jnp.sum(act.fn(x, key))

    x = jnp.array([[1.0, 2.0, -1.0]])
    g = jax.grad(loss)(x)
    # Relu'(x): 1 where x>0 else 0 — unaffected by the additive noise term.
    assert jnp.allclose(g, jnp.array([[1.0, 1.0, 0.0]]))


def test_stochastic_hashability_and_equality():
    """Stochastic instances are hashable (needed for JIT static-arg caching)."""
    a = Stochastic(Relu(), sigma=0.5)
    b = Stochastic(Relu(), sigma=0.5)
    c = Stochastic(Relu(), sigma=0.9)
    assert hash(a) == hash(b)
    assert a == b
    assert a != c
    assert hash(a) != hash(c)


def test_stochastic_rejects_nested_memory_or_stochastic():
    with pytest.raises(ValueError):
        Stochastic(Stochastic())
    with pytest.raises(ValueError):
        Stochastic(Leaky(Relu()))


def test_layer_stochastic_runs_and_injects_noise(sample_data):
    """A Stochastic layer activation runs end-to-end; noise reaches free layers
    but never the clamped layers, and is reproducible for a fixed net seed."""
    def _build():
        net = pcn.PCNetwork(seed=0)
        with net:
            l1 = pcn.Layer(dim=16, label="input")
            l2 = pcn.Layer(dim=8, activation=Stochastic(Tanh(), sigma=0.5),
                           label="hidden")
            l3 = pcn.Layer(dim=4, label="output")
            pcn.Predict(l1, l2)
            pcn.Predict(l2, l3)
        net.build()
        return net

    net = _build()
    assert [a.needs_key for a in net.layer_activations] == [False, True, False]

    sim = pcn.Simulation(net)
    dm = {net._layers[0]: 'input'}
    ra = sim.test([sample_data], data_map=dm, iterations_per_sample=15)
    rb = sim.test([sample_data], data_map=dm, iterations_per_sample=15)
    # values is indexed [layer] -> (batch, n_logs, dim)
    assert jnp.allclose(ra['values'][0], rb['values'][0])      # clamped input
    assert not jnp.allclose(ra['values'][1], rb['values'][1])  # free hidden
    assert jnp.all(jnp.isfinite(ra['values'][1]))

    # Reproducible: a fresh Simulation (same net seed) repeats the first run.
    sim2 = pcn.Simulation(_build())
    rc = sim2.test([sample_data], data_map=dm, iterations_per_sample=15)
    assert jnp.allclose(ra['values'][1], rc['values'][1])


def test_error_stochastic_runs_and_injects_noise(sample_data):
    """Stochastic as a Predict error_activation runs and perturbs free layers."""
    net = pcn.PCNetwork(seed=0)
    with net:
        l1 = pcn.Layer(dim=16, label="input")
        l2 = pcn.Layer(dim=8, activation=pcn.Relu(), label="hidden")
        l3 = pcn.Layer(dim=4, label="output")
        pcn.Predict(l2, l1, error_activation=Stochastic(Direct(), sigma=0.3))
        pcn.Predict(l3, l2, error_activation=Stochastic(Direct(), sigma=0.3))
    net.build()

    assert all(a.needs_key for a in net.predict_error_activations)

    sim = pcn.Simulation(net)
    dm = {net._layers[0]: 'input'}
    ra = sim.test([sample_data], data_map=dm, iterations_per_sample=15)
    rb = sim.test([sample_data], data_map=dm, iterations_per_sample=15)
    assert not jnp.allclose(ra['values'][1], rb['values'][1])
    for conn_errs in ra['errors'][0]:
        assert jnp.all(jnp.isfinite(conn_errs))


def test_stochastic_trains_stably(sample_data):
    """A network with stochastic layer + error activations trains without NaNs."""
    net = pcn.PCNetwork(seed=0)
    with net:
        l1 = pcn.Layer(dim=16, label="input")
        l2 = pcn.Layer(dim=8, activation=Stochastic(Tanh(), sigma=0.2),
                       label="hidden")
        l3 = pcn.Layer(dim=4, label="output")
        pcn.Predict(l1, l2)
        pcn.Predict(l2, l3, error_activation=Stochastic(Direct(), sigma=0.1))
    net.build()

    sim = pcn.Simulation(net)
    sim.train(
        [sample_data],
        data_map={net._layers[0]: 'input', net._layers[-1]: 'output'},
        epochs=1,
        iterations_per_sample=10,
    )
    for e in sim.train_energies:
        assert jnp.all(jnp.isfinite(e))
