"""Tests for Predict(precision_input=...) — precision keyed on arbitrary sources.

Semantics under test (see implementation.md):
  - default (None): precision keyed on the conn's own pre activation,
    bit-compatible with the historical behaviour
  - value sources: current iteration's activated layer value, live in the
    energy gradient (mirrors the default pre read)
  - error/precision sources: previous iteration's carried arrays (Jacobi
    rule) — acyclic, including a conn reading its own error
"""

import jax
import jax.numpy as jnp
import pytest

import pcn


@pytest.fixture
def sample_data():
    key = jax.random.PRNGKey(7)
    k1, k2, k3 = jax.random.split(key, 3)
    return {
        'input': jax.random.normal(k1, (4, 16)),
        'output': jax.random.normal(k2, (4, 4)),
        'ctx': jax.random.normal(k3, (4, 6)),
    }


def _softplus(x):
    return jnp.logaddexp(x, 0.0)


def test_default_spec_fields_empty():
    """Without precision_input the spec fields stay empty and shapes unchanged."""
    net = pcn.PCNetwork(seed=0)
    with net:
        l1 = pcn.Layer(dim=16, label="input")
        l2 = pcn.Layer(dim=4, label="output")
        pcn.Predict(l1, l2)
    net.build()

    spec = net.structure.predict_conns[0]
    assert spec.precision_input_idx == ()
    assert spec.precision_input_node_types == ()
    assert net.params.precision_weights[0].shape == (4, 16)


def test_precision_input_other_layer_shapes():
    """precision_input=<layer> replaces pre: ppw is (post_dim, src_dim)."""
    net = pcn.PCNetwork(seed=0)
    with net:
        l1 = pcn.Layer(dim=16, label="input")
        l2 = pcn.Layer(dim=4, label="output")
        l_ctx = pcn.Layer(dim=6, label="ctx")
        pcn.Predict(l1, l2, precision_input=l_ctx)
    net.build()

    spec = net.structure.predict_conns[0]
    assert spec.precision_input_idx == (l_ctx._idx,)
    assert spec.precision_input_node_types == (0,)
    assert net.params.precision_weights[0].shape == (4, 6)


def test_precision_tracks_source_not_pre(sample_data):
    """With nonzero ppw, precision varies with the source layer, not with pre."""
    net = pcn.PCNetwork(seed=0)
    with net:
        l1 = pcn.Layer(dim=16, activation=pcn.Direct(), label="input")
        l2 = pcn.Layer(dim=4, label="output")
        l_ctx = pcn.Layer(dim=6, activation=pcn.Direct(), label="ctx")
        pcn.Predict(l1, l2, precision_input=l_ctx)
    net.build()

    sim = pcn.Simulation(net)
    sim.params.precision_weights[0] = jnp.ones((4, 6)) * 0.5

    def run(sample):
        results = sim.test(
            [sample], data_map={l1: 'input', l2: 'output', l_ctx: 'ctx'},
            iterations_per_sample=1, return_logs=True)
        # conn 0, all samples, last log: (batch, 4)
        return jnp.asarray(results['precisions'][0])[:, -1, :]

    base = dict(sample_data)
    prec_base = run(base)

    diff_ctx = dict(base, ctx=base['ctx'] + 1.0)
    prec_diff_ctx = run(diff_ctx)
    assert not jnp.allclose(prec_base, prec_diff_ctx), \
        "precision should depend on the precision_input source"

    diff_pre = dict(base, input=base['input'] + 1.0)
    prec_diff_pre = run(diff_pre)
    assert jnp.allclose(prec_base, prec_diff_pre, atol=1e-5), \
        "precision should NOT depend on the conn's pre once replaced"


def test_precision_input_multi_source_mixed(sample_data):
    """Mixed [layer, other_conn.error] sources concatenate correctly and run."""
    net = pcn.PCNetwork(seed=0)
    with net:
        l1 = pcn.Layer(dim=16, label="input")
        l2 = pcn.Layer(dim=8, activation=pcn.Relu(), label="hidden")
        l3 = pcn.Layer(dim=4, label="output")
        p1 = pcn.Predict(l2, l1)
        pcn.Predict(l3, l2, precision_input=[l3, p1.error])
    net.build()

    # src_dim = l3.dim + p1.post_dim = 4 + 16
    assert net.params.precision_weights[1].shape == (8, 20)

    sim = pcn.Simulation(net)
    sim.train([sample_data], data_map={l1: 'input', l3: 'output'},
              epochs=1, iterations_per_sample=10)
    results = sim.test([sample_data], data_map={l1: 'input', l3: 'output'},
                       iterations_per_sample=5, return_logs=True)
    for prec in results['precisions']:  # list over conns
        assert jnp.all(jnp.isfinite(prec))
        assert jnp.all(prec > 0)


def test_precision_learning_moves_custom_weights(sample_data):
    """Energy-gradient learning updates ppw of a custom-source precision."""
    net = pcn.PCNetwork(seed=0)
    with net:
        l1 = pcn.Layer(dim=16, label="input")
        l2 = pcn.Layer(dim=4, label="output")
        l_ctx = pcn.Layer(dim=6, label="ctx")
        pcn.Predict(l1, l2, precision_input=l_ctx)
    net.build()

    sim = pcn.Simulation(net)
    initial = jnp.array(sim.params.precision_weights[0])
    sim.train([sample_data],
              data_map={l1: 'input', l2: 'output', l_ctx: 'ctx'},
              epochs=2, iterations_per_sample=20)
    final = jnp.array(sim.params.precision_weights[0])
    assert not jnp.allclose(initial, final), \
        "custom-source precision weights should learn from the energy"


def test_precision_input_self_error(sample_data):
    """A conn reading its own error (assigned post-construction) builds and
    responds to the previous iteration's error magnitude."""
    net = pcn.PCNetwork(seed=0)
    with net:
        l1 = pcn.Layer(dim=16, label="input")
        l2 = pcn.Layer(dim=4, label="output")
        p = pcn.Predict(l1, l2)
        p.precision_input = [p.error]
    net.build()

    assert net.params.precision_weights[0].shape == (4, 4)
    spec = net.structure.predict_conns[0]
    assert spec.precision_input_idx == (0,)
    assert spec.precision_input_node_types == (1,)

    sim = pcn.Simulation(net)
    # Negative weights: large previous error -> lower precision (adaptive).
    sim.params.precision_weights[0] = -0.5 * jnp.eye(4)
    results = sim.test([sample_data], data_map={l1: 'input', l2: 'output'},
                       iterations_per_sample=3, return_logs=True)
    prec = jnp.asarray(results['precisions'][0])[:, -1, :]
    assert jnp.all(jnp.isfinite(prec))
    # ppw != 0 and carried errors are nonzero, so precision must deviate
    # from the bias-only init value (softplus-inverted init_precision = 1.0).
    assert not jnp.allclose(prec, jnp.ones_like(prec), atol=1e-4)


def test_precision_input_precision_source(sample_data):
    """precision_input=<other conn's precision> exercises the carry-init path."""
    net = pcn.PCNetwork(seed=0)
    with net:
        l1 = pcn.Layer(dim=16, label="input")
        l2 = pcn.Layer(dim=8, activation=pcn.Relu(), label="hidden")
        l3 = pcn.Layer(dim=4, label="output")
        p1 = pcn.Predict(l2, l1)
        pcn.Predict(l3, l2, precision_input=p1.precision)
    net.build()

    # src_dim = p1.post_dim (p1 has learned per-dim precision)
    assert net.params.precision_weights[1].shape == (8, 16)

    sim = pcn.Simulation(net)
    results = sim.test([sample_data], data_map={l1: 'input', l3: 'output'},
                       iterations_per_sample=3, return_logs=True)
    for prec in results['precisions']:  # list over conns
        assert jnp.all(jnp.isfinite(prec))
        assert jnp.all(prec > 0)


def test_precision_input_static_precision_dim():
    """A static-precision source conn carries (batch, 1); dim resolves to 1."""
    net = pcn.PCNetwork(seed=0)
    with net:
        l1 = pcn.Layer(dim=16, label="input")
        l2 = pcn.Layer(dim=4, label="output")
        p1 = pcn.Predict(l1, l2, learn_precision=False)
        pcn.Predict(l2, l1, precision_input=p1.precision)
    net.build()

    assert net.params.precision_weights[0].shape == (1, 16)  # static conn
    assert net.params.precision_weights[1].shape == (16, 1)  # keyed on (batch,1)


def test_static_precision_with_custom_source(sample_data):
    """learn flags off + custom source: precision stays at init_precision."""
    net = pcn.PCNetwork(seed=0)
    with net:
        l1 = pcn.Layer(dim=16, label="input")
        l2 = pcn.Layer(dim=4, label="output")
        l_ctx = pcn.Layer(dim=6, label="ctx")
        pcn.Predict(l1, l2, precision_input=l_ctx, learn_precision=False,
                    init_precision=2.0)
    net.build()

    assert net.params.precision_weights[0].shape == (1, 6)
    sim = pcn.Simulation(net)
    results = sim.test([sample_data],
                       data_map={l1: 'input', l2: 'output', l_ctx: 'ctx'},
                       iterations_per_sample=3, return_logs=True)
    prec = jnp.asarray(results['precisions'][0])
    assert jnp.allclose(prec, 2.0, atol=1e-5)


def test_value_source_gradient_is_live():
    """The energy gradient flows through a value precision source: a free
    layer referenced ONLY via precision_input moves during inference."""
    net = pcn.PCNetwork(seed=0)
    with net:
        l1 = pcn.Layer(dim=8, label="input")
        l2 = pcn.Layer(dim=4, label="output")
        l_ctx = pcn.Layer(dim=6, activation=pcn.Direct(), label="ctx")
        pcn.Predict(l1, l2, precision_input=l_ctx)
    net.build()

    sim = pcn.Simulation(net)
    sim.params.precision_weights[0] = jnp.ones((4, 6)) * 0.5

    key = jax.random.PRNGKey(3)
    sample = {
        'input': jax.random.normal(key, (2, 8)),
        'output': jax.random.normal(jax.random.fold_in(key, 1), (2, 4)),
    }
    results = sim.test([sample], data_map={l1: 'input', l2: 'output'},
                       iterations_per_sample=5, return_logs=True)
    # results['values'] is a list over layers: (batch, n_logged, dim)
    ctx_values = jnp.asarray(results['values'][l_ctx._idx])[:, -1, :]
    # l_ctx is unclamped, predicted by nothing, and read only by the
    # precision function — any movement comes from the live precision read.
    assert not jnp.allclose(ctx_values, jnp.zeros_like(ctx_values)), \
        "live precision read should drive the source layer's inference"


def test_precision_input_save_load(tmp_path, sample_data):
    """Roundtrip preserves the non-default precision weight shapes/values."""
    import numpy as np

    def build():
        net = pcn.PCNetwork(seed=0)
        with net:
            l1 = pcn.Layer(dim=16, label="input")
            l2 = pcn.Layer(dim=4, label="output")
            l_ctx = pcn.Layer(dim=6, label="ctx")
            pcn.Predict(l1, l2, precision_input=l_ctx)
        net.build()
        return net, l1, l2, l_ctx

    net, l1, l2, l_ctx = build()
    sim = pcn.Simulation(net)
    sim.train([sample_data],
              data_map={l1: 'input', l2: 'output', l_ctx: 'ctx'},
              epochs=1, iterations_per_sample=10)
    path = str(tmp_path / "precinput.h5")
    net.save(path)
    saved_ppw = np.array(net.params.precision_weights[0])

    net2, *_ = build()
    net2.load(path)
    assert net2.params.precision_weights[0].shape == (4, 6)
    assert np.allclose(np.array(net2.params.precision_weights[0]), saved_ppw)


def test_stochastic_with_custom_source(sample_data):
    """is_stochastic noise path consumes the custom precision source."""
    net = pcn.PCNetwork(seed=0)
    with net:
        l1 = pcn.Layer(dim=16, label="input")
        l2 = pcn.Layer(dim=4, label="output")
        l_ctx = pcn.Layer(dim=6, label="ctx")
        pcn.Predict(l1, l2, precision_input=l_ctx)
    net.build()

    sim = pcn.Simulation(net)
    results = sim.test([sample_data],
                       data_map={l1: 'input', l2: 'output', l_ctx: 'ctx'},
                       iterations_per_sample=3, is_stochastic=True, return_logs=True)
    for prec in results['precisions']:  # list over conns
        assert jnp.all(jnp.isfinite(prec))


def test_precision_input_validation():
    """Flow nodes, empty lists, and bad types are rejected."""
    net = pcn.PCNetwork(seed=0)
    with net:
        l1 = pcn.Layer(dim=16, label="input")
        l2 = pcn.Layer(dim=4, label="output")
        p1 = pcn.Predict(l1, l2)

        with pytest.raises(ValueError, match="flow nodes"):
            pcn.Predict(l2, l1, precision_input=p1.flow_to_pre)
        with pytest.raises(ValueError, match="empty"):
            pcn.Predict(l2, l1, precision_input=[])
        with pytest.raises(TypeError):
            pcn.Predict(l2, l1, precision_input="ctx")
