"""Tests for Predict(init_precision_weights=...) — hand-set W_rho.

The precision analogue of ``init_weight``: seeds the weights of the
log-precision function ``g(W_rho . src + b_rho)``. Combined with
``learn_precision_weights=False`` it *fixes* a designed precision path, e.g.
keying precision on a clamped control layer so the precision can be steered at
test time without retraining (see experiments.md exp-...-precgain).

Semantics under test:
  - default (None) keeps the historical zeros init -> precision == init_precision
  - scalar / 1-D / (1, pin) / (rows, pin) inits are broadcast to (rows, pin)
  - the shape check is against the RESOLVED precision input dim, so weights
    sized for the pre raise when precision_input keys precision elsewhere
  - a frozen hand-set W_rho survives training and is NOT folded out by the
    unit-precision fast path
"""

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

import pcn


def _net_with(**predict_kwargs):
    """Single Predict(16 -> 4) net, built, with the given Predict kwargs."""
    net = pcn.PCNetwork(seed=0)
    with net:
        l1 = pcn.Layer(dim=16, label="input")
        l2 = pcn.Layer(dim=4, label="output")
        p = pcn.Predict(l1, l2, **predict_kwargs)
    net.build()
    return net, l1, l2, p


# --------------------------------------------------------------------------- #
#  Defaults unchanged                                                          #
# --------------------------------------------------------------------------- #
def test_default_is_zeros():
    """No init_precision_weights -> zeros, so precision == init_precision."""
    net, _, _, _ = _net_with(init_precision=3.0, precision_activation="exp",
                             learn_precision_weights=False,
                             learn_precision_bias=False)
    pw = net.params.precision_weights[0]
    assert pw.shape == (1, 16)
    assert jnp.all(pw == 0.0)
    # bias is the exp-inverse of init_precision
    assert float(jnp.exp(net.params.precision_biases[0][0])) == pytest.approx(3.0)


# --------------------------------------------------------------------------- #
#  Broadcast forms                                                             #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("value,rows", [
    (0.5, 1),                       # scalar
    (np.full(16, 0.5), 1),          # (pin,)
    (np.full((1, 16), 0.5), 1),     # (1, pin)
])
def test_broadcast_forms_frozen(value, rows):
    """Scalar / 1-D / row inits all broadcast to (rows, pin_dim)."""
    net, _, _, _ = _net_with(init_precision_weights=value,
                             learn_precision_weights=False,
                             learn_precision_bias=False)
    pw = net.params.precision_weights[0]
    assert pw.shape == (rows, 16)
    assert jnp.allclose(pw, 0.5)


def test_broadcast_to_post_rows_when_learning():
    """With a learn flag on, rows == post_dim and a (1, pin) init is tiled."""
    net, _, _, _ = _net_with(init_precision_weights=np.full((1, 16), 0.25),
                             learn_precision_weights=True)
    pw = net.params.precision_weights[0]
    assert pw.shape == (4, 16)
    assert jnp.allclose(pw, 0.25)


def test_full_matrix_preserved():
    """A full (rows, pin) matrix is used verbatim."""
    W = np.arange(4 * 16, dtype=np.float32).reshape(4, 16)
    net, _, _, _ = _net_with(init_precision_weights=W,
                             learn_precision_weights=True)
    assert jnp.allclose(net.params.precision_weights[0], W)


# --------------------------------------------------------------------------- #
#  Shape validation against the RESOLVED precision input                       #
# --------------------------------------------------------------------------- #
def test_wrong_dim_raises():
    with pytest.raises(ValueError, match="init_precision_weights"):
        _net_with(init_precision_weights=np.ones(8),  # pre_dim is 16
                  learn_precision_weights=False, learn_precision_bias=False)


def test_wrong_rows_raises():
    with pytest.raises(ValueError, match="expected"):
        _net_with(init_precision_weights=np.ones((3, 16)),
                  learn_precision_weights=True)


def test_rank3_raises_at_construction():
    net = pcn.PCNetwork(seed=0)
    with pytest.raises(ValueError, match="scalar"):
        with net:
            l1 = pcn.Layer(dim=16)
            l2 = pcn.Layer(dim=4)
            pcn.Predict(l1, l2, init_precision_weights=np.ones((2, 2, 16)))


def test_checked_against_precision_input_not_pre():
    """precision_input replaces the pre, so the check follows the NEW dim: an
    init sized for the pre (16) raises when precision reads the 1-d gain."""
    bad = pcn.PCNetwork(seed=0)
    with bad:
        a = pcn.Layer(dim=16, label="input")
        b = pcn.Layer(dim=4, label="output")
        g = pcn.Layer(dim=1, activation=pcn.Direct(), label="gain")
        pcn.Predict(a, b, precision_input=g, init_precision_weights=np.ones(16),
                    learn_precision_weights=False, learn_precision_bias=False)
    with pytest.raises(ValueError, match="precision_input"):
        bad.build()

    ok = pcn.PCNetwork(seed=0)
    with ok:
        a = pcn.Layer(dim=16, label="input")
        b = pcn.Layer(dim=4, label="output")
        g = pcn.Layer(dim=1, activation=pcn.Direct(), label="gain")
        p = pcn.Predict(a, b, precision_input=g, init_precision_weights=1.0,
                        learn_precision_weights=False, learn_precision_bias=False)
    ok.build()
    assert p.precision_input_dim == 1
    assert ok.params.precision_weights[0].shape == (1, 1)


def test_precision_input_assigned_after_construction():
    """precision_input may be set post-construction; the check uses the final dim."""
    net = pcn.PCNetwork(seed=0)
    with net:
        l1 = pcn.Layer(dim=16, label="input")
        l2 = pcn.Layer(dim=4, label="output")
        l_gain = pcn.Layer(dim=1, activation=pcn.Direct(), label="gain")
        p = pcn.Predict(l1, l2, init_precision_weights=1.0,
                        learn_precision_weights=False, learn_precision_bias=False)
        p.precision_input = l_gain          # after construction
    net.build()
    assert net.params.precision_weights[0].shape == (1, 1)


# --------------------------------------------------------------------------- #
#  The point: a frozen gain path that actually drives precision                #
# --------------------------------------------------------------------------- #
def _gain_net(init_precision=1.0):
    """Predict whose precision is exp(1*gain + log(init_precision)) with the
    gain-reading weight FROZEN at 1 — the precgain construction."""
    net = pcn.PCNetwork(seed=0)
    with net:
        l1 = pcn.Layer(dim=8, label="input")
        l2 = pcn.Layer(dim=4, label="output")
        l_gain = pcn.Layer(dim=1, activation=pcn.Direct(), label="gain")
        p = pcn.Predict(l1, l2, precision_input=l_gain,
                        init_precision=init_precision,
                        precision_activation="exp",
                        init_precision_weights=1.0,
                        learn_precision_weights=False, learn_precision_bias=False)
    net.build()
    return net, l1, l2, l_gain, p


def test_unit_precision_fastpath_not_taken_with_hand_set_weights():
    """init_precision==1.0 + frozen params would otherwise look 'provably 1.0';
    a hand-set W_rho makes precision live, so the fold-out must be disabled."""
    net, _, _, _, _ = _gain_net(init_precision=1.0)
    assert net.structure.predict_conns[0].unit_precision is False

    # control: same net without the hand-set weights IS folded out
    net2 = pcn.PCNetwork(seed=0)
    with net2:
        a = pcn.Layer(dim=8)
        b = pcn.Layer(dim=4)
        pcn.Predict(a, b, init_precision=1.0, learn_precision_weights=False,
                    learn_precision_bias=False)
    net2.build()
    assert net2.structure.predict_conns[0].unit_precision is True


@pytest.mark.parametrize("gamma", [-1.0, 0.0, 1.0])
def test_clamped_gain_sets_precision(gamma):
    """Clamping the gain layer to gamma yields precision = init_precision*e^gamma."""
    net, l1, l2, l_gain, _ = _gain_net(init_precision=0.5)
    sim = pcn.Simulation(net)
    n = 3
    batch = {"x": np.ones((n, 8), np.float32),
             "g": np.full((n, 1), gamma, np.float32)}
    r = sim.test([batch], data_map={l1: "x", l_gain: "g"},
                 iterations_per_sample=3, verbose=False, return_logs=True)
    prec = np.asarray(r["precisions"][0])[:, -1, :]
    assert prec.mean() == pytest.approx(0.5 * np.exp(gamma), rel=1e-4)


def test_frozen_gain_weights_survive_training():
    """learn_precision_weights=False keeps the hand-set W_rho exactly fixed."""
    net, l1, l2, l_gain, _ = _gain_net(init_precision=0.5)
    sim = pcn.Simulation(net)
    before = np.asarray(net.params.precision_weights[0]).copy()
    rng = np.random.default_rng(0)
    batches = [{"x": rng.normal(size=(4, 8)).astype(np.float32),
                "y": rng.normal(size=(4, 4)).astype(np.float32),
                "g": np.zeros((4, 1), np.float32)} for _ in range(3)]
    sim.train(batches, data_map={l1: "x", l2: "y", l_gain: "g"}, epochs=1,
              iterations_per_sample=0, learning_iterations_per_sample=4,
              params_optimizer=optax.adam(1e-2), verbose=False)
    assert np.allclose(np.asarray(net.params.precision_weights[0]), before)


def test_gain_steers_precision_after_training():
    """The trained net's precision still tracks the test-time gain (the whole
    point: one trained net, many operating points)."""
    net, l1, l2, l_gain, _ = _gain_net(init_precision=0.5)
    sim = pcn.Simulation(net)
    rng = np.random.default_rng(1)
    batches = [{"x": rng.normal(size=(4, 8)).astype(np.float32),
                "y": rng.normal(size=(4, 4)).astype(np.float32),
                "g": np.zeros((4, 1), np.float32)} for _ in range(3)]
    sim.train(batches, data_map={l1: "x", l2: "y", l_gain: "g"}, epochs=1,
              iterations_per_sample=0, learning_iterations_per_sample=4,
              params_optimizer=optax.adam(1e-2), verbose=False)
    got = []
    for gamma in (-1.0, 1.0):
        b = {"x": np.ones((2, 8), np.float32),
             "g": np.full((2, 1), gamma, np.float32)}
        r = sim.test([b], data_map={l1: "x", l_gain: "g"},
                     iterations_per_sample=3, verbose=False, return_logs=True)
        got.append(float(np.asarray(r["precisions"][0])[:, -1, :].mean()))
    assert got[1] / got[0] == pytest.approx(np.exp(2.0), rel=1e-4)
