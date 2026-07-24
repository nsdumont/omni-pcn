"""
Tests for temporal-delay reads on Predict / Project / Modulate (Phase 1).

A connection with ``delay=d >= 1`` reads its (value) pre from a per-node history
ring buffer instead of the live ``values``:

- ``delay_unit='iteration'`` (**sliding**): pre = ``v_pre[i-d]``, advancing every
  inference iteration. Index math (verified by hand in ``delay_redesign.md``):
  write slot ``i % S`` at the top of each loop body, read slot ``(i-d) % S``,
  ``S = depth+1``; the first ``d`` reads are the pre-fill zeros.
- ``delay_unit='timestep'`` (**latched**): pre = the end-of-frame snapshot of the
  previous input timestep, held constant across all ``iters_per_timestep``
  iterations of the current frame (the tPC/Kalman prior).

``delay=0`` (default) is the historical live-read path and must stay
bit-identical. The delayed read is one-directional (the buffer is a carry
constant, not a differentiation variable), so no error flows back to the delayed
pre — no ``stop_gradient`` needed.

Note on the ``delay_unit`` units at K=1: the design doc's throwaway "coincides
with sliding at K=1" line is inconsistent with its own hand-verified,
end-of-previous-frame latched formula (which this implementation follows). The
true relationship is an off-by-one: latched ``delay=d`` at K=1 equals sliding
``delay=d-1`` at K=1 (both read a buffered constant with the same gradient
structure), so latched ``delay=1`` and sliding ``delay=1`` differ even at K=1.
``test_latched_offset_matches_sliding_at_k1`` pins the honest coincidence.
"""

import numpy as np
import jax
import jax.numpy as jnp
import pytest

import pcn
from pcn import PCNetwork, Layer, Predict, Project, Modulate, Simulation
from pcn.backend.simulation import _read_delayed


D = 2


# ── helpers ─────────────────────────────────────────────────────────────────

def _test_run(net, u, clamp_layer, iters, **kw):
    """Run inference over a (1, T, D) temporal clamp; return the full log dict."""
    sim = Simulation(net)
    return sim.test([{'seq': u}], data_map={net[clamp_layer]: 'seq'},
                    iterations_per_sample=iters, log_every=1,
                    feedforward_init=False, return_logs=True, **kw)


def _errors(res, conn_idx):
    """(n_logged, D) error log for a predict connection (batch 0)."""
    return np.asarray(res['errors'][conn_idx])[0]


def _values(net, res, label):
    """(n_logged, D) value log for a layer (batch 0)."""
    return np.asarray(res['values'][net[label]])[0]


# ── sliding Predict self-edge (clamped pre => fully deterministic) ───────────

class TestSlidingPredict:
    @pytest.mark.parametrize("d", [1, 2, 3])
    def test_sliding_self_edge_matches_numpy(self, d):
        """Clamped self-edge Predict(z, z, delay=d): the transition error is
        ``z[i] - W f(z[i-d])`` with pre-fill zeros for i < d (Direct f)."""
        rng = np.random.default_rng(d)
        T = 6
        u = rng.normal(size=(1, T, D)).astype(np.float32)
        W = rng.normal(size=(D, D)).astype(np.float32)

        net = PCNetwork(seed=0)
        with net:
            z = Layer(dim=D, activation=pcn.Direct(), label='z')
            Predict(z, z, delay=d, init_weight=W, use_bias=False,
                    learn_precision_weights=False, learn_precision_bias=False)
        net.build()
        assert net.structure.hist_specs == ((net['z'], d),)
        assert net.structure.hist_unit_ts == (False,)

        e = _errors(_test_run(net, u, 'z', T), 0)

        ref = np.zeros((T, D), np.float32)
        for i in range(T):
            delayed = u[0, i - d] if i - d >= 0 else np.zeros(D, np.float32)
            ref[i] = u[0, i] - delayed @ W.T
        np.testing.assert_allclose(e, ref, atol=1e-5)

    def test_prefill_zeros_first_delay_iterations(self):
        """The first ``delay`` reads return zeros: the transition error over
        those iterations equals the raw clamped value (no prediction)."""
        rng = np.random.default_rng(11)
        T, d = 5, 2
        u = rng.normal(size=(1, T, D)).astype(np.float32)
        W = rng.normal(size=(D, D)).astype(np.float32)
        net = PCNetwork(seed=0)
        with net:
            z = Layer(dim=D, activation=pcn.Direct(), label='z')
            Predict(z, z, delay=d, init_weight=W, use_bias=False,
                    learn_precision_weights=False, learn_precision_bias=False)
        net.build()
        e = _errors(_test_run(net, u, 'z', T), 0)
        # i < d: delayed pre is zeros -> error == clamped value.
        for i in range(d):
            np.testing.assert_allclose(e[i], u[0, i], atol=1e-6)


# ── latched Predict self-edge (free latent; tPC prior) ──────────────────────

class TestLatchedPredict:
    @pytest.mark.parametrize("K", [2, 3])
    def test_latched_held_constant_and_prev_frame(self, K):
        """Free-latent self-edge Predict(z, z, delay=1, unit='timestep', W=I):
        the delayed pre is held constant across the K iterations of each frame
        and equals the previous frame's last value (0 for the first frame)."""
        rng = np.random.default_rng(100 + K)
        T = 4
        u = rng.normal(size=(1, T, D)).astype(np.float32)
        net = PCNetwork(seed=1)
        with net:
            x = Layer(dim=D, activation=pcn.Direct(), label='x')      # clamped obs
            z = Layer(dim=D, activation=pcn.Direct(), label='z')      # free latent
            Predict(z, x, use_bias=False, learn_precision_weights=False,
                    learn_precision_bias=False)                       # observation
            Predict(z, z, delay=1, delay_unit='timestep',
                    init_weight=jnp.eye(D), use_bias=False,
                    learn_precision_weights=False, learn_precision_bias=False)
        net.build()
        assert net.structure.hist_specs == ((net['z'], 1),)
        assert net.structure.hist_unit_ts == (True,)

        res = _test_run(net, u, 'x', T * K)
        z_log = _values(net, res, 'z')
        e_trans = _errors(res, 1)               # transition conn
        # W=I, Direct f  =>  e = z - delayed_pre  =>  delayed_pre = z - e.
        delayed = z_log - e_trans
        for t in range(T):
            frame = delayed[t * K:(t + 1) * K]
            # held constant across the frame
            np.testing.assert_allclose(
                frame, np.broadcast_to(frame[0], frame.shape), atol=1e-5)
            expect = np.zeros(D, np.float32) if t == 0 else z_log[t * K - 1]
            np.testing.assert_allclose(frame, np.broadcast_to(expect, frame.shape),
                                       atol=1e-5)


# ── sliding vs latched relationship ─────────────────────────────────────────

def _build_selfedge(unit, delay, gain=0.3):
    net = PCNetwork(seed=3)
    with net:
        x = Layer(dim=D, activation=pcn.Direct(), label='x')
        z = Layer(dim=D, activation=pcn.Relu(), label='z')
        Predict(z, x, use_bias=False, learn_precision_weights=False,
                learn_precision_bias=False)
        Predict(z, z, delay=delay, delay_unit=unit, init_weight=gain * jnp.eye(D),
                use_bias=False, learn_precision_weights=False,
                learn_precision_bias=False)
    net.build()
    return net


def _run_z(net, u, K):
    return _values(net, _test_run(net, u, 'x', u.shape[1] * K), 'z')


class TestSlidingVsLatched:
    def test_differ_at_k_gt_1(self):
        """At K>1 sliding and latched read different quantities."""
        rng = np.random.default_rng(303)
        u = rng.normal(size=(1, 4, D)).astype(np.float32)
        sliding = _run_z(_build_selfedge('iteration', 1), u, 3)
        latched = _run_z(_build_selfedge('timestep', 1), u, 3)
        assert not np.allclose(sliding, latched, atol=1e-5)

    def test_differ_at_k1_same_delay(self):
        """Even at K=1 sliding(d=1) != latched(d=1): the latched buffer stores
        the end-of-previous-frame snapshot, one step ahead of the sliding slot
        (design doc's "coincide at K=1" is inconsistent with its own formula)."""
        rng = np.random.default_rng(304)
        u = rng.normal(size=(1, 5, D)).astype(np.float32)
        sliding = _run_z(_build_selfedge('iteration', 1), u, 1)
        latched = _run_z(_build_selfedge('timestep', 1), u, 1)
        assert not np.allclose(sliding, latched, atol=1e-6)

    def test_latched_offset_matches_sliding_at_k1(self):
        """The honest K=1 coincidence: latched(d=2) == sliding(d=1) at K=1."""
        rng = np.random.default_rng(305)
        u = rng.normal(size=(1, 5, D)).astype(np.float32)
        latched2 = _run_z(_build_selfedge('timestep', 2), u, 1)
        sliding1 = _run_z(_build_selfedge('iteration', 1), u, 1)
        np.testing.assert_allclose(latched2, sliding1, atol=1e-6)


# ── delayed Project (delayed identity copy) ─────────────────────────────────

class TestDelayedProject:
    @pytest.mark.parametrize("d", [1, 2])
    def test_delayed_identity_project_is_delayed_copy(self, d):
        """A ``-I`` leak + delayed ``+I`` copy register: dst[i] == src[i-d]
        (0 for i < d) — a delayed identity shortcut."""
        rng = np.random.default_rng(200 + d)
        T = 7
        u = rng.normal(size=(1, T, D)).astype(np.float32)
        net = PCNetwork(seed=2)
        with net:
            src = Layer(dim=D, activation=pcn.Direct(), label='src')
            dst = Layer(dim=D, activation=pcn.Direct(), label='dst')
            Project(dst.value, dst.value, update_rule=pcn.NoLearning(),
                    init_weight=-jnp.eye(D), use_bias=False)              # leak
            Project(src.value, dst.value, update_rule=pcn.NoLearning(),
                    init_weight=jnp.eye(D), use_bias=False, delay=d)      # delayed copy
        net.build()
        assert net.structure.hist_specs == ((net['src'], d),)

        dst_log = _values(net, _test_run(net, u, 'src', T), 'dst')
        ref = np.zeros((T, D), np.float32)
        for i in range(T):
            ref[i] = u[0, i - d] if i - d >= 0 else np.zeros(D, np.float32)
        np.testing.assert_allclose(dst_log, ref, atol=1e-5)


# ── buffer bookkeeping ──────────────────────────────────────────────────────

class TestBufferBookkeeping:
    def test_single_latched_buffer(self):
        """Predict(z, z, delay=1, unit='timestep') allocates exactly 1 buffer."""
        net = PCNetwork(seed=0)
        with net:
            z = Layer(dim=D, activation=pcn.Direct(), label='z')
            Predict(z, z, delay=1, delay_unit='timestep')
        net.build()
        assert len(net.structure.hist_specs) == 1
        assert net.structure.hist_specs == ((net['z'], 1),)
        assert net.structure.hist_unit_ts == (True,)

    def test_depth_is_max_delay_over_readers(self):
        """Two conns reading the same node/unit share one buffer sized to the
        max delay."""
        net = PCNetwork(seed=0)
        with net:
            z = Layer(dim=D, activation=pcn.Direct(), label='z')
            a = Layer(dim=D, activation=pcn.Direct(), label='a')
            b = Layer(dim=D, activation=pcn.Direct(), label='b')
            Predict(z, a, delay=1)
            Predict(z, b, delay=4)
        net.build()
        assert net.structure.hist_specs == ((net['z'], 4),)
        assert net.structure.hist_unit_ts == (False,)

    def test_separate_buffers_per_unit(self):
        """The same node read at both units gets two separate rings."""
        net = PCNetwork(seed=0)
        with net:
            z = Layer(dim=D, activation=pcn.Direct(), label='z')
            a = Layer(dim=D, activation=pcn.Direct(), label='a')
            b = Layer(dim=D, activation=pcn.Direct(), label='b')
            Predict(z, a, delay=1, delay_unit='iteration')
            Predict(z, b, delay=1, delay_unit='timestep')
        net.build()
        # sorted by (node_id, unit_ts): (z, False) then (z, True)
        assert net.structure.hist_specs == ((net['z'], 1), (net['z'], 1))
        assert net.structure.hist_unit_ts == (False, True)

    def test_no_delay_net_has_no_buffers(self):
        net = PCNetwork(seed=0)
        with net:
            a = Layer(dim=D, label='a')
            b = Layer(dim=D, activation=pcn.Relu(), label='b')
            Predict(b, a)
        net.build()
        assert net.structure.hist_specs == ()
        assert net.structure.hist_unit_ts == ()

    def test_spec_fields_default(self):
        net = PCNetwork(seed=0)
        with net:
            a = Layer(dim=D, label='a')
            b = Layer(dim=D, activation=pcn.Relu(), label='b')
            Predict(b, a)
        net.build()
        spec = net.structure.predict_conns[0]
        assert spec.delay == 0
        assert spec.delay_unit_ts is False
        assert spec.pre_buffer_indices == ()


# ── delay==0 bit-identity regression ────────────────────────────────────────

class TestDelayZeroBitIdentical:
    def test_predict_project_modulate_no_delay_bit_identical(self):
        """A Predict+Project+Modulate net with no delays allocates no buffers
        (the delay machinery is a static no-op) and is deterministic — running
        it twice is bit-identical."""
        rng = np.random.default_rng(4)
        T = 4
        u = rng.normal(size=(2, T, 6)).astype(np.float32)

        def build():
            net = PCNetwork(seed=7)
            with net:
                l_in = Layer(dim=6, activation=pcn.Direct(), label='in')
                l_h = Layer(dim=4, activation=pcn.Relu(), label='hid')
                l_o = Layer(dim=3, activation=pcn.Relu(), label='out')
                Predict(l_h, l_in)
                Predict(l_o, l_h)
                Project(l_o.value, l_h.value, update_rule=pcn.Hebbian(learning_rate=1e-3))
                Modulate(l_h.value, l_o.value, update_rule=pcn.Hebbian(learning_rate=1e-3))
            net.build()
            return net

        net_a, net_b = build(), build()
        # No delay anywhere -> the whole feature compiles out.
        assert net_a.structure.hist_specs == ()
        a = _values(net_a, _test_run(net_a, u, 'in', T * 3), 'hid')
        b = _values(net_b, _test_run(net_b, u, 'in', T * 3), 'hid')
        np.testing.assert_array_equal(a, b)


# ── validation errors ───────────────────────────────────────────────────────

class TestDelayValidation:
    def test_negative_delay_raises(self):
        net = PCNetwork(seed=0)
        with net:
            a = Layer(dim=D, label='a')
            b = Layer(dim=D, label='b')
            with pytest.raises(ValueError, match="non-negative int"):
                Predict(b, a, delay=-1)

    def test_bogus_unit_raises(self):
        net = PCNetwork(seed=0)
        with net:
            a = Layer(dim=D, label='a')
            b = Layer(dim=D, label='b')
            with pytest.raises(ValueError, match="'iteration' or 'timestep'"):
                Predict(b, a, delay=1, delay_unit='bogus')

    def test_project_bogus_unit_raises(self):
        net = PCNetwork(seed=0)
        with net:
            a = Layer(dim=D, label='a')
            b = Layer(dim=D, label='b')
            with pytest.raises(ValueError, match="'iteration' or 'timestep'"):
                Project(a.value, b.value, delay=1, delay_unit='bogus')

    def test_delay_on_error_pre_raises_notimplemented(self):
        """Phase 1 only supports value pre nodes: a delayed conn reading an
        error node must raise NotImplementedError."""
        net = PCNetwork(seed=0)
        with net:
            a = Layer(dim=D, label='a')
            b = Layer(dim=D, label='b')
            c = Layer(dim=D, label='c')
            p = Predict(b, a)
            with pytest.raises(NotImplementedError, match="Phase 2"):
                Project(p.error, c.value, delay=1)

    def test_delay_on_error_pre_modulate_raises(self):
        net = PCNetwork(seed=0)
        with net:
            a = Layer(dim=D, label='a')
            b = Layer(dim=D, label='b')
            c = Layer(dim=D, label='c')
            p = Predict(b, a)
            with pytest.raises(NotImplementedError, match="Phase 2"):
                Modulate(p.error, c.value, delay=2)


# ── traceability ────────────────────────────────────────────────────────────

def test_read_delayed_traceable_under_jit():
    """``_read_delayed`` uses only array indexing (no Python ``if`` on the loop
    tracer), so it traces cleanly with a traced iteration index."""
    hist = (jnp.arange(3 * 2 * D, dtype=jnp.float32).reshape(3, 2, D),)

    @jax.jit
    def f(i):
        # sliding, delay=1, buffer 0
        return _read_delayed(hist, 0, 1, False, 0, i, 1)

    for i in range(1, 3):
        got = np.asarray(f(jnp.int32(i)))
        np.testing.assert_array_equal(got, np.asarray(hist[0][(i - 1) % 3]))
    # pre-fill: i=0 reads slot (0-1)%3 = 2 (not yet written in a real run)
    np.testing.assert_array_equal(np.asarray(f(jnp.int32(0))),
                                  np.asarray(hist[0][2]))


def test_delayed_net_runs_under_jit():
    """End-to-end: a delayed net runs through the jit-compiled run_batch."""
    rng = np.random.default_rng(9)
    u = rng.normal(size=(1, 4, D)).astype(np.float32)
    net = PCNetwork(seed=0)
    with net:
        z = Layer(dim=D, activation=pcn.Direct(), label='z')
        Predict(z, z, delay=1, init_weight=jnp.eye(D), use_bias=False,
                learn_precision_weights=False, learn_precision_bias=False)
    net.build()
    res = _test_run(net, u, 'z', 4)
    assert np.asarray(res['errors'][0]).shape[0] == 1  # batch present
