"""
Tests for ``advance='timestep'`` on Project / Modulate connections.

``advance='timestep'`` makes a value-targeting Project/Modulate fire only on the
FIRST inference iteration of each input timestep (``i % iters_per_timestep == 0``)
instead of on every iteration.  This lets a state operator advance once per input
frame while the latent relaxes for ``iters_per_timestep`` iterations against a
held frame.

Verifies:
- Host API validation (bad string, non-value targets).
- Spec plumbing (``advance_timestep`` reaches ``Project/ModulateConnSpec``,
  defaults to False, stays the LAST field so positional construction is safe).
- Backend gating: gated Project contributes 0 off boundary; gated Modulate is
  the identity (1.0) off boundary — both checked against explicit numpy
  references and against the ungated behavior.
- Regression: ungated networks are unaffected (bit-identical), and gated
  connections with ``iters_per_timestep == 1`` reduce exactly to ungated.
"""

import numpy as np
import jax
import jax.numpy as jnp
import pytest

import pcn
from pcn import PCNetwork, Layer, Predict, Project, Modulate, Simulation
from pcn.core.structure import ProjectConnSpec, ModulateConnSpec
from pcn.backend.simulation import _apply_project_modulate_values, ACTIVATIONS


D = 2
T = 4
K = 3   # iters_per_timestep


# ── helpers ─────────────────────────────────────────────────────────────────

def _run(net, layer_label, src_layer, u, iters):
    """Run inference over a (1, T, D) temporal clamp; return the (iters, D) log."""
    sim = Simulation(net)
    res = sim.test([{'src': u}], data_map={src_layer: 'src'},
                   iterations_per_sample=iters, log_every=1,
                   feedforward_init=False, return_logs=True)
    return np.asarray(res['values'][net[layer_label]])[0]


def _build_register(advance):
    """Delay register: out += -I·out + I·src  ⇒  out = src on every fire."""
    net = PCNetwork(seed=0)
    with net:
        l_src = Layer(dim=D, activation=pcn.Direct(), label='src')
        l_out = Layer(dim=D, activation=pcn.Direct(), label='out')
        Project(l_out.value, l_out.value, update_rule=pcn.NoLearning(),
                init_weight=-jnp.eye(D), use_bias=False, advance=advance)
        Project(l_src.value, l_out.value, update_rule=pcn.NoLearning(),
                init_weight=jnp.eye(D), use_bias=False, advance=advance)
    net.build()
    return net, l_src


def _build_accumulator(advance):
    """Integrator: out += I·src on every fire (no leak)."""
    net = PCNetwork(seed=0)
    with net:
        l_src = Layer(dim=D, activation=pcn.Direct(), label='src')
        l_acc = Layer(dim=D, activation=pcn.Direct(), label='acc')
        Project(l_src.value, l_acc.value, update_rule=pcn.NoLearning(),
                init_weight=jnp.eye(D), use_bias=False, advance=advance)
    net.build()
    return net, l_src


def _build_chain(advance):
    """src -> r1 -> r2, each a -I/+I register.  Propagates one hop per fire."""
    net = PCNetwork(seed=0)
    with net:
        l_src = Layer(dim=D, activation=pcn.Direct(), label='src')
        r1 = Layer(dim=D, activation=pcn.Direct(), label='r1')
        r2 = Layer(dim=D, activation=pcn.Direct(), label='r2')
        for tgt, src in ((r1, l_src), (r2, r1)):
            Project(tgt.value, tgt.value, update_rule=pcn.NoLearning(),
                    init_weight=-jnp.eye(D), use_bias=False, advance=advance)
            Project(src.value, tgt.value, update_rule=pcn.NoLearning(),
                    init_weight=jnp.eye(D), use_bias=False, advance=advance)
    net.build()
    return net, l_src


def _build_modulate_decay(advance, factor=0.5):
    """Ungated integrator + (possibly gated) multiplicative decay by `factor`."""
    net = PCNetwork(seed=0)
    with net:
        l_src = Layer(dim=D, activation=pcn.Direct(), label='src')
        l_out = Layer(dim=D, activation=pcn.Direct(), label='out')
        # Always-on accumulator (fires every iteration).
        Project(l_src.value, l_out.value, update_rule=pcn.NoLearning(),
                init_weight=jnp.eye(D), use_bias=False)
        # Decay; use_bias=False so the modulation is exactly factor*I @ src.
        Modulate(l_src.value, l_out.value, update_rule=pcn.NoLearning(),
                 init_weight=factor * jnp.eye(D), use_bias=False,
                 advance=advance)
    net.build()
    return net, l_src


# ── host API validation ─────────────────────────────────────────────────────

class TestAdvanceValidation:
    def test_project_rejects_bogus_advance(self):
        net = PCNetwork(seed=0)
        with net:
            a = Layer(dim=D, label='a')
            b = Layer(dim=D, label='b')
            with pytest.raises(ValueError, match="'iteration' or 'timestep'"):
                Project(a.value, b.value, advance='bogus')

    def test_modulate_rejects_bogus_advance(self):
        net = PCNetwork(seed=0)
        with net:
            a = Layer(dim=D, label='a')
            b = Layer(dim=D, label='b')
            with pytest.raises(ValueError, match="'iteration' or 'timestep'"):
                Modulate(a.value, b.value, advance='bogus')

    def test_project_timestep_on_error_target_raises(self):
        net = PCNetwork(seed=0)
        with net:
            a = Layer(dim=D, label='a')
            b = Layer(dim=D, label='b')
            p = Predict(b, a)
            with pytest.raises(ValueError, match="value-targeting"):
                Project(b.value, p.error, advance='timestep')

    def test_modulate_timestep_on_error_target_raises(self):
        net = PCNetwork(seed=0)
        with net:
            a = Layer(dim=D, label='a')
            b = Layer(dim=D, label='b')
            p = Predict(b, a)
            with pytest.raises(ValueError, match="value-targeting"):
                Modulate(b.value, p.error, advance='timestep')

    def test_timestep_on_precision_target_raises(self):
        net = PCNetwork(seed=0)
        with net:
            a = Layer(dim=D, label='a')
            b = Layer(dim=D, label='b')
            p = Predict(b, a)
            with pytest.raises(ValueError, match="value-targeting"):
                Project(b.value, p.precision, advance='timestep')
            with pytest.raises(ValueError, match="value-targeting"):
                Modulate(b.value, p.precision, advance='timestep')

    def test_error_target_iteration_still_allowed(self):
        """The default must keep working for error targets."""
        net = PCNetwork(seed=0)
        with net:
            a = Layer(dim=D, label='a')
            b = Layer(dim=D, label='b')
            p = Predict(b, a)
            m = Modulate(b.value, p.error)
        net.build()
        assert m.advance == 'iteration'
        assert len(net.structure.modulate_conns_internal) == 1


# ── spec plumbing ───────────────────────────────────────────────────────────

class TestSpecPlumbing:
    def test_field_is_last_and_defaults_false(self):
        for cls in (ProjectConnSpec, ModulateConnSpec):
            # The Phase-1 delay redesign appended delay fields after
            # advance_timestep, so it is no longer the very last field; it must
            # still exist and default to False, and the 7-arg positional
            # construction (checked separately) must stay valid.
            assert 'advance_timestep' in cls._fields
            assert cls._field_defaults['advance_timestep'] is False
            assert cls._fields[-1] == 'pre_buffer_indices'
            assert cls._field_defaults['delay'] == 0

    def test_positional_construction_still_valid(self):
        """Pre-existing positional construction must not break."""
        spec = ProjectConnSpec((0,), 0, 1, 0, 0, 1e-3, -1)
        assert spec.advance_timestep is False
        spec = ModulateConnSpec((0,), 0, 1, 0, 0, 1e-3, -1)
        assert spec.advance_timestep is False

    def test_default_network_is_ungated(self):
        net, _ = _build_accumulator('iteration')
        specs = net.structure.project_conns_value
        assert len(specs) == 1
        assert all(not s.advance_timestep for _, s in specs)

    def test_gated_network_flag_reaches_spec(self):
        net, _ = _build_register('timestep')
        specs = net.structure.project_conns_value
        assert len(specs) == 2
        assert all(s.advance_timestep for _, s in specs)

    def test_modulate_flag_reaches_spec(self):
        net, _ = _build_modulate_decay('timestep')
        (_, pspec), = net.structure.project_conns_value
        (_, mspec), = net.structure.modulate_conns_value
        assert pspec.advance_timestep is False
        assert mspec.advance_timestep is True

    def test_conn_attribute_default(self):
        net = PCNetwork(seed=0)
        with net:
            a = Layer(dim=D, label='a')
            b = Layer(dim=D, label='b')
            pr = Project(a.value, b.value)
            mo = Modulate(a.value, b.value)
        net.build()
        assert pr.advance == 'iteration'
        assert mo.advance == 'iteration'


# ── backend gating: Project ─────────────────────────────────────────────────

class TestGatedProject:
    def test_register_holds_value_across_timestep(self):
        """Gated -I/+I register: piecewise-constant, one update per frame."""
        rng = np.random.default_rng(0)
        u = rng.normal(size=(1, T, D)).astype(np.float32)
        net, l_src = _build_register('timestep')
        out = _run(net, 'out', l_src, u, T * K)

        # numpy reference: v holds within a frame, snaps to the frame value on
        # the frame's first iteration.
        v = np.zeros(D, dtype=np.float32)
        ref = []
        for i in range(T * K):
            t = i // K
            if i % K == 0:                      # boundary: v += -v + src
                v = v + (-v) + u[0, t]
            ref.append(v.copy())
        ref = np.asarray(ref)

        assert out.shape == (T * K, D)
        np.testing.assert_allclose(out, ref, atol=1e-6)
        # Explicitly: constant within each frame, and equal to that frame's src.
        for t in range(T):
            frame = out[t * K:(t + 1) * K]
            np.testing.assert_allclose(frame, np.broadcast_to(u[0, t], frame.shape),
                                       atol=1e-6)

    def test_gated_accumulator_fires_once_per_frame(self):
        """Integrator: gated sums T frames, ungated sums T*K iterations."""
        rng = np.random.default_rng(1)
        u = rng.normal(size=(1, T, D)).astype(np.float32)

        net_g, src_g = _build_accumulator('timestep')
        gated = _run(net_g, 'acc', src_g, u, T * K)
        net_u, src_u = _build_accumulator('iteration')
        ungated = _run(net_u, 'acc', src_u, u, T * K)

        ref_g, ref_u = [], []
        vg = np.zeros(D, dtype=np.float32)
        vu = np.zeros(D, dtype=np.float32)
        for i in range(T * K):
            t = i // K
            if i % K == 0:
                vg = vg + u[0, t]
            vu = vu + u[0, t]
            ref_g.append(vg.copy())
            ref_u.append(vu.copy())

        np.testing.assert_allclose(gated, np.asarray(ref_g), atol=1e-5)
        np.testing.assert_allclose(ungated, np.asarray(ref_u), atol=1e-5)
        # The gate must actually change something.
        assert not np.allclose(gated, ungated)
        # Gated total = sum of frames; ungated total = K x that.
        np.testing.assert_allclose(gated[-1], u[0].sum(axis=0), atol=1e-5)

    def test_chain_propagates_one_hop_per_frame(self):
        """Gated chain lags by one FRAME; ungated lags by one ITERATION."""
        rng = np.random.default_rng(2)
        u = rng.normal(size=(1, T, D)).astype(np.float32)

        net_g, src_g = _build_chain('timestep')
        r2_g = _run(net_g, 'r2', src_g, u, T * K)
        net_u, src_u = _build_chain('iteration')
        r2_u = _run(net_u, 'r2', src_u, u, T * K)

        # Gated: r2 == src of the previous frame, held for the whole frame.
        for t in range(T):
            frame = r2_g[t * K:(t + 1) * K]
            expect = np.zeros(D, dtype=np.float32) if t == 0 else u[0, t - 1]
            np.testing.assert_allclose(frame, np.broadcast_to(expect, frame.shape),
                                       atol=1e-6)
        # Ungated: within a frame r2 catches up to the current frame (K >= 2).
        np.testing.assert_allclose(r2_u[K - 1], u[0, 0], atol=1e-6)
        assert not np.allclose(r2_g, r2_u)

    def test_single_timestep_input_fires_once(self):
        """A static (B, D) clamp is one frame: a gated conn fires exactly once."""
        u = np.ones((1, D), dtype=np.float32)
        net, l_src = _build_accumulator('timestep')
        out = _run(net, 'acc', l_src, u, 5)
        # Fires only at i == 0, then holds.
        np.testing.assert_allclose(out, np.ones((5, D), dtype=np.float32), atol=1e-6)


# ── backend gating: Modulate ────────────────────────────────────────────────

class TestGatedModulate:
    def test_modulate_is_identity_off_boundary(self):
        factor = 0.5
        u = np.ones((1, T, D), dtype=np.float32)

        net_g, src_g = _build_modulate_decay('timestep', factor)
        gated = _run(net_g, 'out', src_g, u, T * K)
        net_u, src_u = _build_modulate_decay('iteration', factor)
        ungated = _run(net_u, 'out', src_u, u, T * K)

        ref_g, ref_u = [], []
        vg = np.zeros(D, dtype=np.float32)
        vu = np.zeros(D, dtype=np.float32)
        for i in range(T * K):
            t = i // K
            vg = vg + u[0, t]
            if i % K == 0:
                vg = vg * factor           # gated: decay once per frame
            else:
                pass                       # identity off boundary
            vu = (vu + u[0, t]) * factor   # ungated: decay every iteration
            ref_g.append(vg.copy())
            ref_u.append(vu.copy())

        np.testing.assert_allclose(gated, np.asarray(ref_g), atol=1e-5)
        np.testing.assert_allclose(ungated, np.asarray(ref_u), atol=1e-5)
        assert not np.allclose(gated, ungated)

    def test_modulate_off_boundary_does_not_zero_target(self):
        """Regression: the off-boundary factor is 1.0, never 0.0."""
        u = np.ones((1, T, D), dtype=np.float32)
        net, l_src = _build_modulate_decay('timestep', factor=0.5)
        out = _run(net, 'out', l_src, u, T * K)
        # Off-boundary iterations of frame 0: 0.5, then +1 each iteration.
        np.testing.assert_allclose(out[0], 0.5 * np.ones(D), atol=1e-6)
        np.testing.assert_allclose(out[1], 1.5 * np.ones(D), atol=1e-6)
        np.testing.assert_allclose(out[2], 2.5 * np.ones(D), atol=1e-6)


# ── regression: ungated behavior unchanged ──────────────────────────────────

class TestUngatedRegression:
    def test_gated_with_one_iter_per_timestep_equals_ungated(self):
        """iters_per_timestep == 1 makes every iteration a boundary."""
        rng = np.random.default_rng(3)
        u = rng.normal(size=(1, T, D)).astype(np.float32)
        net_g, src_g = _build_accumulator('timestep')
        net_u, src_u = _build_accumulator('iteration')
        gated = _run(net_g, 'acc', src_g, u, T)      # iters_per_timestep == 1
        ungated = _run(net_u, 'acc', src_u, u, T)
        np.testing.assert_array_equal(gated, ungated)

    def test_ungated_full_network_bit_identical(self):
        """A network with no gated conn must be untouched by the change.

        Runs a Predict+Project+Modulate net twice: once as-built (all conns
        ungated, so run_batch passes ``is_boundary=None``) and once through
        ``_apply_project_modulate_values`` with ``is_boundary=None`` on
        explicitly ungated specs — the results must match exactly.
        """
        rng = np.random.default_rng(4)
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
            return net, l_in

        net_a, in_a = build()
        net_b, in_b = build()
        assert all(not s.advance_timestep for _, s in net_a.structure.project_conns_value)
        assert all(not s.advance_timestep for _, s in net_a.structure.modulate_conns_value)

        a = _run(net_a, 'hid', in_a, u, T * K)
        b = _run(net_b, 'hid', in_b, u, T * K)
        np.testing.assert_array_equal(a, b)

    def test_gated_conns_with_is_boundary_none_match_ungated(self):
        """``is_boundary=None`` takes the original, un-gated code path exactly."""
        rng = np.random.default_rng(5)
        values = (jnp.asarray(rng.normal(size=(3, D)), dtype=jnp.float32),
                  jnp.asarray(rng.normal(size=(3, D)), dtype=jnp.float32))
        clamped = (jnp.zeros((3, D)), jnp.zeros((3, D)))
        W = jnp.asarray(rng.normal(size=(D, D)), dtype=jnp.float32)
        M = jnp.asarray(rng.normal(size=(D, D)), dtype=jnp.float32)
        acts = (ACTIVATIONS[0], ACTIVATIONS[0])

        def specs(gated):
            p = ProjectConnSpec((0,), 0, 1, 0, 0, 0.0, -1, advance_timestep=gated)
            m = ModulateConnSpec((0,), 0, 1, 0, 0, 0.0, -1, advance_timestep=gated)
            return ((0, p),), ((0, m),)

        def call(gated, is_boundary):
            pv, mv = specs(gated)
            return _apply_project_modulate_values(
                values, (), (W,), (M,), pv, mv, acts, clamped,
                read_values=values, is_boundary=is_boundary)

        ungated = call(False, None)
        gated_none = call(True, None)
        for a, b in zip(ungated, gated_none):
            np.testing.assert_array_equal(np.asarray(a), np.asarray(b))

        # Ungated specs ignore is_boundary entirely.
        for a, b in zip(ungated, call(False, jnp.bool_(False))):
            np.testing.assert_array_equal(np.asarray(a), np.asarray(b))

    def test_gating_math_on_and_off_boundary(self):
        """Unit check: off boundary Project adds 0 and Modulate multiplies by 1."""
        rng = np.random.default_rng(6)
        v0 = jnp.asarray(rng.normal(size=(3, D)), dtype=jnp.float32)
        v1 = jnp.asarray(rng.normal(size=(3, D)), dtype=jnp.float32)
        values = (v0, v1)
        clamped = (jnp.zeros((3, D)), jnp.zeros((3, D)))
        W = jnp.asarray(rng.normal(size=(D, D)), dtype=jnp.float32)
        M = jnp.asarray(rng.normal(size=(D, D)), dtype=jnp.float32)
        acts = (ACTIVATIONS[0], ACTIVATIONS[0])
        p = ProjectConnSpec((0,), 0, 1, 0, 0, 0.0, -1, advance_timestep=True)
        m = ModulateConnSpec((0,), 0, 1, 0, 0, 0.0, -1, advance_timestep=True)

        def call(is_boundary):
            return _apply_project_modulate_values(
                values, (), (W,), (M,), ((0, p),), ((0, m),), acts, clamped,
                read_values=values, is_boundary=is_boundary)

        off = call(jnp.bool_(False))
        np.testing.assert_allclose(np.asarray(off[1]), np.asarray(v1), atol=1e-6)

        on = call(jnp.bool_(True))
        expected = (np.asarray(v1) + np.asarray(v0) @ np.asarray(W).T) \
            * (np.asarray(v0) @ np.asarray(M).T)
        np.testing.assert_allclose(np.asarray(on[1]), expected, rtol=1e-5, atol=1e-5)

        # Float (non-bool) is_boundary must work too (traced-scalar tolerance).
        np.testing.assert_allclose(np.asarray(call(jnp.float32(0.0))[1]),
                                   np.asarray(off[1]), atol=1e-6)
        np.testing.assert_allclose(np.asarray(call(jnp.float32(1.0))[1]),
                                   np.asarray(on[1]), atol=1e-6)

    def test_gating_is_traceable_under_jit(self):
        """The gate must not use a Python ``if`` on the tracer."""
        rng = np.random.default_rng(7)
        values = (jnp.asarray(rng.normal(size=(3, D)), dtype=jnp.float32),
                  jnp.asarray(rng.normal(size=(3, D)), dtype=jnp.float32))
        clamped = (jnp.zeros((3, D)), jnp.zeros((3, D)))
        W = jnp.asarray(rng.normal(size=(D, D)), dtype=jnp.float32)
        M = jnp.asarray(rng.normal(size=(D, D)), dtype=jnp.float32)
        acts = (ACTIVATIONS[0], ACTIVATIONS[0])
        p = ProjectConnSpec((0,), 0, 1, 0, 0, 0.0, -1, advance_timestep=True)
        m = ModulateConnSpec((0,), 0, 1, 0, 0, 0.0, -1, advance_timestep=True)

        @jax.jit
        def f(i):
            return _apply_project_modulate_values(
                values, (), (W,), (M,), ((0, p),), ((0, m),), acts, clamped,
                read_values=values, is_boundary=(i % K) == 0)

        held = np.asarray(f(jnp.int32(1))[1])
        np.testing.assert_allclose(held, np.asarray(values[1]), atol=1e-6)
        fired = np.asarray(f(jnp.int32(0))[1])
        assert not np.allclose(held, fired)


# ── learning path ───────────────────────────────────────────────────────────

class TestLearningLoopGating:
    def test_gating_holds_through_learning_iterations(self):
        """``learning_body`` uses boundary = (n_iterations + i) % ipt == 0."""
        rng = np.random.default_rng(8)
        u = rng.normal(size=(1, T, D)).astype(np.float32)

        net = PCNetwork(seed=0)
        with net:
            l_src = Layer(dim=D, activation=pcn.Direct(), label='src')
            l_acc = Layer(dim=D, activation=pcn.Direct(), label='acc')
            # Separate island so the learning machinery has a Predict to train
            # without perturbing the gated accumulator.
            l_a = Layer(dim=D, activation=pcn.Direct(), label='a')
            l_b = Layer(dim=D, activation=pcn.Relu(), label='b')
            Predict(l_b, l_a)
            Project(l_src.value, l_acc.value, update_rule=pcn.NoLearning(),
                    init_weight=jnp.eye(D), use_bias=False, advance='timestep')
        net.build()

        sim = Simulation(net)
        # Deliberately split mid-frame: 8 inference + 4 learning iterations with
        # K == 3.  The only boundary inside the learning loop is global iter 9,
        # i.e. learning i == 1 — so a wrong (un-offset) ``i % K`` boundary would
        # fire at learning i == 0 and i == 3 instead and change the trace.
        sim.train([{'src': u, 'a': np.zeros((1, T, D), dtype=np.float32)}],
                  data_map={l_src: 'src', l_a: 'a'},
                  iterations_per_sample=8,
                  learning_iterations_per_sample=4,
                  log_every=1, feedforward_init=False, save_logs=True)
        # Raw train logs are (n_logs, batch, dim).
        acc = np.asarray(sim.logs['values'][0][net['acc']])[:, 0, :]

        ref = []
        v = np.zeros(D, dtype=np.float32)
        for i in range(T * K):
            t = i // K
            if i % K == 0:
                v = v + u[0, t]
            ref.append(v.copy())
        np.testing.assert_allclose(acc, np.asarray(ref), atol=1e-5)


# ── save/load round-trip ────────────────────────────────────────────────────

def test_save_load_roundtrip_with_gated_conn(tmp_path):
    """The new spec field must not break HDF5 structure serialization."""
    net, _ = _build_register('timestep')
    path = tmp_path / 'gated.h5'
    net.save(path)
    net2, _ = _build_register('timestep')
    net2.load(path)
    for w1, w2 in zip(net.params.project_weights, net2.params.project_weights):
        np.testing.assert_allclose(np.asarray(w1), np.asarray(w2), atol=1e-6)
