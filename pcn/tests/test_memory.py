"""
Tests for the Memory (LMU / HiPPO) composite.

Verifies:
- The network's recurrent layer realizes m[t+1] = Ā m[t] + B̄ u[t] exactly.
- LegT C(t) reconstructs constant and delayed inputs.
- Deterministic mode isolates the recurrent layer from readout energy; the
  output mirror tracks the recurrent state (one-iteration lag, Jacobi).
- Multi-channel block-diagonal structure; mode wiring; arg validation.
"""

import numpy as np
import jax.numpy as jnp
import pytest

import pcn
from pcn import PCNetwork, Layer, Memory, Simulation, Predict


def _numpy_recurrence(Abar, Bbar, u):
    """m[0]=0; m_{k+1}=Abar m_k + Bbar u_k; return stacked (T, D)."""
    D = Abar.shape[0]
    m = np.zeros(D)
    out = []
    for k in range(u.shape[0]):
        m = Abar @ m + Bbar @ u[k]
        out.append(m.copy())
    return np.array(out)


class TestMemoryRecurrence:
    def test_legt_recurrence_matches_numpy(self):
        net = PCNetwork(seed=0)
        with net:
            l_in = Layer(dim=1, activation=pcn.Direct(), label='in')
            mem = Memory(l_in, dims_per_input=8, theta=1.0, dt=0.1)
        net.build()

        T = 30
        u = np.sin(np.linspace(0, 6, T)).astype(np.float32)
        traj = _numpy_recurrence(mem._Abar_full, mem._Bbar_full, u.reshape(T, 1))

        sim = Simulation(net)
        res = sim.test([{'in': u.reshape(1, T, 1)}], data_map={l_in: 'in'},
                       iterations_per_sample=T, log_every=1, feedforward_init=False, return_logs=True)
        rec_net = np.array(res['values'][net['memory_rec']])[0]   # (T, D)
        assert np.max(np.abs(rec_net - traj)) < 1e-4

    def test_multichannel_blockdiag(self):
        """Two input channels -> two independent d-dim memories; D = 2d."""
        net = PCNetwork(seed=0)
        with net:
            l_in = Layer(dim=2, activation=pcn.Direct(), label='in')
            mem = Memory(l_in, dims_per_input=6, theta=1.0, dt=0.2)
        net.build()
        assert mem.dim == 12
        # Block-diagonal: cross-channel blocks of Ā_full are zero.
        A = mem._Abar_full
        assert np.allclose(A[:6, 6:], 0.0) and np.allclose(A[6:, :6], 0.0)

        T = 20
        u = np.stack([np.sin(np.linspace(0, 5, T)),
                      np.cos(np.linspace(0, 3, T))], axis=1).astype(np.float32)
        traj = _numpy_recurrence(mem._Abar_full, mem._Bbar_full, u)
        sim = Simulation(net)
        res = sim.test([{'in': u[None]}], data_map={l_in: 'in'},
                       iterations_per_sample=T, log_every=1, feedforward_init=False, return_logs=True)
        rec_net = np.array(res['values'][net['memory_rec']])[0]
        assert np.max(np.abs(rec_net - traj)) < 1e-4


class TestMemoryDecode:
    def test_constant_input_reconstruction(self):
        net = PCNetwork(seed=0)
        with net:
            l_in = Layer(dim=1, activation=pcn.Direct(), label='in')
            mem = Memory(l_in, dims_per_input=10, theta=1.0, dt=0.05)
        net.build()
        Abar, Bbar = mem._Abar_full, mem._Bbar_full
        m_ss = np.linalg.solve(np.eye(mem.dim) - Abar, Bbar[:, 0])
        for t in [0.0, 0.3, 0.6, 1.0]:
            recon = float(mem.decode(jnp.asarray(m_ss[None]), t)[0, 0])
            assert abs(recon - 1.0) < 1e-2

    def test_delay_reconstruction_tracks_signal(self):
        net = PCNetwork(seed=0)
        with net:
            l_in = Layer(dim=1, activation=pcn.Direct(), label='in')
            mem = Memory(l_in, dims_per_input=12, theta=1.0, dt=0.05)
        net.build()
        T = 60
        u = np.sin(np.linspace(0, 4, T)).astype(np.float32)
        traj = _numpy_recurrence(mem._Abar_full, mem._Bbar_full, u.reshape(T, 1))
        m_final = traj[-1]
        for tau in [0.0, 0.5, 1.0]:
            k = int(round(tau / mem.dt))
            recon = float(mem.decode(jnp.asarray(m_final[None]), tau)[0, 0])
            assert abs(recon - u[T - 1 - k]) < 0.1

    def test_C_shape(self):
        net = PCNetwork(seed=0)
        with net:
            l_in = Layer(dim=3, activation=pcn.Direct(), label='in')
            mem = Memory(l_in, dims_per_input=5, theta=1.0, dt=0.1)
        net.build()
        C = mem.C(0.5)
        assert C.shape == (3, 15)

    def test_lagt_recurrence_runs_but_C_unimplemented(self):
        net = PCNetwork(seed=0)
        with net:
            l_in = Layer(dim=1, activation=pcn.Direct(), label='in')
            mem = Memory(l_in, dims_per_input=6, memory_type='lagt',
                         theta=1.0, dt=0.1)
        net.build()
        # Discretization produced finite, stable (contractive) dynamics.
        assert np.all(np.isfinite(mem._Abar_full))
        assert np.max(np.abs(np.linalg.eigvals(mem._Abar_full))) < 1.0 + 1e-6
        with pytest.raises(NotImplementedError):
            mem.C(0.5)


class TestMemoryModes:
    def test_deterministic_isolates_recurrent_from_readout(self):
        """A learned readout on Memory.value must not perturb the recurrent
        layer: rec still matches the pure numpy recurrence."""
        net = PCNetwork(seed=0)
        with net:
            l_in = Layer(dim=1, activation=pcn.Direct(), label='in')
            mem = Memory(l_in, dims_per_input=8, theta=1.0, dt=0.1,
                         mode='deterministic')
            l_out = Layer(dim=2, activation=pcn.Direct(), label='out')
            Predict(mem.value, l_out)   # readout reads the output mirror
        net.build()

        T = 25
        u = np.sin(np.linspace(0, 5, T)).astype(np.float32)
        y = np.ones((1, T, 2), dtype=np.float32)
        traj = _numpy_recurrence(mem._Abar_full, mem._Bbar_full, u.reshape(T, 1))

        sim = Simulation(net)
        sim.train([{'in': u.reshape(1, T, 1), 'y': y}],
                  data_map={l_in: 'in', l_out: 'y'},
                  iterations_per_sample=T, learning_iterations_per_sample=0,
                  log_every=1, feedforward_init=False)
        # Re-run inference to read the (isolated) recurrent trajectory.
        res = sim.test([{'in': u.reshape(1, T, 1)}], data_map={l_in: 'in'},
                       iterations_per_sample=T, log_every=1, feedforward_init=False, return_logs=True)
        rec_net = np.array(res['values'][net['memory_rec']])[0]
        assert np.max(np.abs(rec_net - traj)) < 1e-4

    def test_output_mirrors_recurrent_with_one_step_lag(self):
        net = PCNetwork(seed=0)
        with net:
            l_in = Layer(dim=1, activation=pcn.Direct(), label='in')
            mem = Memory(l_in, dims_per_input=6, theta=1.0, dt=0.1,
                         mode='deterministic')
        net.build()
        T = 20
        u = np.cos(np.linspace(0, 4, T)).astype(np.float32)
        sim = Simulation(net)
        res = sim.test([{'in': u.reshape(1, T, 1)}], data_map={l_in: 'in'},
                       iterations_per_sample=T, log_every=1, feedforward_init=False, return_logs=True)
        rec_net = np.array(res['values'][net['memory_rec']])[0]
        out_net = np.array(res['values'][net['memory_out']])[0]
        # Jacobi mirror: out[t] == rec[t-1].
        assert np.max(np.abs(out_net[1:] - rec_net[:-1])) < 1e-4

    def test_value_points_to_right_layer(self):
        net = PCNetwork(seed=0)
        with net:
            l_in = Layer(dim=1, activation=pcn.Direct(), label='in')
            mem_d = Memory(l_in, dims_per_input=4, label='md',
                           mode='deterministic')
            mem_e = Memory(l_in, dims_per_input=4, label='me',
                           mode='energy_coupled')
        net.build()
        assert mem_d.value.owner.label == 'md_out'
        assert mem_d.output is not None
        assert mem_e.value.owner.label == 'me_rec'
        assert mem_e.output is None


class TestMemoryValidation:
    def test_bad_memory_type(self):
        net = PCNetwork(seed=0)
        with net:
            l_in = Layer(dim=1, label='in')
            with pytest.raises(ValueError):
                Memory(l_in, dims_per_input=4, memory_type='legs')

    def test_bad_mode(self):
        net = PCNetwork(seed=0)
        with net:
            l_in = Layer(dim=1, label='in')
            with pytest.raises(ValueError):
                Memory(l_in, dims_per_input=4, mode='hybrid')

    def test_bad_dims(self):
        net = PCNetwork(seed=0)
        with net:
            l_in = Layer(dim=1, label='in')
            with pytest.raises(ValueError):
                Memory(l_in, dims_per_input=0)


class TestAdvanceTimestep:
    """Memory(advance='timestep') gates the recurrence once per input frame."""

    def _mem_states(self, advance, iters_per_frame, seq):
        import optax
        T, Din = seq.shape[1], seq.shape[2]
        net = PCNetwork(seed=0)
        with net:
            l_in = Layer(Din, activation=pcn.Direct(), label='in')
            mem = Memory(l_in.value, dims_per_input=4, memory_type='legt',
                         theta=float(T), dt=1.0, mode='deterministic',
                         advance=advance, label='m')
        net.build()
        sim = Simulation(net)
        freeze = {mem.recurrent._idx, mem.output._idx}
        labels = tuple('f' if i in freeze else 'r' for i in range(len(net._layers)))
        vopt = optax.multi_transform(
            {'r': optax.sgd(0.5), 'f': optax.set_to_zero()}, labels)
        res = sim.test([{'in': seq}], data_map={l_in: 'in'},
                       iterations_per_sample=T * iters_per_frame,
                       log_every=iters_per_frame, feedforward_init=True,
                       values_optimizer=vopt, return_logs=True)
        return np.array(res['values'][net['m_rec']])[0, -T:, :]

    def test_advance_arg_validation(self):
        net = PCNetwork(seed=0)
        with net:
            l = Layer(2, activation=pcn.Direct(), label='in')
            with pytest.raises(ValueError, match="advance"):
                Memory(l.value, dims_per_input=3, advance='bogus')

    def test_repr_reports_advance(self):
        net = PCNetwork(seed=0)
        with net:
            l = Layer(2, activation=pcn.Direct(), label='in')
            mem = Memory(l.value, dims_per_input=3, advance='timestep')
        assert 'advance=timestep' in repr(mem)

    def test_timestep_invariant_to_iters_per_frame(self):
        """The per-frame memory state must not depend on relaxation depth."""
        rng = np.random.RandomState(0)
        seq = rng.randn(1, 6, 2).astype(np.float32)
        base = self._mem_states('timestep', 1, seq)
        assert np.mean(np.abs(np.diff(base, axis=0))) > 1e-3   # actually integrating
        for ipf in (2, 3, 5):
            s = self._mem_states('timestep', ipf, seq)
            np.testing.assert_allclose(s, base, atol=1e-4)

    def test_timestep_at_one_matches_iteration(self):
        """advance='timestep' with 1 iter/frame == legacy advance='iteration'."""
        rng = np.random.RandomState(1)
        seq = rng.randn(1, 6, 2).astype(np.float32)
        np.testing.assert_allclose(self._mem_states('timestep', 1, seq),
                                   self._mem_states('iteration', 1, seq), atol=1e-4)

    def test_iteration_over_advances(self):
        """Contrast: advance='iteration' integrates iters_per_frame times too fast."""
        rng = np.random.RandomState(2)
        seq = rng.randn(1, 6, 2).astype(np.float32)
        one = self._mem_states('iteration', 1, seq)
        three = self._mem_states('iteration', 3, seq)
        assert np.abs(three - one).max() > 0.1
