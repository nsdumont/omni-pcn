"""Tests for the BPTT backend (run_bptt_batch) and BPTTSimulation frontend.

Key invariants:
- The forward trajectory is identical to run_batch's inference phase.
- truncation=0 + energy_final reproduces the PC gradient dW E(v_T, W) exactly.
- Full BPTT adds a through-time term (differs from truncation=0 pre-convergence).
"""

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from pcn.core.state import NetworkParams


def _copy_params(params):
    """Fresh device copies — run_batch/run_bptt_batch donate their inputs."""
    return NetworkParams(**{f: [jnp.array(w) for w in getattr(params, f)] for f in params._fields})


def _mk_sample(rng_key, batch_size=8):
    k1, k2 = jax.random.split(rng_key)
    return {
        "input": jax.random.normal(k1, (batch_size, 16)),
        "output": jax.nn.softmax(jax.random.normal(k2, (batch_size, 4)), axis=-1),
    }


DATA_MAP = ((0, "input"), (2, "output"))


class TestBPTTBackend:
    def test_forward_trajectory_matches_pc(self, simple_network, rng_key):
        """Per-step energies must match run_batch's eval trajectory: BPTT's
        step energy i+1 (pre-update at v_{i+1}) == run_batch's logged energy i
        (recompute at v_{i+1}), and the final recompute matches the last log.
        """
        from pcn.backend.bptt_simulation import run_bptt_batch
        from pcn.backend.simulation import run_batch

        net, _ = simple_network
        T = 6
        sample = _mk_sample(rng_key)

        *_, pc_energies = run_batch(
            {k: jnp.array(v) for k, v in sample.items()},
            _copy_params(net.params),
            net.structure,
            DATA_MAP,
            T,
            1,
            learning=False,
        )

        *_, bptt_energies, _ = run_bptt_batch(
            {k: jnp.array(v) for k, v in sample.items()},
            _copy_params(net.params),
            net.structure,
            DATA_MAP,
            T,
        )

        assert pc_energies.shape == (T,)
        assert bptt_energies.shape == (T + 1,)
        np.testing.assert_allclose(
            np.asarray(bptt_energies[1:]), np.asarray(pc_energies), rtol=1e-5, atol=1e-6
        )

    def test_final_values_match_pc(self, simple_network, rng_key):
        """Final values out of the unroll must bit-match run_batch's."""
        from pcn.backend.bptt_simulation import run_bptt_batch
        from pcn.backend.simulation import run_batch

        net, _ = simple_network
        T = 6
        sample = _mk_sample(rng_key)

        _, _, _, pc_values_log, *_ = run_batch(
            {k: jnp.array(v) for k, v in sample.items()},
            _copy_params(net.params),
            net.structure,
            DATA_MAP,
            T,
            T,
            learning=False,
        )

        _, _, _, bptt_values_log, *_ = run_bptt_batch(
            {k: jnp.array(v) for k, v in sample.items()},
            _copy_params(net.params),
            net.structure,
            DATA_MAP,
            T,
        )

        for pv, bv in zip(pc_values_log, bptt_values_log, strict=False):
            np.testing.assert_allclose(np.asarray(pv[-1]), np.asarray(bv[-1]), rtol=1e-5, atol=1e-6)

    def test_trunc0_matches_pc_equilibrium_gradient(self, simple_network, rng_key):
        """truncation=0 + energy_final + sgd(lr) must step the predict weights
        by exactly -lr * dW E(v_T, W) with v_T held fixed (the PC rule).
        """
        from pcn.backend.bptt_simulation import run_bptt_batch
        from pcn.backend.simulation import _compute_energy, _recompute_errors_precisions, run_batch
        from pcn.core.activations import ACTIVATIONS

        net, _ = simple_network
        T, lr = 8, 1e-2
        sample = _mk_sample(rng_key)
        structure = net.structure

        # Reference: forward-only trajectory to v_T, then grad of the energy
        # at fixed v_T w.r.t. predict weights.
        _, _, _, values_log, *_ = run_batch(
            {k: jnp.array(v) for k, v in sample.items()},
            _copy_params(net.params),
            net.structure,
            DATA_MAP,
            T,
            T,
            learning=False,
        )
        vT = tuple(jnp.array(v[-1]) for v in values_log)

        act_fns = tuple(ACTIVATIONS[l.activation_type] for l in structure.layers)
        pw0 = tuple(jnp.array(w) for w in net.params.predict_weights)
        pb0 = tuple(jnp.array(b) for b in net.params.predict_biases)
        ppw0 = tuple(jnp.array(w) for w in net.params.precision_weights)
        ppb0 = tuple(jnp.array(b) for b in net.params.precision_biases)

        def ref_loss(pw):
            e, p = _recompute_errors_precisions(
                vT, structure.predict_conns, pw, pb0, ppw0, ppb0, act_fns
            )
            return _compute_energy(e, p, structure.predict_pre_scales)

        g_ref = jax.grad(ref_loss)(pw0)

        new_params, *_ = run_bptt_batch(
            {k: jnp.array(v) for k, v in sample.items()},
            _copy_params(net.params),
            net.structure,
            DATA_MAP,
            T,
            loss_mode="energy_final",
            truncation=0,
            params_optimizer=optax.sgd(lr),
        )

        for nw, ow, g in zip(new_params.predict_weights, pw0, g_ref, strict=False):
            np.testing.assert_allclose(
                np.asarray(nw - ow), np.asarray(-lr * g), rtol=1e-4, atol=1e-7
            )

    def test_full_bptt_adds_through_time_term(self, simple_network, rng_key):
        """Pre-convergence, the full-BPTT update must differ from trunc=0."""
        from pcn.backend.bptt_simulation import run_bptt_batch

        net, _ = simple_network
        T, lr = 8, 1e-2
        sample = _mk_sample(rng_key)
        pw0 = tuple(jnp.array(w) for w in net.params.predict_weights)

        def run(truncation):
            new_params, *_ = run_bptt_batch(
                {k: jnp.array(v) for k, v in sample.items()},
                _copy_params(net.params),
                net.structure,
                DATA_MAP,
                T,
                loss_mode="energy_final",
                truncation=truncation,
                params_optimizer=optax.sgd(lr),
            )
            return tuple(jnp.array(w) for w in new_params.predict_weights)

        pw_full = run(None)
        pw_pc = run(0)

        # Both moved the weights...
        assert any(
            float(jnp.max(jnp.abs(nw - ow))) > 1e-8 for nw, ow in zip(pw_full, pw0, strict=False)
        )
        # ...but not identically.
        assert any(
            float(jnp.max(jnp.abs(a - b))) > 1e-8 for a, b in zip(pw_full, pw_pc, strict=False)
        )

    def test_objective_mode_trains(self, simple_network, rng_key):
        """RNN-style training: input clamped, output free, CE at final iterate."""
        from pcn.backend.bptt_simulation import run_bptt_batch

        net, _ = simple_network
        T = 8
        batch_size = 64
        k1, k2 = jax.random.split(rng_key)
        x = jax.random.normal(k1, (batch_size, 16))
        A = jax.random.normal(k2, (16, 4))
        y = jax.nn.one_hot(jnp.argmax(x @ A, axis=-1), 4)

        def ce(values, sample, t):
            return -jnp.mean(jnp.sum(sample["output"] * jax.nn.log_softmax(values[2]), axis=-1))

        data_map = ((0, "input"),)  # output NOT clamped
        params = _copy_params(net.params)
        opt = optax.adam(3e-2)
        opt_state = None
        losses = []
        for _ in range(150):
            sample = {"input": jnp.array(x), "output": jnp.array(y)}
            params, opt_state, _, *_, loss = run_bptt_batch(
                sample,
                params,
                net.structure,
                data_map,
                T,
                loss_mode="objective",
                objective_fn=ce,
                params_optimizer=opt,
                params_opt_state=opt_state,
            )
            losses.append(float(loss))

        assert np.isfinite(losses).all()
        assert np.mean(losses[-5:]) < 0.7 * np.mean(losses[:5])

    def test_objective_sum_equals_objective_on_static_task(self, simple_network, rng_key):
        """With one timestep, objective_sum's only contribution is the final
        iterate, so its loss equals 'objective'.
        """
        from pcn.backend.bptt_simulation import run_bptt_batch

        net, _ = simple_network
        T = 6
        sample = _mk_sample(rng_key)

        def ce(values, sample_, t):
            return -jnp.mean(jnp.sum(sample_["output"] * jax.nn.log_softmax(values[2]), axis=-1))

        losses = {}
        for mode in ("objective", "objective_sum"):
            *_, loss = run_bptt_batch(
                {k: jnp.array(v) for k, v in sample.items()},
                _copy_params(net.params),
                net.structure,
                ((0, "input"),),
                T,
                loss_mode=mode,
                objective_fn=ce,
                params_optimizer=optax.sgd(1e-3),
            )
            losses[mode] = float(loss)

        assert losses["objective"] == pytest.approx(losses["objective_sum"], rel=1e-5)

    def test_energy_sum_and_truncation_smoke(self, simple_network, rng_key):
        from pcn.backend.bptt_simulation import run_bptt_batch

        net, _ = simple_network
        pw0 = tuple(jnp.array(w) for w in net.params.predict_weights)
        for kwargs in (
            {"loss_mode": "energy_sum"},
            {"loss_mode": "energy_final", "truncation": 2},
            {"loss_mode": "energy_final", "remat": False},
        ):
            new_params, *_, energies, loss = run_bptt_batch(
                {k: jnp.array(v) for k, v in _mk_sample(rng_key).items()},
                _copy_params(net.params),
                net.structure,
                DATA_MAP,
                6,
                params_optimizer=optax.sgd(1e-2),
                **kwargs,
            )
            assert np.isfinite(float(loss))
            assert np.isfinite(np.asarray(energies)).all()
            assert any(
                float(jnp.max(jnp.abs(nw - ow))) > 1e-8
                for nw, ow in zip(new_params.predict_weights, pw0, strict=False)
            )

    def test_learn_precision_false_freezes_precision_params(self, rng_key):
        """Per-conn learn_precision=False freezes precision under BPTT, as in
        run_batch (more flag cases: test_precision_schedule.py)."""
        import pcn
        from pcn.backend.bptt_simulation import run_bptt_batch

        # simple_network with precision learning off on both conns.
        net = pcn.PCNetwork(seed=0)
        with net:
            l1 = pcn.Layer(dim=16, activation=pcn.Direct(), label="input")
            l2 = pcn.Layer(dim=8, activation=pcn.Relu(), label="hidden")
            l3 = pcn.Layer(dim=4, activation=pcn.Softmax(), label="output")
            pcn.Predict(l2, l1, learn_precision=False)
            pcn.Predict(l3, l2, learn_precision=False)
        net.build()
        ppw0 = tuple(jnp.array(w) for w in net.params.precision_weights)
        ppb0 = tuple(jnp.array(b) for b in net.params.precision_biases)

        new_params, *_ = run_bptt_batch(
            {k: jnp.array(v) for k, v in _mk_sample(rng_key).items()},
            _copy_params(net.params),
            net.structure,
            DATA_MAP,
            6,
            params_optimizer=optax.adamw(1e-2, weight_decay=0.5),
        )

        for nw, ow in zip(new_params.precision_weights, ppw0, strict=False):
            np.testing.assert_array_equal(np.asarray(nw), np.asarray(ow))
        for nb, ob in zip(new_params.precision_biases, ppb0, strict=False):
            np.testing.assert_array_equal(np.asarray(nb), np.asarray(ob))

    def test_temporal_unroll(self, simple_network, rng_key):
        """3-timestep clamp with 2 iterations per timestep."""
        from pcn.backend.bptt_simulation import run_bptt_batch

        net, _ = simple_network
        batch_size, n_ts, T = 4, 3, 6
        k1, k2 = jax.random.split(rng_key)
        sample = {
            "input": jax.random.normal(k1, (batch_size, n_ts, 16)),
            "output": jax.nn.softmax(jax.random.normal(k2, (batch_size, n_ts, 4)), axis=-1),
        }

        *_, energies, loss = run_bptt_batch(
            sample,
            _copy_params(net.params),
            net.structure,
            DATA_MAP,
            T,
            params_optimizer=optax.sgd(1e-3),
        )
        assert energies.shape == (T + 1,)
        assert np.isfinite(float(loss))

    def test_invalid_args_raise(self, simple_network, rng_key):
        from pcn.backend.bptt_simulation import run_bptt_batch

        net, _ = simple_network
        sample = _mk_sample(rng_key)

        with pytest.raises(ValueError, match="loss_mode"):
            run_bptt_batch(
                sample, _copy_params(net.params), net.structure, DATA_MAP, 6, loss_mode="nope"
            )
        with pytest.raises(ValueError, match="objective_fn"):
            run_bptt_batch(
                sample, _copy_params(net.params), net.structure, DATA_MAP, 6, loss_mode="objective"
            )
        with pytest.raises(ValueError, match="truncation=0"):
            run_bptt_batch(
                sample,
                _copy_params(net.params),
                net.structure,
                DATA_MAP,
                6,
                loss_mode="objective",
                objective_fn=lambda v, s, t: 0.0,
                truncation=0,
            )


class TestBPTTSimulation:
    def test_train_and_test_roundtrip(self, simple_network, rng_key):
        import pcn

        net, (l1, l2, l3) = simple_network
        batch_size = 8
        loader = []
        for i in range(3):
            k = jax.random.fold_in(rng_key, i)
            loader.append(
                {
                    "input": np.asarray(jax.random.normal(k, (batch_size, 16))),
                    "output": np.asarray(
                        jax.nn.one_hot(jax.random.randint(k, (batch_size,), 0, 4), 4)
                    ),
                }
            )

        pw_before = tuple(jnp.array(w) for w in net.params.predict_weights)

        sim = pcn.BPTTSimulation(net, loss_mode="energy_final")
        sim.train(
            loader,
            data_map={l1: "input", l3: "output"},
            epochs=2,
            iterations_per_sample=6,
            params_optimizer=optax.adam(1e-3),
        )

        assert len(sim.train_losses) == 6
        assert np.isfinite(sim.train_losses).all()
        assert any(
            float(jnp.max(jnp.abs(nw - ow))) > 1e-8
            for nw, ow in zip(net.params.predict_weights, pw_before, strict=False)
        )

        def acc(values, targets):
            return float(jnp.mean(jnp.argmax(values, axis=-1) == jnp.argmax(targets, axis=-1)))

        results = sim.test(
            loader, data_map={l1: "input"}, record_map={"acc": ((l3.value, "output"), acc)}
        )
        assert len(results["acc"]) == 3

    def test_objective_fn_two_arg_wrapping(self, simple_network, rng_key):
        import pcn

        net, (l1, l2, l3) = simple_network
        batch_size = 8
        k = rng_key
        loader = [
            {
                "input": np.asarray(jax.random.normal(k, (batch_size, 16))),
                "output": np.asarray(jax.nn.one_hot(jax.random.randint(k, (batch_size,), 0, 4), 4)),
            }
        ]

        def ce(values, sample):
            return -jnp.mean(jnp.sum(sample["output"] * jax.nn.log_softmax(values[2]), axis=-1))

        sim = pcn.BPTTSimulation(net, loss_mode="objective", objective_fn=ce)
        sim.train(loader, data_map={l1: "input"}, epochs=1, iterations_per_sample=4)
        assert len(sim.train_losses) == 1
        assert np.isfinite(sim.train_losses[0])
