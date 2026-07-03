"""
Test simulation runs and outputs have correct shapes.
"""

import pytest
import jax
import jax.numpy as jnp

from pcn.core.state import NetworkParams


class TestRunBatch:
    """Test the consolidated run_batch function."""

    def test_run_batch_shapes(self, simple_network, rng_key):
        """Test run_batch returns correct shapes."""
        from pcn.backend.simulation import run_batch

        net, _ = simple_network
        batch_size = 8
        n_iterations = 20
        log_every = 5

        sample = {
            'input': jax.random.normal(rng_key, (batch_size, 16)),
            'output': jax.nn.softmax(jax.random.normal(rng_key, (batch_size, 4)), axis=-1)
        }
        data_map = ((0, 'input'), (2, 'output'))

        new_params, _, _, values_log, errors_log, precisions_log, _, energies = run_batch(
            sample, net.params,
            net.structure,
            data_map, n_iterations, log_every,
            learning=True
        )

        # Check weights
        assert isinstance(new_params, NetworkParams)
        assert len(new_params.predict_weights) == len(net.params.predict_weights)
        for nw, ow in zip(new_params.predict_weights, net.params.predict_weights):
            assert nw.shape == ow.shape
        assert len(new_params.project_weights) == len(net.params.project_weights)
        assert len(new_params.modulate_weights) == len(net.params.modulate_weights)

        # Check values_log (n_logged, batch, dim) per layer; errors_log per predict connection
        expected_n_logged = (n_iterations + log_every - 1) // log_every
        assert len(values_log) == 3
        assert len(errors_log) == 2  # 2 predict connections
        assert values_log[0].shape == (expected_n_logged, batch_size, 16)
        assert values_log[1].shape == (expected_n_logged, batch_size, 8)
        assert values_log[2].shape == (expected_n_logged, batch_size, 4)
        # Error shapes match predict connection post_dim
        assert errors_log[0].shape == (expected_n_logged, batch_size, 16)  # hidden->input, post_dim=16
        assert errors_log[1].shape == (expected_n_logged, batch_size, 8)   # output->hidden, post_dim=8

        # Check energies: should have expected_n_logged entries
        assert energies.shape == (expected_n_logged, )

    def test_run_batch_no_learning(self, simple_network, rng_key):
        """Test run_batch with learning=False doesn't change weights."""
        from pcn.backend.simulation import run_batch

        net, _ = simple_network
        batch_size = 4

        sample = {
            'input': jax.random.normal(rng_key, (batch_size, 16)),
            'output': jax.nn.softmax(jax.random.normal(rng_key, (batch_size, 4)), axis=-1)
        }
        data_map = ((0, 'input'), (2, 'output'))

        # Snapshot weights before run_batch donates net.params buffers.
        predict_before = tuple(jnp.array(w) for w in net.params.predict_weights)
        project_before = tuple(jnp.array(w) for w in net.params.project_weights)
        modulate_before = tuple(jnp.array(w) for w in net.params.modulate_weights)

        new_params, _, _, _, _, _, _, _ = run_batch(
            sample, net.params,
            net.structure,
            data_map, 50, 50,
            learning=False
        )

        # Weights should be unchanged
        for nw, ow in zip(new_params.predict_weights, predict_before):
            assert jnp.allclose(nw, ow)
        for nw, ow in zip(new_params.project_weights, project_before):
            assert jnp.allclose(nw, ow)
        for nw, ow in zip(new_params.modulate_weights, modulate_before):
            assert jnp.allclose(nw, ow)

    def test_early_stopping_produces_valid_results(self, simple_network, rng_key):
        """Early stopping should produce valid (non-NaN) results."""
        from pcn.backend.simulation import run_batch

        net, _ = simple_network
        batch_size = 8

        sample = {
            'input': jax.random.normal(rng_key, (batch_size, 16)),
            'output': jax.nn.softmax(jax.random.normal(rng_key, (batch_size, 4)), axis=-1)
        }
        data_map = ((0, 'input'), (2, 'output'))

        _, _, _, values_log, errors_log, _, _, energies = run_batch(
            sample, net.params,
            net.structure,
            data_map, 200, 200,
            learning=False,
            convergence_threshold=1e-3
        )

        # Values and errors should be valid (check final logged state)
        for v in values_log:
            assert not jnp.any(jnp.isnan(v[-1]))
        for e in errors_log:
            assert not jnp.any(jnp.isnan(e[-1]))

    def test_early_stopping_matches_full_run(self, simple_network, rng_key):
        """Early stopping should give similar results to a full run when converged."""
        import equinox as eqx
        from pcn.backend.simulation import run_batch

        net, _ = simple_network
        batch_size = 8

        sample = {
            'input': jax.random.normal(rng_key, (batch_size, 16)),
            'output': jax.nn.softmax(jax.random.normal(rng_key, (batch_size, 4)), axis=-1)
        }
        data_map = ((0, 'input'), (2, 'output'))

        # run_batch donates its array inputs; clone them so the second call has
        # live buffers.
        clone = lambda tree: jax.tree.map(
            lambda x: jnp.array(x) if eqx.is_array(x) else x, tree
        )
        sample2 = clone(sample)
        params2 = clone(net.params)
        structure2 = clone(net.structure)

        # Run without early stopping (full 200 iterations)
        _, _, _, values_log_full, _, _, _, _ = run_batch(
            sample, net.params,
            net.structure,
            data_map, 200, 200,
            learning=False,
            convergence_threshold=0.0
        )

        # Run with early stopping (loose threshold, still enough to converge)
        _, _, _, values_log_es, _, _, _, _ = run_batch(
            sample2, params2,
            structure2,
            data_map, 200, 200,
            learning=False,
            convergence_threshold=1e-5
        )

        # Both should give very similar results (compare final logged state)
        for v_full, v_es in zip(values_log_full, values_log_es):
            assert jnp.allclose(v_full[-1], v_es[-1], atol=1e-4)

    def test_early_stopping_disabled_by_default(self, simple_network, rng_key):
        """With threshold=0 (default), all iterations should run."""
        from pcn.backend.simulation import run_batch

        net, _ = simple_network
        batch_size = 4

        sample = {
            'input': jax.random.normal(rng_key, (batch_size, 16)),
            'output': jax.nn.softmax(jax.random.normal(rng_key, (batch_size, 4)), axis=-1)
        }
        data_map = ((0, 'input'), (2, 'output'))

        # Run with default threshold (0.0) - should run all iterations
        _, _, _, values_default, _, _, _, energies = run_batch(
            sample, net.params,
            net.structure,
            data_map, 20, 20,
            learning=False,
            convergence_threshold=0.0
        )

        # Energy should be logged (non-zero final entry)
        assert energies.shape[0] == 1  # 20//20 = 1 logged entry
        assert jnp.any(energies[-1] != 0.0)
