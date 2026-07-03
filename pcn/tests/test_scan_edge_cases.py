"""
Edge case tests for lax.scan-based inference and learning loops.
Tests temporal clamping, PM error-reading, convergence, log_every variations,
and mixed inference+learning phases.
"""
import pytest
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pcn
from pcn.backend.simulation import run_batch


@pytest.fixture
def simple_net():
    """Standard 3-layer discriminative network."""
    net = pcn.PCNetwork(seed=42)
    net.config(use_bias=True, learn_precision=False)
    with net:
        l_in = pcn.Layer(dim=8, activation=pcn.LeakyRelu(), label="in")
        l_h = pcn.Layer(dim=4, activation=pcn.LeakyRelu(), label="h")
        l_out = pcn.Layer(dim=3, activation=pcn.Softmax(), label="out")
        pcn.Predict(l_in, l_h)
        pcn.Predict(l_h, l_out)
    net.build()
    return net, l_in, l_h, l_out


@pytest.fixture
def pm_error_net():
    """Network with Modulate targeting error (uses project_conns_internal/modulate_conns_internal)."""
    net = pcn.PCNetwork(seed=42)
    net.config(use_bias=True, learn_precision=False)
    with net:
        l_in = pcn.Layer(dim=8, activation=pcn.LeakyRelu(), label="in")
        l_h = pcn.Layer(dim=4, activation=pcn.LeakyRelu(), label="h")
        l_out = pcn.Layer(dim=3, activation=pcn.Softmax(), label="out")
        p1 = pcn.Predict(l_in, l_h)
        p2 = pcn.Predict(l_h, l_out)
        pcn.Modulate(l_in, p2.error)
    net.build()
    return net, l_in, l_h, l_out


def _make_sample(batch_size, in_dim, out_dim, key=None):
    if key is None:
        key = jax.random.PRNGKey(0)
    k1, k2 = jax.random.split(key)
    images = jax.random.normal(k1, (batch_size, in_dim))
    labels = jax.nn.one_hot(jax.random.randint(k2, (batch_size,), 0, out_dim), out_dim)
    return {'image': images, 'label': labels}


def _run(net, l_in, l_out, sample, n_iter, log_every, learning, n_learn, convergence=0.0):
    data_map = ((l_in._idx, 'image'), (l_out._idx, 'label'))
    vo = optax.adam(0.5)
    po = optax.adam(1e-3) if learning else None
    return run_batch(
        sample, net.params, net.structure,
        data_map, n_iter, log_every,
        learning=learning, n_learning_iterations=n_learn,
        convergence_threshold=convergence,
        key=jax.random.PRNGKey(1),
        values_optimizer=vo, values_opt_state=None,
        params_optimizer=po,
        params_opt_state=po.init({
            'predict_weights': tuple(net.params.predict_weights),
            'predict_biases': tuple(net.params.predict_biases),
            'project_biases': tuple(net.params.project_biases),
            'modulate_biases': tuple(net.params.modulate_biases),
            'precision_weights': tuple(net.params.precision_weights),
            'precision_biases': tuple(net.params.precision_biases),
            'gd_loss_project_weights': (),
            'gd_loss_modulate_weights': (),
        }) if learning else None,
        spatial_neighborhoods=(),
    )


# --- Test: scan path with inference only ---

class TestScanInference:
    def test_inference_only_basic(self, simple_net):
        """Inference-only: scan path produces correct shapes."""
        net, l_in, _, l_out = simple_net
        sample = _make_sample(4, 8, 3)
        result = _run(net, l_in, l_out, sample, n_iter=5, log_every=5, learning=False, n_learn=0)
        _, _, _, vl, el, pl, _, en = result
        assert en.shape == (1,), f"Expected 1 logged energy, got {en.shape}"
        assert vl[0].shape == (1, 4, 8), f"values_log shape mismatch: {vl[0].shape}"

    def test_inference_log_every_1(self, simple_net):
        """log_every=1: every iteration is logged."""
        net, l_in, _, l_out = simple_net
        sample = _make_sample(4, 8, 3)
        result = _run(net, l_in, l_out, sample, n_iter=5, log_every=1, learning=False, n_learn=0)
        _, _, _, vl, el, pl, _, en = result
        assert en.shape == (5,), f"Expected 5 logged energies, got {en.shape}"
        assert vl[0].shape == (5, 4, 8)

    def test_inference_log_every_2(self, simple_net):
        """log_every=2 with 6 iterations: 3 log entries."""
        net, l_in, _, l_out = simple_net
        sample = _make_sample(4, 8, 3)
        result = _run(net, l_in, l_out, sample, n_iter=6, log_every=2, learning=False, n_learn=0)
        _, _, _, vl, el, pl, _, en = result
        assert en.shape == (3,), f"Expected 3 logged energies, got {en.shape}"

    def test_inference_energy_decreases(self, simple_net):
        """Energy should generally decrease during inference (log_every=1)."""
        net, l_in, _, l_out = simple_net
        sample = _make_sample(4, 8, 3)
        result = _run(net, l_in, l_out, sample, n_iter=10, log_every=1, learning=False, n_learn=0)
        _, _, _, vl, el, pl, _, en = result
        # Energy at end should be less than at start
        assert en[-1] < en[0], f"Energy did not decrease: {en[0]} -> {en[-1]}"


# --- Test: scan path with learning only ---

class TestScanLearning:
    def test_learning_only(self, simple_net):
        """Learning only (n_iterations=0, n_learning_iterations=5)."""
        net, l_in, _, l_out = simple_net
        sample = _make_sample(4, 8, 3)
        # Snapshot predict weights before _run; run_batch donates net.params.
        predict_before = tuple(jnp.array(w) for w in net.params.predict_weights)
        result = _run(net, l_in, l_out, sample, n_iter=0, log_every=5, learning=True, n_learn=5)
        new_params, _, _, vl, el, pl, _, en = result
        assert en.shape == (1,), f"Expected 1 logged energy, got {en.shape}"
        # Weights should have changed
        for w_old, w_new in zip(predict_before, new_params.predict_weights):
            assert not jnp.allclose(w_old, w_new), "Weights should change after learning"

    def test_learning_log_every_1(self, simple_net):
        """Learning with log_every=1: every learning step logged."""
        net, l_in, _, l_out = simple_net
        sample = _make_sample(4, 8, 3)
        result = _run(net, l_in, l_out, sample, n_iter=0, log_every=1, learning=True, n_learn=5)
        _, _, _, vl, el, pl, _, en = result
        assert en.shape == (5,)


# --- Test: mixed inference + learning ---

class TestScanMixed:
    def test_inference_then_learning(self, simple_net):
        """Both inference and learning phases, each contributing log entries."""
        net, l_in, _, l_out = simple_net
        sample = _make_sample(4, 8, 3)
        # 5 inference + 5 learning = 10 total, log_every=5 -> 2 entries
        result = _run(net, l_in, l_out, sample, n_iter=5, log_every=5, learning=True, n_learn=5)
        _, _, _, vl, el, pl, _, en = result
        assert en.shape == (2,), f"Expected 2 logged energies, got {en.shape}"

    def test_mixed_log_every_3(self, simple_net):
        """Mixed with log_every=3, n_iter=3, n_learn=6 -> total=9 -> 3 entries."""
        net, l_in, _, l_out = simple_net
        sample = _make_sample(4, 8, 3)
        result = _run(net, l_in, l_out, sample, n_iter=3, log_every=3, learning=True, n_learn=6)
        _, _, _, vl, el, pl, _, en = result
        # Logged at global i=2,5,8 -> 3 entries
        assert en.shape == (3,), f"Expected 3 energies, got {en.shape}"

    def test_mixed_uneven_log(self, simple_net):
        """Mixed with log_every=3, n_iter=2, n_learn=7 -> total=9 -> 3 entries.
        No inference logs (iterations 0,1 never hit (i+1)%3==0).
        All 3 logs come from learning phase."""
        net, l_in, _, l_out = simple_net
        sample = _make_sample(4, 8, 3)
        result = _run(net, l_in, l_out, sample, n_iter=2, log_every=3, learning=True, n_learn=7)
        _, _, _, vl, el, pl, _, en = result
        # Global logged at i=2,5,8 -> i=2 is in inference, i=5,8 in learning
        assert en.shape == (3,), f"Expected 3 energies, got {en.shape}"


# --- Test: convergence path (fori_loop, not scan) ---

class TestConvergencePath:
    def test_convergence_basic(self, simple_net):
        """Convergence path should still work (fori_loop)."""
        net, l_in, _, l_out = simple_net
        sample = _make_sample(4, 8, 3)
        result = _run(net, l_in, l_out, sample, n_iter=10, log_every=10,
                      learning=False, n_learn=0, convergence=1e-3)
        _, _, _, vl, el, pl, _, en = result
        assert en.shape == (1,)

    def test_convergence_with_learning(self, simple_net):
        """Convergence path with learning phase."""
        net, l_in, _, l_out = simple_net
        sample = _make_sample(4, 8, 3)
        result = _run(net, l_in, l_out, sample, n_iter=5, log_every=5,
                      learning=True, n_learn=5, convergence=1e-3)
        new_params, _, _, vl, el, pl, _, en = result
        assert en.shape == (2,)


# --- Test: PM error-reading networks ---

class TestPMErrorReading:
    def test_pm_inference(self, pm_error_net):
        """PM network inference: errors recomputed every iteration."""
        net, l_in, _, l_out = pm_error_net
        sample = _make_sample(4, 8, 3)
        result = _run(net, l_in, l_out, sample, n_iter=5, log_every=5, learning=False, n_learn=0)
        _, _, _, vl, el, pl, _, en = result
        assert en.shape == (1,)
        assert not jnp.isnan(en[0])

    def test_pm_learning(self, pm_error_net):
        """PM network with learning."""
        net, l_in, _, l_out = pm_error_net
        sample = _make_sample(4, 8, 3)
        result = _run(net, l_in, l_out, sample, n_iter=0, log_every=5, learning=True, n_learn=5)
        new_params, _, _, vl, el, pl, _, en = result
        assert en.shape == (1,)
        assert not jnp.isnan(en[0])


# --- Test: temporal clamping (3D data) ---

class TestTemporalClamping:
    def test_temporal_clamp_inference(self, simple_net):
        """Temporal clamping with 3D input data."""
        net, l_in, _, l_out = simple_net
        batch_size = 4
        n_iter = 6
        n_timesteps = 3  # Must divide total_iterations
        # 3D data: (batch, n_timesteps, dim)
        images_3d = jax.random.normal(jax.random.PRNGKey(0), (batch_size, n_timesteps, 8))
        labels_3d = jnp.broadcast_to(
            jax.nn.one_hot(jnp.array([0, 1, 2, 0]), 3)[:, None, :],
            (batch_size, n_timesteps, 3))
        sample = {'image': images_3d, 'label': labels_3d}

        data_map = ((l_in._idx, 'image'), (l_out._idx, 'label'))
        vo = optax.adam(0.5)
        result = run_batch(
            sample, net.params, net.structure,
            data_map, n_iter, n_iter,
            learning=False, n_learning_iterations=0,
            key=jax.random.PRNGKey(1),
            values_optimizer=vo, values_opt_state=None,
            spatial_neighborhoods=(),
        )
        _, _, _, vl, el, pl, _, en = result
        assert en.shape == (1,)
        assert not jnp.isnan(en[0])

    def test_temporal_clamp_with_learning(self, simple_net):
        """Temporal clamping with both inference and learning."""
        net, l_in, _, l_out = simple_net
        batch_size = 4
        n_iter = 4
        n_learn = 4
        n_timesteps = 2  # Must divide total (4+4=8)
        images_3d = jax.random.normal(jax.random.PRNGKey(0), (batch_size, n_timesteps, 8))
        labels_3d = jnp.broadcast_to(
            jax.nn.one_hot(jnp.array([0, 1, 2, 0]), 3)[:, None, :],
            (batch_size, n_timesteps, 3))
        sample = {'image': images_3d, 'label': labels_3d}

        data_map = ((l_in._idx, 'image'), (l_out._idx, 'label'))
        vo = optax.adam(0.5)
        po = optax.adam(1e-3)
        result = run_batch(
            sample, net.params, net.structure,
            data_map, n_iter, n_iter,
            learning=True, n_learning_iterations=n_learn,
            key=jax.random.PRNGKey(1),
            values_optimizer=vo, values_opt_state=None,
            params_optimizer=po,
            params_opt_state=po.init({
                'predict_weights': tuple(net.params.predict_weights),
                'predict_biases': tuple(net.params.predict_biases),
                'project_biases': tuple(net.params.project_biases),
                'modulate_biases': tuple(net.params.modulate_biases),
                'precision_weights': tuple(net.params.precision_weights),
                'precision_biases': tuple(net.params.precision_biases),
                'gd_loss_project_weights': (),
                'gd_loss_modulate_weights': (),
            }),
            spatial_neighborhoods=(),
        )
        new_params, _, _, vl, el, pl, _, en = result
        # 8 total iterations / 4 log_every = 2 entries
        assert en.shape == (2,)
        assert not jnp.any(jnp.isnan(en))


# --- Test: single-pass shortcut ---

class TestSinglePassShortcut:
    def test_single_iteration(self, simple_net):
        """n_iterations=1, no learning: uses single-pass shortcut."""
        net, l_in, _, l_out = simple_net
        sample = _make_sample(4, 8, 3)
        result = _run(net, l_in, l_out, sample, n_iter=1, log_every=1, learning=False, n_learn=0)
        _, _, _, vl, el, pl, _, en = result
        assert en.shape == (1,)
        assert vl[0].shape == (1, 4, 8)
