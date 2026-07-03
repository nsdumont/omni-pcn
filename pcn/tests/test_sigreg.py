"""Tests for SIGReg (Sketched Isotropic Gaussian Regularization).

Three layers of validation:

1. Statistical correctness of the Epps-Pulley statistic: near zero for
   N(0,1) samples, large for shifted / scaled / heavy-tailed / bimodal
   alternatives, and consistent (shrinks with N) under the null.
2. The sliced loss detects departures from *isotropic* N(0, I):
   correlation, anisotropic scaling, global scale.
3. Gradient descent on the loss actually Gaussianizes variables —
   directly on a point cloud, and through an MLP's weights (the NN
   context), measured by mean, covariance-to-identity distance, and
   marginal kurtosis.
"""

import jax
import jax.numpy as jnp
import optax
import pytest

from pcn.core.regularization import SIGReg, _epps_pulley_statistic, _sigreg_loss


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mean_norm(x):
    return float(jnp.linalg.norm(jnp.mean(x, axis=0)))


def _cov_identity_dist(x):
    """Frobenius distance between the sample covariance and I."""
    c = jnp.cov(x, rowvar=False)
    return float(jnp.linalg.norm(c - jnp.eye(c.shape[0])))


def _mean_excess_kurtosis(x):
    """Mean over dims of |kurtosis - 3| (0 for a Gaussian)."""
    mu = jnp.mean(x, axis=0)
    sd = jnp.std(x, axis=0) + 1e-8
    z = (x - mu) / sd
    kurt = jnp.mean(z ** 4, axis=0)
    return float(jnp.mean(jnp.abs(kurt - 3.0)))


# ---------------------------------------------------------------------------
# 1. Epps-Pulley statistic correctness
# ---------------------------------------------------------------------------

class TestEppsPulleyStatistic:
    N = 4096

    def _stat(self, samples):
        return float(_epps_pulley_statistic(samples))

    def test_standard_normal_is_near_zero(self):
        key = jax.random.PRNGKey(0)
        stat = self._stat(jax.random.normal(key, (self.N,)))
        # E[stat] under the null is O(1/N) ~ 2.4e-4 at N=4096.
        assert stat < 5e-3

    @pytest.mark.parametrize("name,sampler", [
        ("shifted", lambda k, n: jax.random.normal(k, (n,)) + 1.0),
        ("scaled_up", lambda k, n: 2.0 * jax.random.normal(k, (n,))),
        ("scaled_down", lambda k, n: 0.5 * jax.random.normal(k, (n,))),
        ("uniform", lambda k, n: jax.random.uniform(k, (n,), minval=-1.0, maxval=1.0)),
        # Unit-variance Laplace: detected purely via tail shape.
        ("laplace", lambda k, n: jax.random.laplace(k, (n,)) / jnp.sqrt(2.0)),
        ("bimodal", lambda k, n: jax.random.normal(k, (n,)) * 0.5
            + jnp.where(jax.random.bernoulli(jax.random.fold_in(k, 1), 0.5, (n,)), 2.0, -2.0)),
    ])
    def test_alternatives_score_higher(self, name, sampler):
        key = jax.random.PRNGKey(0)
        stat_null = self._stat(jax.random.normal(key, (self.N,)))
        stat_alt = self._stat(sampler(key, self.N))
        assert stat_alt > 10.0 * stat_null, (
            f"{name}: stat {stat_alt:.4g} not >> null {stat_null:.4g}")

    def test_consistency_statistic_shrinks_with_n(self):
        """Under the null the statistic is O(1/N); average over seeds."""
        def avg_stat(n):
            stats = [self._stat(jax.random.normal(jax.random.PRNGKey(s), (n,)))
                     for s in range(5)]
            return sum(stats) / len(stats)
        assert avg_stat(8192) < avg_stat(128) / 4.0


# ---------------------------------------------------------------------------
# 2. Sliced loss detects non-isotropy
# ---------------------------------------------------------------------------

class TestSlicedLossDetection:
    N, D, M = 2048, 8, 256

    def _loss(self, x, seed=0):
        return float(_sigreg_loss(x, jax.random.PRNGKey(seed),
                                  num_slices=self.M, num_points=17, t_max=3.0))

    def test_isotropic_gaussian_is_near_zero(self):
        x = jax.random.normal(jax.random.PRNGKey(0), (self.N, self.D))
        assert self._loss(x) < 5e-3

    def test_detects_correlation(self):
        key = jax.random.PRNGKey(0)
        iso = jax.random.normal(key, (self.N, 2))
        # Correlated Gaussian, still unit marginal variance.
        rho = 0.9
        mix = jnp.array([[1.0, 0.0], [rho, jnp.sqrt(1 - rho ** 2)]])
        corr = iso @ mix.T
        assert self._loss(corr) > 10.0 * self._loss(iso)

    def test_detects_anisotropic_scaling(self):
        key = jax.random.PRNGKey(0)
        iso = jax.random.normal(key, (self.N, self.D))
        scales = jnp.linspace(0.25, 2.0, self.D)
        assert self._loss(iso * scales) > 10.0 * self._loss(iso)

    def test_detects_global_scale(self):
        """The target is N(0, I) exactly — not just shape, also scale."""
        key = jax.random.PRNGKey(0)
        iso = jax.random.normal(key, (self.N, self.D))
        assert self._loss(2.0 * iso) > 10.0 * self._loss(iso)
        assert self._loss(0.5 * iso) > 10.0 * self._loss(iso)


# ---------------------------------------------------------------------------
# 3. SIGReg interface
# ---------------------------------------------------------------------------

class TestSIGRegInterface:
    def test_strength_scales_linearly(self):
        x = jax.random.normal(jax.random.PRNGKey(0), (256, 8)) * 1.7
        key = jax.random.PRNGKey(1)
        l1 = SIGReg(strength=1.0, num_slices=64).apply(x, key=key)
        l3 = SIGReg(strength=3.0, num_slices=64).apply(x, key=key)
        assert jnp.allclose(l3, 3.0 * l1, rtol=1e-5)

    def test_seed_fallback_is_deterministic(self):
        x = jax.random.normal(jax.random.PRNGKey(0), (256, 8))
        reg = SIGReg(num_slices=64, seed=7)
        assert jnp.allclose(reg.apply(x), reg.apply(x))

    def test_jit_and_grad(self):
        x = jax.random.uniform(jax.random.PRNGKey(0), (256, 8))
        reg = SIGReg(num_slices=64)
        loss_fn = jax.jit(lambda v: reg.apply(v, key=jax.random.PRNGKey(1)))
        g = jax.grad(lambda v: loss_fn(v))(x)
        assert g.shape == x.shape
        assert bool(jnp.all(jnp.isfinite(g)))
        assert float(jnp.linalg.norm(g)) > 0.0


# ---------------------------------------------------------------------------
# 4. Optimization: SIGReg Gaussianizes variables
# ---------------------------------------------------------------------------

class TestGaussianization:
    def test_point_cloud_becomes_isotropic_gaussian(self):
        """Gradient descent on raw points: uniform cube -> ~N(0, I)."""
        N, D = 512, 8
        x0 = jax.random.uniform(jax.random.PRNGKey(0), (N, D))  # mean .5, cov I/12
        reg = SIGReg(num_slices=256)

        @jax.jit
        def step(x, opt_state, key):
            loss, g = jax.value_and_grad(
                lambda v: reg.apply(v, key=key))(x)
            updates, opt_state = optim.update(g, opt_state, x)
            return optax.apply_updates(x, updates), opt_state, loss

        optim = optax.adam(2e-2)
        x, opt_state = x0, optim.init(x0)
        base = jax.random.PRNGKey(42)
        # Mean/cov converge within ~200 steps; kurtosis (tail shape) is a
        # weakly-weighted higher moment and needs ~3k steps to settle.
        for i in range(3000):
            x, opt_state, loss = step(x, opt_state, jax.random.fold_in(base, i))

        eval_key = jax.random.PRNGKey(999)  # held-out directions
        loss0 = float(reg.apply(x0, key=eval_key))
        loss1 = float(reg.apply(x, key=eval_key))

        assert loss1 < loss0 / 20.0
        assert _mean_norm(x) < 0.2          # from ~1.41
        assert _cov_identity_dist(x) < 0.5  # from ~2.6
        assert _mean_excess_kurtosis(x) < 0.4  # uniform starts at ~1.2

    def test_mlp_outputs_become_isotropic_gaussian(self):
        """NN context: train MLP weights so its embeddings match N(0, I)."""
        N, D_in, D_hid, D_out = 1024, 16, 64, 8
        kx, k1, k2 = jax.random.split(jax.random.PRNGKey(0), 3)
        inputs = jax.random.uniform(kx, (N, D_in), minval=-1.0, maxval=1.0)
        params = {
            'W1': jax.random.normal(k1, (D_in, D_hid)) / jnp.sqrt(D_in),
            'b1': jnp.zeros(D_hid),
            'W2': jax.random.normal(k2, (D_hid, D_out)) / jnp.sqrt(D_hid),
            'b2': jnp.zeros(D_out),
        }

        def forward(p, x):
            h = jax.nn.relu(x @ p['W1'] + p['b1'])
            return h @ p['W2'] + p['b2']

        reg = SIGReg(num_slices=256)

        @jax.jit
        def step(p, opt_state, key):
            loss, g = jax.value_and_grad(
                lambda q: reg.apply(forward(q, inputs), key=key))(p)
            updates, opt_state = optim.update(g, opt_state, p)
            return optax.apply_updates(p, updates), opt_state, loss

        out0 = forward(params, inputs)
        optim = optax.adam(3e-3)
        opt_state = optim.init(params)
        base = jax.random.PRNGKey(43)
        for i in range(1500):
            params, opt_state, loss = step(params, opt_state, jax.random.fold_in(base, i))
        out1 = forward(params, inputs)

        eval_key = jax.random.PRNGKey(999)  # held-out directions
        loss0 = float(reg.apply(out0, key=eval_key))
        loss1 = float(reg.apply(out1, key=eval_key))

        assert loss1 < loss0 / 10.0
        assert _mean_norm(out1) < 0.3
        assert _cov_identity_dist(out1) < min(0.6, _cov_identity_dist(out0) / 4.0)
        assert _mean_excess_kurtosis(out1) < _mean_excess_kurtosis(out0)
        assert _mean_excess_kurtosis(out1) < 0.6
