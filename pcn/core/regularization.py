"""
Regularization classes for layer activations.

Regularizations can be applied to layer activities during inference
(inference_reg) and/or training (train_reg). Each regularization class
is a NamedTuple with an ``apply(x, key=None, labels=None)`` method that
returns a scalar energy contribution, compatible with JAX tracing and
JIT compilation. ``labels`` is the per-batch class label tensor piped
through the backend from ``data_map`` (resolved via the ``'class'``
sentinel key in ``data_map``). Most regularizers ignore it; label-aware
regs such as :class:`SupConLoss` use it to compute per-batch
class-conditional losses.

Usage:
    import pcn

    with net:
        l1 = pcn.Layer(dim=784, label="input")
        l2 = pcn.Layer(dim=256, inference_reg=pcn.L1Norm(strength=0.01))
        l3 = pcn.Layer(dim=128, train_reg=pcn.SIGReg(strength=0.1))

For the label-aware SupConLoss you also need to thread labels through:

    sim.train(loader,
              data_map={l_input: 'image', l_output: 'label',
                        'class': 'label_idx'},     # <-- labels sentinel
              ...)
"""

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp


class SumReg(NamedTuple):
    """Combine multiple regularizers on the same layer.

    Each child reg's ``apply(x, key, labels)`` is called with a child-specific
    PRNG sub-key (via ``jax.random.fold_in``) so any randomness (e.g. SIGReg's
    random projections) stays independent across children. The sum is returned.

    Example: combine a label-aware SupCon term with a small L2 magnitude penalty
    on the same latent layer::

        l_lat = pcn.Layer(dim=256, activation=pcn.Gelu(),
                           inference_reg=pcn.SumReg(regs=(
                               pcn.SupConLoss(strength=0.3, temperature=0.1),
                               pcn.L2Norm(strength=1e-3),
                           )))

    Attributes:
        regs: tuple of regularizer instances. Each must implement ``apply``.
    """
    regs: tuple = ()

    def apply(self, x, key=None, labels=None):
        total = jnp.array(0.0, dtype=jnp.float32)
        for i, reg in enumerate(self.regs):
            sub = jax.random.fold_in(key, i) if key is not None else None
            total = total + reg.apply(x, key=sub, labels=labels)
        return total


class L1Norm(NamedTuple):
    """L1 norm regularization on layer activations.

    Adds strength * sum(mean(|x|, axis=0)) to the energy,
    encouraging sparse activations (many units driven toward zero).

    Attributes:
        strength: Regularization coefficient.
    """
    strength: float = 1.0

    def apply(self, x, key=None, labels=None):
        """Compute L1 regularization energy.

        Args:
            x: Layer activity with shape (batch_size, dim).
            key: Unused. Accepted for interface compatibility.
            labels: Unused. Accepted for interface compatibility.

        Returns:
            Scalar energy contribution.
        """
        return self.strength * jnp.sum(jnp.mean(jnp.abs(x), axis=0))


class L2Norm(NamedTuple):
    """L2 norm regularization on layer activations.

    Adds strength * sum(mean(x^2, axis=0)) to the energy,
    encouraging small activations.

    Attributes:
        strength: Regularization coefficient.
    """
    strength: float = 1.0

    def apply(self, x, key=None, labels=None):
        """Compute L2 regularization energy.

        Args:
            x: Layer activity with shape (batch_size, dim).
            key: Unused. Accepted for interface compatibility.
            labels: Unused. Accepted for interface compatibility.

        Returns:
            Scalar energy contribution.
        """
        return self.strength * jnp.sum(jnp.mean(x ** 2, axis=0))


class UnitNorm(NamedTuple):
    """Unit-norm regularization on layer activations.

    Penalises deviation of each sample's L2 norm from 1, encouraging
    activations to lie on the unit sphere. Energy is
    strength * mean_b (||x_b||_2 - 1)^2.

    Attributes:
        strength: Regularization coefficient.
        eps: Small constant added inside the sqrt for gradient stability
            near zero norm.
    """
    strength: float = 1.0
    eps: float = 1e-8

    def apply(self, x, key=None, labels=None):
        """Compute unit-norm regularization energy.

        Args:
            x: Layer activity with shape (batch_size, dim).
            key: Unused. Accepted for interface compatibility.
            labels: Unused. Accepted for interface compatibility.

        Returns:
            Scalar energy contribution.
        """
        norms = jnp.sqrt(jnp.sum(x ** 2, axis=-1) + self.eps)
        return self.strength * jnp.mean((norms - 1.0) ** 2)


# ---------------------------------------------------------------------------
# SIGReg helpers
# ---------------------------------------------------------------------------

def _epps_pulley_statistic(
    samples: jnp.ndarray,
    num_points: int = 17,
    t_max: float = 3.0,
) -> jnp.ndarray:
    """Epps-Pulley characteristic function test statistic.

    Measures the squared distance between the empirical characteristic
    function and the N(0,1) characteristic function, weighted by a
    Gaussian kernel.

    Args:
        samples: 1D array of shape [N] — projected scalar embeddings.
        num_points: Number of quadrature points.
        t_max: Integration bound.

    Returns:
        Scalar test statistic (lower = more Gaussian).
    """
    x = samples  # [N]

    # Quadrature grid
    t = jnp.linspace(-t_max, t_max, num_points)  # [T]
    dt = t[1] - t[0]

    # Empirical CF: phi_hat(t) = (1/N) sum_j exp(i t x_j)
    tx = t[:, None] * x[None, :]  # [T, N]
    ecf_real = jnp.mean(jnp.cos(tx), axis=1)  # [T]
    ecf_imag = jnp.mean(jnp.sin(tx), axis=1)  # [T]

    # Gaussian CF for N(0,1): phi(t) = exp(-t^2 / 2)
    gcf = jnp.exp(-0.5 * t ** 2)  # [T]

    # Squared difference |phi_hat - phi|^2
    sq_diff = (ecf_real - gcf) ** 2 + ecf_imag ** 2  # [T]

    # Weight by Gaussian kernel and integrate (trapezoidal rule)
    weight = jnp.exp(-0.5 * t ** 2)  # [T]
    return jnp.trapezoid(sq_diff * weight, dx=dt)


def _sigreg_loss(
    embeddings: jnp.ndarray,
    key: jnp.ndarray,
    num_slices: int,
    num_points: int,
    t_max: float,
) -> jnp.ndarray:
    """Compute SIGReg loss via Cramér-Wold slicing + Epps-Pulley test.

    Args:
        embeddings: [N, D] batch of embedding vectors.
        key: JAX PRNG key for sampling random directions.
        num_slices: Number of random projection directions.
        num_points: Quadrature points for the EP test.
        t_max: Integration bound for the EP test.

    Returns:
        Scalar SIGReg loss.
    """
    N, D = embeddings.shape

    # Sample random unit directions on S^{D-1}
    directions = jax.random.normal(key, shape=(D, num_slices))  # [D, M]
    directions = directions / (jnp.linalg.norm(directions, axis=0, keepdims=True) + 1e-8)

    # Project: [N, D] @ [D, M] -> [N, M]
    projections = embeddings @ directions  # [N, M]

    # Apply EP test to each projection (vectorised via vmap)
    ep_fn = partial(_epps_pulley_statistic, num_points=num_points, t_max=t_max)
    statistics = jax.vmap(ep_fn)(projections.T)  # [M]

    return jnp.mean(statistics)


class SIGReg(NamedTuple):
    """SIGReg (Sketched Isotropic Gaussian Regularization).

    Pushes layer activations toward an isotropic Gaussian distribution
    using the Epps-Pulley characteristic function test with random 1D
    projections (Cramér-Wold slicing).

    Reference:
        Balestriero & LeCun, "LeJEPA: Provable and Scalable
        Self-Supervised Learning Without the Heuristics", 2025.

    Attributes:
        strength: Regularization coefficient.
        num_slices: Number of random projection directions.
        num_points: Quadrature points for the Epps-Pulley test.
        t_max: Integration bound for the Epps-Pulley test.
        seed: Random seed for generating projection directions.
    """
    strength: float = 1.0
    num_slices: int = 1024
    num_points: int = 17
    t_max: float = 3.0
    seed: int = 0

    def apply(self, x, key=None, labels=None):
        """Compute SIGReg regularization energy.

        Args:
            x: Layer activity with shape (batch_size, dim).
            key: Optional JAX PRNG key. If provided, used for random
                projections. Otherwise falls back to self.seed.
            labels: Unused. Accepted for interface compatibility.

        Returns:
            Scalar energy contribution.
        """
        if key is None:
            key = jax.random.PRNGKey(self.seed)
        return self.strength * _sigreg_loss(
            x, key, self.num_slices, self.num_points, self.t_max)


class SupConLoss(NamedTuple):
    """Supervised contrastive loss (Khosla et al., 2020) as a layer-level
    inference regularizer.

    For each sample in the batch, pulls features toward other samples in
    the same class and pushes away from samples of different classes,
    via the standard normalised-cosine + softmax formulation:

        L_i = -1/|P(i)| * sum_{p in P(i)} log( exp(sim(z_i, z_p) / T)
                                                / sum_{a != i} exp(sim(z_i, z_a) / T) )
        L   = mean_i L_i  (over samples that have at least one positive)

    where P(i) is the set of other samples in the batch sharing the
    class of sample i, and ``sim`` is cosine similarity after L2-normalising
    the feature vectors.

    Because this is attached as an ``inference_reg`` on a layer, the loss
    is added directly to the energy. Its gradient back-propagates through
    the layer's value during inference, pulling within-class samples
    together in feature space (and pushing between-class samples apart)
    at the same time as the standard PC energy.

    Labels are passed in via the standard ``data_map`` ``'class'`` sentinel:

        sim.train(loader,
                  data_map={l_input: 'image', l_output: 'label',
                            'class': 'label_idx'},  # <-- per-batch class indices
                  ...)

    If ``labels is None`` at apply-time, the loss returns 0 (silent
    no-op) so the same network definition can run on label-less data
    (e.g. unlabelled SSL splits) without errors.

    Attributes:
        strength: Regularization coefficient (added to the energy).
        temperature: Softmax temperature. 0.07-0.5 is the typical SupCon range.
        eps: Numerical stabiliser added under the L2 normalisation sqrt.
    """
    strength: float = 1.0
    temperature: float = 0.1
    eps: float = 1e-8

    def apply(self, x, key=None, labels=None):
        """Compute supervised contrastive loss.

        Args:
            x: Layer activity (batch_size, dim).
            key: Unused. Accepted for interface compatibility.
            labels: Class labels for the batch — accepts either (B,) class
                indices or (B, n_classes) one-hot. If ``None``, the loss is
                a no-op (returns zero).

        Returns:
            Scalar SupCon loss × strength.
        """
        if labels is None:
            return jnp.array(0.0, dtype=jnp.float32)
        # Normalise to one-hot semantics regardless of input form.
        if labels.ndim > 1:
            labels = jnp.argmax(labels, axis=-1)
        labels = labels.astype(jnp.int32)

        # L2-normalise features for cosine similarity.
        x_norm = x / (jnp.linalg.norm(x, axis=-1, keepdims=True) + self.eps)

        # Pairwise scaled similarity matrix.
        sim = jnp.matmul(x_norm, x_norm.T) / self.temperature

        # Masks: identity diagonal (exclude self), positive (same class & not self).
        B = sim.shape[0]
        self_mask = jnp.eye(B, dtype=jnp.float32)
        diag_off = 1.0 - self_mask
        same_class = (labels[:, None] == labels[None, :]).astype(jnp.float32)
        pos_mask = same_class * diag_off

        # log-sum-exp over all anchors (excluding self) — denominator of the
        # softmax inside the log.
        sim_masked = jnp.where(self_mask > 0.5, -1e9, sim)
        log_denom = jax.scipy.special.logsumexp(sim_masked, axis=-1)
        log_prob = sim - log_denom[:, None]

        # Per-sample mean log-prob over its positives. Avoid div-by-zero for
        # samples with no in-batch positives.
        n_pos = jnp.sum(pos_mask, axis=-1)
        safe_n_pos = jnp.maximum(n_pos, 1.0)
        loss_per = -jnp.sum(pos_mask * log_prob, axis=-1) / safe_n_pos

        # Mean over samples with at least one positive.
        has_pos = (n_pos > 0).astype(jnp.float32)
        loss = jnp.sum(loss_per * has_pos) / jnp.maximum(jnp.sum(has_pos), 1.0)
        return self.strength * loss
