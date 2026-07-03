"""Natural gradient preconditioner for precision parameters.

Provides an optax GradientTransformationExtraArgs that preconditions
the gradients of precision_weights using the empirical Fisher matrix of
the pre-activations, and scales precision_biases by 2.

Usage:
    import optax
    import pcn

    params_optimizer = optax.chain(
        pcn.natural_gradient_precision(damping=1e-4),
        optax.adam(learning_rate=1e-3),
    )
    sim.train(..., params_optimizer=params_optimizer)

The natural gradient update for precision weights A is:
    Delta A = -eta * 2 * C_f^{-1} (grad_A L)^T   (acting on rows of A)
which in matrix form (grad_A is (n_out, d_pre)) is:
    nat_grad_A = 2 * grad_A @ C_f^{-1}
where C_f = E[f f^T] is the empirical feature covariance (d_pre, d_pre).

Backend note:
    The full natural gradient uses jnp.linalg.eigh on a (d_pre, d_pre) matrix.
    On jax-mps (MLX backend) eigh is only GPU-accelerated for matrices up to
    63×63. For larger d_pre a diagonal Fisher approximation is used automatically:
        nat_grad_A ≈ 2 * grad_A / diag(C_f)
    This is computed at JAX trace time (shapes are static), so no runtime branch
    overhead is incurred.
"""

from typing import Any, NamedTuple

import jax.numpy as jnp
import optax


class NatGradPrecisionState(NamedTuple):
    """Empty state — the natural gradient preconditioner is stateless."""
    pass


def natural_gradient_precision(damping: float = 1e-4) -> optax.GradientTransformationExtraArgs:
    """Natural gradient preconditioner for precision parameters.

    Transforms precision_weights gradients via  2 * grad_A @ C_f^{-1}
    and scales precision_biases gradients by 2. All other parameters
    (predict_weights, predict_biases) are passed through unchanged.

    Args:
        damping: Tikhonov regularization added to C_f before inversion.

    Returns:
        An optax GradientTransformationExtraArgs. Its update_fn accepts a
        'features' keyword argument: a tuple of (batch, d_pre) arrays, one
        per predict connection, containing the pre-activations used to
        estimate C_f. If features is None the gradients pass through unchanged.

    Note:
        To use with standard optax transforms in optax.chain, optax >= 0.2
        forwards extra kwargs only to steps that accept them, so chaining
        with e.g. optax.adam works directly:
            optax.chain(natural_gradient_precision(), optax.adam(lr))

        When d_pre > 63 (jax-mps MLX eigh limit), a diagonal Fisher
        approximation is used automatically (trace-time static branch).
    """
    def init_fn(params: Any) -> NatGradPrecisionState:
        return NatGradPrecisionState()

    def update_fn(
        updates: Any,
        state: NatGradPrecisionState,
        params: Any = None,
        *,
        features=None,
    ) -> tuple:
        if features is None:
            return updates, state

        def _precondition_weight(grad_A, f):
            # grad_A: (n_out, d_pre), f: (batch, d_pre)
            C_f = jnp.einsum('bi,bj->ij', f, f) / f.shape[0]
            C_f_reg = C_f + damping * jnp.eye(C_f.shape[-1])
            d_pre = C_f_reg.shape[-1]
            if d_pre <= 63:
                # Full natural gradient via eigendecomposition.
                # C_f_reg = V diag(λ) V^T  =>  grad_A @ C_f_reg^{-1} = (grad_A @ V) * (1/λ) @ V^T
                eigvals, eigvecs = jnp.linalg.eigh(C_f_reg)
                return 2.0 * (grad_A @ eigvecs) * (1.0 / eigvals) @ eigvecs.T
            else:
                # Diagonal Fisher approximation (d_pre > 63: jax-mps eigh GPU limit).
                # Equivalent to scaling each pre-dimension independently by 1/C_f[i,i].
                diag_inv = 1.0 / jnp.diag(C_f_reg)  # (d_pre,)
                return 2.0 * grad_A * diag_inv[None, :]  # (n_out, d_pre)

        nat_pw = tuple(
            _precondition_weight(g_A, f)
            for g_A, f in zip(updates['precision_weights'], features)
        )
        nat_pb = tuple(2.0 * g_b for g_b in updates['precision_biases'])

        new_updates = dict(updates)
        new_updates['precision_weights'] = nat_pw
        new_updates['precision_biases'] = nat_pb
        return new_updates, state

    return optax.GradientTransformationExtraArgs(init_fn, update_fn)
