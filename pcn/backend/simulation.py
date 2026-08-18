"""
Consolidated PCN simulation backend.

Single module with one main function that handles:
- State initialization
- Data clamping
- Value reset via forward predictions
- Inference loop (with lax.fori_loop)
- Weight updates

Optimized for speed with minimal Python overhead.
"""

from typing import Dict, Tuple, List, Optional, Any
import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from jax import lax

from ..core.structure import NetworkStructure
from ..core.learning_rules import Hebbian, ThreeFactorHebbian
from ..core.state import NetworkState, NetworkParams
from ..core.network import _make_band_mask
from ..core.activations import ACTIVATIONS, _nwta  # re-exported for back-compat


# ============================================================================
# Memory-aware error activation dispatch
# ============================================================================

def _activated_error(predict_error_activations, conn, i, residual, prev_error,
                     key=None):
    """Apply the error activation for predict-conn ``i`` to a raw residual.

    If a memory-aware activation is supplied for that conn, it is given the
    previous-iteration activated error from the loop carry (already
    stop_gradient'd inside :meth:`MemoryActivation.apply`). If a stochastic
    activation is supplied, it is given a per-conn key folded from ``key``
    (``None`` on key-free paths, which yields the deterministic output). For
    plain stateless activations we defer to
    :meth:`PredictConnSpec.error_transform`, keeping the original code path
    (including the Direct-identity short-circuit).
    """
    if predict_error_activations:
        act = predict_error_activations[i]
        if act.has_memory:
            return act.apply(residual, prev_error)
        if act.needs_key:
            sub = jax.random.fold_in(key, i) if key is not None else None
            return act.apply(residual, key=sub)
    return conn.error_transform(residual)


def _activated_precision(predict_precision_activations, conn, i, log_prec,
                         prev_precision, key=None):
    """Apply the precision activation for predict-conn ``i`` to log-precision.

    Mirrors :func:`_activated_error`. For memory-aware activations, the
    previous-iteration precision is supplied (zeros on the first iteration
    when ``prev_precision is None``). For stochastic activations, a per-conn
    key folded from ``key`` is supplied (``None`` yields the deterministic
    output). Stateless activations defer to
    :meth:`PredictConnSpec.precision_transform`, preserving the original
    code path and the closed-form bias-init guarantees.
    """
    if predict_precision_activations:
        act = predict_precision_activations[i]
        if act.has_memory:
            if prev_precision is None:
                prev_precision = jnp.zeros_like(log_prec)
            return act.apply(log_prec, prev_precision)
        if act.needs_key:
            sub = jax.random.fold_in(key, i) if key is not None else None
            return act.apply(log_prec, key=sub)
    return conn.precision_transform(log_prec)


def _add_prediction_noise(conn, i, prediction, prec_in,
                          precision_weights, precision_bias,
                          is_stochastic, key):
    """Add precision-scaled Gaussian noise to a prediction (is_stochastic).

    The noise is ``eps / sqrt(precision)`` with ``eps ~ N(0, I)`` and the
    connection's own ``precision_fn``, matching :meth:`stochastic_prediction`.
    ``prec_in`` is the precision function's input — the conn's pre activation
    by default, or the gathered ``precision_input`` sources.
    It is added as a ``stop_gradient``'d offset so it perturbs the resulting
    error (and hence the value dynamics) like an injected Langevin term,
    without contributing a gradient through the noise scale — variance is
    still learned via the energy's ``prec*err^2 - log prec`` term.

    No-op when ``is_stochastic`` is False or ``key`` is None (deterministic
    paths). The per-conn fold (``i + 1000``) matches :func:`_single_pass`, so
    the same iteration key yields the same noise realisation across the energy
    step and the error recompute.

    ``is_stochastic`` may be a bool (``True`` → temperature 1) or a float
    sampling **temperature** ``T``: the noise std is scaled by ``sqrt(T)``
    (variance ∝ T), the Langevin temperature knob. ``T>1`` heats the sampler so
    a strong generator explores within-class exemplars; ``T<1`` cools it.
    """
    if not is_stochastic or key is None or not getattr(conn, 'stochastic', True):
        return prediction
    skey = jax.random.fold_in(key, i + 1000)
    eps = jax.random.normal(skey, prediction.shape)
    precision = conn.precision_fn(prec_in, precision_weights, precision_bias)
    std = (float(is_stochastic) ** 0.5) / jnp.sqrt(jnp.clip(precision, 1e-8, None))
    return prediction + jax.lax.stop_gradient(std * eps)


# ============================================================================
# Loss function helpers
# ============================================================================

def _gather_loss_fn_args(resolved_inputs, node_arrays, sample_arrays):
    """Gather arrays for a loss function from nodes and sample dict.

    Args:
        resolved_inputs: A single resolved key or tuple of resolved keys.
            Each element is either (node_type, idx) or a str (sample key).
        node_arrays: (values_tuple, errors_tuple, precisions_tuple)
        sample_arrays: Dict mapping sample key -> jnp.ndarray

    Returns:
        List of arrays to pass as positional args.
    """
    def _get_one(elem):
        if isinstance(elem, str):
            return sample_arrays[elem]
        else:
            node_type, idx = elem
            return node_arrays[node_type][idx]

    if isinstance(resolved_inputs, str):
        return [sample_arrays[resolved_inputs]]
    elif isinstance(resolved_inputs, tuple) and len(resolved_inputs) == 2 and isinstance(resolved_inputs[0], int):
        # Single node ref: (node_type, idx)
        node_type, idx = resolved_inputs
        return [node_arrays[node_type][idx]]
    elif isinstance(resolved_inputs, tuple):
        return [_get_one(elem) for elem in resolved_inputs]
    else:
        raise TypeError(f"Unexpected resolved_inputs format: {resolved_inputs}")


# ============================================================================
# Sliced write-back helpers
# ============================================================================

def _apply_mask(mask, original, updated):
    """Soft-clamp blend: mask=1 → original, mask=0 → updated, intermediate → linear mix."""
    return mask * original + (1 - mask) * updated


def _write_additive(arr, contribution, post_slice):
    """Add contribution to arr, potentially to a slice of dimensions."""
    if post_slice:
        return arr.at[:, post_slice[0]:post_slice[1]].add(contribution)
    return arr + contribution


def _write_multiplicative(arr, modulation, post_slice):
    """Multiply arr by modulation, potentially on a slice of dimensions."""
    if post_slice:
        sliced = arr[:, post_slice[0]:post_slice[1]]
        return arr.at[:, post_slice[0]:post_slice[1]].set(sliced * modulation)
    return arr * modulation


# ============================================================================
# Temporal-delay history buffers (Phase 1: value pre nodes)
# ============================================================================

def _read_delayed(hist, buf_idx, delay, unit_ts, tick_base, i, ipt):
    """Read a value ``delay`` steps back from its history ring buffer.

    ``tick_base + i`` is the global iteration index (``tick_base`` offsets the
    learning loop by ``n_iterations``). For sliding buffers the tick is the
    iteration; for latched ('timestep') buffers it is the frame ``//ipt``.
    Returns ``hist[buf_idx][(tick - delay) % S]`` — a plain array index, hence
    automatically a constant w.r.t. ``value_and_grad(values)`` (the
    one-directional property, no ``stop_gradient`` needed). Slots not yet
    written hold the pre-fill zeros.
    """
    tick = (tick_base + i) if not unit_ts else (tick_base + i) // ipt
    S = hist[buf_idx].shape[0]
    return hist[buf_idx][(tick - delay) % S]


def _delayed_srcs(conn, hist, tick_base, iter_idx, ipt):
    """Delayed pre values for ``conn`` (parallel to ``conn.pre_idx``), or None.

    Returns None on the static ``delay == 0`` path so the caller's ``get_pre``
    takes the historical live-read branch (bit-identical). Only built when the
    conn statically asks for a delay. Also returns None when no ``hist`` is
    threaded (the initial ``_single_pass`` runs before the ring buffers exist —
    previously an IndexError for delayed Project/Modulate at the init pass;
    the live read there matches the zero-iteration semantics).
    """
    if conn.delay == 0 or not hist:
        return None
    return [
        _read_delayed(hist, conn.pre_buffer_indices[k], conn.delay,
                      conn.delay_unit_ts, tick_base, iter_idx, ipt)
        for k in range(len(conn.pre_idx))
    ]


# ============================================================================
# Project / Modulate helpers
# ============================================================================

def _apply_project_modulate_internal(
    errors, values,
    project_weights, modulate_weights,
    project_conns_internal, modulate_conns_internal,
    activation_fns,
    precisions=(),
    project_biases=(),
    modulate_biases=(),
    hist=(), tick_base=0, iter_idx=0, iters_per_timestep=1,
):
    """Apply Project (additive) then Modulate (multiplicative) to errors.

    Uses stop_gradient on pre_act so value dynamics don't see routing paths.
    Weights are passed as-is — caller is responsible for stop_gradient on
    weights (allowing GD connections to have live weights for gradient flow).

    ``hist``/``tick_base``/``iter_idx``/``iters_per_timestep`` carry the delay
    buffers; a value-pre conn with ``delay>=1`` reads its pre from ``hist``.
    Defaults leave every delay==0 conn on the live-read path (bit-identical).
    """
    new_errors = list(errors)

    for weight_idx, conn in project_conns_internal:
        dsrc = _delayed_srcs(conn, hist, tick_base, iter_idx, iters_per_timestep)
        pre_act = conn.get_pre(values, errors, activation_fns,
                               precisions=precisions, delayed_srcs=dsrc)
        p_bias = project_biases[weight_idx] if project_biases else 0.0
        contribution = conn.apply(
            jax.lax.stop_gradient(pre_act),
            project_weights[weight_idx]) + p_bias
        new_errors[conn.post_idx] = _write_additive(
            new_errors[conn.post_idx], contribution, conn.post_slice)

    for weight_idx, conn in modulate_conns_internal:
        dsrc = _delayed_srcs(conn, hist, tick_base, iter_idx, iters_per_timestep)
        pre_act = conn.get_pre(values, tuple(new_errors), activation_fns,
                               precisions=precisions, delayed_srcs=dsrc)
        bias = modulate_biases[weight_idx] if modulate_biases else 0.0
        modulation = conn.apply(
            jax.lax.stop_gradient(pre_act),
            modulate_weights[weight_idx]) + bias
        new_errors[conn.post_idx] = _write_multiplicative(
            new_errors[conn.post_idx], modulation, conn.post_slice)

    return tuple(new_errors)


def _apply_project_modulate_internal_for_loss(
    errors, values,
    project_weights, modulate_weights,
    project_conns_internal, modulate_conns_internal,
    activation_fns,
    precisions=(),
    project_biases=(),
    modulate_biases=(),
):
    """Apply Project/Modulate to errors with full gradient flow.

    Like _apply_project_modulate_internal but WITHOUT stop_gradient on
    pre_act.  Used inside loss_objective for true backprop.
    """
    new_errors = list(errors)

    for weight_idx, conn in project_conns_internal:
        pre_act = conn.get_pre(values, errors, activation_fns, precisions=precisions)
        p_bias = project_biases[weight_idx] if project_biases else 0.0
        contribution = conn.apply(pre_act, project_weights[weight_idx]) + p_bias
        new_errors[conn.post_idx] = _write_additive(
            new_errors[conn.post_idx], contribution, conn.post_slice)

    for weight_idx, conn in modulate_conns_internal:
        pre_act = conn.get_pre(values, tuple(new_errors), activation_fns, precisions=precisions)
        bias = modulate_biases[weight_idx] if modulate_biases else 0.0
        modulation = conn.apply(pre_act, modulate_weights[weight_idx]) + bias
        new_errors[conn.post_idx] = _write_multiplicative(
            new_errors[conn.post_idx], modulation, conn.post_slice)

    return tuple(new_errors)


def _apply_project_modulate_values(
    values, errors,
    project_weights, modulate_weights,
    project_conns_value, modulate_conns_value,
    activation_fns, clamped,
    project_biases=(),
    modulate_biases=(),
    read_values=None,
    is_boundary=None,
    hist=(), tick_base=0, iter_idx=0, iters_per_timestep=1,
    precisions=(),
):
    """Apply Project (additive) then Modulate (multiplicative) targeting values.

    Called after the optax value update, as an explicit state operator
    (integrating drive) that persists into the carried values.

    **Jacobi combination.** All pre-activations are read from ``read_values``
    (the frozen pre-update state ``v[t]``) — not from the progressively-updated
    output — so every value Project/Modulate contribution is computed from the
    same ``v[t]`` and summed. ``read_values`` defaults to ``values`` (the
    write-target) for backward compatibility. Callers pass the pre-update
    ``v[t]`` so the per-iteration update is
    ``v[t+1] = v[t] + (energy step) + Σ W f(v[t]_pre)`` (additive projects),
    then ``× modulation(v[t]_pre)``. Order-independent across projects; chained
    value->value->value routing propagates one hop per iteration.

    ``errors`` is the frozen carried error state (read by error->value routing).
    Respects clamping (soft blend against the write-target).

    **``advance='timestep'`` gating.** ``is_boundary`` is a traced scalar that is
    truthy on the first inference iteration of each input timestep
    (``i % iters_per_timestep == 0``). Connections whose spec has
    ``advance_timestep=True`` only fire there: off boundary an additive Project
    contributes exactly 0 and a multiplicative Modulate factor is replaced by
    the identity (1.0), so the target passes through untouched. Because ``i`` is
    a tracer inside ``lax.fori_loop`` the gate is applied arithmetically, never
    via a Python ``if``. ``is_boundary=None`` means "always fire" and takes the
    original, un-gated code path verbatim (bit-identical numerics, no extra ops)
    — callers should pass ``None`` whenever no value conn is gated.

    Note: ``_single_pass`` deliberately does NOT gate — the feedforward seed is
    a forward seed, not a timestep update, so value Project/Modulate always run
    there regardless of ``advance``.
    """
    new_values = list(values)
    original_values = values  # for re-clamping
    read_src = tuple(values) if read_values is None else tuple(read_values)

    gate_bool = None
    if is_boundary is not None:
        _b = jnp.asarray(is_boundary)
        gate_bool = _b if _b.dtype == jnp.bool_ else (_b != 0)

    for weight_idx, conn in project_conns_value:
        dsrc = _delayed_srcs(conn, hist, tick_base, iter_idx, iters_per_timestep)
        pre_act = conn.get_pre(read_src, errors, activation_fns,
                               precisions=precisions, delayed_srcs=dsrc)
        p_bias = project_biases[weight_idx] if project_biases else 0.0
        contribution = conn.apply(pre_act, project_weights[weight_idx]) + p_bias
        if gate_bool is not None and getattr(conn, 'advance_timestep', False):
            # Off boundary the additive contribution is zeroed out.
            contribution = contribution * gate_bool.astype(contribution.dtype)
        new_values[conn.post_idx] = _write_additive(
            new_values[conn.post_idx], contribution, conn.post_slice)

    for weight_idx, conn in modulate_conns_value:
        dsrc = _delayed_srcs(conn, hist, tick_base, iter_idx, iters_per_timestep)
        pre_act = conn.get_pre(read_src, errors, activation_fns,
                               precisions=precisions, delayed_srcs=dsrc)
        bias = modulate_biases[weight_idx] if modulate_biases else 0.0
        modulation = conn.apply(pre_act, modulate_weights[weight_idx]) + bias
        if gate_bool is not None and getattr(conn, 'advance_timestep', False):
            # Off boundary the factor must be the IDENTITY, not zero.
            modulation = jnp.where(gate_bool, modulation,
                                   jnp.ones_like(modulation))
        new_values[conn.post_idx] = _write_multiplicative(
            new_values[conn.post_idx], modulation, conn.post_slice)

    # Re-apply clamping (soft blend)
    return tuple(
        _apply_mask(clamped[j], original_values[j], new_values[j])
        for j in range(len(values)))


def _apply_project_modulate_values_in_energy(
    values, prev_errors,
    project_weights, modulate_weights,
    project_conns_value, modulate_conns_value,
    activation_fns,
    project_biases=(),
    modulate_biases=(),
    precisions=(),
):
    """Apply value-targeting Project/Modulate inside the energy function.

    stop_gradient on pre_act for value locality. No clamping (energy context).
    Weights are passed as-is — caller handles stop_gradient.
    prev_errors: errors from the previous iteration (for conns reading from errors).
    """
    new_values = list(values)

    for weight_idx, conn in project_conns_value:
        pre_act = conn.get_pre(tuple(new_values), prev_errors, activation_fns,
                               precisions=precisions)
        p_bias = project_biases[weight_idx] if project_biases else 0.0
        new_values[conn.post_idx] = _write_additive(
            new_values[conn.post_idx],
            conn.apply(jax.lax.stop_gradient(pre_act),
                       project_weights[weight_idx]) + p_bias,
            conn.post_slice)

    for weight_idx, conn in modulate_conns_value:
        pre_act = conn.get_pre(tuple(new_values), prev_errors, activation_fns,
                               precisions=precisions)
        bias = modulate_biases[weight_idx] if modulate_biases else 0.0
        new_values[conn.post_idx] = _write_multiplicative(
            new_values[conn.post_idx],
            conn.apply(jax.lax.stop_gradient(pre_act),
                       modulate_weights[weight_idx]) + bias,
            conn.post_slice)

    return tuple(new_values)


def _apply_project_modulate_values_for_loss(
    values, prev_errors,
    project_weights, modulate_weights,
    project_conns_value, modulate_conns_value,
    activation_fns,
    project_biases=(),
    modulate_biases=(),
    precisions=(),
):
    """Apply value-targeting Project/Modulate with full gradient flow.

    Like _apply_project_modulate_values_in_energy but WITHOUT stop_gradient
    on pre_act.  Used inside loss_objective so that gradients flow through
    the entire chain of Project connections (true backprop).
    """
    new_values = list(values)

    for weight_idx, conn in project_conns_value:
        pre_act = conn.get_pre(tuple(new_values), prev_errors, activation_fns,
                               precisions=precisions)
        p_bias = project_biases[weight_idx] if project_biases else 0.0
        new_values[conn.post_idx] = _write_additive(
            new_values[conn.post_idx],
            conn.apply(pre_act, project_weights[weight_idx]) + p_bias,
            conn.post_slice)

    for weight_idx, conn in modulate_conns_value:
        pre_act = conn.get_pre(tuple(new_values), prev_errors, activation_fns,
                               precisions=precisions)
        bias = modulate_biases[weight_idx] if modulate_biases else 0.0
        new_values[conn.post_idx] = _write_multiplicative(
            new_values[conn.post_idx],
            conn.apply(pre_act, modulate_weights[weight_idx]) + bias,
            conn.post_slice)

    return tuple(new_values)


def _apply_project_modulate_precision(
    precisions, values, errors,
    project_weights, modulate_weights,
    project_conns_precision, modulate_conns_precision,
    activation_fns,
    project_biases=(),
    modulate_biases=(),
    hist=(), tick_base=0, iter_idx=0, iters_per_timestep=1,
):
    """Apply Project (additive) then Modulate (multiplicative) to precisions.

    Uses stop_gradient on pre_act for locality, same as error-targeting.
    Returns unmodified precisions if no precision-targeting connections exist.

    ``hist``/``tick_base``/``iter_idx``/``iters_per_timestep`` carry the delay
    buffers; a value-pre conn with ``delay>=1`` reads its pre delayed. Defaults
    keep every delay==0 conn on the live-read path (bit-identical).
    """
    if not project_conns_precision and not modulate_conns_precision:
        return precisions
    new_precisions = list(precisions)

    for weight_idx, conn in project_conns_precision:
        dsrc = _delayed_srcs(conn, hist, tick_base, iter_idx, iters_per_timestep)
        pre_act = conn.get_pre(
            values, errors, activation_fns, precisions=tuple(new_precisions),
            delayed_srcs=dsrc)
        p_bias = project_biases[weight_idx] if project_biases else 0.0
        contribution = conn.apply(
            jax.lax.stop_gradient(pre_act),
            project_weights[weight_idx]) + p_bias
        new_precisions[conn.post_idx] = _write_additive(
            new_precisions[conn.post_idx], contribution, conn.post_slice)

    for weight_idx, conn in modulate_conns_precision:
        dsrc = _delayed_srcs(conn, hist, tick_base, iter_idx, iters_per_timestep)
        pre_act = conn.get_pre(
            values, errors, activation_fns, precisions=tuple(new_precisions),
            delayed_srcs=dsrc)
        bias = modulate_biases[weight_idx] if modulate_biases else 0.0
        modulation = conn.apply(
            jax.lax.stop_gradient(pre_act),
            modulate_weights[weight_idx]) + bias
        new_precisions[conn.post_idx] = _write_multiplicative(
            new_precisions[conn.post_idx], modulation, conn.post_slice)

    return tuple(new_precisions)


# ============================================================================
# Core inference step (internal, used by run_batch)
# ============================================================================


def _combined_step(
    values: tuple,
    prev_errors: tuple,
    clamped: tuple,
    predict_weights: tuple,
    predict_biases: tuple,
    project_weights: tuple,
    project_biases: tuple,
    modulate_weights: tuple,
    modulate_biases: tuple,
    precision_weights: tuple,
    precision_biases: tuple,
    predict_conns: tuple,
    project_conns: tuple,
    modulate_conns: tuple,
    project_conns_internal: tuple,
    project_conns_value: tuple,
    modulate_conns_internal: tuple,
    modulate_conns_value: tuple,
    activation_fns: tuple,
    values_optimizer,
    values_opt_state,
    params_optimizer,
    params_opt_state,
    gd_loss_project: tuple = (),
    gd_loss_modulate: tuple = (),
    reward_fns: tuple = (),
    loss_fns: tuple = (),
    loss_fn_sample_arrays: dict = {},
    spatial_layers: tuple = (), #TODO: no longer used
    spatial_neighborhoods: tuple = (),
    inference_regs: tuple = (),
    train_regs: tuple = (),
    key = None,
    labels = None,
    update_precision: bool = True,
    is_stochastic: bool = False,
    # Mechanism 1: precision-targeting Project/Modulate
    project_conns_precision: tuple = (),
    modulate_conns_precision: tuple = (),
    # Mechanism 2: per-leg flow gating
    modulate_conns_flow_pre: tuple = (),
    modulate_conns_flow_post: tuple = (),
    predict_has_flow_gates: tuple = (),
    # Mechanism 3: structural attention
    structural_attention_groups: tuple = (),
    pre_scales: tuple = (),
    # Per-connection weight masks for transformation='masked' (one entry per
    # connection of each type; dummy scalar for non-masked conns, never read).
    predict_weight_masks: tuple = (),
    project_weight_masks: tuple = (),
    modulate_weight_masks: tuple = (),
    # Per-Predict-conn error Activation instances. When entry is a
    # MemoryActivation (has_memory=True), the backend dispatches
    # ``act.apply(residual, prev_error)`` using the iteration's
    # ``prev_errors`` instead of ``conn.error_transform(residual)``.
    predict_error_activations: tuple = (),
    # Same idea for the precision slot: memory entries read
    # ``prev_precisions[i]`` from the carry. Stateless entries take the
    # standard ``conn.precision_transform`` path.
    predict_precision_activations: tuple = (),
    prev_precisions: tuple = (),
    # Per-layer Activation instances. Stochastic (needs_key) entries inject
    # noise into the prediction pre-activation, keyed per layer from ``key``.
    layer_activations: tuple = (),
    # Traced scalar, truthy on the first iteration of each input timestep.
    # None => no value Project/Modulate is advance='timestep' gated.
    is_boundary=None,
    # Delay history buffers threaded from run_batch. hist=() disables the
    # feature (delay==0 path is bit-identical). tick_base offsets the learning
    # loop by n_iterations; iter_idx is the loop tracer; iters_per_timestep (ipt)
    # scales latched reads.
    hist=(), tick_base=0, iter_idx=0, iters_per_timestep=1,
) -> tuple:
    """Combined inference + learning step.

    Computes the variational free energy as a function of both values and
    trainable parameters, takes gradients w.r.t. both in a single backward
    pass, and applies the respective optimizers.

    Value-targeting Project/Modulate are NOT applied inside the energy; they
    are explicit state operators applied after the value update (integrating
    drive) and persist into the carried values, so the energy reads them from
    the carried state. Error- and precision-targeting Project/Modulate are
    applied to errors/precisions inside the energy (and recomputed identically
    for the carried/logged state). All use stop_gradient on pre_act for value
    locality.

    GradientDescent (type 2) project/modulate weights always carry a loss_fn:
    their weights are learned by a separate gradient call on L using the
    resolved inputs spec (the energy-based no-loss_fn path has been removed).

    Args:
        update_precision: If False, precision params are stop_gradiented
            (no gradient, no effective update). Used to freeze precision
            during iterative learning loops and update precision only in
            the final step, preventing the precision-error feedback loop.

    Returns:
        (new_values, errors, precisions,
         new_pw, new_pb,
         new_project_weights, new_project_biases,
         new_modulate_weights, new_modulate_biases,
         new_ppw, new_ppb,
         new_values_opt_state, new_params_opt_state)
    """
    n_layers = len(values)
    batch_size = values[0].shape[0]
    n_pc_conn = len(predict_conns)

    # Trainable dict: predict/precision params + GD-loss project/modulate weights
    # plus project/modulate biases (frozen via stop_gradient when has_bias=False)
    trainable = {
        'predict_weights': predict_weights,
        'predict_biases': predict_biases,
        'project_biases': project_biases,
        'modulate_biases': modulate_biases,
        'precision_weights': precision_weights,
        'precision_biases': precision_biases,
        'gd_loss_project_weights': tuple(project_weights[idx] for idx, _ in gd_loss_project),
        'gd_loss_modulate_weights': tuple(modulate_weights[idx] for idx, _ in gd_loss_modulate),
    }

    def energy_with_aux(vals, params):
        pw = params['predict_weights']
        pb = params['predict_biases']
        proj_b_in = params['project_biases']
        mod_b_in = params['modulate_biases']
        ppw = params['precision_weights']
        ppb = params['precision_biases']

        # Stop gradient on biases for connections without use_bias (dummy zeros)
        proj_b = tuple(
            proj_b_in[i] if project_conns[i].has_bias else jax.lax.stop_gradient(proj_b_in[i])
            for i in range(len(project_conns))
        )
        mod_b = tuple(
            mod_b_in[i] if modulate_conns[i].has_bias else jax.lax.stop_gradient(mod_b_in[i])
            for i in range(len(modulate_conns))
        )

        # Project/Modulate weights never receive gradients from the energy
        # (GD-on-energy removed); they are learned by Hebbian/Oja/3-factor or
        # GD-on-loss. Stop-gradient them so error/precision routing and flow
        # gates stay weight-local in the energy backward pass.
        sg_proj_w = tuple(jax.lax.stop_gradient(w) for w in project_weights)
        sg_mod_w = tuple(jax.lax.stop_gradient(w) for w in modulate_weights)

        # 1. Value-targeting Project/Modulate are NOT applied in the energy;
        # they persist into the carried values (applied after the value update
        # below), so the energy reads them directly from vals.
        mod_vals = vals

        # 1b. Compute per-leg flow gates (Mechanism 2)
        flow_gates_pre = []
        flow_gates_post = []
        for i, conn in enumerate(predict_conns):
            post_dim = mod_vals[conn.post_idx].shape[-1]
            flow_gates_pre.append(jnp.ones((batch_size, post_dim)))
            flow_gates_post.append(jnp.ones((batch_size, post_dim)))
        for weight_idx, mc in modulate_conns_flow_pre:
            pre_act = mc.get_pre(mod_vals, prev_errors, activation_fns)
            gate = mc.apply(
                jax.lax.stop_gradient(pre_act), sg_mod_w[weight_idx])
            flow_gates_pre[mc.post_idx] = flow_gates_pre[mc.post_idx] * gate
        for weight_idx, mc in modulate_conns_flow_post:
            pre_act = mc.get_pre(mod_vals, prev_errors, activation_fns)
            gate = mc.apply(
                jax.lax.stop_gradient(pre_act), sg_mod_w[weight_idx])
            flow_gates_post[mc.post_idx] = flow_gates_post[mc.post_idx] * gate
        flow_gates_pre = tuple(flow_gates_pre)
        flow_gates_post = tuple(flow_gates_post)

        # 2. Compute predictions, errors, precisions from mod_vals
        E = 0.
        all_errors = []
        all_precisions = []
        all_e_pre = []   # split errors for per-leg gating
        all_e_post = []
        for i, conn in enumerate(predict_conns):
            dsrc = _delayed_srcs(conn, hist, tick_base, iter_idx, iters_per_timestep)
            pre_act = conn.get_pre(mod_vals, (), activation_fns,
                                   layer_activations, key, delayed_srcs=dsrc)

            if conn.has_fixed_weights:
                pw_i = jax.lax.stop_gradient(pw[i])
                pb_i = jax.lax.stop_gradient(pb[i])
                ppw_i = jax.lax.stop_gradient(ppw[i])
                ppb_i = jax.lax.stop_gradient(ppb[i])
            else:
                pw_i = pw[i]
                pb_i = pb[i] if conn.has_bias else jax.lax.stop_gradient(pb[i])
                # When update_precision=False, always stop_gradient precision
                # params to prevent the precision-error feedback loop in
                # iterative combined_step loops.
                _learn_ppw = conn.learn_precision_weights and update_precision
                _learn_ppb = conn.learn_precision_bias and update_precision
                ppw_i = ppw[i] if _learn_ppw else jax.lax.stop_gradient(ppw[i])
                ppb_i = ppb[i] if _learn_ppb else jax.lax.stop_gradient(ppb[i])

            pre_value = conn.get_pre_value(mod_vals, delayed_srcs=dsrc) if conn.is_res else None
            prediction = conn.prediction(pre_act, pw_i, pb_i, pre_value)
            # Precision input: pre_act by default; custom precision_input
            # sources read current values (live) / prev-iteration carries.
            prec_in = conn.get_precision_input(
                pre_act, mod_vals, prev_errors, prev_precisions,
                activation_fns, layer_activations, key)
            prediction = _add_prediction_noise(
                conn, i, prediction, prec_in, ppw_i, ppb_i, is_stochastic, key)
            if conn.unit_precision:
                precision = jnp.ones((prediction.shape[0], 1), dtype=prediction.dtype)  # provably 1.0 — see _inference_step
            else:
                log_prec = conn.log_precision_fn(prec_in, ppw_i, ppb_i)
                prev_p_i = prev_precisions[i] if prev_precisions else None
                precision = _activated_precision(
                    predict_precision_activations, conn, i, log_prec, prev_p_i,
                    key=key)
            post_val = conn.get_post(mod_vals)
            prev_e_i = prev_errors[i] if prev_errors else None
            residual = _activated_error(
                predict_error_activations, conn, i,
                post_val - prediction, prev_e_i, key=key)
            all_errors.append(residual)
            all_precisions.append(precision)

            # Split errors for per-leg flow gating (Mechanism 2)
            if predict_has_flow_gates and predict_has_flow_gates[i]:
                e_pre = _activated_error(
                    predict_error_activations, conn, i,
                    jax.lax.stop_gradient(post_val) - prediction, prev_e_i,
                    key=key)
                e_post = _activated_error(
                    predict_error_activations, conn, i,
                    post_val - jax.lax.stop_gradient(prediction), prev_e_i,
                    key=key)
                all_e_pre.append(e_pre)
                all_e_post.append(e_post)
            else:
                all_e_pre.append(residual)
                all_e_post.append(residual)

        # 2b. Apply precision-targeting Project/Modulate (Mechanism 1)
        mod_precisions = _apply_project_modulate_precision(
            tuple(all_precisions), mod_vals, tuple(all_errors),
            sg_proj_w, sg_mod_w,
            project_conns_precision, modulate_conns_precision,
            activation_fns,
            project_biases=proj_b,
            modulate_biases=mod_b,
            hist=hist, tick_base=tick_base, iter_idx=iter_idx,
            iters_per_timestep=iters_per_timestep)

        # 3. Apply error-targeting Project/Modulate
        mod_errors = _apply_project_modulate_internal(
            tuple(all_errors), mod_vals,
            sg_proj_w, sg_mod_w,
            project_conns_internal, modulate_conns_internal,
            activation_fns,
            precisions=mod_precisions,
            project_biases=proj_b,
            modulate_biases=mod_b,
            hist=hist, tick_base=tick_base, iter_idx=iter_idx,
            iters_per_timestep=iters_per_timestep)

        # 3b. Structural attention: softmax reweighting (Mechanism 3)
        if structural_attention_groups:
            mod_errors_list = list(mod_errors)
            for group in structural_attention_groups:
                neg_energies = []
                for idx in group.conn_indices:
                    e_i = mod_errors_list[idx]
                    p_i = mod_precisions[idx]
                    neg_e = -0.5 * jnp.sum(
                        p_i * e_i ** 2, axis=-1) / group.temperature
                    neg_energies.append(neg_e)
                alphas = jax.nn.softmax(
                    jnp.stack(neg_energies, axis=-1), axis=-1)
                for k, idx in enumerate(group.conn_indices):
                    mod_errors_list[idx] = (
                        mod_errors_list[idx] * alphas[:, k:k+1])
            mod_errors = tuple(mod_errors_list)

        # 4. Energy summation
        for i in range(n_pc_conn):
            scale = pre_scales[i] if pre_scales else 1.0
            if predict_has_flow_gates and predict_has_flow_gates[i]:
                # Split energy: independent per-leg gating
                E = E + scale * 0.5 * jnp.sum(jnp.mean(
                    mod_precisions[i] * flow_gates_pre[i] * all_e_pre[i] ** 2
                    + mod_precisions[i] * flow_gates_post[i] * all_e_post[i] ** 2
                    - jnp.log(mod_precisions[i]), axis=-1))
            else:
                E = E + scale * 0.5 * jnp.sum(
                    jnp.mean(mod_precisions[i] * mod_errors[i] ** 2
                             - jnp.log(mod_precisions[i]), axis=-1))

        # Inference regularization
        for layer_idx, reg in inference_regs:
            reg_key = jax.random.fold_in(key, layer_idx) if key is not None else None
            E = E + reg.apply(mod_vals[layer_idx], key=reg_key, labels=labels)

        # Train regularization #TODO: have only one toye of reg?? think about it
        for layer_idx, reg in train_regs:
            reg_key = jax.random.fold_in(key, layer_idx) if key is not None else None
            E = E + reg.apply(mod_vals[layer_idx], key=reg_key, labels=labels)

        return E, (mod_errors, tuple(mod_precisions), mod_vals)

    (_, (pre_update_errors, pre_update_precisions, pre_update_mod_vals)), (val_grads, param_grads) = jax.value_and_grad(
        energy_with_aux, argnums=(0, 1), has_aux=True)(values, trainable)

    # Scale param grads to mean-over-batch convention
    param_grads = jax.tree.map(lambda g: g / batch_size, param_grads)

    # Compute loss grads for GD-loss connections and merge before optimizer call
    if gd_loss_project or gd_loss_modulate:
        loss_gd_pw = tuple(project_weights[idx] for idx, _ in gd_loss_project)
        loss_gd_mw = tuple(modulate_weights[idx] for idx, _ in gd_loss_modulate)

        def loss_objective(gd_pw_args, gd_mw_args):
            l_proj_w = list(project_weights)
            for k, (idx, _) in enumerate(gd_loss_project):
                l_proj_w[idx] = gd_pw_args[k]
            l_proj_w = tuple(jax.lax.stop_gradient(w) if i not in {idx for idx, _ in gd_loss_project} else w
                             for i, w in enumerate(l_proj_w))
            l_mod_w = list(modulate_weights)
            for k, (idx, _) in enumerate(gd_loss_modulate):
                l_mod_w[idx] = gd_mw_args[k]
            l_mod_w = tuple(jax.lax.stop_gradient(w) if i not in {idx for idx, _ in gd_loss_modulate} else w
                            for i, w in enumerate(l_mod_w))

            l_mod_vals = _apply_project_modulate_values_for_loss(
                values, prev_errors, l_proj_w, l_mod_w,
                project_conns_value, modulate_conns_value, activation_fns,
                project_biases=tuple(jax.lax.stop_gradient(b) for b in project_biases),
                modulate_biases=tuple(jax.lax.stop_gradient(b) for b in modulate_biases),
                precisions=pre_update_precisions)

            l_errors = []
            l_precisions = []
            for i, conn in enumerate(predict_conns):
                pre_act = conn.get_pre(l_mod_vals, (), activation_fns,
                                       layer_activations, key)
                pre_value = conn.get_pre_value(l_mod_vals) if conn.is_res else None
                prediction = conn.prediction(
                    pre_act, predict_weights[i], predict_biases[i], pre_value)
                prev_e_i = prev_errors[i] if prev_errors else None
                l_errors.append(_activated_error(
                    predict_error_activations, conn, i,
                    conn.get_post(l_mod_vals) - prediction, prev_e_i, key=key))
                l_prec_in = conn.get_precision_input(
                    pre_act, l_mod_vals, prev_errors, prev_precisions,
                    activation_fns, layer_activations, key)
                l_log_prec = conn.log_precision_fn(
                    l_prec_in, precision_weights[i], precision_biases[i])
                prev_p_i = prev_precisions[i] if prev_precisions else None
                l_precisions.append(_activated_precision(
                    predict_precision_activations, conn, i, l_log_prec, prev_p_i,
                    key=key))

            l_mod_errors = _apply_project_modulate_internal_for_loss(
                tuple(l_errors), l_mod_vals, l_proj_w, l_mod_w,
                project_conns_internal, modulate_conns_internal, activation_fns,
                precisions=tuple(l_precisions),
                project_biases=tuple(jax.lax.stop_gradient(b) for b in project_biases),
                modulate_biases=tuple(jax.lax.stop_gradient(b) for b in modulate_biases))

            node_arrays = (l_mod_vals, tuple(l_mod_errors), tuple(l_precisions))

            L = 0.
            seen = set()
            for _, lfn_idx in list(gd_loss_project) + list(gd_loss_modulate):
                if lfn_idx not in seen:
                    seen.add(lfn_idx)
                    resolved_inputs, fn = loss_fns[lfn_idx]
                    args = _gather_loss_fn_args(
                        resolved_inputs, node_arrays, loss_fn_sample_arrays)
                    L = L + fn(*args)
            return L

        gd_pw_grads, gd_mw_grads = jax.grad(
            loss_objective, argnums=(0, 1))(loss_gd_pw, loss_gd_mw)

        param_grads = {**param_grads,
            'gd_loss_project_weights': tuple(g / batch_size for g in gd_pw_grads),
            'gd_loss_modulate_weights': tuple(g / batch_size for g in gd_mw_grads),
        }

    # Apply values optimizer
    updates, new_values_opt_state = values_optimizer.update(val_grads, values_opt_state, values)
    new_values_raw = optax.apply_updates(values, updates)
    new_values = tuple(
        _apply_mask(clamped[j], values[j], new_values_raw[j])
        for j in range(n_layers)
    )

    # Apply value-targeting Project/Modulate as explicit state operators
    # (integrating drive), persisting into the carried values. Jacobi: pre is
    # read from the frozen pre-update v[t] (= `values`), not the post-energy
    # state, so the energy step and routing both combine from v[t]. Re-clamps.
    new_values = _apply_project_modulate_values(
        new_values, pre_update_errors,
        project_weights, modulate_weights,
        project_conns_value, modulate_conns_value,
        activation_fns, clamped,
        project_biases=project_biases, modulate_biases=modulate_biases,
        read_values=values, is_boundary=is_boundary,
        hist=hist, tick_base=tick_base, iter_idx=iter_idx,
        iters_per_timestep=iters_per_timestep,
        precisions=pre_update_precisions)

    # Apply params optimizer
    if isinstance(params_optimizer, optax.GradientTransformationExtraArgs):
        features = tuple(
            conn.get_pre(values, (), activation_fns)
            for conn in predict_conns
        )
        updates, new_params_opt_state = params_optimizer.update(
            param_grads, params_opt_state, trainable, features=features
        )
    else:
        updates, new_params_opt_state = params_optimizer.update(
            param_grads, params_opt_state, trainable
        )
    new_trainable = optax.apply_updates(trainable, updates)

    # Post-processing: restore fixed/non-learned predict params
    new_pw = list(new_trainable['predict_weights'])
    new_pb = list(new_trainable['predict_biases'])
    new_ppw = list(new_trainable['precision_weights'])
    new_ppb = list(new_trainable['precision_biases'])
    new_proj_b = list(new_trainable['project_biases'])
    new_mod_b = list(new_trainable['modulate_biases'])

    for i, conn in enumerate(predict_conns):
        if conn.has_fixed_weights:
            new_pw[i] = predict_weights[i]
            new_pb[i] = predict_biases[i]
            new_ppw[i] = precision_weights[i]
            new_ppb[i] = precision_biases[i]
        else:
            if not conn.has_bias:
                new_pb[i] = predict_biases[i]
            if not conn.learn_precision_weights:
                new_ppw[i] = precision_weights[i]
            if not conn.learn_precision_bias:
                new_ppb[i] = precision_biases[i]
        if conn.n_bands > 0:
            m, n = new_pw[i].shape
            new_pw[i] = new_pw[i] * _make_band_mask(m, n, conn.n_bands)
        if conn.is_masked:
            new_pw[i] = new_pw[i] * predict_weight_masks[i]

    # Restore project/modulate biases for connections without use_bias
    for i, conn in enumerate(project_conns):
        if not conn.has_bias:
            new_proj_b[i] = project_biases[i]
    for i, conn in enumerate(modulate_conns):
        if not conn.has_bias:
            new_mod_b[i] = modulate_biases[i]

    # Learn Project/Modulate weights:
    # 1) Hebbian/ThreeFactorHebbian — uses pre-update mod state
    new_project_weights, new_modulate_weights = _learn_project_modulate_weights(
        pre_update_mod_vals, pre_update_errors,
        project_weights, modulate_weights,
        project_conns, modulate_conns,
        activation_fns, batch_size,
        reward_fns,
        precisions=pre_update_precisions,
        reward_fn_sample_arrays=loss_fn_sample_arrays)

    # 2) GD-loss weights already updated by params_optimizer — overwrite
    new_project_weights = list(new_project_weights)
    new_modulate_weights = list(new_modulate_weights)
    for j, (idx, _) in enumerate(gd_loss_project):
        new_project_weights[idx] = new_trainable['gd_loss_project_weights'][j]
    for j, (idx, _) in enumerate(gd_loss_modulate):
        new_modulate_weights[idx] = new_trainable['gd_loss_modulate_weights'][j]

    # Re-mask banded / custom-masked project/modulate weights
    for i, conn in enumerate(project_conns):
        if conn.n_bands > 0 and not (conn.is_conv or conn.is_transconv):
            m, n = new_project_weights[i].shape
            new_project_weights[i] = new_project_weights[i] * _make_band_mask(m, n, conn.n_bands)
        if conn.is_masked:
            new_project_weights[i] = new_project_weights[i] * project_weight_masks[i]
    for i, conn in enumerate(modulate_conns):
        if conn.n_bands > 0 and not (conn.is_conv or conn.is_transconv):
            m, n = new_modulate_weights[i].shape
            new_modulate_weights[i] = new_modulate_weights[i] * _make_band_mask(m, n, conn.n_bands)
        if conn.is_masked:
            new_modulate_weights[i] = new_modulate_weights[i] * modulate_weight_masks[i]

    new_project_weights = tuple(new_project_weights)
    new_modulate_weights = tuple(new_modulate_weights)

    # Return pre-update errors/precisions from energy; callers recompute at new state when logging
    return (
        new_values, pre_update_errors, pre_update_precisions,
        tuple(new_pw), tuple(new_pb),
        new_project_weights, tuple(new_proj_b),
        new_modulate_weights, tuple(new_mod_b),
        tuple(new_ppw), tuple(new_ppb),
        new_values_opt_state, new_params_opt_state,
    )


def _recompute_errors_precisions(
    values, predict_conns, predict_weights, predict_biases,
    precision_weights, precision_biases, activation_fns,
    predict_error_activations: tuple = (),
    prev_errors: tuple = (),
    predict_precision_activations: tuple = (),
    prev_precisions: tuple = (),
    is_stochastic: bool = False,
    key = None,
    project_weights: tuple = (),
    modulate_weights: tuple = (),
    project_biases: tuple = (),
    modulate_biases: tuple = (),
    project_conns_precision: tuple = (),
    modulate_conns_precision: tuple = (),
    project_conns_internal: tuple = (),
    modulate_conns_internal: tuple = (),
    hist=(), tick_base=0, iter_idx=0, iters_per_timestep=1,
):
    """Compute errors and precisions at the given values (forward pass only).

    ``prev_errors`` and ``prev_precisions`` are used by memory-aware
    activations and by error/precision ``precision_input`` sources (which
    always read the previous iteration's carry); stateless slots with default
    precision sourcing take the standard transform path. When
    ``is_stochastic`` and ``key`` are supplied, predictions get the same
    precision-scaled noise as the energy step (see :func:`_add_prediction_noise`),
    so the logged/carried errors reflect the stochastic dynamics.

    Precision- and error-targeting Project/Modulate connections are applied to
    the recomputed precisions and errors (same order as the energy: precision
    routing first, then error routing) so the carried/logged errors and
    precisions match what the energy consumed. They are no-ops when the
    corresponding connection lists are empty.
    """
    errors = []
    precisions = []
    for i, conn in enumerate(predict_conns):
        dsrc = _delayed_srcs(conn, hist, tick_base, iter_idx, iters_per_timestep)
        pre_act = conn.get_pre(values, (), activation_fns, delayed_srcs=dsrc)
        pre_value = conn.get_pre_value(values, delayed_srcs=dsrc) if conn.is_res else None
        prediction = conn.prediction(
            pre_act, predict_weights[i], predict_biases[i], pre_value)
        prec_in = conn.get_precision_input(
            pre_act, values, prev_errors, prev_precisions, activation_fns)
        prediction = _add_prediction_noise(
            conn, i, prediction, prec_in,
            precision_weights[i], precision_biases[i], is_stochastic, key)
        log_prec = conn.log_precision_fn(
            prec_in, precision_weights[i], precision_biases[i])
        prev_p_i = prev_precisions[i] if prev_precisions else None
        precision = _activated_precision(
            predict_precision_activations, conn, i, log_prec, prev_p_i)
        prev_e_i = prev_errors[i] if prev_errors else None
        errors.append(_activated_error(
            predict_error_activations, conn, i,
            conn.get_post(values) - prediction, prev_e_i))
        precisions.append(precision)

    precisions = _apply_project_modulate_precision(
        tuple(precisions), values, tuple(errors),
        project_weights, modulate_weights,
        project_conns_precision, modulate_conns_precision,
        activation_fns,
        project_biases=project_biases, modulate_biases=modulate_biases,
        hist=hist, tick_base=tick_base, iter_idx=iter_idx,
        iters_per_timestep=iters_per_timestep)
    errors = _apply_project_modulate_internal(
        tuple(errors), values,
        project_weights, modulate_weights,
        project_conns_internal, modulate_conns_internal,
        activation_fns,
        precisions=precisions,
        project_biases=project_biases, modulate_biases=modulate_biases,
        hist=hist, tick_base=tick_base, iter_idx=iter_idx,
        iters_per_timestep=iters_per_timestep)
    return tuple(errors), tuple(precisions)


def _inference_step(
    values: tuple,
    errors: tuple,
    clamped: tuple,
    predict_weights: tuple,
    predict_biases: tuple,
    project_weights: tuple,
    project_biases: tuple,
    modulate_weights: tuple,
    modulate_biases: tuple,
    precision_weights: tuple,
    precision_biases: tuple,
    predict_conns: tuple,
    project_conns_internal: tuple,
    project_conns_value: tuple,
    modulate_conns_internal: tuple,
    modulate_conns_value: tuple,
    activation_fns: tuple,
    values_optimizer,
    values_opt_state,
    is_poisson_types: tuple = None,
    key = None,
    is_stochastic: bool = False,
    spatial_layers: tuple = (),
    spatial_neighborhoods: tuple = (),
    inference_regs: tuple = (),
    labels = None,
    # Attention mechanisms
    project_conns_precision: tuple = (),
    modulate_conns_precision: tuple = (),
    modulate_conns_flow_pre: tuple = (),
    modulate_conns_flow_post: tuple = (),
    predict_has_flow_gates: tuple = (),
    structural_attention_groups: tuple = (),
    pre_scales: tuple = (),
    predict_error_activations: tuple = (),
    predict_precision_activations: tuple = (),
    precisions: tuple = (),
    layer_activations: tuple = (),
    # Traced scalar, truthy on the first iteration of each input timestep.
    # None => no value Project/Modulate is advance='timestep' gated.
    is_boundary=None,
    # Delay history buffers threaded from run_batch. hist=() => delay==0 path
    # (bit-identical). See _combined_step for the argument semantics.
    hist=(), tick_base=0, iter_idx=0, iters_per_timestep=1,
) -> Tuple[tuple, tuple, tuple, jnp.ndarray, Any]:
    """
    Single inference step: compute energy-based value gradients and update values.

    Uses value_and_grad with has_aux to return errors/precisions from the
    energy forward pass without a redundant recomputation.

    Returns:
        (new_values, pre_update_errors, pre_update_precisions, energy, new_values_opt_state)

    Note: errors and precisions are at the PRE-UPDATE values (computed during
    the energy forward pass). Caller is responsible for recomputing at new
    values when needed (e.g., for logging).
    """
    n_layers = len(values)
    batch_size = values[0].shape[0]

    sg_proj_w = project_weights
    sg_mod_w = modulate_weights

    def energy_with_aux(vals):
        # 1. Value-targeting Project/Modulate are NOT applied in the energy.
        # They are explicit state operators applied after the value update
        # (see _apply_project_modulate_values below), so the carried values
        # already include them and the energy reads them directly.
        mod_vals = vals

        # 1b. Compute per-leg flow gates (Mechanism 2)
        flow_gates_pre = []
        flow_gates_post = []
        for i, conn in enumerate(predict_conns):
            post_dim = mod_vals[conn.post_idx].shape[-1]
            flow_gates_pre.append(jnp.ones((batch_size, post_dim)))
            flow_gates_post.append(jnp.ones((batch_size, post_dim)))
        for weight_idx, mc in modulate_conns_flow_pre:
            pre_act = mc.get_pre(mod_vals, errors, activation_fns)
            gate = mc.apply(
                jax.lax.stop_gradient(pre_act), sg_mod_w[weight_idx])
            flow_gates_pre[mc.post_idx] = flow_gates_pre[mc.post_idx] * gate
        for weight_idx, mc in modulate_conns_flow_post:
            pre_act = mc.get_pre(mod_vals, errors, activation_fns)
            gate = mc.apply(
                jax.lax.stop_gradient(pre_act), sg_mod_w[weight_idx])
            flow_gates_post[mc.post_idx] = flow_gates_post[mc.post_idx] * gate
        flow_gates_pre = tuple(flow_gates_pre)
        flow_gates_post = tuple(flow_gates_post)

        # 2. Compute predictions, errors, precisions from mod_vals
        E = 0.
        all_errors = []
        all_precisions = []
        all_e_pre = []
        all_e_post = []
        for i, conn in enumerate(predict_conns):
            dsrc = _delayed_srcs(conn, hist, tick_base, iter_idx, iters_per_timestep)
            pre_act = conn.get_pre(mod_vals, (), activation_fns,
                                   layer_activations, key, delayed_srcs=dsrc)
            pre_value = conn.get_pre_value(mod_vals, delayed_srcs=dsrc) if conn.is_res else None
            prediction = conn.prediction(
                pre_act, predict_weights[i], predict_biases[i], pre_value)
            prec_in = conn.get_precision_input(
                pre_act, mod_vals, errors, precisions,
                activation_fns, layer_activations, key)
            prediction = _add_prediction_noise(
                conn, i, prediction, prec_in,
                precision_weights[i], precision_biases[i], is_stochastic, key)
            if conn.unit_precision:
                # Precision is provably 1.0 — skip the (loop-invariant) softplus
                # of a runtime bias; a compile-time ones constant lets XLA fold
                # ``1*err^2 - log 1`` to ``err^2`` in the value gradient. Shape
                # (batch, 1) matches the non-learned precision's broadcast shape
                # (bias is (1,)), keeping the carried/recomputed logs consistent.
                precision = jnp.ones((prediction.shape[0], 1), dtype=prediction.dtype)
            else:
                log_prec = conn.log_precision_fn(prec_in, precision_weights[i], precision_biases[i])
                prev_p_i = precisions[i] if precisions else None
                precision = _activated_precision(
                    predict_precision_activations, conn, i, log_prec, prev_p_i,
                    key=key)
            post_val = conn.get_post(mod_vals)
            prev_e_i = errors[i] if errors else None
            residual = _activated_error(
                predict_error_activations, conn, i,
                post_val - prediction, prev_e_i, key=key)
            all_errors.append(residual)
            all_precisions.append(precision)

            if predict_has_flow_gates and predict_has_flow_gates[i]:
                all_e_pre.append(_activated_error(
                    predict_error_activations, conn, i,
                    jax.lax.stop_gradient(post_val) - prediction, prev_e_i,
                    key=key))
                all_e_post.append(_activated_error(
                    predict_error_activations, conn, i,
                    post_val - jax.lax.stop_gradient(prediction), prev_e_i,
                    key=key))
            else:
                all_e_pre.append(residual)
                all_e_post.append(residual)

        # 2b. Apply precision-targeting Project/Modulate (Mechanism 1)
        mod_precisions = _apply_project_modulate_precision(
            tuple(all_precisions), mod_vals, tuple(all_errors),
            sg_proj_w, sg_mod_w,
            project_conns_precision, modulate_conns_precision,
            activation_fns,
            project_biases=project_biases,
            modulate_biases=modulate_biases,
            hist=hist, tick_base=tick_base, iter_idx=iter_idx,
            iters_per_timestep=iters_per_timestep)

        # 3. Apply error-targeting Project/Modulate
        mod_errors = _apply_project_modulate_internal(
            tuple(all_errors), mod_vals,
            sg_proj_w, sg_mod_w,
            project_conns_internal, modulate_conns_internal,
            activation_fns,
            precisions=mod_precisions,
            project_biases=project_biases,
            modulate_biases=modulate_biases,
            hist=hist, tick_base=tick_base, iter_idx=iter_idx,
            iters_per_timestep=iters_per_timestep)

        # 3b. Structural attention (Mechanism 3)
        if structural_attention_groups:
            mod_errors_list = list(mod_errors)
            for group in structural_attention_groups:
                neg_energies = []
                for idx in group.conn_indices:
                    e_i = mod_errors_list[idx]
                    p_i = mod_precisions[idx]
                    neg_e = -0.5 * jnp.sum(
                        p_i * e_i ** 2, axis=-1) / group.temperature
                    neg_energies.append(neg_e)
                alphas = jax.nn.softmax(
                    jnp.stack(neg_energies, axis=-1), axis=-1)
                for k, idx in enumerate(group.conn_indices):
                    mod_errors_list[idx] = (
                        mod_errors_list[idx] * alphas[:, k:k+1])
            mod_errors = tuple(mod_errors_list)

        # 4. Energy summation
        for i in range(len(predict_conns)):
            scale = pre_scales[i] if pre_scales else 1.0
            if predict_has_flow_gates and predict_has_flow_gates[i]:
                E = E + scale * 0.5 * jnp.sum(jnp.mean(
                    mod_precisions[i] * flow_gates_pre[i] * all_e_pre[i] ** 2
                    + mod_precisions[i] * flow_gates_post[i] * all_e_post[i] ** 2
                    - jnp.log(mod_precisions[i]), axis=-1))
            else:
                E = E + scale * 0.5 * jnp.sum(
                    jnp.mean(mod_precisions[i] * mod_errors[i] ** 2
                             - jnp.log(mod_precisions[i]), axis=-1))

        # Inference regularization
        for layer_idx, reg in inference_regs:
            reg_key = jax.random.fold_in(key, layer_idx) if key is not None else None
            E = E + reg.apply(mod_vals[layer_idx], key=reg_key, labels=labels)

        return E, (tuple(all_errors), tuple(mod_precisions))

    (energy_val, (pre_errors, pre_precisions)), val_grads = jax.value_and_grad(
        energy_with_aux, has_aux=True)(values)

    # Apply optax, respecting clamping (soft blend)
    updates, new_values_opt_state = values_optimizer.update(val_grads, values_opt_state, values)
    new_values_raw = optax.apply_updates(values, updates)
    new_values = tuple(
        _apply_mask(clamped[j], values[j], new_values_raw[j])
        for j in range(n_layers)
    )

    # Apply value-targeting Project/Modulate as explicit state operators
    # (integrating drive), persisting into the carried values. Jacobi: pre is
    # read from the frozen pre-update v[t] (= `values`), not the post-energy
    # state, so the energy step and routing both combine from v[t]. Re-clamps.
    new_values = _apply_project_modulate_values(
        new_values, errors,
        project_weights, modulate_weights,
        project_conns_value, modulate_conns_value,
        activation_fns, clamped,
        project_biases=project_biases, modulate_biases=modulate_biases,
        read_values=values, is_boundary=is_boundary,
        hist=hist, tick_base=tick_base, iter_idx=iter_idx,
        iters_per_timestep=iters_per_timestep,
        precisions=precisions)

    return tuple(new_values), pre_errors, pre_precisions, energy_val, new_values_opt_state


def _compute_energy(errors: tuple, precisions: tuple, pre_scales: tuple = ()) -> jnp.ndarray:
    """Compute total variational free energy: 0.5*precision*residual^2 - log(precision),
    mean over dimensions, mean over batch, summed across predict connections.
    pre_scales: per-connection scale factors (1/n_predicts_to_pre); defaults to 1.0."""
    energy = 0
    for i, (error, prec) in enumerate(zip(errors, precisions)):
        scale = pre_scales[i] if pre_scales else 1.0
        safe_prec = jnp.clip(prec, 1e-8, None)
        energy = energy + scale * 0.5 * jnp.mean(jnp.mean(prec * error ** 2 - jnp.log(safe_prec), axis=-1))
    return energy


def _compute_deltas(errors: tuple, precisions: tuple) -> tuple:
    """Second-order errors: delta = 0.5 * (error**2 - 1/precision), per Predict conn.
    `error` is the raw (non-precision-modulated) residual."""
    return tuple(
        0.5 * (e ** 2 - 1.0 / jnp.clip(p, 1e-8, None))
        for e, p in zip(errors, precisions)
    )


def _single_pass(
    values: tuple,
    errors: tuple,
    clamped: tuple,
    predict_weights: tuple,
    predict_biases: tuple,
    project_weights: tuple,
    project_biases: tuple,
    modulate_weights: tuple,
    modulate_biases: tuple,
    precision_weights: tuple,
    precision_biases: tuple,
    predict_conns: tuple,
    project_conns: tuple, # todo: these should be used in single pass too, same order relative to everything as in the combined/inference
    modulate_conns: tuple, 
    activation_fns: tuple,
    project_conns_value: tuple = (),
    modulate_conns_value: tuple = (),
    is_poisson_types: tuple = None,
    key = None,
    is_stochastic: bool = False,
    predict_error_activations: tuple = (),
    predict_precision_activations: tuple = (),
    layer_activations: tuple = (),
    feedforward_init: bool = True,
    project_conns_precision: tuple = (),
    modulate_conns_precision: tuple = (),
    project_conns_internal: tuple = (),
    modulate_conns_internal: tuple = (),
    precisions_carry: tuple = (),
) -> Tuple[tuple, tuple]:
    """
    Single forward pass: propagate clamped values through the network sequentially
    and compute errors. Each connection uses values already updated by earlier
    connections, so predictions propagate correctly through multi-layer chains.
    Clamped layer values are always preserved.

    Also applies value-targeting Project/Modulate connections so that networks
    using Project (e.g. with GradientDescent) get non-zero initial values.

    ``feedforward_init`` (Python-static): when True (default) the predictions/
    projections/modulations overwrite the non-clamped layer values, seeding
    inference with a forward pass. When False, layer values are left at their
    initial state (zeros, except clamped inputs) and only the errors and
    precisions are computed against them — inference then starts from zeros.

    ``precisions_carry`` provides the "previous iteration" arrays read by
    precision-node ``precision_input`` sources on this very first pass
    (run_batch passes the activated init precision, ``g(precision_bias)``).
    Error sources read the ``errors`` argument (zeros at init). Only needed
    when some conn has a precision-node source.
    """
    new_values = list(values)
    new_errors = [jnp.zeros_like(e) for e in errors]
    new_precisions = []

    for i, conn in enumerate(predict_conns):
        subkey = jax.random.fold_in(key, i)

        # Handle Poisson per-layer for multi-pre
        has_poisson = is_poisson_types is not None and any(
            is_poisson_types[idx] for idx in conn.pre_idx) and key is not None
        if has_poisson:
            parts = []
            for k, idx in enumerate(conn.pre_idx):
                act_fn = activation_fns[idx]
                if is_poisson_types[idx]:
                    pk = jax.random.fold_in(subkey, k)
                    rate = jnp.clip(act_fn(new_values[idx]), 0.0, 1e6)
                    arr = jax.random.poisson(pk, rate).astype(jnp.float32)
                else:
                    arr = act_fn(new_values[idx])
                sl = conn.pre_slices[k] if conn.pre_slices else None
                if sl is not None:
                    arr = arr[:, sl[0]:sl[1]]
                parts.append(arr)
            pre_act = parts[0] if len(parts) == 1 else jnp.concatenate(parts, axis=-1)
        else:
            pre_act = conn.get_pre(tuple(new_values), (), activation_fns,
                                   layer_activations, subkey)

        pre_value = conn.get_pre_value(tuple(new_values)) if conn.is_res else None
        prec_in = conn.get_precision_input(
            pre_act, tuple(new_values), errors, precisions_carry,
            activation_fns, layer_activations, subkey)
        if is_stochastic and key is not None and getattr(conn, 'stochastic', True):
            stoch_key = jax.random.fold_in(key, i + 1000)
            prediction = conn.stochastic_prediction(
                pre_act, predict_weights[i], predict_biases[i],
                precision_weights[i], precision_biases[i], stoch_key, pre_value,
                precision_input=prec_in, noise_scale=float(is_stochastic) ** 0.5,
            )
        else:
            prediction = conn.prediction(pre_act, predict_weights[i], predict_biases[i], pre_value)
        log_prec = conn.log_precision_fn(
            prec_in, precision_weights[i], precision_biases[i])
        precision = _activated_precision(
            predict_precision_activations, conn, i, log_prec, None, key=subkey)
        post_val = conn.get_post(tuple(new_values))
        prev_e_i = errors[i] if errors else None
        new_errors[i] = _activated_error(
            predict_error_activations, conn, i,
            post_val - prediction, prev_e_i, key=subkey)
        new_precisions.append(precision)
        # Set post-layer values to a soft blend of clamped input and prediction.
        # Skipped when feedforward_init is off so non-clamped values stay at init.
        if feedforward_init:
            if conn.post_slice:
                s, e = conn.post_slice
                clamp_slice = clamped[conn.post_idx][:, s:e] if clamped[conn.post_idx].ndim > 1 else clamped[conn.post_idx]
                updated = _apply_mask(clamp_slice, post_val, prediction)
                new_values[conn.post_idx] = new_values[conn.post_idx].at[:, s:e].set(updated)
            else:
                new_values[conn.post_idx] = _apply_mask(
                    clamped[conn.post_idx], new_values[conn.post_idx], prediction
                )

    if feedforward_init:
        # Apply value-targeting Project connections (additive)
        for weight_idx, conn in project_conns_value:
            pre_act = conn.get_pre(tuple(new_values), errors, activation_fns,
                                   precisions=tuple(new_precisions))
            p_bias = project_biases[weight_idx] if project_biases else 0.0
            projection = conn.apply(pre_act, project_weights[weight_idx]) + p_bias
            updated = _write_additive(new_values[conn.post_idx], projection, conn.post_slice)
            new_values[conn.post_idx] = _apply_mask(clamped[conn.post_idx], new_values[conn.post_idx], updated)

        # Apply value-targeting Modulate connections (multiplicative)
        for weight_idx, conn in modulate_conns_value:
            pre_act = conn.get_pre(tuple(new_values), errors, activation_fns,
                                   precisions=tuple(new_precisions))
            bias = modulate_biases[weight_idx] if modulate_biases else 0.0
            modulation = conn.apply(pre_act, modulate_weights[weight_idx]) + bias
            updated = _write_multiplicative(new_values[conn.post_idx], modulation, conn.post_slice)
            new_values[conn.post_idx] = _apply_mask(clamped[conn.post_idx], new_values[conn.post_idx], updated)

    # Apply precision- then error-targeting Project/Modulate so the initial
    # carried errors/precisions match what the inference energy will consume.
    new_precisions = _apply_project_modulate_precision(
        tuple(new_precisions), tuple(new_values), tuple(new_errors),
        project_weights, modulate_weights,
        project_conns_precision, modulate_conns_precision,
        activation_fns,
        project_biases=project_biases, modulate_biases=modulate_biases)
    new_errors = _apply_project_modulate_internal(
        tuple(new_errors), tuple(new_values),
        project_weights, modulate_weights,
        project_conns_internal, modulate_conns_internal,
        activation_fns,
        precisions=new_precisions,
        project_biases=project_biases, modulate_biases=modulate_biases)

    return tuple(new_values), tuple(new_errors), tuple(new_precisions)


def _learn_project_modulate_weights(
    values, errors,
    project_weights, modulate_weights,
    project_conns, modulate_conns,
    activation_fns, batch_size,
    reward_fns=(),
    precisions=(),
    reward_fn_sample_arrays: dict = {},
):
    """Learn Project/Modulate weights using Hebbian/ThreeFactorHebbian rules.

    GradientDescent (type 2) connections are skipped — they are handled by
    the energy backward pass (loss_fn=None) or a separate loss gradient
    (loss_fn provided) in _combined_step.
    """
    new_pw = list(project_weights)
    new_mw = list(modulate_weights)
    node_arrays = (tuple(values), tuple(errors), tuple(precisions))

    def _apply_rule(W, conn, weights_list, idx):
        if conn.learning_rule_type == 2:  # GradientDescent — handled elsewhere
            return

        pre_act = conn.get_pre(values, errors, activation_fns, precisions=precisions)
        post_arr = conn.get_post(values, errors, precisions=precisions)

        if conn.learning_rule_type == 0:  # Hebbian
            dW = conn.learning_rate * jnp.mean(
                post_arr[:, :, None] * pre_act[:, None, :], axis=0)
            weights_list[idx] = W + dW
        elif conn.learning_rule_type == 3:  # Oja
            hebbian = post_arr[:, :, None] * pre_act[:, None, :]
            decay = (post_arr ** 2)[:, :, None] * W[None, :, :]
            dW = conn.learning_rate * jnp.mean(hebbian - decay, axis=0)
            weights_list[idx] = W + dW
        elif conn.learning_rule_type == 1:  # ThreeFactorHebbian
            if conn.reward_fn_idx >= 0 and len(reward_fns) > conn.reward_fn_idx:
                resolved_inputs, fn = reward_fns[conn.reward_fn_idx]
                args = _gather_loss_fn_args(
                    resolved_inputs, node_arrays, reward_fn_sample_arrays)
                reward = fn(*args)
                if reward.ndim == 0:
                    reward = jnp.broadcast_to(reward, (batch_size,))
                dW = conn.learning_rate * jnp.mean(
                    reward[:, None, None] * post_arr[:, :, None] * pre_act[:, None, :],
                    axis=0)
                weights_list[idx] = W + dW

    for i, conn in enumerate(project_conns):
        _apply_rule(project_weights[i], conn, new_pw, i)

    for i, conn in enumerate(modulate_conns):
        _apply_rule(modulate_weights[i], conn, new_mw, i)

    return tuple(new_pw), tuple(new_mw)



# ============================================================================
# Main simulation function
# ============================================================================

@eqx.filter_jit(donate='all')
def run_batch(
    sample: Dict[str, jnp.ndarray],
    params: NetworkParams,
    structure: NetworkStructure,
    data_map: tuple,
    n_iterations: int,
    log_every: int,
    learning: bool = True,
    n_learning_iterations: int = 0,
    reward_fns: tuple = (),
    loss_fns: tuple = (),
    convergence_threshold: float = 0.0,
    key: jnp.ndarray = None,
    values_optimizer = None,
    values_opt_state = None,
    params_optimizer = None,
    params_opt_state = None,
    is_stochastic: bool = False,
    spatial_neighborhoods: tuple = (),
    log_initial: bool = False,
    predict_weight_masks: tuple = (),
    project_weight_masks: tuple = (),
    modulate_weight_masks: tuple = (),
    predict_error_activations: tuple = (),
    predict_precision_activations: tuple = (),
    layer_activations: tuple = (),
    feedforward_init: bool = True,
) -> Tuple[NetworkParams, Any, Any, tuple, tuple, tuple, tuple, jnp.ndarray]:
    """
    Run a complete batch: init -> clamp -> reset -> inference -> weight update.

    Returns:
        new_params, new_params_opt_state, new_values_opt_state,
        values_log, errors_log, precisions_log, deltas_log, energies

    `deltas_log[i] = 0.5 * (errors_log[i]**2 - 1 / precisions_log[i])` — the
    per-element second-order error (variance residual) for each Predict conn.

    If `log_initial=True`, the logs (and `energies`) have one extra leading slot
    holding the state right after the initial forward pass (i.e. before any
    inference step). All subsequent log writes shift by +1. Default False.
    """
    # Unpack params
    predict_weights = tuple(params.predict_weights)
    predict_biases = tuple(params.predict_biases)
    project_weights = tuple(params.project_weights)
    project_biases = tuple(params.project_biases)
    modulate_weights = tuple(params.modulate_weights)
    modulate_biases = tuple(params.modulate_biases)
    precision_weights = tuple(params.precision_weights)
    precision_biases = tuple(params.precision_biases)

    layer_dims = structure.layer_dims
    predict_error_dims = structure.predict_error_dims

    predict_conns = structure.predict_conns
    predict_pre_scales = structure.predict_pre_scales
    project_conns = structure.project_conns
    modulate_conns = structure.modulate_conns

    # Pre-sorted connection lists
    project_conns_internal = structure.project_conns_internal
    project_conns_value = structure.project_conns_value
    modulate_conns_internal = structure.modulate_conns_internal
    modulate_conns_value = structure.modulate_conns_value

    spatial_layers = structure.spatial_layers
    inference_regs = structure.inference_regs
    train_regs = structure.train_regs

    # Mechanism 1: precision-targeting
    project_conns_precision = structure.project_conns_precision
    modulate_conns_precision = structure.modulate_conns_precision
    # Mechanism 2: per-leg flow gating
    modulate_conns_flow_pre = structure.modulate_conns_flow_pre
    modulate_conns_flow_post = structure.modulate_conns_flow_post
    predict_has_flow_gates = structure.predict_has_flow_gates
    # Mechanism 3: structural attention
    structural_attention_groups = structure.structural_attention_groups

    # GradientDescent (loss-based) connection indices
    gd_loss_project = structure.gd_loss_project
    gd_loss_modulate = structure.gd_loss_modulate
    loss_fn_sample_keys = structure.loss_fn_sample_keys

    # Extract sample arrays needed by loss functions
    loss_fn_sample_arrays = {k: sample[k] for k in loss_fn_sample_keys} if loss_fn_sample_keys else {}

    activation_types = tuple(layer.activation_type for layer in structure.layers)
    activation_temps = tuple(
        float(getattr(layer, 'activation_temperature', 1.0)) for layer in structure.layers)
    activation_winners = tuple(
        int(getattr(layer, 'activation_num_winners', 0)) for layer in structure.layers)
    activation_thresholds = tuple(
        tuple(getattr(layer, 'activation_thresholds', ())) for layer in structure.layers)
    # Apply input temperature as activation(x / T); T==1.0 is the plain fn.
    # NWTA layers (num_winners > 0) bake the winner count into the closure,
    # mirroring how Softmax bakes in its temperature. ThresholdRelu layers
    # (non-empty thresholds) bake the per-neuron subtractive thresholds:
    # f(x) = max(x - theta, 0).
    def _build_activation_fn(t, T, nw, thr):
        if thr:
            return lambda x, _a=jnp.asarray(thr): jnp.maximum(x - _a, 0)
        if nw > 0:
            return lambda x, _nw=nw: _nwta(x, _nw)
        if T == 1.0:
            return ACTIVATIONS[t]
        return lambda x, _t=t, _T=T: ACTIVATIONS[_t](x / _T)
    activation_fns = tuple(
        _build_activation_fn(t, T, nw, thr)
        for t, T, nw, thr in zip(activation_types, activation_temps,
                                 activation_winners, activation_thresholds))
    is_poisson_types = tuple(layer.is_poisson for layer in structure.layers)
    # Per-layer Activation instances are only threaded into the prediction
    # pre-activation when at least one layer is stochastic; otherwise the
    # plain ``activation_fns`` path is used (zero behaviour change).
    _layer_acts = layer_activations if any(
        getattr(a, 'needs_key', False) for a in layer_activations) else ()
    # Per-layer dropout probability; 0.0 means no dropout. Plumbed to the
    # predict-loop so Bernoulli masks are applied to activated values used as
    # downstream predictions during *learning* (training) only.
    dropout_probs = tuple(
        float(getattr(layer, 'dropout_prob', 0.0)) for layer in structure.layers
    )

    # Use PRNGKey(0) if no key provided
    if key is None:
        key = jax.random.PRNGKey(0)

    # Get batch size from first data array
    first_key = list(sample.keys())[0]
    batch_size = sample[first_key].shape[0]

    # Generate per-layer Bernoulli dropout masks (shape (B, D), values 0 or 1)
    # ONCE per batch. Mask is the same across all inference iterations of the
    # batch — re-applied at the start of each iteration so dropped dims stay
    # at zero. Only active when ``learning`` is True and the layer's
    # ``dropout_prob`` is > 0; otherwise mask is all-ones (a no-op).
    #
    # We deliberately use NON-inverted dropout (mask only, no 1/(1-p) scaling).
    # The inverted variant blows up the prediction-error energy in the first
    # epochs because the freshly-initialized predict weights, scaled by
    # 1/(1-p), overshoot the data target. With the non-inverted variant the
    # predict weights learn an implicit (1-p) compensation, and test-time
    # predictions match the training scale up to the standard MC-dropout
    # train/test bias (negligible for our task).
    _dropout_active = bool(learning) and any(p > 0.0 for p in dropout_probs)
    if _dropout_active:
        _dropout_masks = tuple(
            jax.random.bernoulli(
                jax.random.fold_in(key, 0xD20 + i),
                1.0 - p, shape=(batch_size, dim),
            ).astype(jnp.float32)
            if p > 0.0 else jnp.ones((batch_size, dim), dtype=jnp.float32)
            for i, (p, dim) in enumerate(zip(dropout_probs, layer_dims))
        )
    else:
        _dropout_masks = tuple(
            jnp.ones((batch_size, dim), dtype=jnp.float32) for dim in layer_dims
        )

    # Initialize state
    n_layers = len(layer_dims)
    values_list = [jnp.zeros((batch_size, dim)) for dim in layer_dims]
    clamped_list = [jnp.zeros((batch_size, dim), dtype=jnp.float32) for dim in layer_dims]

    # Extract class labels if provided via sentinel (-1, sample_key)
    labels = None
    for entry in data_map:
        if entry[0] == -1:
            labels = sample[entry[1]]
            break

    # Detect temporal dimension from clamped data
    n_timesteps = 1
    for entry in data_map:
        layer_idx = entry[0]
        if layer_idx == -1:
            continue
        data_key = entry[1] if isinstance(entry[1], str) else entry[1][0]
        data = sample[data_key]
        if data.ndim == 3:
            T = data.shape[1]
            if n_timesteps == 1:
                n_timesteps = T
            elif T != n_timesteps:
                raise ValueError(
                    f"Inconsistent temporal dimensions: expected {n_timesteps}, got {T}")

    # Validate temporal dimension
    total_iterations = n_iterations + (n_learning_iterations if learning else 0)
    if n_timesteps > 1:
        if n_timesteps > total_iterations:
            raise ValueError(
                f"n_timesteps ({n_timesteps}) exceeds total_iterations ({total_iterations})")
        if total_iterations % n_timesteps != 0:
            raise ValueError(
                f"total_iterations ({total_iterations}) is not an integer multiple of "
                f"n_timesteps ({n_timesteps})")
    iters_per_timestep = max(total_iterations, 1) // n_timesteps

    # advance='timestep' gating: only pay for the boundary indicator when at
    # least one value-targeting Project/Modulate actually asked for it. This is
    # a plain Python check over the static specs (trace time), so networks with
    # no gated conn get `is_boundary=None` and the original, un-gated code path.
    _has_ts_gated_value_pm = any(
        getattr(s, 'advance_timestep', False)
        for _, s in tuple(project_conns_value) + tuple(modulate_conns_value))

    def _timestep_boundary(iter_idx):
        """Traced bool: True on the first inference iteration of a timestep."""
        if not _has_ts_gated_value_pm:
            return None
        return (iter_idx % iters_per_timestep) == 0

    # Out-of-loop `_combined_step` calls (the `n_learning_iterations == 0`
    # equilibrium step and the final precision update) run on `clamped_last`
    # AFTER the sequence has been consumed, so they are not timestep
    # boundaries: a gated conn must not advance there. `None` when nothing is
    # gated, which keeps the original code path.
    _not_a_boundary = jnp.bool_(False) if _has_ts_gated_value_pm else None

    # ---- Delay + one-step-carry history buffers ----
    # Phase 1: value pre nodes read at delay>=1 (node_type 0). One ring per
    # (node, unit) read at delay>=1; entry k is (depth_k+1, B, dim).
    # Phase 2: the former dedicated ``prev_errors`` / ``prev_precisions``
    # one-step carries are folded in as depth-1, unit='iteration' buffers
    # (node_type 1 = error, 2 = precision; node_id is the predict-conn index).
    # Value buffers are written at the TOP of each body from the carry-in
    # ``values``; error/precision buffers at the BOTTOM from the step's
    # freshly-recomputed carry-out. Empty when nothing is delayed AND nothing
    # reads a carried error/precision, so delay==0 / no-consumer nets stay
    # bit-identical.
    _hist_specs = structure.hist_specs
    _hist_unit_ts = structure.hist_unit_ts
    # Legacy/manual structures may omit node types -> Phase-1 all-value default.
    _hist_node_types = structure.hist_node_types or tuple(0 for _ in _hist_specs)
    # Runtime precision arrays are (B, error_dim) (broadcast from the bias),
    # except unit_precision conns which are (B, 1). Sizing rings from the
    # STORED bias shape breaks for scalar ``init_precision`` biases of shape
    # (1,) as soon as a precision consumer exists (write of (B, D) into a
    # (depth, B, 1) ring).
    _precision_dims = tuple(
        1 if conn.unit_precision else ed
        for conn, ed in zip(predict_conns, predict_error_dims))

    def _hist_buf_dim(node_type, node_id):
        if node_type == 0:
            return layer_dims[node_id]
        if node_type == 1:
            return predict_error_dims[node_id]
        return _precision_dims[node_id]

    hist = tuple(
        jnp.zeros((depth + 1, batch_size, _hist_buf_dim(nt, node_id)))
        for (node_id, depth), nt in zip(_hist_specs, _hist_node_types)
    )

    # Per-predict-conn maps: error/precision node -> hist buffer index. Empty
    # tuples when that node-type is not consumed (the dropped-carry path).
    _n_predict = len(predict_conns)
    errors_consumed = any(nt == 1 for nt in _hist_node_types)
    precisions_consumed = any(nt == 2 for nt in _hist_node_types)
    if errors_consumed:
        _emap = {node_id: k for k, ((node_id, _d), nt)
                 in enumerate(zip(_hist_specs, _hist_node_types)) if nt == 1}
        err_buf_idx = tuple(_emap[i] for i in range(_n_predict))
    else:
        err_buf_idx = ()
    if precisions_consumed:
        _pmap = {node_id: k for k, ((node_id, _d), nt)
                 in enumerate(zip(_hist_specs, _hist_node_types)) if nt == 2}
        prec_buf_idx = tuple(_pmap[i] for i in range(_n_predict))
    else:
        prec_buf_idx = ()

    def _write_hist(hist, values, global_iter):
        """Advance the VALUE delay rings from the post-clamp carry-in ``values``.

        Called at the TOP of each loop body (after temporal-clamp+dropout, before
        the step). ``global_iter`` is ``tick_base + i``. Error/precision buffers
        (node_type != 0) are skipped here — they are bottom-written by
        ``_write_hist_ep`` from the recomputed carry-out. Static no-op when there
        are no value buffers.
        """
        if not _hist_specs:
            return hist
        new = list(hist)
        for k, (node_id, depth) in enumerate(_hist_specs):
            if _hist_node_types[k] != 0:
                continue  # error/precision buffer: bottom-write, not here
            S = depth + 1
            if _hist_unit_ts[k]:  # latched ('timestep'): push at frame boundary
                tick = global_iter // iters_per_timestep
                push = ((global_iter % iters_per_timestep) == 0) & (global_iter > 0)
                slot = (tick - 1) % S
                new[k] = jnp.where(
                    push, hist[k].at[slot].set(values[node_id]), hist[k])
            else:                 # sliding ('iteration'): write every iteration
                new[k] = hist[k].at[global_iter % S].set(values[node_id])
        return tuple(new)

    def _reconstruct_ep(hist, buf_idx_map, tick_base, iter_idx):
        """Reconstruct a per-conn (errors or precisions) tuple from ``hist`` at
        delay 1 — the value the deleted one-step carry would have held at the top
        of this iteration. Returns () when the node-type is dropped, so the
        ``... if x else None`` guards downstream reproduce today's behaviour.
        """
        if not buf_idx_map:
            return ()
        return tuple(
            _read_delayed(hist, buf_idx_map[i], 1, False,
                          tick_base, iter_idx, iters_per_timestep)
            for i in range(len(buf_idx_map)))

    def _write_hist_ep(hist, err_arrays, prec_arrays, global_iter):
        """Write the step's carry-out errors/precisions into their depth-1
        iteration buffers at slot ``global_iter % S`` (S == 2). Bottom-of-body,
        after the recompute/aux carry-out is known. Static no-op when neither
        node-type is buffered.
        """
        if not err_buf_idx and not prec_buf_idx:
            return hist
        new = list(hist)
        for i, bi in enumerate(err_buf_idx):
            S = new[bi].shape[0]
            new[bi] = new[bi].at[global_iter % S].set(err_arrays[i])
        for i, bi in enumerate(prec_buf_idx):
            S = new[bi].shape[0]
            new[bi] = new[bi].at[global_iter % S].set(prec_arrays[i])
        return tuple(new)

    # Clamp data
    temporal_clamp_list = [jnp.zeros((batch_size, n_timesteps, dim)) for dim in layer_dims]
    for entry in data_map:
        layer_idx = entry[0]
        if layer_idx == -1:
            continue
        sample_key_or_pair = entry[1]
        if isinstance(sample_key_or_pair, str):
            data = sample[sample_key_or_pair]
            if data.ndim == 2:
                data = data[:, None, :]
            temporal_clamp_list[layer_idx] = data
            values_list[layer_idx] = data[:, 0, :]
            clamped_list[layer_idx] = jnp.ones((batch_size, layer_dims[layer_idx]), dtype=jnp.float32)
        else:
            data_key, mask_key = sample_key_or_pair
            data = sample[data_key]
            if data.ndim == 2:
                data = data[:, None, :]
            mask = sample[mask_key]
            temporal_clamp_list[layer_idx] = data
            clamped_list[layer_idx] = mask
            mask_t0 = mask[:, 0, :] if mask.ndim == 3 else mask
            values_list[layer_idx] = _apply_mask(mask_t0, data[:, 0, :], values_list[layer_idx])
    temporal_clamp_values = tuple(temporal_clamp_list)
    clamped = tuple(
        c if c.ndim == 3 else jnp.broadcast_to(c[:, None, :], (batch_size, n_timesteps, c.shape[-1]))
        for c in clamped_list
    )

    # Forward pass
    errors_init = tuple(jnp.zeros((batch_size, dim)) for dim in predict_error_dims)
    # Precision-node precision_input sources need a "previous iteration"
    # precision on the very first pass: use the activated init precision
    # g(precision_bias). Only built when some conn actually has such a source,
    # keeping the default graph unchanged.
    if any(2 in c.precision_input_node_types for c in predict_conns
           if c.precision_input_node_types):
        precisions_carry_init = tuple(
            jnp.broadcast_to(conn.precision_transform(pb)[None, :],
                             (batch_size, pb.shape[0]))
            for conn, pb in zip(predict_conns, precision_biases))
    else:
        precisions_carry_init = ()
    clamped_t0 = tuple(c[:, 0, :] for c in clamped)
    values, errors, precisions = _single_pass(
        tuple(values_list), errors_init, clamped_t0,
        predict_weights, predict_biases,
        project_weights, project_biases,
        modulate_weights, modulate_biases,
        precision_weights, precision_biases,
        predict_conns, project_conns, modulate_conns,
        activation_fns, project_conns_value, modulate_conns_value,
        is_poisson_types, key, is_stochastic,
        predict_error_activations=predict_error_activations,
        predict_precision_activations=predict_precision_activations,
        layer_activations=_layer_acts,
        feedforward_init=feedforward_init,
        project_conns_precision=project_conns_precision,
        modulate_conns_precision=modulate_conns_precision,
        project_conns_internal=project_conns_internal,
        modulate_conns_internal=modulate_conns_internal,
        precisions_carry=precisions_carry_init,
    )

    # Phase 2 bootstrap: pre-fill the error/precision one-step buffers so
    # iteration 0's delay-1 read (slot (0-1)%S == 1) returns the SAME seed the
    # old dedicated carry used — the ``_single_pass`` output errors/precisions
    # (post-routing; the g(bias) precision bootstrap is already baked in). Slot
    # 0 stays zeros, overwritten by iteration 0's bottom write.
    if err_buf_idx or prec_buf_idx:
        _h = list(hist)
        for _i, _bi in enumerate(err_buf_idx):
            _h[_bi] = _h[_bi].at[(-1) % _h[_bi].shape[0]].set(errors[_i])
        for _i, _bi in enumerate(prec_buf_idx):
            _h[_bi] = _h[_bi].at[(-1) % _h[_bi].shape[0]].set(precisions[_i])
        hist = tuple(_h)

    # Initialize values optimizer
    _values_optimizer = values_optimizer if values_optimizer is not None else optax.sgd(1.0)
    _values_opt_state = (
        _values_optimizer.init(values) if values_opt_state is None else values_opt_state
    )

    if (n_iterations == 1) and (learning == False) and (n_learning_iterations == 0):
        values_log = tuple(v[None] for v in values)
        errors_log = tuple(e[None] for e in errors)
        precisions_log = tuple(p[None] for p in precisions)
        deltas = _compute_deltas(errors, precisions)
        deltas_log = tuple(d[None] for d in deltas)
        energies = jnp.atleast_1d(_compute_energy(errors, precisions, predict_pre_scales))
        new_params = NetworkParams(
            predict_weights=list(predict_weights),
            predict_biases=list(predict_biases),
            project_weights=list(project_weights),
            project_biases=list(project_biases),
            modulate_weights=list(modulate_weights),
            modulate_biases=list(modulate_biases),
            precision_weights=list(precision_weights),
            precision_biases=list(precision_biases),
        )
        return new_params, params_opt_state, _values_opt_state, values_log, errors_log, precisions_log, deltas_log, energies

    else:
        # log_initial=True reserves slot 0 for the post-_single_pass state and
        # shifts every subsequent write by +1. Captured by the body closures
        # below via Python-static `log_initial_offset`.
        log_initial_offset = 1 if log_initial else 0
        n_logged = (total_iterations + log_every - 1) // log_every + log_initial_offset
        _has_pm_error_reading = bool(project_conns_internal or modulate_conns_internal)

        def _state_log(i, energies_list, values_log, errors_log, precisions_log, deltas_log, values, errors, precisions):
            """Log state conditionally: only computes energy and writes on logging iterations."""
            should_log = (i + 1) % log_every == 0
            log_idx = i // log_every + log_initial_offset

            def _do_log(_):
                energy = _compute_energy(errors, precisions, predict_pre_scales)
                deltas = _compute_deltas(errors, precisions)
                new_energies = energies_list.at[log_idx].set(energy)
                new_vl = tuple(vl.at[log_idx].set(v) for vl, v in zip(values_log, values))
                new_el = tuple(el.at[log_idx].set(e) for el, e in zip(errors_log, errors))
                new_pl = tuple(pl.at[log_idx].set(p) for pl, p in zip(precisions_log, precisions))
                new_dl = tuple(dl.at[log_idx].set(d) for dl, d in zip(deltas_log, deltas))
                return new_energies, new_vl, new_el, new_pl, new_dl

            def _skip_log(_):
                return energies_list, values_log, errors_log, precisions_log, deltas_log

            return lax.cond(should_log, _do_log, _skip_log, None)

        # Initialize log storage
        energies = jnp.zeros((n_logged,))
        values_log = tuple(jnp.zeros((n_logged, batch_size, dim)) for dim in layer_dims)
        errors_log = tuple(jnp.zeros((n_logged, batch_size, dim)) for dim in predict_error_dims)
        # Same sizing rule as the delay rings: runtime precisions are
        # (B, error_dim) once any precision routing applies (a modulated
        # (B, 1) bias precision broadcasts to the gate's full dim), so logs
        # sized from the stored bias shape would reject the write. (B, 1)
        # entries still broadcast INTO a full-dim slot, so this is
        # shape-compatible for all previously working nets.
        precision_dims = _precision_dims
        precisions_log = tuple(jnp.zeros((n_logged, batch_size, dim)) for dim in precision_dims)
        # deltas share the per-Predict-connection error shape
        deltas_log = tuple(jnp.zeros((n_logged, batch_size, dim)) for dim in predict_error_dims)

        if log_initial:
            init_deltas = _compute_deltas(errors, precisions)
            init_energy = _compute_energy(errors, precisions, predict_pre_scales)
            energies = energies.at[0].set(init_energy)
            values_log = tuple(vl.at[0].set(v) for vl, v in zip(values_log, values))
            errors_log = tuple(el.at[0].set(e) for el, e in zip(errors_log, errors))
            precisions_log = tuple(pl.at[0].set(p) for pl, p in zip(precisions_log, precisions))
            deltas_log = tuple(dl.at[0].set(d) for dl, d in zip(deltas_log, init_deltas))

        # Phase 1: Inference-only iterations
        if n_iterations > 0:

            if convergence_threshold > 0:
                # Full inference body with convergence checking
                def inference_body(i, carry):
                    # The convergence path keeps a full errors/precisions carry:
                    # its freeze-on-converge (skip_step returns the operand
                    # unchanged) must return the SAME pytree structure as
                    # do_step's full recompute, so the depth-1 hist reconstruction
                    # (which is () for dropped node-types) cannot stand in here.
                    # The carry-out is still written into hist at the bottom so a
                    # downstream learning/precision step reconstructs correctly.
                    values, errors, precisions, energies_list, values_log, errors_log, precisions_log, deltas_log, converged, vos, hist = carry

                    t = i // iters_per_timestep
                    clamped_t = tuple(c[:, t, :] for c in clamped)
                    values = tuple(
                        _apply_mask(clamped_t[j], temporal_clamp_values[j][:, t, :], values[j])
                        for j in range(n_layers)
                    )
                    # Latent dropout: zero out free (unclamped) positions
                    # according to per-layer Bernoulli mask. dropped dims are
                    # held at 0 throughout the iteration; gradient on them is 0.
                    values = tuple(
                        jnp.where(clamped_t[j] > 0.5, values[j],
                                  values[j] * _dropout_masks[j])
                        for j in range(n_layers)
                    )
                    # Advance value delay rings from the post-clamp carry-in.
                    hist = _write_hist(hist, values, i)
                    prev_t = jnp.maximum(i - 1, 0) // iters_per_timestep
                    converged = jnp.where((t != prev_t) & (i > 0), jnp.bool_(False), converged)

                    iter_key = jax.random.fold_in(key, i)
                    is_boundary = _timestep_boundary(i)

                    def do_step(operand):
                        v, e, prec, vos = operand
                        new_v, pre_e, pre_prec, energy, new_vos = _inference_step(
                            v, e, clamped_t,
                            predict_weights, predict_biases,
                            project_weights, project_biases,
                            modulate_weights, modulate_biases,
                            precision_weights, precision_biases,
                            predict_conns,
                            project_conns_internal, project_conns_value,
                            modulate_conns_internal, modulate_conns_value,
                            activation_fns, _values_optimizer, vos, is_poisson_types, iter_key,
                            is_stochastic,
                            spatial_layers, spatial_neighborhoods, inference_regs, labels,
                            project_conns_precision=project_conns_precision,
                            modulate_conns_precision=modulate_conns_precision,
                            modulate_conns_flow_pre=modulate_conns_flow_pre,
                            modulate_conns_flow_post=modulate_conns_flow_post,
                            predict_has_flow_gates=predict_has_flow_gates,
                            structural_attention_groups=structural_attention_groups,
                            pre_scales=predict_pre_scales,
                            predict_error_activations=predict_error_activations,
                            predict_precision_activations=predict_precision_activations,
                            precisions=prec,
                            layer_activations=_layer_acts,
                            is_boundary=is_boundary,
                            hist=hist, tick_base=0, iter_idx=i,
                            iters_per_timestep=iters_per_timestep,
                        )
                        new_e, new_prec = _recompute_errors_precisions(
                            new_v, predict_conns, predict_weights, predict_biases,
                            precision_weights, precision_biases, activation_fns,
                            predict_error_activations=predict_error_activations,
                            prev_errors=e,
                            predict_precision_activations=predict_precision_activations,
                            prev_precisions=prec,
                            project_weights=project_weights, modulate_weights=modulate_weights,
                            project_biases=project_biases, modulate_biases=modulate_biases,
                            project_conns_precision=project_conns_precision,
                            modulate_conns_precision=modulate_conns_precision,
                            project_conns_internal=project_conns_internal,
                            modulate_conns_internal=modulate_conns_internal,
                            is_stochastic=is_stochastic, key=iter_key,
                            hist=hist, tick_base=0, iter_idx=i,
                            iters_per_timestep=iters_per_timestep)
                        return new_v, new_e, new_prec, new_vos

                    def skip_step(operand):
                        return operand

                    new_values, new_errors, new_precisions, new_vos = lax.cond(
                        converged, skip_step, do_step, (values, errors, precisions, vos)
                    )

                    # Re-apply dropout mask to free positions so dropped dims
                    # stay at exactly zero throughout the iteration (otherwise
                    # the value optimizer would push them away from zero,
                    # defeating the dropout). Done BEFORE logging so the log
                    # reflects the actual masked state used by inference.
                    new_values = tuple(
                        jnp.where(clamped_t[j] > 0.5, new_values[j],
                                  new_values[j] * _dropout_masks[j])
                        for j in range(n_layers)
                    )

                    max_delta = jnp.float32(0.0)
                    for j in range(n_layers):
                        delta = jnp.max(jnp.abs(new_values[j] - values[j]))
                        max_delta = jnp.maximum(max_delta, delta)
                    new_converged = jnp.logical_and(
                        convergence_threshold > 0.0,
                        max_delta < convergence_threshold
                    )
                    converged = jnp.logical_or(converged, new_converged)

                    energies_list, values_log, errors_log, precisions_log, deltas_log = _state_log(
                        i, energies_list, values_log, errors_log, precisions_log, deltas_log,
                        new_values, new_errors, new_precisions
                    )
                    converged = jnp.logical_or(converged, jnp.isnan(energies_list[-1]))
                    # Bottom write: mirror the full carry-out into the hist rings
                    # (E[i+1]) so any following learning/precision step, which
                    # reconstructs its prev from hist, stays consistent. No-op for
                    # dropped node-types.
                    hist = _write_hist_ep(hist, new_errors, new_precisions, i)
                    return new_values, new_errors, new_precisions, energies_list, values_log, errors_log, precisions_log, deltas_log, converged, new_vos, hist

                values, errors, precisions, energies, values_log, errors_log, precisions_log, deltas_log, _, _values_opt_state, hist = lax.fori_loop(
                    0, n_iterations, inference_body,
                    (values, errors, precisions, energies, values_log, errors_log, precisions_log, deltas_log, jnp.bool_(False), _values_opt_state, hist)
                )
            else:
                # Fast inference body: no convergence check, merged recompute+log via lax.cond
                def inference_body_fast(i, carry):
                    values, energies_list, values_log, errors_log, precisions_log, deltas_log, vos, hist = carry

                    t = i // iters_per_timestep
                    clamped_t = tuple(c[:, t, :] for c in clamped)
                    values = tuple(
                        _apply_mask(clamped_t[j], temporal_clamp_values[j][:, t, :], values[j])
                        for j in range(n_layers)
                    )
                    values = tuple(
                        jnp.where(clamped_t[j] > 0.5, values[j],
                                  values[j] * _dropout_masks[j])
                        for j in range(n_layers)
                    )
                    # Advance value delay rings from the post-clamp carry-in.
                    hist = _write_hist(hist, values, i)
                    # Reconstruct the one-step error/precision carries (E[i]/P[i]).
                    errors = _reconstruct_ep(hist, err_buf_idx, 0, i)
                    precisions = _reconstruct_ep(hist, prec_buf_idx, 0, i)

                    iter_key = jax.random.fold_in(key, i)
                    is_boundary = _timestep_boundary(i)

                    new_values, pre_errors, pre_precisions, energy, new_vos = _inference_step(
                        values, errors, clamped_t,
                        predict_weights, predict_biases,
                        project_weights, project_biases,
                        modulate_weights, modulate_biases,
                        precision_weights, precision_biases,
                        predict_conns,
                        project_conns_internal, project_conns_value,
                        modulate_conns_internal, modulate_conns_value,
                        activation_fns, _values_optimizer, vos, is_poisson_types, iter_key,
                        is_stochastic,
                        spatial_layers, spatial_neighborhoods, inference_regs, labels,
                        project_conns_precision=project_conns_precision,
                        modulate_conns_precision=modulate_conns_precision,
                        modulate_conns_flow_pre=modulate_conns_flow_pre,
                        modulate_conns_flow_post=modulate_conns_flow_post,
                        predict_has_flow_gates=predict_has_flow_gates,
                        structural_attention_groups=structural_attention_groups,
                        pre_scales=predict_pre_scales,
                        predict_error_activations=predict_error_activations,
                        predict_precision_activations=predict_precision_activations,
                        precisions=precisions,
                        layer_activations=_layer_acts,
                        is_boundary=is_boundary,
                        hist=hist, tick_base=0, iter_idx=i,
                        iters_per_timestep=iters_per_timestep,
                    )

                    # Re-apply dropout mask BEFORE logging so the log reflects
                    # the actual values carried into the next iteration.
                    new_values = tuple(
                        jnp.where(clamped_t[j] > 0.5, new_values[j],
                                  new_values[j] * _dropout_masks[j])
                        for j in range(n_layers)
                    )

                    should_log = (i + 1) % log_every == 0

                    if _has_pm_error_reading:
                        new_errors, new_precisions = _recompute_errors_precisions(
                            new_values, predict_conns, predict_weights, predict_biases,
                            precision_weights, precision_biases, activation_fns,
                            predict_error_activations=predict_error_activations,
                            prev_errors=errors,
                            predict_precision_activations=predict_precision_activations,
                            prev_precisions=precisions,
                            project_weights=project_weights, modulate_weights=modulate_weights,
                            project_biases=project_biases, modulate_biases=modulate_biases,
                            project_conns_precision=project_conns_precision,
                            modulate_conns_precision=modulate_conns_precision,
                            project_conns_internal=project_conns_internal,
                            modulate_conns_internal=modulate_conns_internal,
                            is_stochastic=is_stochastic, key=iter_key,
                            hist=hist, tick_base=0, iter_idx=i,
                            iters_per_timestep=iters_per_timestep)
                        energies_list, values_log, errors_log, precisions_log, deltas_log = _state_log(
                            i, energies_list, values_log, errors_log, precisions_log, deltas_log,
                            new_values, new_errors, new_precisions
                        )
                    else:
                        log_idx = i // log_every + log_initial_offset
                        def _do_recompute_and_log(_):
                            ne, np_ = _recompute_errors_precisions(
                                new_values, predict_conns, predict_weights, predict_biases,
                                precision_weights, precision_biases, activation_fns,
                                predict_error_activations=predict_error_activations,
                                prev_errors=errors,
                                predict_precision_activations=predict_precision_activations,
                                prev_precisions=precisions,
                                project_weights=project_weights, modulate_weights=modulate_weights,
                                project_biases=project_biases, modulate_biases=modulate_biases,
                                project_conns_precision=project_conns_precision,
                                modulate_conns_precision=modulate_conns_precision,
                                project_conns_internal=project_conns_internal,
                                modulate_conns_internal=modulate_conns_internal,
                                is_stochastic=is_stochastic, key=iter_key,
                                hist=hist, tick_base=0, iter_idx=i,
                                iters_per_timestep=iters_per_timestep)
                            energy = _compute_energy(ne, np_)
                            nd_ = _compute_deltas(ne, np_)
                            new_energies = energies_list.at[log_idx].set(energy)
                            new_vl = tuple(vl.at[log_idx].set(v) for vl, v in zip(values_log, new_values))
                            new_el = tuple(el.at[log_idx].set(e) for el, e in zip(errors_log, ne))
                            new_pl = tuple(pl.at[log_idx].set(p) for pl, p in zip(precisions_log, np_))
                            new_dl = tuple(dl.at[log_idx].set(d) for dl, d in zip(deltas_log, nd_))
                            return ne, np_, new_energies, new_vl, new_el, new_pl, new_dl

                        def _skip_all(_):
                            return pre_errors, pre_precisions, energies_list, values_log, errors_log, precisions_log, deltas_log

                        new_errors, new_precisions, energies_list, values_log, errors_log, precisions_log, deltas_log = lax.cond(
                            should_log, _do_recompute_and_log, _skip_all, None)

                    # Bottom write: store the carry-out errors/precisions (E[i+1]).
                    hist = _write_hist_ep(hist, new_errors, new_precisions, i)
                    return new_values, energies_list, values_log, errors_log, precisions_log, deltas_log, new_vos, hist

                values, energies, values_log, errors_log, precisions_log, deltas_log, _values_opt_state, hist = lax.fori_loop(
                    0, n_iterations, inference_body_fast,
                    (values, energies, values_log, errors_log, precisions_log, deltas_log, _values_opt_state, hist)
                )

        # Resolve params optimizer
        new_params_opt_state = params_opt_state
        if learning:
            _params_optimizer = params_optimizer if params_optimizer is not None else optax.adam(1e-4)
            if params_opt_state is None:
                _trainable = {
                    'predict_weights': predict_weights,
                    'predict_biases': predict_biases,
                    'project_biases': project_biases,
                    'modulate_biases': modulate_biases,
                    'precision_weights': precision_weights,
                    'precision_biases': precision_biases,
                    'gd_loss_project_weights': tuple(project_weights[idx] for idx, _ in gd_loss_project),
                    'gd_loss_modulate_weights': tuple(modulate_weights[idx] for idx, _ in gd_loss_modulate),
                }
                new_params_opt_state = _params_optimizer.init(_trainable)

        # Phase 2: Learning
        if learning:
            if n_learning_iterations == 0:
                clamped_last = tuple(c[:, -1, :] for c in clamped)
                learn_key = jax.random.fold_in(key, n_iterations)
                # Equilibrium step after inference: reconstruct the one-step
                # error/precision carries from the rings inference left behind.
                errors = _reconstruct_ep(hist, err_buf_idx, n_iterations, 0)
                precisions = _reconstruct_ep(hist, prec_buf_idx, n_iterations, 0)
                (values, errors, precisions,
                 predict_weights, predict_biases,
                 project_weights, project_biases,
                 modulate_weights, modulate_biases,
                 precision_weights, precision_biases,
                 _values_opt_state, new_params_opt_state) = _combined_step(
                    values, errors, clamped_last,
                    predict_weights, predict_biases,
                    project_weights, project_biases,
                    modulate_weights, modulate_biases,
                    precision_weights, precision_biases,
                    predict_conns, project_conns, modulate_conns,
                    project_conns_internal, project_conns_value,
                    modulate_conns_internal, modulate_conns_value,
                    activation_fns, _values_optimizer, _values_opt_state,
                    _params_optimizer, new_params_opt_state,
                    gd_loss_project, gd_loss_modulate,
                    reward_fns, loss_fns, loss_fn_sample_arrays,
                    spatial_layers, spatial_neighborhoods,
                    inference_regs, train_regs, learn_key, labels,
                    is_boundary=_not_a_boundary,
                    project_conns_precision=project_conns_precision,
                    modulate_conns_precision=modulate_conns_precision,
                    modulate_conns_flow_pre=modulate_conns_flow_pre,
                    modulate_conns_flow_post=modulate_conns_flow_post,
                    predict_has_flow_gates=predict_has_flow_gates,
                    structural_attention_groups=structural_attention_groups,
                    pre_scales=predict_pre_scales,
                    predict_weight_masks=predict_weight_masks,
                    project_weight_masks=project_weight_masks,
                    modulate_weight_masks=modulate_weight_masks,
                    predict_error_activations=predict_error_activations,
                    predict_precision_activations=predict_precision_activations,
                    prev_precisions=precisions,
                    layer_activations=_layer_acts,
                    is_stochastic=is_stochastic,
                    # Equilibrium learning step after the sequence: read delayed
                    # from the rings left by inference (no new write here).
                    hist=hist, tick_base=n_iterations, iter_idx=0,
                    iters_per_timestep=iters_per_timestep,
                )
                # errors/precisions are pre-update; not used after this point
            else:
                # Precision is frozen during the iterative learning loop
                # (update_precision=False) to prevent the precision-error
                # feedback loop.  A single precision update happens after
                # the loop via a final combined_step with update_precision=True.
                def learning_body(i, carry):
                    values, energies_list, values_log, errors_log, precisions_log, deltas_log, pw, pb, prw, prb, mw, mb, ppw, ppb, params_opt_st, vos, hist = carry

                    t = (n_iterations + i) // iters_per_timestep
                    clamped_t = tuple(c[:, t, :] for c in clamped)
                    values = tuple(
                        _apply_mask(clamped_t[j], temporal_clamp_values[j][:, t, :], values[j])
                        for j in range(n_layers)
                    )
                    values = tuple(
                        jnp.where(clamped_t[j] > 0.5, values[j],
                                  values[j] * _dropout_masks[j])
                        for j in range(n_layers)
                    )
                    # Advance value delay rings; learning loop offsets by n_iterations.
                    hist = _write_hist(hist, values, n_iterations + i)
                    # Reconstruct the one-step error/precision carries.
                    errors = _reconstruct_ep(hist, err_buf_idx, n_iterations, i)
                    precisions = _reconstruct_ep(hist, prec_buf_idx, n_iterations, i)

                    learn_key = jax.random.fold_in(key, n_iterations + i)
                    is_boundary = _timestep_boundary(n_iterations + i)
                    (values, pre_errors, pre_precisions,
                     pw, pb, prw, prb, mw, mb, ppw, ppb,
                     vos, params_opt_st) = _combined_step(
                        values, errors, clamped_t,
                        pw, pb, prw, prb, mw, mb, ppw, ppb,
                        predict_conns, project_conns, modulate_conns,
                        project_conns_internal, project_conns_value,
                        modulate_conns_internal, modulate_conns_value,
                        activation_fns, _values_optimizer, vos,
                        _params_optimizer, params_opt_st,
                        gd_loss_project, gd_loss_modulate,
                        reward_fns, loss_fns, loss_fn_sample_arrays,
                        spatial_layers, spatial_neighborhoods,
                        inference_regs, train_regs, learn_key, labels,
                        update_precision=False,
                        is_stochastic=is_stochastic,
                        project_conns_precision=project_conns_precision,
                        modulate_conns_precision=modulate_conns_precision,
                        modulate_conns_flow_pre=modulate_conns_flow_pre,
                        modulate_conns_flow_post=modulate_conns_flow_post,
                        predict_has_flow_gates=predict_has_flow_gates,
                        structural_attention_groups=structural_attention_groups,
                        pre_scales=predict_pre_scales,
                        predict_weight_masks=predict_weight_masks,
                        project_weight_masks=project_weight_masks,
                        modulate_weight_masks=modulate_weight_masks,
                        predict_error_activations=predict_error_activations,
                        predict_precision_activations=predict_precision_activations,
                        prev_precisions=precisions,
                        layer_activations=_layer_acts,
                        is_boundary=is_boundary,
                        hist=hist, tick_base=n_iterations, iter_idx=i,
                        iters_per_timestep=iters_per_timestep,
                    )

                    # Re-apply dropout mask BEFORE logging.
                    values = tuple(
                        jnp.where(clamped_t[j] > 0.5, values[j],
                                  values[j] * _dropout_masks[j])
                        for j in range(n_layers)
                    )

                    should_log = ((n_iterations + i) + 1) % log_every == 0

                    if _has_pm_error_reading:
                        errors, precisions = _recompute_errors_precisions(
                            values, predict_conns,
                            tuple(pw), tuple(pb), tuple(ppw), tuple(ppb),
                            activation_fns,
                            predict_error_activations=predict_error_activations,
                            prev_errors=errors,
                            predict_precision_activations=predict_precision_activations,
                            prev_precisions=precisions,
                            project_weights=prw, modulate_weights=mw,
                            project_biases=prb, modulate_biases=mb,
                            project_conns_precision=project_conns_precision,
                            modulate_conns_precision=modulate_conns_precision,
                            project_conns_internal=project_conns_internal,
                            modulate_conns_internal=modulate_conns_internal,
                            is_stochastic=is_stochastic, key=learn_key,
                            hist=hist, tick_base=n_iterations, iter_idx=i,
                            iters_per_timestep=iters_per_timestep)
                        energies_list, values_log, errors_log, precisions_log, deltas_log = _state_log(
                            n_iterations + i, energies_list, values_log, errors_log, precisions_log, deltas_log,
                            values, errors, precisions
                        )
                    else:
                        log_idx = (n_iterations + i) // log_every + log_initial_offset
                        _prev_errors_for_log = errors
                        _prev_precisions_for_log = precisions
                        def _do_recompute_and_log(_):
                            ne, np_ = _recompute_errors_precisions(
                                values, predict_conns,
                                tuple(pw), tuple(pb), tuple(ppw), tuple(ppb),
                                activation_fns,
                                predict_error_activations=predict_error_activations,
                                prev_errors=_prev_errors_for_log,
                                predict_precision_activations=predict_precision_activations,
                                prev_precisions=_prev_precisions_for_log,
                                project_weights=prw, modulate_weights=mw,
                                project_biases=prb, modulate_biases=mb,
                                project_conns_precision=project_conns_precision,
                                modulate_conns_precision=modulate_conns_precision,
                                project_conns_internal=project_conns_internal,
                                modulate_conns_internal=modulate_conns_internal,
                                is_stochastic=is_stochastic, key=learn_key,
                                hist=hist, tick_base=n_iterations, iter_idx=i,
                                iters_per_timestep=iters_per_timestep)
                            energy = _compute_energy(ne, np_, predict_pre_scales)
                            nd_ = _compute_deltas(ne, np_)
                            new_energies = energies_list.at[log_idx].set(energy)
                            new_vl = tuple(vl.at[log_idx].set(v) for vl, v in zip(values_log, values))
                            new_el = tuple(el.at[log_idx].set(e) for el, e in zip(errors_log, ne))
                            new_pl = tuple(pl.at[log_idx].set(p) for pl, p in zip(precisions_log, np_))
                            new_dl = tuple(dl.at[log_idx].set(d) for dl, d in zip(deltas_log, nd_))
                            return ne, np_, new_energies, new_vl, new_el, new_pl, new_dl

                        def _skip_all(_):
                            return pre_errors, pre_precisions, energies_list, values_log, errors_log, precisions_log, deltas_log

                        errors, precisions, energies_list, values_log, errors_log, precisions_log, deltas_log = lax.cond(
                            should_log, _do_recompute_and_log, _skip_all, None)
                    # Bottom write: store the carry-out errors/precisions.
                    hist = _write_hist_ep(hist, errors, precisions, n_iterations + i)
                    return values, energies_list, values_log, errors_log, precisions_log, deltas_log, pw, pb, prw, prb, mw, mb, ppw, ppb, params_opt_st, vos, hist

                (values, energies, values_log, errors_log, precisions_log, deltas_log,
                 predict_weights, predict_biases,
                 project_weights, project_biases,
                 modulate_weights, modulate_biases,
                 precision_weights, precision_biases, new_params_opt_state, _values_opt_state, hist) = lax.fori_loop(
                    0, n_learning_iterations, learning_body,
                    (values, energies, values_log, errors_log, precisions_log, deltas_log,
                     predict_weights, predict_biases,
                     project_weights, project_biases,
                     modulate_weights, modulate_biases,
                     precision_weights, precision_biases, new_params_opt_state, _values_opt_state, hist)
                )

                # Final step: update precision with the converged values/weights.
                # This gives precision one gradient step per sample, at the
                # equilibrium point, rather than N steps during the transient.
                _has_learnable_prec = any(
                    c.learn_precision_weights or c.learn_precision_bias
                    for c in predict_conns)
                if _has_learnable_prec:
                    clamped_last = tuple(c[:, -1, :] for c in clamped)
                    prec_key = jax.random.fold_in(
                        key, n_iterations + n_learning_iterations)
                    # Reconstruct the one-step carries from the rings the
                    # learning loop left behind (tick_base past its last write).
                    errors = _reconstruct_ep(
                        hist, err_buf_idx, n_iterations + n_learning_iterations, 0)
                    precisions = _reconstruct_ep(
                        hist, prec_buf_idx, n_iterations + n_learning_iterations, 0)
                    (values, errors, precisions,
                     predict_weights, predict_biases,
                     project_weights, project_biases,
                     modulate_weights, modulate_biases,
                     precision_weights, precision_biases,
                     _values_opt_state, new_params_opt_state) = _combined_step(
                        values, errors, clamped_last,
                        predict_weights, predict_biases,
                        project_weights, project_biases,
                        modulate_weights, modulate_biases,
                        precision_weights, precision_biases,
                        predict_conns, project_conns, modulate_conns,
                        project_conns_internal, project_conns_value,
                        modulate_conns_internal, modulate_conns_value,
                        activation_fns, _values_optimizer, _values_opt_state,
                        _params_optimizer, new_params_opt_state,
                        gd_loss_project, gd_loss_modulate,
                        reward_fns, loss_fns, loss_fn_sample_arrays,
                        spatial_layers, spatial_neighborhoods,
                        inference_regs, train_regs, prec_key, labels,
                        is_boundary=_not_a_boundary,
                        update_precision=True,
                        is_stochastic=is_stochastic,
                        project_conns_precision=project_conns_precision,
                        modulate_conns_precision=modulate_conns_precision,
                        modulate_conns_flow_pre=modulate_conns_flow_pre,
                        modulate_conns_flow_post=modulate_conns_flow_post,
                        predict_has_flow_gates=predict_has_flow_gates,
                        structural_attention_groups=structural_attention_groups,
                        pre_scales=predict_pre_scales,
                        predict_weight_masks=predict_weight_masks,
                        project_weight_masks=project_weight_masks,
                        modulate_weight_masks=modulate_weight_masks,
                        predict_error_activations=predict_error_activations,
                        predict_precision_activations=predict_precision_activations,
                        prev_precisions=precisions,
                        layer_activations=_layer_acts,
                        # Final precision step after the sequence: read delayed
                        # from the rings left by the learning loop (no new write).
                        hist=hist, tick_base=n_iterations + n_learning_iterations,
                        iter_idx=0, iters_per_timestep=iters_per_timestep,
                    )

    new_params = NetworkParams(
        predict_weights=list(predict_weights),
        predict_biases=list(predict_biases),
        project_weights=list(project_weights),
        project_biases=list(project_biases),
        modulate_weights=list(modulate_weights),
        modulate_biases=list(modulate_biases),
        precision_weights=list(precision_weights),
        precision_biases=list(precision_biases),
    )

    return new_params, new_params_opt_state, _values_opt_state, values_log, errors_log, precisions_log, deltas_log, energies
