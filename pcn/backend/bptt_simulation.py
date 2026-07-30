"""
BPTT baseline backend: backprop-through-time over the PC inference dynamics.

The forward trajectory is IDENTICAL to ``run_batch``'s inference phase — values
are updated once per iteration by the values optimizer on dE/dv via the
unchanged ``_inference_step`` (locality stop_gradients and all). What changes
is weight learning: instead of the local per-iteration PC update, the whole
T-iteration unroll is differentiated end-to-end (``lax.scan`` + optional remat)
and the trainable parameters get ONE optax update per batch from the gradient
of a trajectory loss.

Loss modes:
    'energy_final'  E(v_T; W) — free energy recomputed at the final values.
    'energy_sum'    mean of the T per-step energies plus E(v_T; W); the
                    trajectory-average free energy.
    'objective'     objective_fn(values_T, sample, t_last) — task loss at the
                    final iterate. The supervised target must NOT be clamped.
    'objective_sum' objective_fn evaluated at the last iteration of every
                    timestep, averaged over timesteps (temporal tasks).

``objective_fn`` has signature ``(values: tuple, sample: dict, t) -> scalar``
where ``t`` is the (traced) timestep index.

Truncation (TBPTT): ``truncation=K >= 1`` stop-gradients the scan carry every K
iterations, so gradient flows through windows of at most K value updates.
``truncation=0`` stop-gradients the carry every step AND the final state before
the loss, which for the energy modes reduces exactly to the PC learning rule
∇_W E at fixed values. ``truncation=None`` (default) is full BPTT.

Energies are reported as batch-MEAN (matching ``run_batch``'s logged energies
and ``_compute_energy``), and the energy losses use the same scale. The PC
backend's internal weight gradient derives from the batch-SUM energy, so at
``truncation=0`` the BPTT update direction matches PC exactly but the
effective learning rate differs by a factor of batch_size.

Deviations from ``run_batch`` (documented, not accidental):
- The error/precision one-step hist buffers are bottom-written every iteration
  with the energy pass's pre-update errors/precisions (run_batch's non-logging
  path); run_batch writes a fresh recompute on logging iterations. Only nets
  with memory error/precision activations or error/precision
  ``precision_input`` sources can observe the difference.
- The final energy recompute omits inference regularizers (they do appear in
  the per-step energies).
- Logs hold only the final state (leading axis 1); ``energies`` has T+1
  entries: the T per-step (pre-update) energies then the final recompute.

Not supported: convergence_threshold (fixed T only), Project/Modulate weight
learning (their weights are frozen; state-operator dynamics still run),
Poisson layers (non-differentiable sampling).
"""

from typing import Dict, Tuple, Any
import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from jax import lax

from ..core.structure import NetworkStructure
from ..core.state import NetworkParams
from ..core.activations import ACTIVATIONS, _nwta
from .simulation import (
    _inference_step,
    _single_pass,
    _recompute_errors_precisions,
    _compute_energy,
    _compute_deltas,
    _apply_mask,
    _read_delayed,
)

_LOSS_MODES = ('energy_final', 'energy_sum', 'objective', 'objective_sum')


@eqx.filter_jit(donate='all')
def run_bptt_batch(
    sample: Dict[str, jnp.ndarray],
    params: NetworkParams,
    structure: NetworkStructure,
    data_map: tuple,
    n_iterations: int,
    loss_mode: str = 'energy_final',
    objective_fn=None,
    truncation=None,
    key: jnp.ndarray = None,
    values_optimizer=None,
    values_opt_state=None,
    params_optimizer=None,
    params_opt_state=None,
    is_stochastic: bool = False,
    update_precision: bool = True,
    remat: bool = True,
    spatial_neighborhoods: tuple = (),
    predict_weight_masks: tuple = (),
    predict_error_activations: tuple = (),
    predict_precision_activations: tuple = (),
    layer_activations: tuple = (),
    feedforward_init: bool = True,
) -> Tuple[NetworkParams, Any, Any, tuple, tuple, tuple, tuple, jnp.ndarray, jnp.ndarray]:
    """
    Run one BPTT batch: init -> clamp -> unrolled inference -> one weight update.

    Trainable (receives the BPTT gradient): predict weights/biases and — when
    ``update_precision`` — precision weights/biases. Project/Modulate weights
    and biases pass through unchanged.

    Returns:
        new_params, new_params_opt_state, new_values_opt_state,
        values_log, errors_log, precisions_log, deltas_log, energies, loss

    Logs have a leading axis of 1 (final state only). ``energies`` has
    ``n_iterations + 1`` entries (per-step pre-update energies, then the final
    recompute at v_T). ``loss`` is the scalar the weight update descended.
    """
    if loss_mode not in _LOSS_MODES:
        raise ValueError(f"loss_mode must be one of {_LOSS_MODES}, got {loss_mode!r}")
    if loss_mode in ('objective', 'objective_sum') and objective_fn is None:
        raise ValueError(f"loss_mode={loss_mode!r} requires objective_fn")
    if truncation == 0 and loss_mode in ('objective', 'objective_sum'):
        raise ValueError(
            "truncation=0 stop-gradients the final state, which kills all "
            "gradient flow into an objective loss; use an energy mode or K>=1")
    if n_iterations < 1:
        raise ValueError("run_bptt_batch requires n_iterations >= 1")

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
    project_conns_internal = structure.project_conns_internal
    project_conns_value = structure.project_conns_value
    modulate_conns_internal = structure.modulate_conns_internal
    modulate_conns_value = structure.modulate_conns_value
    spatial_layers = structure.spatial_layers
    inference_regs = structure.inference_regs
    project_conns_precision = structure.project_conns_precision
    modulate_conns_precision = structure.modulate_conns_precision
    modulate_conns_flow_pre = structure.modulate_conns_flow_pre
    modulate_conns_flow_post = structure.modulate_conns_flow_post
    predict_has_flow_gates = structure.predict_has_flow_gates
    structural_attention_groups = structure.structural_attention_groups

    activation_types = tuple(layer.activation_type for layer in structure.layers)
    activation_temps = tuple(
        float(getattr(layer, 'activation_temperature', 1.0)) for layer in structure.layers)
    activation_winners = tuple(
        int(getattr(layer, 'activation_num_winners', 0)) for layer in structure.layers)

    def _build_activation_fn(t, T, nw):
        if nw > 0:
            return lambda x, _nw=nw: _nwta(x, _nw)
        if T == 1.0:
            return ACTIVATIONS[t]
        return lambda x, _t=t, _T=T: ACTIVATIONS[_t](x / _T)
    activation_fns = tuple(
        _build_activation_fn(t, T, nw)
        for t, T, nw in zip(activation_types, activation_temps, activation_winners))
    is_poisson_types = tuple(layer.is_poisson for layer in structure.layers)
    _layer_acts = layer_activations if any(
        getattr(a, 'needs_key', False) for a in layer_activations) else ()
    dropout_probs = tuple(
        float(getattr(layer, 'dropout_prob', 0.0)) for layer in structure.layers)

    if key is None:
        key = jax.random.PRNGKey(0)

    first_key = list(sample.keys())[0]
    batch_size = sample[first_key].shape[0]

    # Per-layer dropout masks, fixed for the whole unroll (this backend always
    # trains, so dropout is active whenever a layer requests it).
    if any(p > 0.0 for p in dropout_probs):
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

    n_layers = len(layer_dims)
    values_init_list = [jnp.zeros((batch_size, dim)) for dim in layer_dims]
    clamped_list = [jnp.zeros((batch_size, dim), dtype=jnp.float32) for dim in layer_dims]

    labels = None
    for entry in data_map:
        if entry[0] == -1:
            labels = sample[entry[1]]
            break

    # Temporal dimension from clamped data
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
    if n_timesteps > 1:
        if n_timesteps > n_iterations:
            raise ValueError(
                f"n_timesteps ({n_timesteps}) exceeds n_iterations ({n_iterations})")
        if n_iterations % n_timesteps != 0:
            raise ValueError(
                f"n_iterations ({n_iterations}) is not an integer multiple of "
                f"n_timesteps ({n_timesteps})")
    iters_per_timestep = max(n_iterations, 1) // n_timesteps

    _has_ts_gated_value_pm = any(
        getattr(s, 'advance_timestep', False)
        for _, s in tuple(project_conns_value) + tuple(modulate_conns_value))

    def _timestep_boundary(iter_idx):
        if not _has_ts_gated_value_pm:
            return None
        return (iter_idx % iters_per_timestep) == 0

    # ---- Delay + one-step-carry history buffers (mirrors run_batch) ----
    _hist_specs = structure.hist_specs
    _hist_unit_ts = structure.hist_unit_ts
    _hist_node_types = structure.hist_node_types or tuple(0 for _ in _hist_specs)
    _precision_dims = tuple(pb.shape[0] for pb in precision_biases)

    def _hist_buf_dim(node_type, node_id):
        if node_type == 0:
            return layer_dims[node_id]
        if node_type == 1:
            return predict_error_dims[node_id]
        return _precision_dims[node_id]

    hist_init = tuple(
        jnp.zeros((depth + 1, batch_size, _hist_buf_dim(nt, node_id)))
        for (node_id, depth), nt in zip(_hist_specs, _hist_node_types)
    )

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
        if not _hist_specs:
            return hist
        new = list(hist)
        for k, (node_id, depth) in enumerate(_hist_specs):
            if _hist_node_types[k] != 0:
                continue
            S = depth + 1
            if _hist_unit_ts[k]:
                tick = global_iter // iters_per_timestep
                push = ((global_iter % iters_per_timestep) == 0) & (global_iter > 0)
                slot = (tick - 1) % S
                new[k] = jnp.where(
                    push, hist[k].at[slot].set(values[node_id]), hist[k])
            else:
                new[k] = hist[k].at[global_iter % S].set(values[node_id])
        return tuple(new)

    def _reconstruct_ep(hist, buf_idx_map, tick_base, iter_idx):
        if not buf_idx_map:
            return ()
        return tuple(
            _read_delayed(hist, buf_idx_map[i], 1, False,
                          tick_base, iter_idx, iters_per_timestep)
            for i in range(len(buf_idx_map)))

    def _write_hist_ep(hist, err_arrays, prec_arrays, global_iter):
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
            values_init_list[layer_idx] = data[:, 0, :]
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
            values_init_list[layer_idx] = _apply_mask(
                mask_t0, data[:, 0, :], values_init_list[layer_idx])
    temporal_clamp_values = tuple(temporal_clamp_list)
    clamped = tuple(
        c if c.ndim == 3 else jnp.broadcast_to(c[:, None, :], (batch_size, n_timesteps, c.shape[-1]))
        for c in clamped_list
    )
    clamped_t0 = tuple(c[:, 0, :] for c in clamped)
    values_init = tuple(values_init_list)
    errors_init = tuple(jnp.zeros((batch_size, dim)) for dim in predict_error_dims)

    # Values optimizer: init on a same-shaped template (init only reads
    # shape/dtype, so this matches run_batch's post-single-pass init).
    _values_optimizer = values_optimizer if values_optimizer is not None else optax.sgd(1.0)
    if values_opt_state is None:
        values_opt_state = _values_optimizer.init(
            tuple(jnp.zeros((batch_size, dim)) for dim in layer_dims))

    trainable = {
        'predict_weights': predict_weights,
        'predict_biases': predict_biases,
        'precision_weights': precision_weights,
        'precision_biases': precision_biases,
    }

    _needs_prec_carry_init = any(
        2 in c.precision_input_node_types for c in predict_conns
        if c.precision_input_node_types)

    def unroll(trainable_params):
        pw = trainable_params['predict_weights']
        pb = trainable_params['predict_biases']
        ppw = trainable_params['precision_weights']
        ppb = trainable_params['precision_biases']
        if not update_precision:
            ppw = tuple(lax.stop_gradient(w) for w in ppw)
            ppb = tuple(lax.stop_gradient(b) for b in ppb)

        if _needs_prec_carry_init:
            precisions_carry_init = tuple(
                jnp.broadcast_to(conn.precision_transform(b)[None, :],
                                 (batch_size, b.shape[0]))
                for conn, b in zip(predict_conns, ppb))
        else:
            precisions_carry_init = ()

        # Initial forward pass — inside the differentiated function so the
        # feedforward init is part of the trajectory the gradient sees.
        values, errors, precisions = _single_pass(
            values_init, errors_init, clamped_t0,
            pw, pb,
            project_weights, project_biases,
            modulate_weights, modulate_biases,
            ppw, ppb,
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

        hist = hist_init
        if err_buf_idx or prec_buf_idx:
            _h = list(hist)
            for _i, _bi in enumerate(err_buf_idx):
                _h[_bi] = _h[_bi].at[(-1) % _h[_bi].shape[0]].set(errors[_i])
            for _i, _bi in enumerate(prec_buf_idx):
                _h[_bi] = _h[_bi].at[(-1) % _h[_bi].shape[0]].set(precisions[_i])
            hist = tuple(_h)

        def body(carry, i):
            values, vos, hist = carry
            # TBPTT: sever gradient flow through the carry at window
            # boundaries. jnp.where(pred, stop_gradient(x), x) has zero
            # cotangent into x exactly when pred is True.
            if truncation is not None:
                if truncation == 0:
                    values, vos, hist = jax.tree_util.tree_map(
                        lax.stop_gradient, (values, vos, hist))
                else:
                    cut = (i % truncation == 0) & (i > 0)
                    values, vos, hist = jax.tree_util.tree_map(
                        lambda x: jnp.where(cut, lax.stop_gradient(x), x),
                        (values, vos, hist))

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
            hist = _write_hist(hist, values, i)
            errors = _reconstruct_ep(hist, err_buf_idx, 0, i)
            precisions = _reconstruct_ep(hist, prec_buf_idx, 0, i)

            iter_key = jax.random.fold_in(key, i)
            is_boundary = _timestep_boundary(i)

            new_values, pre_errors, pre_precisions, energy, new_vos = _inference_step(
                values, errors, clamped_t,
                pw, pb,
                project_weights, project_biases,
                modulate_weights, modulate_biases,
                ppw, ppb,
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
            new_values = tuple(
                jnp.where(clamped_t[j] > 0.5, new_values[j],
                          new_values[j] * _dropout_masks[j])
                for j in range(n_layers)
            )
            hist = _write_hist_ep(hist, pre_errors, pre_precisions, i)

            # Per-step energy is _inference_step's batch-SUM; log/loss use the
            # batch-mean scale of _compute_energy.
            energy_mean = energy / batch_size
            if loss_mode == 'objective_sum':
                is_ts_end = ((i + 1) % iters_per_timestep) == 0
                obj = objective_fn(new_values, sample, t)
                ys = (energy_mean, jnp.where(is_ts_end, obj, 0.0))
            else:
                ys = energy_mean
            return (new_values, new_vos, hist), ys

        step = jax.checkpoint(body) if remat else body
        (values, new_vos, hist), ys = lax.scan(
            step, (values, values_opt_state, hist), jnp.arange(n_iterations))

        if loss_mode == 'objective_sum':
            step_energies, step_objs = ys
        else:
            step_energies = ys

        # truncation=0: the final loss must see the trajectory as a constant
        # (exact PC gradient: d E(v*, W)/dW at fixed v*).
        if truncation == 0:
            values = tuple(lax.stop_gradient(v) for v in values)
            hist = jax.tree_util.tree_map(lax.stop_gradient, hist)

        # Final recompute at v_T (equilibrium-step context: tick_base past the
        # last write). Note: prev errors/precisions reconstruct from the last
        # bottom write (pre-update at v_{T-1}); see module docstring.
        prev_e = _reconstruct_ep(hist, err_buf_idx, n_iterations, 0)
        prev_p = _reconstruct_ep(hist, prec_buf_idx, n_iterations, 0)
        final_key = jax.random.fold_in(key, n_iterations)
        final_errors, final_precisions = _recompute_errors_precisions(
            values, predict_conns, pw, pb, ppw, ppb, activation_fns,
            predict_error_activations=predict_error_activations,
            prev_errors=prev_e,
            predict_precision_activations=predict_precision_activations,
            prev_precisions=prev_p,
            project_weights=project_weights, modulate_weights=modulate_weights,
            project_biases=project_biases, modulate_biases=modulate_biases,
            project_conns_precision=project_conns_precision,
            modulate_conns_precision=modulate_conns_precision,
            project_conns_internal=project_conns_internal,
            modulate_conns_internal=modulate_conns_internal,
            is_stochastic=is_stochastic, key=final_key,
            hist=hist, tick_base=n_iterations, iter_idx=0,
            iters_per_timestep=iters_per_timestep)
        final_energy = _compute_energy(
            final_errors, final_precisions, predict_pre_scales)

        if loss_mode == 'energy_final':
            loss = final_energy
        elif loss_mode == 'energy_sum':
            loss = (jnp.sum(step_energies) + final_energy) / (n_iterations + 1)
        elif loss_mode == 'objective':
            loss = objective_fn(values, sample, n_timesteps - 1)
        else:  # objective_sum
            loss = jnp.sum(step_objs) / n_timesteps

        aux = (step_energies, final_energy, values,
               final_errors, final_precisions, new_vos)
        return loss, aux

    (loss, aux), grads = jax.value_and_grad(unroll, has_aux=True)(trainable)
    (step_energies, final_energy, final_values,
     final_errors, final_precisions, new_values_opt_state) = aux

    _params_optimizer = params_optimizer if params_optimizer is not None else optax.adam(1e-4)
    if params_opt_state is None:
        params_opt_state = _params_optimizer.init(trainable)
    updates, new_params_opt_state = _params_optimizer.update(
        grads, params_opt_state, trainable)
    new_trainable = optax.apply_updates(trainable, updates)

    new_pw = list(new_trainable['predict_weights'])
    if predict_weight_masks:
        for i, conn in enumerate(predict_conns):
            if conn.is_masked:
                new_pw[i] = new_pw[i] * predict_weight_masks[i]

    new_params = NetworkParams(
        predict_weights=new_pw,
        predict_biases=list(new_trainable['predict_biases']),
        project_weights=list(project_weights),
        project_biases=list(project_biases),
        modulate_weights=list(modulate_weights),
        modulate_biases=list(modulate_biases),
        precision_weights=list(new_trainable['precision_weights']),
        precision_biases=list(new_trainable['precision_biases']),
    )

    values_log = tuple(v[None] for v in final_values)
    errors_log = tuple(e[None] for e in final_errors)
    precisions_log = tuple(p[None] for p in final_precisions)
    deltas_log = tuple(d[None] for d in _compute_deltas(final_errors, final_precisions))
    energies = jnp.concatenate([step_energies, jnp.atleast_1d(final_energy)])

    return (new_params, new_params_opt_state, new_values_opt_state,
            values_log, errors_log, precisions_log, deltas_log, energies, loss)
