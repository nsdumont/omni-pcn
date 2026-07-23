"""
High-level Simulation class for training and inference.

Note: For maximum performance, use the backend.run_batch function directly
(see examples/mnist_discriminative.py). This class provides a convenient interface.
"""

from typing import Any, Dict, List, Optional, Union, Iterator
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
import optax
import time
from tqdm import tqdm
from .core.network import PCNetwork
from .core.layer import Layer, NodeRef
from .core.state import NetworkState, NetworkParams
from .backend import run_batch
from .config import DEFAULTS, load_config, _DEFAULT, validate_keys

def _is_tuple_of_type(obj, item_type):
    return isinstance(obj, tuple) and all(isinstance(item, item_type) for item in obj)


class Simulation:
    """
    High-level simulation manager for training and inference.

    Wraps a built PCNetwork and provides train() and test() methods that
    call the JIT-compiled backend. Weights are stored on the Simulation
    instance and written back to net.params after training.

    Args:
        net: A built PCNetwork (net.build() must have been called).

    Attributes:
        params: NetworkParams pytree containing all learnable parameters.
        train_energies: List of energy arrays from the last train() call.

    Example:
        sim = Simulation(net)

        sim.train(
            train_loader,
            data_map={l_input: 'image', l_output: 'label'},
            epochs=10, iterations_per_sample=100, verbose=True)

        results = sim.test(
            test_loader,
            data_map={l_input: 'image'},
            record_map={'accuracy': ((l_output.value, 'label'), accuracy_fn)})

        print(results['accuracy'])  # list of per-batch results
    """

    def __init__(
        self, net: PCNetwork        
    ):
        if net.structure is None:
            raise RuntimeError(
                "Network must be built before creating Simulation. "
                "Call net.build() first."
            )

        self.net = net
        self._train_defaults = dict(DEFAULTS["train"])
        self._test_defaults = dict(DEFAULTS["test"])
        # RNG key derived from the network seed (independent of weight-init keys).
        # Advanced each batch so different batches get different samples.
        self._rng_key = net.sim_rng
        # Store params as a single NetworkParams pytree
        self.params = NetworkParams(
            predict_weights=list(self.net.params.predict_weights),
            predict_biases=list(self.net.params.predict_biases),
            project_weights=list(self.net.params.project_weights),
            project_biases=list(self.net.params.project_biases),
            modulate_weights=list(self.net.params.modulate_weights),
            modulate_biases=list(self.net.params.modulate_biases),
            precision_weights=list(self.net.params.precision_weights),
            precision_biases=list(self.net.params.precision_biases),
        )

    def config(self, config_file=None, **kwargs) -> 'Simulation':
        """Set default values for simulation parameters.

        Args:
            config_file: Optional path to a JSON config file. The
                ``"train"`` and ``"test"`` sections are extracted; other
                sections are silently ignored.

        Supported kwargs (see ``default_config.json`` for defaults):
            iterations_per_sample, learning_iterations_per_sample,
            log_every.

        kwargs may be applied to **both** train and test defaults (where
        the key exists in that section's defaults).

        Returns:
            self for method chaining.
        """
        if config_file is not None:
            full = load_config(config_file)
            self._train_defaults.update(full.get("train", {}))
            self._test_defaults.update(full.get("test", {}))
        validate_keys(('train', 'test'),
                      [k for k in kwargs if k not in ('train', 'test')])
        for k, v in kwargs.items():
            if k == 'train':
                validate_keys('train', v)
                self._train_defaults.update(v)
            elif k == 'test':
                validate_keys('test', v)
                self._test_defaults.update(v)
            if k in self._train_defaults:
                self._train_defaults[k] = v
            if k in self._test_defaults:
                self._test_defaults[k] = v
        return self

    def _resolve_key(self, key):
        """Resolve a single key element to its internal representation.

        Returns:
            int               for Layer or int keys (layer index)
            (node_type, idx)  for NodeRef keys (idx is layer idx for value,
                              predict conn idx for error/precision)
            str               for sample dict keys (passed through)
        """
        if isinstance(key, int):
            return key
        elif isinstance(key, Layer):
            return key._idx
        elif isinstance(key, NodeRef):
            return (key.node_type_id, key.owner._idx)
        elif isinstance(key, str):
            return key
        else:
            raise TypeError(f"Unsupported key type: {type(key)}")

    def _needs_rng(self, is_stochastic: bool) -> bool:
        """Whether run_batch's graph consumes PRNG randomness for this net.

        When False, train/test pass key=None and skip the per-batch
        jax.random.split dispatch; run_batch's key-dependent paths are all
        deterministic in that case. Conservative: any stochastic flag,
        key-consuming activation, or regularizer (SIGReg draws random
        projections from the key) counts as needing randomness.
        """
        if is_stochastic:
            return True
        structure = self.net.structure
        if getattr(structure, 'inference_regs', ()) or getattr(structure, 'train_regs', ()):
            return True
        if any(float(getattr(l, 'dropout_prob', 0.0)) > 0.0 for l in structure.layers):
            return True
        # GD loss_fns / three-factor reward_fns receive folded sub-keys;
        # assume they may consume them.
        if getattr(self.net, '_reward_fns', ()) or getattr(self.net, '_loss_fns', ()):
            return True
        return any(
            a is not None and getattr(a, 'needs_key', False)
            for acts in (getattr(self.net, 'layer_activations', ()),
                         getattr(self.net, 'predict_error_activations', ()),
                         getattr(self.net, 'predict_precision_activations', ()))
            for a in acts)

    def _convert_map(self, data_map: Dict[Any, Any]):
        """
        Convert a user-defined map with Layer/NodeRef/int/str keys into a
        tuple of (resolved_key, value) pairs using the network's layer indices.

        Keys are resolved as follows:
            int            -> int (layer index, used as-is)
            Layer          -> int (layer._idx)
            NodeRef        -> (node_type_id, layer._idx)  e.g. (0, 2)
            str            -> str (sample dict key, passed through)
            tuple          -> tuple of resolved elements (can mix NodeRef and str)
        """
        converted = {}
        for key, value in data_map.items():
            if isinstance(key, str) and key == 'class':
                converted[-1] = value
            elif isinstance(key, tuple):
                resolved = tuple(self._resolve_key(elem) for elem in key)
                converted[resolved] = value
            else:
                converted[self._resolve_key(key)] = value

        return tuple(converted.items())

    def _apply_sensory_transforms(self, sample, data_map_tuple):
        """Apply fixed ``SensoryInput`` feature transforms to a batch, *outside*
        inference.

        For each ``data_map`` entry whose layer is a :class:`SensoryInput`,
        compute the transformed features once (``layer.encode``) and retarget the
        clamp to a derived sample key. The raw data stays in ``sample`` under its
        original key (and in the caller's ``raw_sample``) so record functions can
        still read it. ``run_batch`` is unchanged — it just clamps the layer value
        to the derived feature key.

        Temporal data is supported: a 3-D raw array ``(B, T, raw_dim)`` (the
        temporal-clamp convention ``run_batch`` reads) is encoded across all
        timesteps in one vectorized call, giving ``(B, T, feat_dim)``.

        A sensory input may be given either a full-clamp ``'key'`` value or a
        ``(data_key, mask_key)`` **soft-clamp** value. The mask is a clamp-space
        strength (applied to the feature clamp, NOT to the data): a scalar or a
        per-sample / per-timestep array that is broadcast to the feature clamp
        shape, so the feature layer is nudged toward ``encode(data)`` at strength
        ``β`` (``v = β·features + (1−β)·v_inf``). It is not derived from a
        data-shaped mask.

        Returns ``(sample, patched_data_map_tuple)``. When no ``SensoryInput`` is
        present both are returned unchanged.
        """
        from .core.sensory.base import SensoryInput
        layers = self.net._layers
        patched = []
        out_sample = sample
        copied = False
        for key, value in data_map_tuple:
            layer = (layers[key]
                     if isinstance(key, int) and 0 <= key < len(layers) else None)
            if isinstance(layer, SensoryInput):
                if not copied:
                    out_sample = dict(sample)
                    copied = True
                data_key = value if isinstance(value, str) else value[0]
                feat = self._encode_sensory(layer, sample[data_key])
                feat_key = f"__sensory_feat_{key}"
                out_sample[feat_key] = feat
                if isinstance(value, str):
                    patched.append((key, feat_key))
                else:
                    mask_key_new = f"__sensory_mask_{key}"
                    out_sample[mask_key_new] = self._broadcast_clamp_mask(
                        jnp.asarray(sample[value[1]]), feat.shape)
                    patched.append((key, (feat_key, mask_key_new)))
            else:
                patched.append((key, value))
        return out_sample, tuple(patched)

    @staticmethod
    def _encode_sensory(layer, data):
        """Encode a (possibly temporal) raw batch to flattened features.

        Static ``(B, raw_dim)`` -> ``(B, feat_dim)``; temporal ``(B, T, raw_dim)``
        -> ``(B, T, feat_dim)`` via a single vectorized encode over ``B*T``.
        """
        data = jnp.asarray(data)
        if data.ndim == 3:                                    # (B, T, raw_dim)
            b, t = data.shape[0], data.shape[1]
            feat = layer.encode(data.reshape(b * t, data.shape[2]))
            return feat.reshape(b, t, feat.shape[-1])
        return layer.encode(data)                             # (B, raw_dim)

    @staticmethod
    def _broadcast_clamp_mask(mask, feat_shape):
        """Broadcast a clamp-strength mask to the feature clamp shape.

        ``mask`` may be a scalar, a leading-dims array (e.g. ``(B,)`` /
        ``(B, T)``), or already the full feature shape; a trailing feature axis is
        added as needed. No reduction of a data-shaped mask is performed.
        """
        mask = jnp.asarray(mask, dtype=jnp.float32)
        if 0 < mask.ndim < len(feat_shape):
            mask = mask.reshape(mask.shape + (1,) * (len(feat_shape) - mask.ndim))
        return jnp.broadcast_to(mask, feat_shape)

    def _convert_record_map(self, record_map):
        """Convert a user-defined record_map into a list of (name, resolved_key, fn) triples.

        record_map format: {name: (inputs, fn)} where:
            - name: str key for the results dict
            - inputs: a NodeRef, str, or tuple of NodeRefs/strs
            - fn: callable receiving the resolved arrays as positional args

        Example:
            {'accuracy': ((l_output.value, 'label'), batch_accuracy),
             'ce_loss':  ((l_output.value, 'label'), ce_loss_fn)}
        """
        result = []
        for name, (inputs, fn) in record_map.items():
            if not isinstance(name, str):
                raise TypeError(f"record_map keys must be strings, got {type(name)}")
            # Wrap non-tuple inputs so resolution is uniform
            if isinstance(inputs, tuple):
                resolved = tuple(self._resolve_key(elem) for elem in inputs)
            else:
                resolved = self._resolve_key(inputs)
            result.append((name, resolved, fn))
        return result

    @staticmethod
    def _resolve_record_args(resolved_key, node_arrays, sample):
        """Resolve a record_map key into a list of arrays to pass to the function.

        Args:
            resolved_key: A resolved key from _convert_map. Can be:
                (node_type, layer_idx) - single node reference
                ((node_type, layer_idx), 'key', ...) - tuple of nodes/sample keys
            node_arrays: [values_tuple, errors_tuple] indexed by node_type
            sample: The current batch sample dict

        Returns:
            List of arrays to pass as positional args to the record function.
        """
        def _get_array(elem):
            if isinstance(elem, str):
                return sample[elem]
            elif isinstance(elem, tuple):
                # (node_type, layer_idx)
                node_type, layer_idx = elem
                return node_arrays[node_type][layer_idx]
            else:
                raise TypeError(f"Unexpected record key element: {elem}")

        if isinstance(resolved_key, str):
            # Single sample key
            return [sample[resolved_key]]
        elif isinstance(resolved_key, tuple) and len(resolved_key) == 2 and isinstance(resolved_key[0], int):
            # Single node: (node_type, layer_idx)
            node_type, layer_idx = resolved_key
            return [node_arrays[node_type][layer_idx]]
        elif isinstance(resolved_key, tuple):
            # Mixed tuple of nodes and/or sample keys
            return [_get_array(elem) for elem in resolved_key]
        else:
            raise TypeError(f"Unexpected resolved_key format: {resolved_key}")

    def train(
        self,
        dataloader: Iterator[Dict[str, Any]],
        data_map: Dict[Union[Layer, int], str],
        epochs: int = 1,
        iterations_per_sample=_DEFAULT,
        learning_iterations_per_sample=_DEFAULT,
        log_every=_DEFAULT,
        convergence_threshold=_DEFAULT,
        values_optimizer: Optional[optax.GradientTransformation] = None,
        params_optimizer: Optional[optax.GradientTransformation] = None,
        reset_values_opt_state: bool = False,
        record_map: Optional[Dict[str, tuple]] = None,
        verbose: bool = False,
        is_stochastic: bool = False,
        log_initial: bool = False,
        feedforward_init: bool = True,
        save_logs: bool = _DEFAULT,
    ) -> 'Simulation':
        """
        Train the network on data from the dataloader.

        Two-phase training per batch:
        1. Inference phase: run iterations_per_sample iterations with weights
           frozen, letting values converge toward equilibrium.
        2. Learning phase: if learning_iterations_per_sample == 0 (default),
           perform a single weight update at equilibrium. Otherwise, run that
           many additional iterations with simultaneous inference + learning.

        Args:
            dataloader: Iterable yielding dicts of batched numpy/JAX arrays.
            data_map: Maps Layer objects or indices to sample dict keys,
                specifying which layers are clamped to which data.
                E.g. {l_input: 'image', l_output: 'label'}.
            epochs: Number of passes through the dataloader.
            iterations_per_sample: See default_config.json for default.
            learning_iterations_per_sample: See default_config.json for default.
            log_every: See default_config.json for default.
            record_map: Named recording functions. Dict mapping string names
                to (inputs, fn) tuples. ``inputs`` can be a NodeRef, str, or
                tuple mixing both. ``fn`` receives resolved arrays as positional
                args and returns a per-batch result. Results are stored in
                self.train_records keyed by name, and epoch means are printed
                when verbose=True.
                E.g. {'accuracy': ((l_output.value, 'label'), accuracy_fn)}.
            values_optimizer: An optax GradientTransformation for inference-time
                value updates. Defaults to optax.sgd(1.0). State is persisted
                across batches (reset when this argument changes or
                reset_values_opt_state=True).
            params_optimizer: An optax GradientTransformation for weight updates.
                Defaults to optax.adam(1e-4). State is managed internally and
                reset whenever this argument changes between train() calls.

                The trainable param dict passed to the optimizer has 
                top-level keys: ``'predict_weights'``, ``'predict_biases'``,
                ``'precision_weights'``, ``'precision_biases'``,
                ``'gd_loss_project_weights'``, ``'gd_loss_modulate_weights'``.
                Each value is a tuple of arrays (one per related connection).

                Examples::

                    # Adam with gradient clipping
                    params_optimizer = optax.chain(
                        optax.clip_by_global_norm(1.0),
                        optax.adam(1e-3),
                    )

                    # SGD with Nesterov momentum and weight decay
                    params_optimizer = optax.chain(
                        optax.add_decayed_weights(1e-4),
                        optax.sgd(1e-3, momentum=0.9, nesterov=True),
                    )
            reset_values_opt_state: If True, reinitialise the values optimizer
                state each batch (no momentum/statistics carry-over).
            verbose: Print per-epoch energy, timing, and record means.
            is_stochastic: If True, use stochastic_prediction (samples from the
                predictive distribution) in both the initial forward pass and
                the inference loop. Default False.
            log_initial: If True, prepend the state right after the initial
                forward pass to every log tensor (and energies). Adds one extra
                leading slot. Useful with ``log_every=1`` for full trajectory
                plotting. Default False.
            feedforward_init: If True (default), seed inference with an initial
                forward pass that propagates predictions/projections through
                the network to set non-clamped layer values. If False, skip
                that propagation — non-clamped values start at zero and
                inference relaxes from there.

        Returns:
            self for method chaining.
        """
        td = self._train_defaults
        # ``save_logs`` controls whether the full per-iteration
        # values/errors/precisions/deltas trajectories are retained in
        # ``self.logs`` for **every batch across the whole epoch** — that is
        # O(n_batches) device memory and OOMs on large datasets / large conv
        # nets. Historically it was implicitly enabled whenever ``log_every``
        # was passed; that conflated within-batch log frequency with cross-batch
        # retention. It can now be set explicitly (default preserves the old
        # behaviour for callers that read ``sim.logs``).
        if save_logs is _DEFAULT:
            save_logs = log_every is not _DEFAULT
        iterations_per_sample = iterations_per_sample if iterations_per_sample is not _DEFAULT else td['iterations_per_sample']
        learning_iterations_per_sample = learning_iterations_per_sample if learning_iterations_per_sample is not _DEFAULT else td['learning_iterations_per_sample']
        log_every = log_every if log_every is not _DEFAULT else td['log_every']
        convergence_threshold = convergence_threshold if convergence_threshold is not _DEFAULT else td.get('convergence_threshold', 0.0)
        if log_every is None:
            log_every = iterations_per_sample + learning_iterations_per_sample
        data_map_tuple = self._convert_map(data_map)

        # Resolve params optimizer and (re)initialize state if changed
        if params_optimizer is None:
            params_optimizer = optax.adam(1e-4)
        if not hasattr(self, '_params_optimizer') or self._params_optimizer is not params_optimizer:
            self._params_optimizer = params_optimizer
            structure = self.net.structure
            trainable = {
                'predict_weights': tuple(self.params.predict_weights),
                'predict_biases': tuple(self.params.predict_biases),
                'project_biases': tuple(self.params.project_biases),
                'modulate_biases': tuple(self.params.modulate_biases),
                'precision_weights': tuple(self.params.precision_weights),
                'precision_biases': tuple(self.params.precision_biases),
                'gd_loss_project_weights': tuple(
                    self.params.project_weights[idx] for idx, _ in structure.gd_loss_project),
                'gd_loss_modulate_weights': tuple(
                    self.params.modulate_weights[idx] for idx, _ in structure.gd_loss_modulate),
            }
            self._params_opt_state = self._params_optimizer.init(trainable)

        # Resolve values optimizer; state is initialised inside run_batch on first call
        if values_optimizer is None:
            values_optimizer = optax.sgd(1.0)
        if not hasattr(self, '_values_optimizer') or self._values_optimizer is not values_optimizer:
            self._values_optimizer = values_optimizer
            self._values_opt_state = None  # will be initialised inside run_batch

        record_entries = []
        if record_map is not None:
            record_entries = self._convert_record_map(record_map)

        all_energies = []
        self.train_records = {name: [] for name, _, _ in record_entries}
        if save_logs:
            self.logs = {'values': [], 'errors': [], 'precisions': [], 'deltas': []}

        # Only pay a per-batch RNG split when the compiled graph actually
        # consumes randomness; key=None takes run_batch's deterministic path.
        needs_rng = self._needs_rng(is_stochastic)

        for epoch in range(epochs):
            epoch_start = time.time()
            epoch_energy_start = len(all_energies)
            n_samples = 0
            epoch_records = {name: [] for name, _, _ in record_entries}

            for sample in tqdm(dataloader, disable=not verbose, leave=False):
                # Convert to JAX arrays. Always copy: run_batch donates its
                # inputs, so passing user-owned device arrays through uncopied
                # would delete them out from under the caller. Keep the
                # caller's arrays for record-fn resolution — the device copies
                # may be deleted by donation aliasing after run_batch.
                raw_sample = sample
                sample = {k: jnp.array(v) for k, v in sample.items()}
                # Fixed sensory front-ends: transform raw data -> features once,
                # outside inference (no-op when no SensoryInput is used).
                sample, batch_data_map = self._apply_sensory_transforms(
                    sample, data_map_tuple)

                # Run batch through consolidated function
                if needs_rng:
                    self._rng_key, batch_key = jax.random.split(self._rng_key)
                else:
                    batch_key = None
                values_opt_to_pass = None if reset_values_opt_state else self._values_opt_state
                self.params, self._params_opt_state, new_values_opt_state, values_log, errors_log, precisions_log, deltas_log, energies = run_batch(
                    sample, self.params,
                    self.net.structure,
                    batch_data_map, iterations_per_sample,
                    log_every, learning=True,
                    n_learning_iterations=learning_iterations_per_sample,
                    reward_fns=getattr(self.net, '_reward_fns', ()),
                    loss_fns=getattr(self.net, '_loss_fns', ()),
                    convergence_threshold=convergence_threshold,
                    key=batch_key,
                    values_optimizer=self._values_optimizer,
                    values_opt_state=values_opt_to_pass,
                    params_optimizer=self._params_optimizer,
                    params_opt_state=self._params_opt_state,
                    is_stochastic=is_stochastic,
                    spatial_neighborhoods=getattr(self.net, 'spatial_neighborhoods', ()),
                    log_initial=log_initial,
                    predict_weight_masks=getattr(self.net, 'predict_weight_masks', ()),
                    project_weight_masks=getattr(self.net, 'project_weight_masks', ()),
                    modulate_weight_masks=getattr(self.net, 'modulate_weight_masks', ()),
                    predict_error_activations=getattr(self.net, 'predict_error_activations', ()),
                    predict_precision_activations=getattr(self.net, 'predict_precision_activations', ()),
                    layer_activations=getattr(self.net, 'layer_activations', ()),
                    feedforward_init=feedforward_init,
                )
                if not reset_values_opt_state:
                    self._values_opt_state = new_values_opt_state
                all_energies.append(energies)
                n_samples += 1
                if save_logs:
                    self.logs['values'].append(values_log)
                    self.logs['errors'].append(errors_log)
                    self.logs['precisions'].append(precisions_log)
                    self.logs['deltas'].append(deltas_log)

                # Apply record functions using final logged state. The
                # node_arrays slices dispatch ~one device op per conn, so
                # build them only when something will consume them.
                if record_entries:
                    node_arrays = [tuple(v[-1] for v in values_log), tuple(e[-1] for e in errors_log), tuple(p[-1] for p in precisions_log)]
                    for name, resolved_key, record_fn in record_entries:
                        args = self._resolve_record_args(resolved_key, node_arrays, raw_sample)
                        result = record_fn(*args)
                        epoch_records[name].append(result)
                        self.train_records[name].append(result)

            train_time = time.time() - epoch_start
            if verbose:
                # Computed lazily from the collected energies (one host sync
                # per epoch) instead of a per-batch device accumulation.
                epoch_energies = all_energies[epoch_energy_start:]
                avg_energy = (float(np.mean([np.asarray(e[-1]) for e in epoch_energies]))
                              if n_samples > 0 else 0)
                parts = [f"Epoch {epoch + 1}/{epochs}", f"Avg Energy {avg_energy:.3f}", f"Time {train_time:.1f}s"]
                for name, vals in epoch_records.items():
                    parts.append(f"{name}: {np.mean(vals):.4f}")
                print(" | ".join(parts))

        # Store final params back to network
        self.net.params = self.params
        self.train_energies = all_energies

        return self


    def test(
        self,
        dataloader: Iterator[Dict[str, Any]],
        data_map: Dict[Union[Layer, int], str],
        iterations_per_sample=_DEFAULT,
        log_every=_DEFAULT,
        convergence_threshold=_DEFAULT,
        values_optimizer: Optional[optax.GradientTransformation] = None,
        record_map: Optional[Dict[str, tuple]] = None,
        verbose: bool = False,
        is_stochastic: bool = False,
        log_initial: bool = False,
        feedforward_init: bool = True,
        return_logs: bool = False,
    ) -> Dict[str, Any]:
        """
        Run inference and apply recording functions to collect results.

        Args:
            dataloader: Iterable yielding dicts of batched numpy/JAX arrays.
            data_map: Maps Layer objects or indices to sample dict keys for
                clamping. Typically only the input layer is clamped during
                testing. E.g. {l_input: 'image'}.
            iterations_per_sample: See default_config.json for default.
            log_every: See default_config.json for default.
            record_map: Named recording functions. Dict mapping string names
                to (inputs, fn) tuples. ``inputs`` can be a NodeRef, str, or
                tuple mixing both. ``fn`` receives resolved arrays as
                positional args and returns a per-batch result.
                E.g. {'accuracy': ((l_output.value, 'label'), accuracy_fn)}.
                Results are keyed by the string name in the returned dict.
            verbose: Print summary energy and record means after test.
            is_stochastic: If True, use stochastic_prediction (samples from the
                predictive distribution) in both the initial forward pass and
                the inference loop. Default False.
            log_initial: If True, prepend the state right after the initial
                forward pass to every log tensor (and energies). Adds one extra
                leading slot. Useful with ``log_every=1`` for full trajectory
                plotting. Default False.
            feedforward_init: If True (default), seed inference with an initial
                forward pass that propagates predictions/projections through
                the network to set non-clamped layer values. If False, skip
                that propagation — non-clamped values start at zero and
                inference relaxes from there.
            return_logs: If True, accumulate the full per-iteration
                ``values``/``errors``/``precisions``/``deltas`` trajectories for
                **every batch** in the returned dict. This holds
                ``O(n_batches * batch * sum(dims) * n_logged)`` arrays on-device
                and OOMs on large loaders / large conv nets, so it is False by
                default: record functions still run on each batch's final
                state, but the trajectory lists are left empty and freed per
                batch. Set True when you need full relaxation trajectories
                (e.g. settling-dynamics plots on a small loader).

        Returns:
            Dict with 'energies' (list of per-batch mean energies) and one
            entry per record_map name (list of per-batch results). When
            ``return_logs`` is True it also contains the reshaped
            ``values``/``errors``/``precisions``/``deltas`` trajectories.
        """
        td = self._test_defaults
        save_logs = log_every is not _DEFAULT
        iterations_per_sample = iterations_per_sample if iterations_per_sample is not _DEFAULT else td['iterations_per_sample']
        log_every = log_every if log_every is not _DEFAULT else td['log_every']
        convergence_threshold = convergence_threshold if convergence_threshold is not _DEFAULT else td.get('convergence_threshold', 0.0)
        if log_every is None:
            log_every = iterations_per_sample

        data_map_tuple = self._convert_map(data_map)

        # Convert record_map to resolved entries
        record_entries = []  # list of (name, resolved_key, record_fn)
        if record_map is not None:
            record_entries = self._convert_record_map(record_map)

        if values_optimizer is None:
            values_optimizer = optax.sgd(1.0)
        if not hasattr(self, '_values_optimizer') or self._values_optimizer is not values_optimizer:
            self._values_optimizer = values_optimizer
            self._values_opt_state = None

        all_energies = []
        time_start = time.time()
        results = {name: [] for name, _, _ in record_entries}
        n_samples = 0
        results['values'] = []
        results['errors'] = []
        results['precisions'] = []
        results['deltas'] = []
        if save_logs:
            self.logs = {'values': [], 'errors': [], 'precisions': [], 'deltas': []}

        needs_rng = self._needs_rng(is_stochastic)

        for sample in tqdm(dataloader, disable=not verbose, leave=False):
            # Always copy (run_batch donates its inputs; see train()).
            raw_sample = sample
            sample = {k: jnp.array(v) for k, v in sample.items()}
            # Fixed sensory front-ends: transform raw data -> features once,
            # outside inference (no-op when no SensoryInput is used).
            sample, batch_data_map = self._apply_sensory_transforms(
                sample, data_map_tuple)

            if needs_rng:
                self._rng_key, batch_key = jax.random.split(self._rng_key)
            else:
                batch_key = None
            self.params, _, _, values_log, errors_log, precisions_log, deltas_log, energies = run_batch(
                sample, self.params,
                self.net.structure,
                batch_data_map, iterations_per_sample,
                log_every, learning=False,
                n_learning_iterations=0,
                convergence_threshold=convergence_threshold,
                key=batch_key,
                values_optimizer=getattr(self, '_values_optimizer', optax.sgd(1.0)),
                values_opt_state=None,
                is_stochastic=is_stochastic,
                spatial_neighborhoods=getattr(self.net, 'spatial_neighborhoods', ()),
                log_initial=log_initial,
                predict_error_activations=getattr(self.net, 'predict_error_activations', ()),
                predict_precision_activations=getattr(self.net, 'predict_precision_activations', ()),
                layer_activations=getattr(self.net, 'layer_activations', ()),
                feedforward_init=feedforward_init,
            )
            if return_logs:
                results['values'].append(values_log)
                results['errors'].append(errors_log)
                results['precisions'].append(precisions_log)
                results['deltas'].append(deltas_log)
            if save_logs:
                self.logs['values'].append(values_log)
                self.logs['errors'].append(errors_log)
                self.logs['precisions'].append(precisions_log)
                self.logs['deltas'].append(deltas_log)

            # Apply record functions using final logged state
            # Resolved keys can be:
            #   (node_type, layer_idx)  - single node
            #   ((node_type, layer_idx), 'key', ...)  - tuple of nodes and/or sample keys
            if record_entries:
                node_arrays = [tuple(v[-1] for v in values_log), tuple(e[-1] for e in errors_log), tuple(p[-1] for p in precisions_log)]
                for name, resolved_key, record_fn in record_entries:
                    args = self._resolve_record_args(resolved_key, node_arrays, raw_sample)
                    results[name].append(record_fn(*args))

            all_energies.append(energies)
            n_samples += 1

        test_time = time.time() - time_start
        if verbose:
            avg_energy = (float(np.mean([np.asarray(e[-1]) for e in all_energies]))
                          if n_samples > 0 else 0)
            parts = [f"Avg Energy {avg_energy:.3f}", f"Time {test_time:.1f}s"]
            for name, vals in results.items():
                parts.append(f"{name}: {jnp.mean(jnp.array(vals)):.4f}")
            print(" | ".join(parts))

        results['energies'] = all_energies
        self.test_energies = all_energies

        if not return_logs:
            # Trajectory logs were not accumulated (memory-safe path). Leave the
            # value/error/precision/delta lists empty; record_map metrics and
            # energies are still returned.
            return results

        n_vs = len(results['values'][0])
        n_es = len(results['errors'][0])
        n_iterations = results['values'][0][0].shape[0]
        dims = [v.shape[-1] for v in results['values'][0]]
        results['values'] = [jnp.stack([v[i] for v in results['values']]).transpose(0, 2, 1, 3).reshape(-1, n_iterations, dims[i]) for i in range(n_vs) ]
        dims = [v.shape[-1] for v in results['errors'][0]]
        results['errors'] = [jnp.stack([v[i] for v in results['errors']]).transpose(0, 2, 1, 3).reshape(-1, n_iterations, dims[i]) for i in range(n_es) ]
        n_ps = len(results['precisions'][0])
        dims = [v.shape[-1] for v in results['precisions'][0]]
        results['precisions'] = [jnp.stack([v[i] for v in results['precisions']]).transpose(0, 2, 1, 3).reshape(-1, n_iterations, dims[i]) for i in range(n_ps) ]
        n_ds = len(results['deltas'][0])
        dims = [v.shape[-1] for v in results['deltas'][0]]
        results['deltas'] = [jnp.stack([v[i] for v in results['deltas']]).transpose(0, 2, 1, 3).reshape(-1, n_iterations, dims[i]) for i in range(n_ds) ]

        return results

    def save(self, path: Union[str, Path]) -> Path:
        """Save network parameters and optimizer states to a checkpoint directory.

        Creates ``path/network.h5`` (via net.save) and ``path/optimizer.pkl``
        (params + values optimizer states serialized as numpy arrays).

        Args:
            path: Checkpoint directory. Created if it doesn't exist.

        Returns:
            The resolved Path of the checkpoint directory.
        """
        import pickle
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Sync simulation params to network, then save via net.save
        self.net.params = self.params
        self.net.save(path / "network.h5")

        # Serialize optimizer states: convert JAX arrays → numpy for pickling
        def _to_numpy(x):
            return np.array(x) if hasattr(x, 'shape') else x

        states = {
            'params_opt_state': jax.tree_util.tree_map(
                _to_numpy, getattr(self, '_params_opt_state', None)
            ),
            'values_opt_state': jax.tree_util.tree_map(
                _to_numpy, getattr(self, '_values_opt_state', None)
            ),
        }
        with open(path / "optimizer.pkl", 'wb') as f:
            pickle.dump(states, f)

        return path

    def load(
        self,
        path: Union[str, Path],
        params_optimizer: Optional[optax.GradientTransformation] = None,
        values_optimizer: Optional[optax.GradientTransformation] = None,
    ) -> 'Simulation':
        """Load network parameters and optimizer states from a checkpoint directory.

        Call this after ``Simulation(net)`` and before ``train()``.  Pass the
        same optimizer objects you will pass to ``train()`` so that the
        reference-equality check in ``train()`` does not reinitialise the
        restored optimizer state.

        Args:
            path: Checkpoint directory created by :meth:`save`.
            params_optimizer: The params optimizer that will be used in the
                next ``train()`` call. If given, the saved params opt state is
                restored and ``train()`` will use it directly.
            values_optimizer: The values optimizer that will be used in the
                next ``train()`` call. If given, the saved values opt state is
                restored.

        Returns:
            self for method chaining.
        """
        import pickle
        path = Path(path)

        # Load network params and sync to self.params
        self.net.load(path / "network.h5")
        self.params = NetworkParams(
            predict_weights=list(self.net.params.predict_weights),
            predict_biases=list(self.net.params.predict_biases),
            project_weights=list(self.net.params.project_weights),
            project_biases=list(self.net.params.project_biases),
            modulate_weights=list(self.net.params.modulate_weights),
            modulate_biases=list(self.net.params.modulate_biases),
            precision_weights=list(self.net.params.precision_weights),
            precision_biases=list(self.net.params.precision_biases),
        )

        # Store optimizer references so train()'s reference-equality check
        # sees the same object and skips reinitialisation
        if params_optimizer is not None:
            self._params_optimizer = params_optimizer
        if values_optimizer is not None:
            self._values_optimizer = values_optimizer

        # Restore optimizer states from pickle
        opt_path = path / "optimizer.pkl"
        if opt_path.exists():
            with open(opt_path, 'rb') as f:
                states = pickle.load(f)
            if params_optimizer is not None and states.get('params_opt_state') is not None:
                self._params_opt_state = jax.tree_util.tree_map(
                    jnp.array, states['params_opt_state']
                )
            if values_optimizer is not None and states.get('values_opt_state') is not None:
                self._values_opt_state = jax.tree_util.tree_map(
                    jnp.array, states['values_opt_state']
                )

        return self
