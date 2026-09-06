"""
BPTTSimulation: backprop-through-time baseline over the PC inference dynamics.

Training runs the SAME per-iteration value dynamics as ``Simulation`` (values
relax on dE/dv each iteration), but weights are learned by differentiating a
trajectory loss through the whole unroll — one optax update per batch —
instead of the local PC weight update. Evaluation (``test``) is inherited from
``Simulation`` unchanged, so test-time inference is identical to the PC
backend by construction.
"""

import inspect
from typing import Any, Dict, Optional, Union, Iterator

import jax
import jax.numpy as jnp
import numpy as np
import optax
import time
from tqdm import tqdm

from .core.network import PCNetwork
from .core.layer import Layer
from .simulation import Simulation
from .config import _DEFAULT
from .backend.bptt_simulation import run_bptt_batch


class BPTTSimulation(Simulation):
    """
    BPTT baseline: PC inference dynamics + backprop-through-time weight learning.

    Args:
        net: A built PCNetwork.
        loss_mode: 'energy_final' (default), 'energy_sum', 'objective', or
            'objective_sum'. See ``run_bptt_batch``.
        objective_fn: Task loss for the objective modes. Signature
            ``(values, sample) -> scalar`` or ``(values, sample, t) -> scalar``
            (``t`` is the traced timestep index; 2-arg functions are wrapped).
            For objective modes the supervised target must NOT be in the
            training data_map — training-time dynamics are test-time dynamics.
        truncation: None (full BPTT), K>=1 (TBPTT window of K iterations), or
            0 (exact PC gradient at the final state — energy modes only).
        update_precision: If False, precision params get zero gradient.
        remat: Checkpoint the scan body (memory O(step) at ~1 extra forward).

    Example:
        sim = BPTTSimulation(net, loss_mode='objective', objective_fn=ce_loss)
        sim.train(loader, data_map={l_in: 'image'}, epochs=10,
                  iterations_per_sample=16,
                  params_optimizer=optax.adam(1e-3))
        results = sim.test(loader, data_map={l_in: 'image'},
                           record_map={'acc': ((l_out.value, 'label'), acc_fn)})
    """

    def __init__(
        self,
        net: PCNetwork,
        loss_mode: str = 'energy_final',
        objective_fn=None,
        truncation: Optional[int] = None,
        update_precision: bool = True,
        remat: bool = True,
    ):
        super().__init__(net)
        self.loss_mode = loss_mode
        self.truncation = truncation
        self.update_precision = update_precision
        self.remat = remat
        # Backend calls objective_fn(values, sample, t); wrap 2-arg user fns
        # ONCE here so the jitted backend sees a stable callable across batches.
        if objective_fn is not None and len(
                inspect.signature(objective_fn).parameters) == 2:
            _user_fn = objective_fn
            objective_fn = lambda values, sample, t: _user_fn(values, sample)
        self.objective_fn = objective_fn

    def train(
        self,
        dataloader: Iterator[Dict[str, Any]],
        data_map: Dict[Union[Layer, int], str],
        epochs: int = 1,
        iterations_per_sample=_DEFAULT,
        values_optimizer: Optional[optax.GradientTransformation] = None,
        params_optimizer: Optional[optax.GradientTransformation] = None,
        reset_values_opt_state: bool = False,
        record_map: Optional[Dict[str, tuple]] = None,
        verbose: bool = False,
        is_stochastic: bool = False,
        feedforward_init: bool = True,
    ) -> 'BPTTSimulation':
        """
        Train with BPTT: unroll ``iterations_per_sample`` inference iterations,
        differentiate the trajectory loss, one weight update per batch.

        Args mirror ``Simulation.train`` where they apply; there is no
        ``learning_iterations_per_sample`` (no phase 2) and no
        ``convergence_threshold`` (fixed-length unroll).

        Returns:
            self for method chaining.
        """
        td = self._train_defaults
        iterations_per_sample = (iterations_per_sample
                                 if iterations_per_sample is not _DEFAULT
                                 else td['iterations_per_sample'])
        data_map_tuple = self._convert_map(data_map)

        if params_optimizer is None:
            params_optimizer = optax.adam(1e-4)
        # Opt state is initialised inside run_bptt_batch on first call; reset
        # whenever the optimizer object changes between train() calls.
        if (not hasattr(self, '_bptt_params_optimizer')
                or self._bptt_params_optimizer is not params_optimizer):
            self._bptt_params_optimizer = params_optimizer
            self._bptt_params_opt_state = None

        if values_optimizer is None:
            values_optimizer = optax.sgd(1.0)
        if (not hasattr(self, '_values_optimizer')
                or self._values_optimizer is not values_optimizer):
            self._values_optimizer = values_optimizer
            self._values_opt_state = None

        record_entries = []
        if record_map is not None:
            record_entries = self._convert_record_map(record_map)

        all_losses = []
        all_energies = []
        self.train_records = {name: [] for name, _, _ in record_entries}

        needs_rng = self._needs_rng(is_stochastic)

        for epoch in range(epochs):
            epoch_start = time.time()
            epoch_loss_start = len(all_losses)
            n_samples = 0
            epoch_records = {name: [] for name, _, _ in record_entries}

            for sample in tqdm(dataloader, disable=not verbose, leave=False):
                # Always copy: run_bptt_batch donates its inputs.
                raw_sample = sample
                sample = {k: jnp.array(v) for k, v in sample.items()}
                sample, batch_data_map = self._apply_sensory_transforms(
                    sample, data_map_tuple)

                if needs_rng:
                    self._rng_key, batch_key = jax.random.split(self._rng_key)
                else:
                    batch_key = None
                values_opt_to_pass = (None if reset_values_opt_state
                                      else self._values_opt_state)

                (self.params, self._bptt_params_opt_state, new_values_opt_state,
                 values_log, errors_log, precisions_log, deltas_log,
                 energies, loss) = run_bptt_batch(
                    sample, self.params,
                    self.net.structure,
                    batch_data_map, iterations_per_sample,
                    loss_mode=self.loss_mode,
                    objective_fn=self.objective_fn,
                    truncation=self.truncation,
                    key=batch_key,
                    values_optimizer=self._values_optimizer,
                    values_opt_state=values_opt_to_pass,
                    params_optimizer=self._bptt_params_optimizer,
                    params_opt_state=self._bptt_params_opt_state,
                    is_stochastic=is_stochastic,
                    update_precision=self.update_precision,
                    remat=self.remat,
                    spatial_neighborhoods=getattr(self.net, 'spatial_neighborhoods', ()),
                    predict_weight_masks=self.net.donatable_weight_masks('predict'),
                    predict_error_activations=getattr(self.net, 'predict_error_activations', ()),
                    predict_precision_activations=getattr(self.net, 'predict_precision_activations', ()),
                    layer_activations=getattr(self.net, 'layer_activations', ()),
                    feedforward_init=feedforward_init,
                )
                if not reset_values_opt_state:
                    self._values_opt_state = new_values_opt_state
                all_losses.append(loss)
                all_energies.append(energies)
                n_samples += 1

                if record_entries:
                    node_arrays = [tuple(v[-1] for v in values_log),
                                   tuple(e[-1] for e in errors_log),
                                   tuple(p[-1] for p in precisions_log)]
                    for name, resolved_key, record_fn in record_entries:
                        args = self._resolve_record_args(
                            resolved_key, node_arrays, raw_sample)
                        result = record_fn(*args)
                        epoch_records[name].append(result)
                        self.train_records[name].append(result)

            train_time = time.time() - epoch_start
            if verbose:
                epoch_losses = all_losses[epoch_loss_start:]
                avg_loss = (float(np.mean([np.asarray(l) for l in epoch_losses]))
                            if n_samples > 0 else 0)
                parts = [f"Epoch {epoch + 1}/{epochs}",
                         f"Avg Loss {avg_loss:.4f}",
                         f"Time {train_time:.1f}s"]
                for name, vals in epoch_records.items():
                    parts.append(f"{name}: {np.mean(vals):.4f}")
                print(" | ".join(parts))

        self.net.params = self.params
        self.train_losses = [float(np.asarray(l)) for l in all_losses]
        self.train_energies = all_energies

        return self
