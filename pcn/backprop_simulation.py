"""
BackpropSimulation: standard backprop baseline using the same network topology.

Forward pass through predict connections + optax.adam weight updates,
with a user-provided objective function. No PCN dynamics.
"""

from typing import Any, Dict, Optional, Union, Iterator
import jax.numpy as jnp
import optax
import time
from tqdm import tqdm
from .core.network import PCNetwork
from .core.layer import Layer, NodeRef
from .backend.backprop_simulation import run_backprop_batch


class BackpropSimulation:
    """
    Backprop baseline using the same network graph as PCN.

    Runs a standard forward pass through predict connections (ignoring
    errors and precisions), computes a user-provided loss, and updates
    weights via optax.adam.

    Args:
        net: A built PCNetwork (net.build() must have been called).
        objective_fn: Loss function with signature
            (values: tuple, sample: dict) -> scalar.
            ``values`` is a tuple of layer value arrays indexed by layer
            index. ``sample`` is the current batch dict.
        learning_rate: Adam learning rate.

    Example:
        def loss_fn(values, sample):
            logits = values[output_idx]
            targets = sample['label']
            return -jnp.mean(jnp.sum(targets * jnp.log(logits + 1e-8), axis=-1))

        sim = BackpropSimulation(net, loss_fn, learning_rate=1e-3)
        sim.train(train_loader, data_map={l_input: 'image'}, epochs=10)
        results = sim.test(test_loader, data_map={l_input: 'image'},
                           record_map={(l_output.value, 'label'): accuracy_fn})
    """

    def __init__(
        self,
        net: PCNetwork,
        objective_fn,
        learning_rate: float = 1e-3,
    ):
        if net.structure is None:
            raise RuntimeError(
                "Network must be built before creating BackpropSimulation. "
                "Call net.build() first."
            )
        self.net = net
        self.objective_fn = objective_fn
        self.predict_weights = tuple(net.params.predict_weights)
        from .core.sparse import SparseWeight
        if any(isinstance(w, SparseWeight) for w in self.predict_weights):
            raise NotImplementedError(
                "BackpropSimulation does not support sparse (sparse=True) "
                "predict weights; use Simulation or BPTTSimulation.")
        self.optimizer = optax.adam(learning_rate)
        self.opt_state = self.optimizer.init(self.predict_weights)

    # ------------------------------------------------------------------
    # Key / map resolution helpers (mirrors Simulation)
    # ------------------------------------------------------------------

    def _resolve_key(self, key):
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

    def _convert_map(self, data_map):
        converted = {}
        for key, value in data_map.items():
            if isinstance(key, tuple):
                resolved = tuple(self._resolve_key(elem) for elem in key)
                converted[resolved] = value
            else:
                converted[self._resolve_key(key)] = value
        return tuple(converted.items())

    @staticmethod
    def _resolve_record_args(resolved_key, values, sample):
        def _get_array(elem):
            if isinstance(elem, str):
                return sample[elem]
            elif isinstance(elem, tuple):
                node_type, layer_idx = elem
                return values[layer_idx]
            elif isinstance(elem, int):
                return values[elem]
            else:
                raise TypeError(f"Unexpected record key element: {elem}")

        if isinstance(resolved_key, str):
            return [sample[resolved_key]]
        elif isinstance(resolved_key, tuple) and len(resolved_key) == 2 and isinstance(resolved_key[0], int):
            node_type, layer_idx = resolved_key
            return [values[layer_idx]]
        elif isinstance(resolved_key, tuple):
            return [_get_array(elem) for elem in resolved_key]
        else:
            raise TypeError(f"Unexpected resolved_key format: {resolved_key}")

    # ------------------------------------------------------------------
    # Train / Test
    # ------------------------------------------------------------------

    def train(
        self,
        dataloader: Iterator[Dict[str, Any]],
        data_map: Dict[Union[Layer, int], str],
        epochs: int = 1,
        record_map: Optional[Dict[Union[NodeRef, tuple], callable]] = None,
        verbose: bool = False,
    ) -> 'BackpropSimulation':
        """
        Train with standard backprop.

        Args:
            dataloader: Iterable yielding dicts of batched arrays.
            data_map: Maps Layer objects or indices to sample dict keys.
                Only the input layer(s) should be clamped.
            epochs: Number of passes through the dataloader.
            record_map: Maps node references to recording functions.
            verbose: Print per-epoch loss and timing.

        Returns:
            self for method chaining.
        """
        data_map_tuple = self._convert_map(data_map)

        record_entries = []
        if record_map is not None:
            record_entries = list(self._convert_map(record_map))

        all_losses = []
        self.train_records = {record_fn.__name__: [] for _, record_fn in record_entries}

        for epoch in range(epochs):
            epoch_start = time.time()
            total_loss = 0.0
            n_samples = 0
            epoch_records = {record_fn.__name__: [] for _, record_fn in record_entries}

            for sample in tqdm(dataloader, disable=not verbose, leave=False):
                sample = {k: jnp.array(v) for k, v in sample.items()}

                self.predict_weights, self.opt_state, values, loss = run_backprop_batch(
                    sample, self.predict_weights,
                    self.net.structure, data_map_tuple,
                    self.objective_fn, self.optimizer, self.opt_state,
                    learning=True,
                )

                total_loss += float(loss)
                n_samples += 1
                all_losses.append(float(loss))

                for resolved_key, record_fn in record_entries:
                    args = self._resolve_record_args(resolved_key, values, sample)
                    result = record_fn(*args)
                    epoch_records[record_fn.__name__].append(result)
                    self.train_records[record_fn.__name__].append(result)

            train_time = time.time() - epoch_start
            if verbose:
                import numpy as np
                avg_loss = total_loss / n_samples if n_samples > 0 else 0
                parts = [f"Epoch {epoch + 1}/{epochs}", f"Avg Loss {avg_loss:.4f}", f"Time {train_time:.1f}s"]
                for name, vals in epoch_records.items():
                    parts.append(f"{name}: {np.mean(vals):.4f}")
                print(" | ".join(parts))

        # Write weights back to network params
        self.net.params = self.net.params._replace(
            predict_weights=list(self.predict_weights)
        )
        self.train_losses = all_losses
        return self

    def test(
        self,
        dataloader: Iterator[Dict[str, Any]],
        data_map: Dict[Union[Layer, int], str],
        record_map: Optional[Dict[Union[NodeRef, tuple], callable]] = None,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """
        Run forward inference and collect results.

        Args:
            dataloader: Iterable yielding dicts of batched arrays.
            data_map: Maps Layer objects or indices to sample dict keys.
            record_map: Maps node references to recording functions.
            verbose: Print summary.

        Returns:
            Dict with 'losses' and one entry per record_map function.
        """
        data_map_tuple = self._convert_map(data_map)

        record_entries = []
        if record_map is not None:
            record_entries = list(self._convert_map(record_map))

        all_losses = []
        time_start = time.time()
        results = {record_fn.__name__: [] for _, record_fn in record_entries}

        for sample in tqdm(dataloader, disable=not verbose, leave=False):
            sample = {k: jnp.array(v) for k, v in sample.items()}

            _, _, values, loss = run_backprop_batch(
                sample, self.predict_weights,
                self.net.structure, data_map_tuple,
                self.objective_fn, self.optimizer, self.opt_state,
                learning=False,
            )

            for resolved_key, record_fn in record_entries:
                args = self._resolve_record_args(resolved_key, values, sample)
                results[record_fn.__name__].append(record_fn(*args))

            all_losses.append(float(loss))

        test_time = time.time() - time_start
        if verbose:
            import numpy as np
            avg_loss = np.mean(all_losses)
            parts = [f"Avg Loss {avg_loss:.4f}", f"Time {test_time:.1f}s"]
            for name, vals in results.items():
                parts.append(f"{name}: {np.mean(vals):.4f}")
            print(" | ".join(parts))

        results['losses'] = all_losses
        return results
