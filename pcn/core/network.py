"""
PCNetwork class - the main class for defining predictive coding networks.

Provides a context manager for network definition and compiles to JAX-friendly
structures for efficient execution.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np

from .state import NetworkParams
from .structure import (
    NetworkStructure,
    LayerSpec,
    PredictConnSpec,
    ProjectConnSpec,
    ModulateConnSpec,
)
from .layer import Layer, NodeRef
from .connections import (
    Predict, PredictRes, PredictConv, PredictTransConv, Project, Modulate,
    _resolve_owner_idx,
)
from ..config import DEFAULTS, load_config, _DEFAULT, validate_keys
from .activations import activation_from_name, Activation, Poisson


def _make_band_mask(m, n, n_bands):
    """Binary mask for a band-diagonal (m, n) matrix with half-bandwidth n_bands.

    Entry (i, j) is 1 if |i - j| <= n_bands, else 0.
    For non-square matrices this tracks the main diagonal by absolute index
    distance. An alternative for uniform receptive-field coverage would use
    scaled indices: |i/m - j/n| <= n_bands/m.
    """
    return (jnp.abs(jnp.arange(m)[:, None] - jnp.arange(n)[None, :]) <= n_bands).astype(jnp.float32)


# Sentinel for "not provided" — distinct from None
_UNSET = object()

# Module-level context for the current network being defined
_current_network: Optional['PCNetwork'] = None


def _get_current_network() -> 'PCNetwork':
    """Get the currently active PCNetwork context."""
    global _current_network
    if _current_network is None:
        raise RuntimeError(
            "No PCNetwork context active. "
            "Create layers and connections inside a 'with net:' block."
        )
    return _current_network


def _set_current_network(net: Optional['PCNetwork']) -> None:
    """Set the current network context."""
    global _current_network
    _current_network = net


def _clear_current_network() -> None:
    """Clear the current network context."""
    global _current_network
    _current_network = None


class PCNetwork:
    """
    A predictive coding network defined as a graph of layers and connections.

    Usage:
        net = PCNetwork(seed=0)
        with net:
            l1 = Layer(dim=128, label="input")
            l2 = Layer(dim=64, activation=Relu(), dynamics_rate=0.1, label="hidden")
            l3 = Layer(dim=10, activation=Softmax(), label="output")
            Predict(l2, l1)
            Predict(l3, l2)
        net.build()

    Attributes:
        seed: Random seed for parameter initialization
        label_to_idx: Map layer labels to indices
        idx_to_label: Map indices to layer labels
        node_to_idx: Map node labels to (index, node_type)
        structure: NetworkStructure - compiled static structure (after build)
        params: NetworkParams - learnable parameters (after build)
    """

    def __init__(self, seed: int = 0):
        self.seed = seed
        # Split the seed into two independent streams so weight init and
        # simulation inference never share the same key sequence.
        self.rng, self.sim_rng = jax.random.split(jax.random.PRNGKey(seed))

        # Defaults: start from JSON defaults, overridden by config()
        self._defaults: Dict = dict(DEFAULTS["model"])

        # Populated during context manager
        self._layers: List[Layer] = []
        self._predict_conns: List[Predict] = []
        self._project_conns: List[Project] = []
        self._modulate_conns: List[Modulate] = []
        self._structural_attention_groups: list = []

        # Populated during build()
        self.structure: Optional[NetworkStructure] = None
        self.params: Optional[NetworkParams] = None
        # Custom weight masks for transformation='masked' (one entry per conn
        # of each type; entries for non-masked conns are dummy scalars).
        self.predict_weight_masks: tuple = ()
        self.project_weight_masks: tuple = ()
        self.modulate_weight_masks: tuple = ()
        # Per-Predict-conn error/precision Activation instances. Threaded to
        # the backend so memory-aware activations (e.g. Leaky) can dispatch
        # against the previous-iteration value from the inference carry.
        # Stateless instances are passed unchanged; the backend short-circuits
        # to the standard ACTIVATIONS path when ``has_memory == False``.
        self.predict_error_activations: tuple = ()
        self.predict_precision_activations: tuple = ()
        # Per-layer Activation instances, threaded to the backend so stochastic
        # (needs_key) layer activations can inject noise into the prediction
        # pre-activation. Stateless layers take the standard ACTIVATIONS path.
        self.layer_activations: tuple = ()

        # Label mappings (populated during build)
        self.label_to_idx: Dict[str, int] = {}
        self.idx_to_label: Dict[int, str] = {}
        self.node_to_idx: Dict[str, Tuple[int, str]] = {}

    def config(self, config_file=None, **kwargs) -> 'PCNetwork':
        """
        Set default values for layers and connections.

        Call before the ``with net:`` block to override built-in defaults.
        Values are resolved in order: explicit kwargs > config_file > default_config.json.

        Args:
            config_file: Optional path to a JSON config file. The ``"model"``
                section is extracted; other sections (e.g. ``"train"``,
                ``"test"``) are silently ignored.

        Supported kwargs (see ``default_config.json`` for defaults):
            dynamics_rate, activation, hebbian_learning_rate,
            init_precision, learn_precision (shorthand expanding to
            ``learn_precision_weights`` and ``learn_precision_bias``),
            learn_precision_weights, learn_precision_bias,
            precision_activation (alias: ``precision_parameterization``),
            error_activation, use_bias, alpha, spatial_structure.

            ``activation``, ``error_activation``, and ``precision_activation``
            accept either a name from
            :data:`pcn.core.activations.ACTIVATION_REGISTRY`
            (e.g. ``'relu'``, ``'softplus'``, ``'linear'``) or an
            :class:`pcn.core.activations.Activation` instance — including
            memory-aware ones such as ``Leaky(Tanh(), leak=0.3)``. Per-conn
            kwargs on :class:`Predict` override the network defaults set here.

            ``init_log_precision`` is also accepted for backwards compatibility
            and is converted via ``init_precision = exp(init_log_precision)``.

        Returns:
            self for method chaining.

        Example:
            net = PCNetwork(seed=0)
            net.config(dynamics_rate=0.05)
            with net:
                l1 = Layer(dim=784, label="input")
                l2 = Layer(dim=256, label="hidden")
                Predict(l1, l2)
        """
        validate_keys('model', kwargs)

        if config_file is not None:
            # Load the user's file directly (without merging with DEFAULTS)
            # so we can apply BC conversion on the keys the user actually set.
            import json as _json
            with open(config_file) as _f:
                user_cfg = _json.load(_f)
            model_cfg = dict(user_cfg.get("model", {}))
            validate_keys('model', model_cfg)
            # Back-compat: legacy key ``init_log_precision`` is converted to
            # ``init_precision`` (unless the file already sets init_precision).
            if 'init_log_precision' in model_cfg and 'init_precision' not in model_cfg:
                model_cfg['init_precision'] = float(np.exp(model_cfg.pop('init_log_precision')))
            else:
                model_cfg.pop('init_log_precision', None)
            self._defaults.update(model_cfg)

        # Convert string activation to Activation instance
        if 'activation' in kwargs and isinstance(kwargs['activation'], str):
            kwargs['activation'] = activation_from_name(kwargs['activation'])

        # Back-compat: accept ``init_log_precision`` kwarg and convert.
        if 'init_log_precision' in kwargs and 'init_precision' not in kwargs:
            kwargs['init_precision'] = float(np.exp(kwargs.pop('init_log_precision')))
        else:
            kwargs.pop('init_log_precision', None)

        # Expand learn_precision shorthand to both individual flags if neither
        # is already explicitly provided in kwargs.
        if 'learn_precision' in kwargs:
            lp = kwargs.pop('learn_precision')
            kwargs.setdefault('learn_precision_weights', lp)
            kwargs.setdefault('learn_precision_bias', lp)

        self._defaults.update(kwargs)

        # Also convert activation if it came from config_file as a string
        act = self._defaults.get('activation')
        if isinstance(act, str):
            self._defaults['activation'] = activation_from_name(act)

        return self

    def structural_attention(self, predict_conns, temperature=1.0):
        """Group predict connections for softmax competition.

        When multiple predictions target the same layer, this enables
        soft selection of the best-predicting source (Singh et al., NeurIPS 2023).

        Args:
            predict_conns: List of Predict connections to compete.
            temperature: Softmax temperature (lower = sharper selection).
        """
        from .structure import StructuralAttentionGroup
        indices = tuple(p._idx for p in predict_conns)
        self._structural_attention_groups.append(
            StructuralAttentionGroup(indices, temperature))

    def __enter__(self) -> 'PCNetwork':
        """Enter the network definition context."""
        _set_current_network(self)
        return self

    def __exit__(self, *args) -> None:
        """Exit the network definition context."""
        _clear_current_network()

    def __getitem__(self, label: str) -> int:
        """
        Get layer index by label.

        Example: net['hidden'] -> 1
        """
        return self.label_to_idx[label]

    def _add_layer(self, layer: Layer) -> None:
        """Register a layer (called by Layer.__init__)."""
        idx = len(self._layers)

        # Auto-generate label if not provided
        if layer.label is None:
            layer.label = f'layer_d{layer.dim}_{idx}'

        self._layers.append(layer)
        layer._idx = idx
        layer._network = self

    def _add_predict(self, conn: Predict) -> None:
        """Register a Predict connection."""
        conn._idx = len(self._predict_conns)
        self._predict_conns.append(conn)

    def _add_project(self, conn: Project) -> None:
        """Register a Project connection."""
        conn._idx = len(self._project_conns)
        # Update indices from owners
        conn.pre_layer_idxs = tuple(_resolve_owner_idx(p) for p in conn._pre_list)
        conn.post_layer_idx = conn.post.owner._idx
        self._project_conns.append(conn)

    def _add_modulate(self, conn: Modulate) -> None:
        """Register a Modulate connection."""
        conn._idx = len(self._modulate_conns)
        conn.pre_layer_idxs = tuple(_resolve_owner_idx(p) for p in conn._pre_list)
        conn.post_layer_idx = conn.post.owner._idx
        self._modulate_conns.append(conn)

    def _add_skip(self, pre: Layer, post: Layer, delay: int, skip_scale: float):
        """Create auxiliary layers and Project connections for a Skip.

        Builds a chain ``pre -> aux_1 -> ... -> aux_delay -> post`` using
        fixed-weight identity Project connections scaled by *skip_scale*.

        Returns:
            (auxiliary_layers, project_conns) — lists of the created objects.
        """
        from .activations import Direct
        from .learning_rules import NoLearning

        dim = pre.dim

        chain = [pre]
        auxiliary_layers = []
        for i in range(delay):
            aux = Layer(dim=dim, activation=Direct(),
                        label=f"skip_aux_{pre.label}_{post.label}_{i}")
            auxiliary_layers.append(aux)
            chain.append(aux)
        chain.append(post)

        project_conns = []
        for i in range(delay + 1):
            proj = Project(
                chain[i].value, chain[i + 1].value,
                update_rule=NoLearning(),
                init_weight=skip_scale * jnp.eye(chain[i].dim),
            )
            project_conns.append(proj)

        return auxiliary_layers, project_conns

    def build(self, weight_initialization='xavier') -> 'PCNetwork':
        """
        Compile the network into JAX-friendly structures.

        This method:
        1. Builds label-to-index mappings
        2. Orders connections (user-specified order, then Predict, Project, Modulate)
        3. Creates NetworkStructure (static)
        4. Initializes NetworkParams (weights via Xavier, precisions to 1)

        Returns:
            self for method chaining
        """
        # Assign and deduplicate labels
        self._all_conns = self._predict_conns + self._project_conns + self._modulate_conns
        self._resolve_labels()

        # Build label mappings for layers
        for idx, layer in enumerate(self._layers):
            self.label_to_idx[layer.label] = idx
            self.idx_to_label[idx] = layer.label
            self.node_to_idx[f'{layer.label}-value'] = (idx, 'value')

        # Build label mappings for predict connection error/precision nodes
        for idx, conn in enumerate(self._predict_conns):
            self.node_to_idx[f'{conn.label}-error'] = (idx, 'error')
            self.node_to_idx[f'{conn.label}-logprecision'] = (idx, 'log_precision')

        # Sort connections by order
        self._sort_connections()

        # Build structure
        self.structure = self._build_structure()

        # Compute spatial locations and neighborhoods
        self._compute_spatial()

        # Initialize parameters
        self.params = self._initialize_params(weight_initialization)

        return self

    def _sort_connections(self) -> None:
        """Sort connections by order parameter, defaulting by type."""
        def sort_key(conn, type_priority, default_idx):
            if conn.order is not None:
                return (conn.order, type_priority, default_idx)
            return (float('inf'), type_priority, default_idx)

        # Assign default indices
        for i, c in enumerate(self._predict_conns):
            c._default_order = i
        for i, c in enumerate(self._project_conns):
            c._default_order = i
        for i, c in enumerate(self._modulate_conns):
            c._default_order = i

        # Sort each type (Modulate connections default to end)
        self._predict_conns.sort(key=lambda c: sort_key(c, 0, c._default_order))
        self._project_conns.sort(key=lambda c: sort_key(c, 1, c._default_order))
        self._modulate_conns.sort(key=lambda c: sort_key(c, 2, c._default_order))

    def _resolve_labels(self) -> None:
        """Assign and deduplicate labels for layers and predict connections."""
        # Deduplicate layer labels
        layer_labels = [l.label for l in self._layers]
        unique_layer_labels = self._make_unique(layer_labels)
        for layer, label in zip(self._layers, unique_layer_labels):
            layer.label = label

        # Auto-generate connection labels where not user-specified
        for conn in self._predict_conns:
            if conn.label is None:
                if len(conn.pre) == 1:
                    pre_label = conn.pre[0].label or f"layer_{conn.pre[0]._idx}"
                else:
                    pre_label = '+'.join(p.label or f"layer_{p._idx}" for p in conn.pre)
                post_label = conn.post.label or f"layer_{conn.post._idx}"
                conn.label = f"predict_{pre_label}_{post_label}"
        for conn in self._project_conns:
            if conn.label is None:
                pre_label = conn._pre_list[0].node_type
                post_label = conn.post.node_type
                if conn.update_rule is not None:
                    gd = 'gd_' if conn.update_rule.type_id == 2 else ''
                else:
                    gd = ''
                conn.label = f"{gd}project_{pre_label}_{post_label}"
        for conn in self._modulate_conns:
            if conn.label is None:
                pre_label = conn._pre_list[0].node_type
                post_label = conn.post.node_type
                if conn.update_rule is not None:
                    gd = 'gd_' if conn.update_rule.type_id == 2 else ''
                else:
                    gd = ''
                conn.label = f"{gd}modulate_{pre_label}_{post_label}"

        # Deduplicate connection labels
        conn_labels = [c.label for c in self._all_conns] 
        unique_conn_labels = self._make_unique(conn_labels)
        for conn, label in zip(self._all_conns, unique_conn_labels):
            conn.label = label

    @staticmethod
    def _make_unique(labels):
        """Make labels unique by appending _{index} for duplicates."""
        seen = set()
        result = []
        for i, label in enumerate(labels):
            original = label
            if label in seen:
                label = f"{original}_{i}"
                counter = 2
                while label in seen:
                    label = f"{original}_{i}_{counter}"
                    counter += 1
            seen.add(label)
            result.append(label)
        return result

    def _resolve_loss_fn_inputs(self, inputs_spec):
        """Resolve a loss_fn inputs spec to backend-friendly form.

        NodeRef  -> (node_type_id, owner._idx)
        str      -> str  (sample dict key, resolved at runtime)
        tuple    -> tuple of resolved elements
        """
        def _resolve_one(elem):
            if isinstance(elem, NodeRef):
                return (elem.node_type_id, elem.owner._idx)
            elif isinstance(elem, str):
                return elem
            else:
                raise TypeError(
                    f"loss_fn inputs must be NodeRef or str, got {type(elem)}")

        if isinstance(inputs_spec, (NodeRef, str)):
            return _resolve_one(inputs_spec)
        elif isinstance(inputs_spec, tuple):
            return tuple(_resolve_one(e) for e in inputs_spec)
        else:
            raise TypeError(
                f"loss_fn inputs must be NodeRef, str, or tuple thereof, "
                f"got {type(inputs_spec)}")

    def _build_structure(self) -> NetworkStructure:
        """Create the static NetworkStructure."""
        layer_specs = tuple(
            LayerSpec(
                dim=layer.dim,
                activation_type=layer.activation.type_id,
                dynamics_rate=layer.dynamics_rate,
                label=layer.label,
                is_poisson=isinstance(layer.activation, Poisson),
                spatial_structure=layer.spatial_structure,
                dropout_prob=getattr(layer, 'dropout_prob', 0.0),
                activation_temperature=float(getattr(layer.activation, 'temperature', 1.0)),
                activation_num_winners=int(getattr(layer.activation, 'num_winners', 0)),
            )
            for layer in self._layers
        )

        def _precision_input_fields(conn):
            """Resolve a Predict's precision_input to spec tuples (idx, node_type, slice)."""
            refs = conn._resolved_precision_input()
            if refs is None:
                return (), (), ()
            idxs, ntypes, slices = [], [], []
            for r in refs:
                if r.owner._idx is None:
                    raise ValueError(
                        f"precision_input source {r!r} is not registered in "
                        f"this network")
                idxs.append(r.owner._idx)
                ntypes.append(r.node_type_id)
                slices.append(r.slice_bounds)
            return tuple(idxs), tuple(ntypes), tuple(slices)

        # A connection's precision is provably the constant 1.0 (so the backend
        # can fold it out of the inference hot loop) only when nothing can make
        # it vary: not learned, init 1.0, a stateless precision activation, and
        # no precision-targeting Project/Modulate anywhere in the graph.
        _has_prec_routing = any(
            getattr(c, 'post_node_type', 0) == 2
            for c in (self._project_conns + self._modulate_conns))

        def _is_unit_precision(conn):
            pact = getattr(conn, 'precision_activation', None)
            return (
                not conn.learn_precision_weights
                and not conn.learn_precision_bias
                and float(getattr(conn, 'init_precision', 1.0)) == 1.0
                and not _has_prec_routing
                and not getattr(pact, 'has_memory', False)
                and not getattr(pact, 'needs_key', False)
            )

        def _predict_spec(conn):
            pin_idx, pin_ntypes, pin_slices = _precision_input_fields(conn)
            return PredictConnSpec(
                pre_idx=tuple(p._idx for p in conn.pre),
                post_idx=conn.post._idx,
                has_fixed_weights=conn.weight is not None,
                learn_precision_weights=conn.learn_precision_weights,
                learn_precision_bias=conn.learn_precision_bias,
                alpha=conn.alpha,
                is_conv=getattr(conn, 'is_conv', False),
                is_transconv=getattr(conn, 'is_transconv', False),
                in_channels=getattr(conn, 'in_channels', 0),
                out_channels=getattr(conn, 'out_channels', 0),
                kernel_size=getattr(conn, 'kernel_size', ()),
                stride=getattr(conn, 'stride', (1, 1)),
                padding=getattr(conn, 'padding', 'SAME'),
                input_spatial=getattr(conn, 'input_shape', ()),
                output_spatial=getattr(conn, 'output_shape', ()),
                pool_type=getattr(conn, 'pool_type', 0),
                pool_size=getattr(conn, 'pool_size', ()),
                pool_stride=getattr(conn, 'pool_stride', ()),
                has_bias=getattr(conn, 'use_bias', True),
                is_res=getattr(conn, 'is_res', False),
                label=conn.label or '',
                n_bands=getattr(conn, 'n_bands', 0),
                pre_slices=conn.pre_slices if any(s is not None for s in conn.pre_slices) else (),
                post_slice=conn.post_slice if conn.post_slice is not None else (),
                precision_activation_type=getattr(conn, 'precision_activation_type',
                                                   getattr(conn, 'precision_param_type', 9)),
                unit_precision=_is_unit_precision(conn),
                error_activation_type=getattr(conn, 'error_activation_type', 0),
                post_activation_type=getattr(conn, 'post_activation_type_id', 0),
                is_masked=getattr(conn, 'is_masked', False),
                precision_input_norm=getattr(conn, 'precision_input_norm', False),
                stochastic=getattr(conn, 'stochastic', True),
                precision_input_idx=pin_idx,
                precision_input_node_types=pin_ntypes,
                precision_input_slices=pin_slices,
            )

        predict_specs = tuple(
            _predict_spec(conn) for conn in self._predict_conns
        )

        # Collect reward/loss functions from Project/Modulate connections.
        # Both reward_fn and loss_fn use the same (inputs, fn) signature, so
        # resolution and sample-key tracking are unified.
        reward_fns, loss_fns = [], []
        sample_keys = set()

        def _collect_keys(resolved):
            if isinstance(resolved, str):
                sample_keys.add(resolved)
            elif isinstance(resolved, tuple):
                for elem in resolved:
                    if isinstance(elem, str):
                        sample_keys.add(elem)

        for conn in list(self._project_conns) + list(self._modulate_conns):
            if hasattr(conn.update_rule, 'reward_fn') and conn.update_rule.reward_fn is not None:
                inputs_spec, fn = conn.update_rule.reward_fn
                resolved = self._resolve_loss_fn_inputs(inputs_spec)
                _collect_keys(resolved)
                conn.reward_fn_idx = len(reward_fns)
                reward_fns.append((resolved, fn))
            if hasattr(conn.update_rule, 'loss_fn') and conn.update_rule.loss_fn is not None:
                inputs_spec, fn = conn.update_rule.loss_fn
                resolved = self._resolve_loss_fn_inputs(inputs_spec)
                _collect_keys(resolved)
                conn.loss_fn_idx = len(loss_fns)
                loss_fns.append((resolved, fn))
        self._reward_fns = tuple(reward_fns)
        self._loss_fns = tuple(loss_fns)
        self._loss_fn_sample_keys = tuple(sorted(sample_keys))

        project_specs = tuple(
            ProjectConnSpec(
                pre_idx=conn.pre_layer_idxs,
                pre_node_type=conn.pre_node_type,
                post_idx=conn.post_layer_idx,
                post_node_type=conn.post_node_type,
                learning_rule_type=conn.update_rule.type_id,
                learning_rate=conn.update_rule.learning_rate,
                reward_fn_idx=getattr(conn, 'reward_fn_idx', -1),
                loss_fn_idx=getattr(conn, 'loss_fn_idx', -1),
                is_conv=getattr(conn, 'is_conv', False),
                is_transconv=getattr(conn, 'is_transconv', False),
                in_channels=getattr(conn, 'in_channels', 0),
                out_channels=getattr(conn, 'out_channels', 0),
                kernel_size=getattr(conn, 'kernel_size', ()),
                stride=getattr(conn, 'stride', (1, 1)),
                padding=getattr(conn, 'padding', 'SAME'),
                input_spatial=getattr(conn, 'input_shape', ()),
                output_spatial=getattr(conn, 'output_shape', ()),
                pool_type=getattr(conn, 'pool_type', 0),
                pool_size=getattr(conn, 'pool_size', ()),
                pool_stride=getattr(conn, 'pool_stride', ()),
                n_bands=getattr(conn, 'n_bands', 0),
                pre_slices=conn.pre_slices if any(s is not None for s in conn.pre_slices) else (),
                post_slice=conn.post_slice if conn.post_slice is not None else (),
                has_bias=getattr(conn, 'use_bias', False),
                post_activation_type=getattr(conn, 'post_activation_type_id', 0),
                is_masked=getattr(conn, 'is_masked', False),
            )
            for conn in self._project_conns
        )

        modulate_specs = tuple(
            ModulateConnSpec(
                pre_idx=conn.pre_layer_idxs,
                pre_node_type=conn.pre_node_type,
                post_idx=conn.post_layer_idx,
                post_node_type=conn.post_node_type,
                learning_rule_type=conn.update_rule.type_id,
                learning_rate=conn.update_rule.learning_rate,
                reward_fn_idx=getattr(conn, 'reward_fn_idx', -1),
                loss_fn_idx=getattr(conn, 'loss_fn_idx', -1),
                is_conv=getattr(conn, 'is_conv', False),
                is_transconv=getattr(conn, 'is_transconv', False),
                in_channels=getattr(conn, 'in_channels', 0),
                out_channels=getattr(conn, 'out_channels', 0),
                kernel_size=getattr(conn, 'kernel_size', ()),
                stride=getattr(conn, 'stride', (1, 1)),
                padding=getattr(conn, 'padding', 'SAME'),
                input_spatial=getattr(conn, 'input_shape', ()),
                output_spatial=getattr(conn, 'output_shape', ()),
                pool_type=getattr(conn, 'pool_type', 0),
                pool_size=getattr(conn, 'pool_size', ()),
                pool_stride=getattr(conn, 'pool_stride', ()),
                n_bands=getattr(conn, 'n_bands', 0),
                pre_slices=conn.pre_slices if any(s is not None for s in conn.pre_slices) else (),
                post_slice=conn.post_slice if conn.post_slice is not None else (),
                has_bias=getattr(conn, 'use_bias', False),
                post_activation_type=getattr(conn, 'post_activation_type_id', 0),
                is_masked=getattr(conn, 'is_masked', False),
            )
            for conn in self._modulate_conns
        )

        # Pre-sort Project/Modulate by target type for efficient dispatch
        project_conns_internal = tuple(
            (i, s) for i, s in enumerate(project_specs) if s.post_node_type == 1
        )
        project_conns_value = tuple(
            (i, s) for i, s in enumerate(project_specs) if s.post_node_type == 0
        )
        modulate_conns_internal = tuple(
            (i, s) for i, s in enumerate(modulate_specs) if s.post_node_type == 1
        )
        modulate_conns_value = tuple(
            (i, s) for i, s in enumerate(modulate_specs) if s.post_node_type == 0
        )
        # Mechanism 1: precision-targeting
        project_conns_precision = tuple(
            (i, s) for i, s in enumerate(project_specs) if s.post_node_type == 2
        )
        modulate_conns_precision = tuple(
            (i, s) for i, s in enumerate(modulate_specs) if s.post_node_type == 2
        )
        # Mechanism 2: per-leg flow gating
        modulate_conns_flow_pre = tuple(
            (i, s) for i, s in enumerate(modulate_specs) if s.post_node_type == 3
        )
        modulate_conns_flow_post = tuple(
            (i, s) for i, s in enumerate(modulate_specs) if s.post_node_type == 4
        )
        _flow_target_predict_idxs = set(
            s.post_idx for s in modulate_specs if s.post_node_type in (3, 4))
        predict_has_flow_gates = tuple(
            i in _flow_target_predict_idxs for i in range(len(predict_specs))
        )

        predict_error_dims = tuple(
            conn.post_dim
            for conn in self._predict_conns
        )

        spatial_layers = tuple(
            i for i, layer in enumerate(self._layers)
            if layer.spatial_structure != 'none'
        )

        # GradientDescent connection indices (type_id=2).
        # GradientDescent always carries a loss_fn (required); weights are
        # learned by a single-step jax.grad of that loss, never from the PC
        # energy backward pass.
        gd_loss_project = tuple(
            (i, s.loss_fn_idx) for i, s in enumerate(project_specs)
            if s.learning_rule_type == 2 and s.loss_fn_idx >= 0
        )
        gd_loss_modulate = tuple(
            (i, s.loss_fn_idx) for i, s in enumerate(modulate_specs)
            if s.learning_rule_type == 2 and s.loss_fn_idx >= 0
        )

        inference_regs = tuple(
            (layer._idx, layer.inference_reg)
            for layer in self._layers
            if layer.inference_reg is not None
        )
        train_regs = tuple(
            (layer._idx, layer.train_reg)
            for layer in self._layers
            if layer.train_reg is not None
        )

        # Per-connection energy scale: 1 / (number of predict conns sharing the same pre layer).
        # When a layer fans out to N predict connections its gradient accumulates N times;
        # dividing each connection's energy by N keeps the effective gradient magnitude
        # independent of fan-out.
        _pre_counts: dict = {}
        for spec in predict_specs:
            for idx in spec.pre_idx:
                _pre_counts[idx] = _pre_counts.get(idx, 0) + 1
        predict_pre_scales = tuple(
            1.0 / max(_pre_counts[idx] for idx in spec.pre_idx)
            for spec in predict_specs
        ) if predict_specs else ()

        return NetworkStructure(
            layers=layer_specs,
            predict_conns=predict_specs,
            project_conns=project_specs,
            modulate_conns=modulate_specs,
            layer_dims=tuple(layer.dim for layer in self._layers),
            predict_error_dims=predict_error_dims,
            project_conns_internal=project_conns_internal,
            project_conns_value=project_conns_value,
            modulate_conns_internal=modulate_conns_internal,
            modulate_conns_value=modulate_conns_value,
            spatial_layers=spatial_layers,
            gd_loss_project=gd_loss_project,
            gd_loss_modulate=gd_loss_modulate,
            loss_fn_sample_keys=self._loss_fn_sample_keys,
            inference_regs=inference_regs,
            train_regs=train_regs,
            project_conns_precision=project_conns_precision,
            modulate_conns_precision=modulate_conns_precision,
            modulate_conns_flow_pre=modulate_conns_flow_pre,
            modulate_conns_flow_post=modulate_conns_flow_post,
            predict_has_flow_gates=predict_has_flow_gates,
            structural_attention_groups=tuple(self._structural_attention_groups),
            predict_pre_scales=predict_pre_scales,
        )

    @staticmethod
    def _make_grid_1d(dim):
        """Create 1D grid locations in [-1, 1]."""
        return np.linspace(-1, 1, dim).reshape(dim, 1)

    @staticmethod
    def _make_grid_2d(dim):
        """Create 2D grid locations in [-1, 1]^2."""
        n_side = int(np.ceil(np.sqrt(dim)))
        coords = np.linspace(-1, 1, n_side)
        xx, yy = np.meshgrid(coords, coords)
        locations = np.stack([xx.ravel(), yy.ravel()], axis=1)[:dim]
        return locations

    @staticmethod
    def _make_hex_2d(dim):
        """Create 2D hexagonal grid locations in [-1, 1]^2."""
        n_side = int(np.ceil(np.sqrt(dim)))
        rows = []
        for r in range(n_side):
            for c in range(n_side):
                x = c + 0.5 * (r % 2)
                y = r * np.sqrt(3) / 2
                rows.append([x, y])
                if len(rows) >= dim:
                    break
            if len(rows) >= dim:
                break
        locations = np.array(rows[:dim])
        # Normalize to [-1, 1]
        lo = locations.min(axis=0)
        hi = locations.max(axis=0)
        span = hi - lo
        span = np.where(span < 1e-8, 1.0, span)
        locations = 2.0 * (locations - lo) / span - 1.0
        return locations

    def _compute_spatial(self):
        """Compute locations and neighborhood matrices for spatial layers."""
        neighborhoods = []
        for idx in self.structure.spatial_layers:
            layer = self._layers[idx]
            dim = layer.dim
            ss = layer.spatial_structure

            if ss == 'grid_1':
                locations = self._make_grid_1d(dim)
            elif ss == 'grid_2':
                locations = self._make_grid_2d(dim)
            elif ss == 'hex_2':
                locations = self._make_hex_2d(dim)
            else:
                raise ValueError(f"Unknown spatial_structure: {ss}")

            # Pairwise distances
            diffs = locations[:, None, :] - locations[None, :, :]
            dists = np.sqrt(np.sum(diffs ** 2, axis=-1))

            # Compute sigma from mean nearest-neighbor distance
            dists_nn = dists.copy()
            np.fill_diagonal(dists_nn, np.inf)
            sigma = np.mean(np.min(dists_nn, axis=1))

            # Gaussian kernel
            neighborhood = np.exp(-dists ** 2 / (2 * sigma ** 2))

            layer.locations = jnp.array(locations)
            layer.neighborhood = jnp.array(neighborhood)
            neighborhoods.append(jnp.array(neighborhood))

        self.spatial_neighborhoods = tuple(neighborhoods)

    def _initialize_params(self, weight_initialization='xavier') -> NetworkParams:
        """Initialize all parameters.

        Also populates ``self.predict_weight_masks``, ``self.project_weight_masks``,
        and ``self.modulate_weight_masks`` — tuples (length = number of conns of
        each type) of ``jnp.ndarray`` masks for ``transformation='masked'``
        connections, with a 0-dim dummy for unmasked positions. The dummies
        are never read because callers Python-guard on ``spec.is_masked``.
        """
        predict_weights = []

        for conn in self._predict_conns:
            if conn.weight is not None:  # user set their own weight matrix
                W = jnp.array(conn.weight)
                if getattr(conn, 'is_masked', False):
                    W = W * jnp.asarray(conn.weight_mask, dtype=W.dtype)
                predict_weights.append(W)
            else:
                if getattr(conn, 'is_conv', False) or getattr(conn, 'is_transconv', False):
                    kH, kW = conn.kernel_size
                    fan_in = conn.in_channels * kH * kW
                    fan_out = conn.out_channels * kH * kW
                    shape = (conn.out_channels, conn.in_channels, kH, kW)
                else:
                    fan_in = conn.pre_dim  # sum of all pre dims (sliced)
                    fan_out = conn.post_dim
                    shape = (fan_out, fan_in)
                self.rng, subkey = jax.random.split(self.rng)
                # Use first pre layer's activation for init type
                first_pre = conn.pre[0]
                if first_pre.f.init_type == 'xavier':
                    fan_avg = fan_in + fan_out
                elif first_pre.f.init_type == 'he':
                    fan_avg = fan_in

                stddev = first_pre.f.init_scale * jnp.sqrt(1.0 / fan_avg)
                W = jax.random.normal(subkey, shape) * stddev
                if conn.n_bands > 0 and not (getattr(conn, 'is_conv', False) or getattr(conn, 'is_transconv', False)):
                    W = W * _make_band_mask(fan_out, fan_in, conn.n_bands)
                if getattr(conn, 'is_masked', False):
                    W = W * jnp.asarray(conn.weight_mask, dtype=W.dtype)
                predict_weights.append(W)

        project_weights = []
        project_biases = []
        for conn in self._project_conns:
            pre_dim = conn.pre_dim
            post_dim = conn.post_dim
            is_conv = getattr(conn, 'is_conv', False) or getattr(conn, 'is_transconv', False)

            if conn.weight is not None:
                W = jnp.array(conn.weight)
                if getattr(conn, 'is_masked', False):
                    W = W * jnp.asarray(conn.weight_mask, dtype=W.dtype)
                project_weights.append(W)
            else:
                if is_conv:
                    kH, kW = conn.kernel_size
                    shape = (conn.out_channels, conn.in_channels, kH, kW)
                    fan_in = conn.in_channels * kH * kW
                    fan_out = conn.out_channels * kH * kW
                else:
                    shape = (post_dim, pre_dim)
                    fan_in = pre_dim
                    fan_out = post_dim
                self.rng, subkey = jax.random.split(self.rng)
                stddev = jnp.sqrt(2.0 / (fan_in + fan_out))
                W = jax.random.normal(subkey, shape) * stddev
                if conn.n_bands > 0 and not is_conv:
                    W = W * _make_band_mask(fan_out, fan_in, conn.n_bands)
                if getattr(conn, 'is_masked', False):
                    W = W * jnp.asarray(conn.weight_mask, dtype=W.dtype)
                project_weights.append(W)

            if conn.use_bias:
                bias_dim = conn.out_channels if is_conv else post_dim
                if conn.bias is not None:
                    project_biases.append(jnp.array(conn.bias, dtype=jnp.float32))
                else:
                    project_biases.append(jnp.zeros(bias_dim, dtype=jnp.float32))
            else:
                project_biases.append(jnp.zeros(1, dtype=jnp.float32))  # dummy, always adds 0

        modulate_weights = []
        modulate_biases = []
        for conn in self._modulate_conns:
            pre_dim = conn.pre_dim
            post_dim = conn.post_dim
            is_conv = getattr(conn, 'is_conv', False) or getattr(conn, 'is_transconv', False)

            if is_conv:
                kH, kW = conn.kernel_size
                shape = (conn.out_channels, conn.in_channels, kH, kW)
            else:
                shape = (post_dim, pre_dim)

            if conn.weight is not None:
                W = jnp.array(conn.weight)
            else:
                self.rng, subkey = jax.random.split(self.rng)
                if conn.use_bias:
                    # Near-zero W so that W @ f(pre) + 1 ≈ 1 at init (identity modulation)
                    W = jax.random.normal(subkey, shape) * 0.01
                else:
                    # Near-1 W (legacy behaviour, no bias)
                    W = jnp.ones(shape) + jax.random.normal(subkey, shape) * 0.01

            if conn.use_bias:
                bias_dim = conn.out_channels if is_conv else post_dim
                if conn.bias is not None:
                    modulate_biases.append(jnp.array(conn.bias, dtype=jnp.float32))
                else:
                    modulate_biases.append(jnp.ones(bias_dim, dtype=jnp.float32))
            else:
                modulate_biases.append(jnp.zeros(1, dtype=jnp.float32))  # dummy, always adds 0
            if conn.n_bands > 0 and not is_conv:
                W = W * _make_band_mask(post_dim, pre_dim, conn.n_bands)
            if getattr(conn, 'is_masked', False):
                W = W * jnp.asarray(conn.weight_mask, dtype=W.dtype)
            modulate_weights.append(W)

        predict_biases = []
        for conn in self._predict_conns:
            if conn.use_bias and conn.bias is not None:
                predict_biases.append(jnp.array(conn.bias, dtype=jnp.float32))
            elif conn.use_bias:
                is_conv = getattr(conn, 'is_conv', False) or getattr(conn, 'is_transconv', False)
                if is_conv:
                    predict_biases.append(jnp.zeros(conn.out_channels, dtype=jnp.float32))
                else:
                    predict_biases.append(jnp.zeros(conn.post_dim, dtype=jnp.float32))
            else:
                predict_biases.append(jnp.zeros(1, dtype=jnp.float32))  # dummy, never updated

        # Precision weights/biases: precision = g(precision_weights @ pre_act.T + precision_bias)
        # where g is the activation selected by ``precision_activation_type`` (an
        # index into pcn.core.activations.ACTIVATIONS). The precision bias is
        # initialised so the starting precision equals ``conn.init_precision``
        # by applying the inverse of g where one is known. We special-case the
        # three recommended parameterisations:
        #   - exp (type_id 10)      :  bias = log(init_precision)
        #   - softplus (type_id 9)  :  bias = log(expm1(init_precision))
        #   - linear/Direct (id 0)  :  bias = init_precision  (identity)
        # For any other activation we set bias = init_precision raw — the user
        # has opted out of the recommended set and is responsible for choosing
        # an init that yields a sensible starting precision under that g.
        from .activations import Exp as _Exp, Softplus as _Softplus, Direct as _Direct
        precision_weights = []
        precision_biases = []
        for conn in self._predict_conns:
            # Feature dim of the precision function's input: the conn's own
            # pre dim by default, or the summed dims of custom
            # ``precision_input`` sources.
            pin_dim = conn.precision_input_dim
            post_dim = conn.post_dim
            init_p = float(conn.init_precision)
            ppt = conn.precision_activation_type

            if ppt == _Exp.type_id:
                bias_val = float(np.log(init_p))
            elif ppt == _Softplus.type_id:
                # Stable inverse softplus: log(expm1(p)) = p + log1p(-exp(-p)).
                # The naive form overflows expm1 for p >~ 700 (e.g. precisions
                # initialised to a layer dimension for sum-convention scaling).
                bias_val = float(init_p + np.log1p(-np.exp(-init_p)))
            elif ppt == _Direct.type_id:
                bias_val = init_p
            else:
                bias_val = init_p

            if not conn.learn_precision_weights and not conn.learn_precision_bias:
                pw = jnp.zeros((1, pin_dim), dtype=jnp.float32)
                pb = jnp.full(1, bias_val, dtype=jnp.float32)
            else:
                pw = jnp.zeros((post_dim, pin_dim), dtype=jnp.float32)
                pb = jnp.full(post_dim, bias_val, dtype=jnp.float32)
            precision_weights.append(pw)
            precision_biases.append(pb)

        # Collect weight masks (one entry per connection; dummy for unmasked
        # positions, never read because the backend Python-guards on is_masked).
        _dummy_mask = jnp.zeros((), dtype=jnp.float32)

        def _mask_for(conn):
            if getattr(conn, 'is_masked', False):
                return jnp.asarray(conn.weight_mask, dtype=jnp.float32)
            return _dummy_mask

        self.predict_weight_masks = tuple(_mask_for(c) for c in self._predict_conns)
        self.project_weight_masks = tuple(_mask_for(c) for c in self._project_conns)
        self.modulate_weight_masks = tuple(_mask_for(c) for c in self._modulate_conns)

        # Cache the Activation instances for backend memory-dispatch.
        self.predict_error_activations = tuple(
            c.error_activation for c in self._predict_conns)
        self.predict_precision_activations = tuple(
            c.precision_activation for c in self._predict_conns)
        self.layer_activations = tuple(
            layer.activation for layer in self._layers)

        return NetworkParams(
            predict_weights=predict_weights,
            predict_biases=predict_biases,
            project_weights=project_weights,
            project_biases=project_biases,
            modulate_weights=modulate_weights,
            modulate_biases=modulate_biases,
            precision_weights=precision_weights,
            precision_biases=precision_biases,
        )

    def get_layer(self, identifier) -> Layer:
        """
        Get a layer by label or index.

        Args:
            identifier: Either a string label or integer index

        Returns:
            The Layer object
        """
        if isinstance(identifier, str):
            idx = self.label_to_idx[identifier]
        else:
            idx = identifier
        return self._layers[idx]

    def multi_transform(self, optim_dict, default_optim):
        """Create an optax.multi_transform using connection labels and param types.

        Builds a label function that maps each parameter to an optimizer based on
        connection labels and parameter types (``predict_weights``,
        ``predict_biases``, ``precision_weights``, ``precision_biases``,
        ``gd_loss_project_weights``, ``gd_loss_modulate_weights``).

        Keys in ``optim_dict`` are matched to parameters in priority order:

        1. ``'{conn_label}_{param_type}'`` — most specific
        2. ``'{conn_label}'`` — all param types for that connection
        3. General class:

           a. exact param_type (e.g. ``'predict_weights'``)
           b. category shorthand (most specific first):

              - ``'gd_loss_project'`` / ``'loss_project'`` / ``'project'`` —
                loss-based GD project weights
              - Same pattern for ``'modulate'``
              - ``'predict'``, ``'precision'``

           c. kind (``'weights'`` or ``'biases'``)

        4. ``default_optim`` — fallback

        If both a category and a kind key match the same parameter (e.g.
        ``'precision'`` and ``'biases'`` both match ``precision_biases``),
        a ``ValueError`` is raised.

        Args:
            optim_dict: Maps labels/types to optax optimizers.
            default_optim: Fallback optimizer for unmatched parameters.

        Returns:
            An ``optax.GradientTransformation`` (multi_transform).

        Example::

            optimizer = net.multi_transform(
                {'predict': optax.adam(1e-3), 'precision': optax.sgd(1e-5)},
                default_optim=optax.adam(1e-4),
            )
        """
        import optax

        if self.structure is None:
            raise RuntimeError(
                "Network must be built before calling multi_transform. "
                "Call net.build() first."
            )

        param_types = ('predict_weights', 'predict_biases',
                       'project_biases', 'modulate_biases',
                       'precision_weights', 'precision_biases',
                       'gd_loss_project_weights', 'gd_loss_modulate_weights')

        predict_conn_labels = [conn.label for conn in self._predict_conns]
        project_conn_labels = [conn.label for conn in self._project_conns]
        modulate_conn_labels = [conn.label for conn in self._modulate_conns]
        gd_loss_project_conn_labels = [
            conn.label for conn in self._project_conns
            if conn.update_rule.type_id == 2 and conn.update_rule.loss_fn is not None]
        gd_loss_modulate_conn_labels = [
            conn.label for conn in self._modulate_conns
            if conn.update_rule.type_id == 2 and conn.update_rule.loss_fn is not None]

        # Map category (from param_type.rsplit('_', 1)[0]) to its connection labels
        conn_labels_for_category = {
            'predict': predict_conn_labels,
            'project': project_conn_labels,
            'modulate': modulate_conn_labels,
            'precision': predict_conn_labels,
            'gd_loss_project': gd_loss_project_conn_labels,
            'gd_loss_modulate': gd_loss_modulate_conn_labels,
        }

        # Category shorthand matching — ordered most-specific to least-specific.
        # First match wins, so 'loss_project' beats 'project' when both are keys.
        subcategory_map = {
            'predict': ('predict',),
            'project': ('project',),
            'modulate': ('modulate',),
            'precision': ('precision',),
            'gd_loss_project': ('gd_loss_project', 'loss_project', 'project'),
            'gd_loss_modulate': ('gd_loss_modulate', 'loss_modulate', 'modulate'),
        }

        default_key = '__default__'
        label_map = {}

        for pt in param_types:
            category, kind = pt.rsplit('_', 1)
            labels_for_pt = []
            conn_labels = conn_labels_for_category[category]

            for conn_label in conn_labels:
                # Priority 1: '{conn_label}_{param_type}'
                specific_key = f'{conn_label}_{pt}'
                if specific_key in optim_dict:
                    labels_for_pt.append(specific_key)
                    continue

                # Priority 2: connection label
                if conn_label in optim_dict:
                    labels_for_pt.append(conn_label)
                    continue

                # Priority 3a: exact param type
                if pt in optim_dict:
                    labels_for_pt.append(pt)
                    continue

                # Priority 3b: category shorthand or kind (conflict if both match)
                matches = []
                for subcat in subcategory_map.get(category, (category,)):
                    if subcat in optim_dict:
                        matches.append(subcat)
                        break  # use most-specific category match
                if kind in optim_dict:
                    matches.append(kind)

                if len(matches) > 1:
                    raise ValueError(
                        f"Ambiguous optim_dict keys {matches} all match "
                        f"connection '{conn_label}', param '{pt}'. "
                        f"Use '{conn_label}_{pt}' to resolve."
                    )
                elif len(matches) == 1:
                    labels_for_pt.append(matches[0])
                    continue

                # Priority 4: default
                labels_for_pt.append(default_key)

            label_map[pt] = tuple(labels_for_pt)

        full_dict = dict(optim_dict)
        full_dict[default_key] = default_optim

        def _label_fn(params):
            return {k: label_map[k] for k in params}

        return optax.multi_transform(full_dict, _label_fn)

    # ------------------------------------------------------------------ #
    #  Save / Load                                                        #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _get_git_hash() -> str:
        """Return the current git revision hash, or 'unknown' on failure."""
        import subprocess
        try:
            return (
                subprocess.check_output(
                    ['git', 'rev-parse', 'HEAD'],
                    stderr=subprocess.DEVNULL,
                )
                .decode('ascii')
                .strip()
            )
        except Exception:
            return 'unknown'

    def save(
        self,
        path: Optional[Union[str, Path]] = None,
        simulation_results: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """Save network structure, parameters, and optional simulation results.

        Everything is stored in a single HDF5 file with the following groups::

            /metadata          – saved_at (ISO timestamp), code_version (git hash)
            /structure/layers  – one dataset per LayerSpec field
            /structure/predict_conns
            /structure/project_conns
            /structure/modulate_conns
            /params/predict_weights/0, 1, …
            /params/project_weights/0, 1, …
            /params/modulate_weights/0, 1, …
            /params/precisions/0, 1, …
            /results/train_energies  (optional)
            /results/test_energies   (optional)
            /results/<name>          (optional, from record_map)

        Args:
            path: File path for the HDF5 file.  If *None*, defaults to
                ``saved_models/<first_layer_label>_<last_layer_label>.h5``.
            simulation_results: Optional dict that may contain:
                ``"train_energies"`` – list/array of training energy arrays
                ``"test_energies"``  – list/array of test energy values
                Plus any extra keys (e.g. record_map results) whose values
                are lists of scalars or arrays.

        Returns:
            The resolved :class:`~pathlib.Path` of the saved file.
        """
        import h5py
        from datetime import datetime, timezone

        if self.structure is None or self.params is None:
            raise RuntimeError("Network must be built before saving. Call net.build() first.")

        # Default path
        if path is None:
            first = self.structure.layers[0].label
            last = self.structure.layers[-1].label
            save_dir = Path('saved_models')
            save_dir.mkdir(parents=True, exist_ok=True)
            path = save_dir / f'{first}_{last}.h5'
        else:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)

        with h5py.File(path, 'w') as f:
            # -- metadata --
            meta = f.create_group('metadata')
            meta.attrs['saved_at'] = datetime.now(timezone.utc).isoformat()
            meta.attrs['code_version'] = self._get_git_hash()

            # -- structure --
            self._save_structure(f, self.structure)

            # -- params --
            self._save_params(f, self.params)

            # -- simulation results --
            if simulation_results is not None:
                self._save_results(f, simulation_results)

        return path

    @staticmethod
    def _save_structure(f, structure: NetworkStructure) -> None:
        """Write NetworkStructure to HDF5 groups."""
        def _set_attr(g, field, value):
            # Spec fields are write-only metadata (load() reads only
            # layer_dims + params). Values h5py cannot store natively —
            # e.g. slice tuples containing None, such as
            # precision_input_slices=(None,) or mixed pre_slices — are
            # stored as their repr string instead.
            try:
                g.attrs[field] = value
            except TypeError:
                g.attrs[field] = repr(value)

        sg = f.create_group('structure')
        sg.attrs['layer_dims'] = list(structure.layer_dims)
        sg.attrs['predict_error_dims'] = list(structure.predict_error_dims)

        # Layers
        lg = sg.create_group('layers')
        for i, spec in enumerate(structure.layers):
            g = lg.create_group(str(i))
            g.attrs['dim'] = spec.dim
            g.attrs['activation_type'] = spec.activation_type
            g.attrs['label'] = spec.label
            g.attrs['dynamics_rate'] = spec.dynamics_rate
            g.attrs['is_poisson'] = spec.is_poisson
            g.attrs['dropout_prob'] = float(spec.dropout_prob)
            g.attrs['activation_temperature'] = float(spec.activation_temperature)
            g.attrs['activation_num_winners'] = int(spec.activation_num_winners)

        # Predict connections
        pg = sg.create_group('predict_conns')
        for i, spec in enumerate(structure.predict_conns):
            g = pg.create_group(str(i))
            for field in spec._fields:
                _set_attr(g, field, getattr(spec, field))

        # Project connections
        prg = sg.create_group('project_conns')
        for i, spec in enumerate(structure.project_conns):
            g = prg.create_group(str(i))
            for field in spec._fields:
                _set_attr(g, field, getattr(spec, field))

        # Modulate connections
        mg = sg.create_group('modulate_conns')
        for i, spec in enumerate(structure.modulate_conns):
            g = mg.create_group(str(i))
            for field in spec._fields:
                _set_attr(g, field, getattr(spec, field))

    @staticmethod
    def _save_params(f, params: NetworkParams) -> None:
        """Write NetworkParams to HDF5 datasets."""
        pg = f.create_group('params')
        for name in ('predict_weights', 'predict_biases',
                     'project_weights', 'project_biases',
                     'modulate_weights', 'modulate_biases',
                     'precision_weights', 'precision_biases'):
            grp = pg.create_group(name)
            for i, arr in enumerate(getattr(params, name)):
                grp.create_dataset(str(i), data=np.asarray(arr, dtype=np.float32))

    @staticmethod
    def _save_results(f, results: Dict[str, Any]) -> None:
        """Write simulation results to HDF5."""
        rg = f.create_group('results')
        for key, value in results.items():
            if value is None:
                continue
            try:
                arr = np.asarray(value, dtype=np.float32)
                if arr.dtype.kind in ('f', 'i', 'u', 'b'):
                    rg.create_dataset(key, data=arr)
            except (ValueError, TypeError):
                # Skip non-numeric results that can't be converted
                pass

    def load(self, path: Union[str, Path]) -> 'PCNetwork':
        """Load network parameters from an HDF5 file.

        Restores ``self.params`` (predict/project/modulate weights and
        log_precisions) from the file.  The network must already be built so
        that structure is available for validation.

        Args:
            path: Path to the HDF5 file previously created by :meth:`save`.

        Returns:
            self for method chaining.

        Raises:
            RuntimeError: If the network has not been built yet.
            ValueError: If the saved structure doesn't match the current one.
        """
        import h5py

        if self.structure is None:
            raise RuntimeError(
                "Network must be built before loading parameters. "
                "Call net.build() first."
            )

        path = Path(path)
        with h5py.File(path, 'r') as f:
            # Validate structure matches
            saved_dims = tuple(f['structure'].attrs['layer_dims'])
            if saved_dims != self.structure.layer_dims:
                raise ValueError(
                    f"Layer dims mismatch: saved {saved_dims} vs "
                    f"current {self.structure.layer_dims}"
                )

            self.params = self._load_params(f)

        return self

    @staticmethod
    def _load_params(f) -> NetworkParams:
        """Read NetworkParams from an HDF5 file."""
        pg = f['params']
        loaded = {}
        for name in ('predict_weights', 'predict_biases',
                     'project_weights', 'project_biases',
                     'modulate_weights', 'modulate_biases',
                     'precision_weights', 'precision_biases'):
            if name not in pg:
                # Backward compat: old files without modulate_biases/project_biases — default to dummy zeros
                if name == 'modulate_biases':
                    n_mw = len(pg.get('modulate_weights', {}))
                    loaded[name] = [jnp.zeros(1, dtype=jnp.float32) for _ in range(n_mw)]
                    continue
                if name == 'project_biases':
                    n_pw = len(pg.get('project_weights', {}))
                    loaded[name] = [jnp.zeros(1, dtype=jnp.float32) for _ in range(n_pw)]
                    continue
            grp = pg[name]
            arrays = []
            for i in range(len(grp)):
                arrays.append(jnp.array(grp[str(i)][...]))
            loaded[name] = arrays
        return NetworkParams(**loaded)
