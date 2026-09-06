"""
Layer and NodeRef classes for defining predictive coding network architecture.

Layers are the main building blocks of PCN networks. Each layer has value nodes.
Error and precision nodes belong to Predict connections, not layers.
NodeRef allows referencing specific nodes for Project and Modulate connections.
"""

from typing import TYPE_CHECKING, Optional
from .activations import Activation
from ..config import _DEFAULT

if TYPE_CHECKING:
    from .network import PCNetwork


class NodeRef:
    """
    Reference to a specific node (value, error, precision) within a layer or
    predict connection, optionally restricted to a slice of dimensions.

    Used as source/target for Project and Modulate connections, allowing
    fine-grained control over which nodes are connected.

    Attributes:
        owner: The Layer or Predict connection containing this node
        owner_type: 'layer' or 'predict'
        node_type: String identifying the node ('value', 'error', or 'precision')
        slice_bounds: Optional (start, stop) tuple restricting to a subset of dims
    """

    def __init__(self, owner, node_type: str, owner_type: str = 'layer',
                 slice_bounds=None):
        self.owner = owner
        self.node_type = node_type
        self.owner_type = owner_type
        self.slice_bounds = slice_bounds
        self.label = owner.label + node_type if owner.label else node_type

    @property
    def _full_dim(self) -> int:
        """Full dimensionality of the underlying node (before slicing)."""
        if self.owner_type == 'layer':
            return self.owner.dim
        else:
            return self.owner.post.dim

    @property
    def dim(self) -> int:
        """Dimensionality of this reference (respects slicing)."""
        if self.slice_bounds is not None:
            return self.slice_bounds[1] - self.slice_bounds[0]
        return self._full_dim

    @property
    def layer(self):
        """returns owner if it's a layer, raises otherwise."""
        if self.owner_type == 'layer':
            return self.owner
        raise AttributeError(
            f"NodeRef owner is a predict connection, not a layer. "
            f"Use .predict_conn instead."
        )

    @property
    def predict_conn(self):
        """Returns owner if it's a predict connection, raises otherwise."""
        if self.owner_type == 'predict':
            return self.owner
        raise AttributeError(
            f"NodeRef owner is a layer, not a predict connection. "
            f"Use .layer instead."
        )

    @property
    def node_type_id(self) -> int:
        """Numeric ID for the node type (used in backend)."""
        return {'value': 0, 'error': 1, 'precision': 2,
                'flow_to_pre': 3, 'flow_to_post': 4, 'perror': 5}[self.node_type]

    @property
    def activation(self):
        """The activation applied at this node.

        - ``layer.value.activation`` → ``Layer.activation`` (the input
          nonlinearity ``f`` applied when reading the value).
        - ``predict.error.activation`` → ``Predict.error_activation`` (the
          nonlinearity ``h`` applied to the raw residual).
        - ``predict.precision.activation`` → ``Predict.precision_activation``
          (the nonlinearity mapping raw log-precision parameters to a
          positive precision).

        Flow nodes (``flow_to_pre`` / ``flow_to_post``) have no
        associated activation and raise ``AttributeError``.
        """
        if self.owner_type == 'layer':
            return self.owner.activation
        if self.node_type == 'error':
            return self.owner.error_activation
        if self.node_type == 'precision':
            return self.owner.precision_activation
        raise AttributeError(
            f"NodeRef.activation is not defined for node_type "
            f"{self.node_type!r} on a predict connection")

    def _parse_key(self, key):
        """Parse an index/slice key into (start, stop) bounds."""
        full = self._full_dim
        if isinstance(key, int):
            if key < 0:
                key += full
            if not (0 <= key < full):
                raise IndexError(
                    f"Index {key} out of range for dimension {full}")
            return (key, key + 1)
        elif isinstance(key, slice):
            if key.step is not None:
                raise ValueError("Step is not supported in NodeRef slicing")
            start = key.start if key.start is not None else 0
            stop = key.stop if key.stop is not None else full
            if start < 0:
                start += full
            if stop < 0:
                stop += full
            start = max(0, min(start, full))
            stop = max(0, min(stop, full))
            if start >= stop:
                raise ValueError(
                    f"Empty slice [{start}:{stop}] for dimension {full}")
            return (start, stop)
        else:
            raise TypeError(
                f"NodeRef indices must be integers or slices, got {type(key).__name__}")

    def __getitem__(self, key):
        if self.slice_bounds is not None:
            raise ValueError("Cannot slice an already-sliced NodeRef")
        start, stop = self._parse_key(key)
        return NodeRef(self.owner, self.node_type, self.owner_type,
                       slice_bounds=(start, stop))

    def __repr__(self):
        if self.owner_type == 'layer':
            label = self.owner.label or f"layer_{self.owner._idx}"
        else:
            label = f"predict_{self.owner._idx}"
        s = f"NodeRef({label}.{self.node_type})"
        if self.slice_bounds is not None:
            s = f"NodeRef({label}.{self.node_type}[{self.slice_bounds[0]}:{self.slice_bounds[1]}])"
        return s


class Layer:
    """
    A layer in the predictive coding network.

    Each layer has:
        - value: Current activity/representation (batch, dim)
        - f: Activation function applied to value before sending predictions

    Layers must be created inside a PCNetwork context manager:

        net = PCNetwork()
        with net:
            l1 = Layer(dim=128, label="input")
            l2 = Layer(dim=64, activation=Relu(), label="hidden")

    Attributes:
        dim: Dimensionality of the layer
        activation: Activation function class instance
        label: Unique identifier (auto-generated if not provided)

    After build():
        _idx: Index in the network's layer list
        _network: Reference to parent PCNetwork
    """

    def __init__(
        self,
        dim: int,
        activation=_DEFAULT,
        dynamics_rate=_DEFAULT,
        spatial_structure=_DEFAULT,
        inference_reg=None,
        train_reg=None,
        dropout_prob: float = 0.0,
        label: Optional[str] = None
    ):
        from .network import _get_current_network
        from .activations import activation_from_name
        net = _get_current_network()
        defaults = net._defaults

        self.dim = int(dim)

        if activation is not _DEFAULT:
            self.activation = activation
        else:
            act = defaults['activation']
            self.activation = activation_from_name(act) if isinstance(act, str) else act

        self.dynamics_rate = dynamics_rate if dynamics_rate is not _DEFAULT else defaults['dynamics_rate']
        self.spatial_structure = spatial_structure if spatial_structure is not _DEFAULT else defaults.get('spatial_structure', 'none')
        self.inference_reg = inference_reg
        self.train_reg = train_reg
        # Per-layer dropout probability applied to the *activated value* before
        # it is consumed by downstream predict connections. Only active during
        # learning (training); during sim.test the value passes through
        # unchanged. 0 disables. See `pcn.backend.simulation` for the
        # sampling dispatch (parallel to is_poisson).
        dp = float(dropout_prob)
        if not 0.0 <= dp < 1.0:
            raise ValueError(f"dropout_prob must be in [0, 1), got {dp}")
        self.dropout_prob = dp
        self.label = label

        # Populated during build() for spatial layers
        self.locations = None
        self.neighborhood = None

        self._idx: Optional[int] = None
        self._network: Optional['PCNetwork'] = None

        # Register with current network context
        net._add_layer(self)

    @property
    def value(self) -> NodeRef:
        """Reference to this layer's value node."""
        return NodeRef(self, 'value', owner_type='layer')

    @property
    def f(self) -> Activation:
        """The activation function (alias for self.activation)."""
        return self.activation

    def __getitem__(self, key):
        """Slice this layer's value node: ``layer[3:7]`` == ``layer.value[3:7]``."""
        return self.value[key]

    def __repr__(self):
        label = self.label or f"layer_{self._idx}"
        return f"Layer(dim={self.dim}, activation={self.activation}, label='{label}')"
