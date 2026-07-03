"""
Connection classes for predictive coding networks.

Three types of connections:
- Predict: Standard PC connection (layer -> layer), owns error and precision nodes
- Project: Additive non-PC connection (node -> node)
- Modulate: Multiplicative non-PC connection (node -> node)

All connection types accept a ``transformation`` parameter:
- 'linear' (default): dense matmul ``W @ f(pre)`` (+ bias if enabled).
- 'linear-<activation>': dense matmul wrapped in a post-nonlinearity,
    ``g(W f(pre) + b)``. ``<activation>`` is any name in
    :data:`pcn.core.activations.ACTIVATION_REGISTRY` (e.g. ``'linear-softplus'``,
    ``'linear-exp'``, ``'linear-relu'``).
- 'conv': spatial convolution. Optionally accepts ``weight_mask`` with shape
    ``(out_channels, in_channels, kH, kW)`` applied element-wise to the kernel
    at init and after every weight update.
- 'transconv': transposed convolution (upsampling). Same optional ``weight_mask``
    semantics as 'conv'.
- 'banded{N}': band-diagonal mask with half-bandwidth N (e.g. 'banded5').
- 'masked': dense matmul with a user-supplied ``weight_mask`` array of shape
    ``(post_dim, pre_dim)`` element-wise multiplied into ``W`` at init and
    after every weight update.

Predict, Project, and Modulate can all take a *list* for ``pre``,
in which case the pre values are concatenated before the transform.
Project and Modulate additionally accept plain Layer objects for
``pre``/``post`` (treated as ``layer.value``).
"""

from typing import TYPE_CHECKING, Optional, Union, List
import numpy as np

from .layer import Layer, NodeRef
from .learning_rules import LearningRule, Hebbian
from .activations import Activation, Direct, activation_from_name
from ..config import _DEFAULT


def _resolve_activation(value, default_instance):
    """Coerce a kwarg value into an Activation instance.

    Accepts an Activation instance, a string name (looked up via
    :func:`activation_from_name`), ``None`` (returns ``default_instance``),
    or ``_DEFAULT`` (also returns ``default_instance``).
    """
    if value is _DEFAULT or value is None:
        return default_instance
    if isinstance(value, str):
        return activation_from_name(value)
    if isinstance(value, Activation):
        return value
    raise TypeError(
        f"activation must be an Activation instance, a name string, or None; "
        f"got {type(value).__name__}")

if TYPE_CHECKING:
    from .network import PCNetwork


# ============================================================================
# Shared transformation setup
# ============================================================================

def _parse_conv_params(pre_dim, post_dim, kernel_size, input_shape,
                       stride, padding, is_transconv=False):
    """Parse convolution parameters and compute output shape + channels.

    Returns a dict with: in_channels, out_channels, kernel_size, stride,
    padding, input_shape, output_shape.
    """
    kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else tuple(kernel_size)
    stride = (stride, stride) if isinstance(stride, int) else tuple(stride)
    input_shape = tuple(input_shape)
    if isinstance(padding, int):
        padding = ((padding, padding), (padding, padding))

    H_in, W_in = input_shape
    kH, kW = kernel_size
    sH, sW = stride

    if is_transconv:
        if padding == 'SAME':
            H_out = H_in * sH
            W_out = W_in * sW
        elif padding == 'VALID':
            H_out = (H_in - 1) * sH + kH
            W_out = (W_in - 1) * sW + kW
        else:
            pH_lo, pH_hi = padding[0]
            pW_lo, pW_hi = padding[1]
            H_out = (H_in - 1) * sH + kH - pH_lo - pH_hi
            W_out = (W_in - 1) * sW + kW - pW_lo - pW_hi
    else:
        if padding == 'SAME':
            H_out = -(-H_in // sH)
            W_out = -(-W_in // sW)
        elif padding == 'VALID':
            H_out = (H_in - kH) // sH + 1
            W_out = (W_in - kW) // sW + 1
        else:
            pH_lo, pH_hi = padding[0]
            pW_lo, pW_hi = padding[1]
            H_out = (H_in + pH_lo + pH_hi - kH) // sH + 1
            W_out = (W_in + pW_lo + pW_hi - kW) // sW + 1

    output_shape = (H_out, W_out)
    spatial_in = H_in * W_in
    spatial_out = H_out * W_out

    if pre_dim % spatial_in != 0:
        raise ValueError(
            f"pre_dim={pre_dim} not divisible by input spatial "
            f"{H_in}*{W_in}={spatial_in}")
    if post_dim % spatial_out != 0:
        raise ValueError(
            f"post_dim={post_dim} not divisible by output spatial "
            f"{H_out}*{W_out}={spatial_out}")

    return {
        'in_channels': pre_dim // spatial_in,
        'out_channels': post_dim // spatial_out,
        'kernel_size': kernel_size,
        'stride': stride,
        'padding': padding,
        'input_shape': input_shape,
        'output_shape': output_shape,
    }


def _attach_kernel_mask(obj, weight_mask):
    """If a kernel-shaped ``weight_mask`` was supplied, validate and store it.

    ``obj`` must already have ``out_channels``, ``in_channels``, and
    ``kernel_size`` set (i.e. after ``_parse_conv_params``). Mask shape must
    match ``(out_channels, in_channels, kH, kW)``.
    """
    if weight_mask is None:
        return
    kH, kW = obj.kernel_size
    expected = (obj.out_channels, obj.in_channels, kH, kW)
    mask = np.asarray(weight_mask, dtype=np.float32)
    if mask.shape != expected:
        raise ValueError(
            f"weight_mask shape {mask.shape} does not match expected "
            f"kernel shape {expected}.")
    obj.is_masked = True
    obj.weight_mask = mask


def _setup_transform(obj, transformation, pre_dim, post_dim,
                     kernel_size=None, input_shape=None,
                     stride=None, padding=None, weight_mask=None):
    """Parse a transformation string and set transform attributes on ``obj``.

    Sets: ``is_conv``, ``is_transconv``, ``n_bands``, ``is_masked``,
    ``weight_mask``, ``post_activation_type_id``, and (for conv/transconv)
    ``in_channels``, ``out_channels``, ``kernel_size``, ``stride``,
    ``padding``, ``input_shape``, ``output_shape``.

    ``post_activation_type_id`` is the ``ACTIVATIONS`` index for the
    post-transform nonlinearity g in ``g(W f(x) + b)``. It defaults to 0
    (Direct = identity) and is set to a real activation when
    ``transformation='linear-<name>'``.

    For ``transformation='masked'`` the caller must supply ``weight_mask``
    with shape ``(post_dim, pre_dim)``. The mask is stored on the object
    and applied multiplicatively to ``W`` at init and after every weight
    update.

    For ``transformation='conv'`` / ``'transconv'`` a ``weight_mask`` is
    optional. If provided, it must have shape
    ``(out_channels, in_channels, kH, kW)`` and is applied multiplicatively
    to the kernel at init and after every weight update.
    """
    obj.is_conv = False
    obj.is_transconv = False
    obj.is_masked = False
    obj.weight_mask = None
    obj.post_activation_type_id = 0  # Direct by default

    if transformation == 'linear':
        pass
    elif transformation.startswith('linear-'):
        name = transformation[len('linear-'):]
        try:
            act = activation_from_name(name)
        except ValueError as e:
            raise ValueError(
                f"Invalid transformation '{transformation}': {e}") from None
        obj.post_activation_type_id = act.type_id
    elif transformation == 'conv':
        if kernel_size is None or input_shape is None:
            raise ValueError("'conv' transformation requires kernel_size and input_shape")
        obj.is_conv = True
        info = _parse_conv_params(
            pre_dim, post_dim, kernel_size, input_shape,
            stride if stride is not None else 1,
            padding if padding is not None else 0,
            is_transconv=False)
        for k, v in info.items():
            setattr(obj, k, v)
        _attach_kernel_mask(obj, weight_mask)
    elif transformation == 'transconv':
        if kernel_size is None or input_shape is None:
            raise ValueError("'transconv' transformation requires kernel_size and input_shape")
        obj.is_transconv = True
        info = _parse_conv_params(
            pre_dim, post_dim, kernel_size, input_shape,
            stride if stride is not None else 2,
            padding if padding is not None else 0,
            is_transconv=True)
        for k, v in info.items():
            setattr(obj, k, v)
        _attach_kernel_mask(obj, weight_mask)
    elif transformation.startswith('banded'):
        try:
            obj.n_bands = int(transformation[6:])
        except ValueError:
            raise ValueError(
                f"Invalid banded transformation '{transformation}'. "
                f"Use format 'banded{{N}}' e.g. 'banded5'.")
    elif transformation == 'masked':
        if weight_mask is None:
            raise ValueError(
                "'masked' transformation requires a weight_mask argument "
                f"of shape ({post_dim}, {pre_dim}).")
        mask = np.asarray(weight_mask, dtype=np.float32)
        if mask.shape != (post_dim, pre_dim):
            raise ValueError(
                f"weight_mask shape {mask.shape} does not match expected "
                f"({post_dim}, {pre_dim}).")
        obj.is_masked = True
        obj.weight_mask = mask
    else:
        raise ValueError(
            f"Unknown transformation '{transformation}'. "
            f"Choices: 'linear', 'linear-<activation>', 'conv', 'transconv', "
            f"'banded{{N}}', 'masked'.")


# ============================================================================
# Resolve helpers for NodeRef
# ============================================================================

def _resolve_owner_idx(node_ref: NodeRef) -> int:
    """Get the index for a NodeRef based on its owner type."""
    return node_ref.owner._idx


def _resolve_owner_dim(node_ref: NodeRef) -> int:
    """Get the dimensionality for a NodeRef (respects slice_bounds)."""
    return node_ref.dim


def _resolve_owner_label(node_ref: NodeRef) -> str:
    """Get a label for a NodeRef based on its owner type."""
    if node_ref.owner_type == 'layer':
        return node_ref.owner.label or f"layer_{node_ref.owner._idx}"
    else:
        return f"predict_{node_ref.owner._idx}"


def _normalize_pre_layers(pre):
    """Normalize a pre argument to (layers, slices).

    Accepts Layer, sliced NodeRef (from ``layer[3:7]``), or a list mixing both.
    Returns a list of Layer objects and a parallel tuple of slice_bounds
    (each ``(start, stop)`` or ``None``).
    """
    if not isinstance(pre, (list, tuple)):
        pre = [pre]
    layers = []
    slices = []
    for p in pre:
        if isinstance(p, Layer):
            layers.append(p)
            slices.append(None)
        elif isinstance(p, NodeRef):
            if p.node_type != 'value' or p.owner_type != 'layer':
                raise ValueError(
                    "Predict pre must be Layers or sliced layer values, "
                    f"got {p!r}")
            layers.append(p.owner)
            slices.append(p.slice_bounds)  # None if unsliced
        else:
            raise TypeError(f"Expected Layer or NodeRef for pre, got {type(p).__name__}")
    return layers, tuple(slices)


def _normalize_pre_noderefs(pre):
    """Normalize a pre argument (NodeRef, Layer, or list) to a list of NodeRefs.

    Layer objects are converted to layer.value.
    """
    if isinstance(pre, (list, tuple)):
        return [p.value if isinstance(p, Layer) else p for p in pre]
    if isinstance(pre, Layer):
        return [pre.value]
    return [pre]


def _normalize_precision_input(precision_input):
    """Normalize and validate a Predict ``precision_input`` argument.

    Accepts a Layer, a NodeRef (value/error/precision, optionally sliced),
    or a list mixing both. Returns a list of NodeRefs, or None when
    ``precision_input`` is None (default: precision keyed on the conn's pre).
    """
    if precision_input is None:
        return None
    refs = _normalize_pre_noderefs(precision_input)
    if not refs:
        raise ValueError("precision_input must not be an empty list")
    for r in refs:
        if not isinstance(r, NodeRef):
            raise TypeError(
                "precision_input entries must be Layers or NodeRefs, "
                f"got {type(r).__name__}")
        if r.node_type_id not in (0, 1, 2):
            raise ValueError(
                "precision_input sources must be value, error, or precision "
                f"nodes; flow nodes are not allowed, got {r!r}")
    return refs


def _precision_source_dim(ref: NodeRef) -> int:
    """Runtime feature dim of a precision-input source array.

    NodeRef.dim is wrong for two predict-owned cases: error arrays follow the
    conn's (possibly sliced) post_dim, and a conn with both precision learn
    flags off carries a (batch, 1) precision regardless of post_dim.
    """
    if ref.slice_bounds is not None:
        return ref.slice_bounds[1] - ref.slice_bounds[0]
    if ref.owner_type == 'layer':
        return ref.owner.dim
    conn = ref.owner
    if ref.node_type == 'precision' and not (
            conn.learn_precision_weights or conn.learn_precision_bias):
        return 1
    return conn.post_dim


# ============================================================================
# Predict
# ============================================================================

class Predict:
    """
    Standard predictive coding connection.

    The pre layer(s) predict the post layer:
        prediction = W @ f(pre.value)
        error = precision * (post.value - prediction)

    Each Predict connection owns its own error and precision nodes.

    Args:
        pre: Source layer(s). A single Layer or a list of Layers.
            For multi-pre, values are concatenated before the transform.
        post: Target layer being predicted.
        transformation: Transform type. One of 'linear' (default),
            'linear-<activation>' (e.g. ``'linear-softplus'``, ``'linear-relu'``,
            ``'linear-exp'``), 'conv', 'transconv', 'banded{N}' (e.g.
            'banded5'), or 'masked'. ``'linear-<activation>'`` wraps the matmul
            in a post-nonlinearity ``g(W f(x) + b)``. ``'masked'`` requires the
            ``weight_mask`` argument.
        kernel_size, input_shape, stride, padding: Required for conv/transconv.
        weight_mask: Required when ``transformation='masked'`` with shape
            ``(post_dim, pre_dim)``. Optional for ``'conv'`` / ``'transconv'``
            with shape ``(out_channels, in_channels, kH, kW)``. Element-wise
            multiplied into ``W`` at init and after every weight update.
            Entries do not need to be binary.
        order: Execution order (lower = earlier). If None, uses definition order.
        init_weight: If provided, weights are fixed to this value and not learned.
        init_precision: Initial precision value (in precision-space, not log).
            The precision bias is initialized such that, under the chosen
            ``precision_parameterization``, the starting precision equals this
            value. Default from config (``1.0``).
        init_log_precision: Deprecated alias — if provided, converted via
            ``init_precision = exp(init_log_precision)``. Kept for backwards
            compatibility.
        learn_precision: Shorthand to enable/disable learning of both precision
            weights and biases. Expands to ``learn_precision_weights`` and
            ``learn_precision_bias``; individual flags take priority.
        learn_precision_weights: Whether to learn input-dependent precision
            (weights of the log-precision function). Default from config.
        learn_precision_bias: Whether to learn per-dimension precision offsets
            (bias of the log-precision function). Default from config.
        error_activation: Activation applied to the raw residual
            ``post_val - prediction`` before it is used downstream (energy,
            Project/Modulate reads, logging). Defaults to ``Direct`` (identity,
            preserving the standard PC residual). Accepts an
            :class:`pcn.core.activations.Activation` instance or a registry
            name (see :data:`pcn.core.activations.ACTIVATION_REGISTRY`).
            **Note:** any non-identity choice changes the energy from the
            standard Gaussian variational free energy.
        precision_activation: Activation function used to map raw precision
            parameters to a positive precision. Replaces / supersedes
            ``precision_parameterization`` (both names are accepted; the new
            name wins if both are given). Any activation name in
            :data:`pcn.core.activations.ACTIVATION_REGISTRY` is accepted, but
            ``'softplus'`` (default; bounded gradient), ``'exp'`` (legacy) and
            ``'linear'`` (identity, no nonlinearity) are the recommended choices
            because they admit a closed-form inverse used to initialise the
            precision bias from ``init_precision``. Other activations are
            accepted; the bias is then initialised raw (no inverse applied).
        precision_parameterization: Deprecated alias for ``precision_activation``.
        precision_input: Optional custom source(s) for the precision function.
            Default ``None`` keys the precision on this connection's own pre
            activation (historical behaviour). Otherwise a Layer, a NodeRef
            (a layer value — possibly sliced, another Predict's ``.error`` or
            ``.precision``), or a list mixing them; multi-source inputs are
            concatenated. **Replaces** the default pre source — include the
            pre layer in the list to augment rather than replace. Value
            sources are read live at the current iteration (the energy
            gradient flows into their inference dynamics, exactly like the
            default pre read). Error/precision sources read the *previous*
            iteration's carried arrays (Jacobi rule), so any reference is
            acyclic — a connection may even read its own error
            (``p.precision_input = [p.error]`` assigned after construction;
            resolution happens at build time).

    Precision learning rate:
        Precision parameters are updated by the same optimizer as predict
        weights. To use a separate learning rate for precision, build the
        optimizer with :meth:`PCNetwork.multi_transform`::

            net.build()
            optimizer = net.multi_transform(
                {'precision': optax.adam(1e-5)},
                default_optim=optax.adam(1e-3),
            )
            sim.train(..., params_optimizer=optimizer)

        This gives precision parameters a lower learning rate while predict
        weights use the default. See :meth:`PCNetwork.multi_transform` for
        the full matching priority (connection labels, param types, etc.).
    """

    def __init__(
        self,
        pre: Union[Layer, List[Layer]],
        post: Layer,
        label: Optional[str] = None,
        order: Optional[int] = None,
        init_weight: Optional[np.ndarray] = None,
        init_bias: Optional[np.ndarray] = None,
        use_bias=_DEFAULT,
        init_precision=_DEFAULT,
        init_log_precision=_DEFAULT,
        learn_precision=_DEFAULT,
        learn_precision_weights=_DEFAULT,
        learn_precision_bias=_DEFAULT,
        precision_activation=_DEFAULT,
        precision_parameterization=_DEFAULT,
        precision_input_norm: bool = False,
        precision_input=None,
        error_activation=_DEFAULT,
        alpha=_DEFAULT,
        n_bands: int = 0,
        transformation: str = 'linear',
        kernel_size=None,
        input_shape=None,
        stride=None,
        padding=None,
        weight_mask=None,
        stochastic: bool = True,
    ):
        from .network import _get_current_network
        net = _get_current_network()
        defaults = net._defaults

        # Normalize pre to list of Layers + parallel slice bounds
        self.pre, self.pre_slices = _normalize_pre_layers(pre)

        # Normalize post (may be a sliced NodeRef from layer[:5])
        if isinstance(post, NodeRef):
            if post.node_type != 'value' or post.owner_type != 'layer':
                raise ValueError(
                    f"Predict post must be a Layer or sliced layer value, got {post!r}")
            self.post = post.owner
            self.post_slice = post.slice_bounds
        else:
            self.post = post
            self.post_slice = None

        self.label = label
        self.order = order
        self.weight = init_weight
        self.bias = init_bias
        self.n_bands = n_bands
        self.stochastic = stochastic   # per-conn is_stochastic noise gate
        self.use_bias = use_bias if use_bias is not _DEFAULT else defaults['use_bias']

        # Resolve init_precision. ``init_log_precision`` is accepted for
        # backwards compatibility: if provided, it is converted via exp.
        # An explicit ``init_precision`` kwarg wins over ``init_log_precision``.
        if init_precision is not _DEFAULT:
            self.init_precision = float(init_precision)
        elif init_log_precision is not _DEFAULT:
            self.init_precision = float(np.exp(init_log_precision))
        elif 'init_precision' in defaults:
            self.init_precision = float(defaults['init_precision'])
        else:
            # Fallback: legacy config files may still specify init_log_precision
            self.init_precision = float(np.exp(defaults['init_log_precision']))

        # Resolve learn_precision_weights and learn_precision_bias.
        # learn_precision is a shorthand: if set and neither individual flag is
        # provided, it expands to both. Individual flags always take priority.
        if learn_precision_weights is not _DEFAULT:
            self.learn_precision_weights = learn_precision_weights
        elif learn_precision is not _DEFAULT:
            self.learn_precision_weights = learn_precision
        else:
            self.learn_precision_weights = defaults.get('learn_precision_weights',
                                                        defaults.get('learn_precision', True))

        if learn_precision_bias is not _DEFAULT:
            self.learn_precision_bias = learn_precision_bias
        elif learn_precision is not _DEFAULT:
            self.learn_precision_bias = learn_precision
        else:
            self.learn_precision_bias = defaults.get('learn_precision_bias',
                                                     defaults.get('learn_precision', True))
        self.alpha = alpha if alpha is not _DEFAULT else defaults['alpha']

        # When True, precision = softplus(W_ρ @ standardize(pre) + b): the
        # pre-activation is per-sample standardised (stop_gradient) inside the
        # precision parameterisation only. Lets precision=f(pre) read an O(1)
        # signal from a magnitude-collapsed latent without destabilising the
        # forward dynamics (exp-multimodal-alphanum-gen-05 phase 3).
        self.precision_input_norm = bool(precision_input_norm)

        # Precision activation: new ``precision_activation`` kwarg supersedes
        # the legacy ``precision_parameterization``. If neither is provided,
        # fall back to the network default (registered under either name).
        if precision_activation is not _DEFAULT:
            pp = precision_activation
        elif precision_parameterization is not _DEFAULT:
            pp = precision_parameterization
        else:
            pp = defaults.get('precision_activation',
                              defaults.get('precision_parameterization', 'softplus'))
        try:
            self.precision_activation = _resolve_activation(pp, default_instance=None)
        except ValueError as e:
            raise ValueError(
                f"Unknown precision_activation '{pp}': {e}") from None
        if self.precision_activation is None:
            raise ValueError("precision_activation must not be None")
        self.precision_activation_type = self.precision_activation.type_id

        # Error activation: defaults to Direct (identity) -- preserves the
        # standard PC residual semantics when not configured.
        ea = error_activation if error_activation is not _DEFAULT else defaults.get('error_activation', None)
        try:
            self.error_activation = _resolve_activation(ea, default_instance=Direct())
        except ValueError as e:
            raise ValueError(
                f"Unknown error_activation '{ea}': {e}") from None
        self.error_activation_type = self.error_activation.type_id

        # Custom precision sources. Stored raw and re-normalized at build time
        # (see _resolved_precision_input) so it can be assigned after
        # construction, e.g. ``p.precision_input = [p.error]`` for a
        # connection reading its own error.
        _normalize_precision_input(precision_input)  # early validation
        self.precision_input = precision_input

        # Compute combined pre dim (respects slicing)
        self.pre_dim = sum(
            (s[1] - s[0]) if s is not None else p.dim
            for p, s in zip(self.pre, self.pre_slices)
        )
        # Post dim (respects slicing)
        self.post_dim = (self.post_slice[1] - self.post_slice[0]) if self.post_slice else self.post.dim

        # Set up transformation (sets is_conv, is_transconv, n_bands, conv params)
        self.is_res = getattr(self, 'is_res', False)
        _setup_transform(self, transformation, self.pre_dim, self.post_dim,
                         kernel_size, input_shape, stride, padding,
                         weight_mask=weight_mask)

        # Validate: slicing incompatible with conv/transconv
        has_slicing = any(s is not None for s in self.pre_slices) or self.post_slice is not None
        if has_slicing and (self.is_conv or self.is_transconv):
            raise ValueError(
                "Slicing is not compatible with conv/transconv transforms")

        self._idx: Optional[int] = None
        self._default_order: int = 0

        # Register with current network context
        net._add_predict(self)

    @property
    def precision_param_type(self) -> int:
        """Deprecated alias for ``precision_activation_type``."""
        return self.precision_activation_type

    @property
    def learn_precision(self) -> bool:
        """Convenience: True if both weights and bias are learned."""
        return self.learn_precision_weights and self.learn_precision_bias

    def _resolved_precision_input(self):
        """Normalized + validated precision_input as a list of NodeRefs.

        None means the default (precision keyed on this conn's pre).
        Re-normalizes on every call because ``precision_input`` may be
        assigned after construction.
        """
        return _normalize_precision_input(self.precision_input)

    @property
    def precision_input_dim(self) -> int:
        """Feature dim of the precision function's input (default: pre_dim)."""
        refs = self._resolved_precision_input()
        if refs is None:
            return self.pre_dim
        return sum(_precision_source_dim(r) for r in refs)

    @property
    def init_log_precision(self) -> float:
        """Backwards-compat view of ``init_precision`` as ``log(init_precision)``."""
        return float(np.log(self.init_precision))

    @property
    def error(self) -> NodeRef:
        """Reference to this connection's error node."""
        return NodeRef(self, 'error', owner_type='predict')

    @property
    def precision(self) -> NodeRef:
        """Reference to this connection's precision node."""
        return NodeRef(self, 'precision', owner_type='predict')

    @property
    def flow_to_pre(self) -> NodeRef:
        """Gate for ascending error pathway (error -> pre value gradient)."""
        return NodeRef(self, 'flow_to_pre', owner_type='predict')

    @property
    def flow_to_post(self) -> NodeRef:
        """Gate for descending error pathway (error -> post value gradient)."""
        return NodeRef(self, 'flow_to_post', owner_type='predict')

    def __repr__(self):
        if len(self.pre) == 1:
            pre_label = self.pre[0].label or f"layer_{self.pre[0]._idx}"
        else:
            pre_label = '+'.join(p.label or f"layer_{p._idx}" for p in self.pre)
        post_label = self.post.label or f"layer_{self.post._idx}"
        return f"Predict({pre_label} -> {post_label})"


class PredictRes(Predict):
    """
    Residual predictive coding connection.

    Like Predict, but adds a skip connection: prediction = W @ f(pre.value) + pre.value

    Requires pre.dim == post.dim (single pre only).
    """

    def __init__(self, pre, post, **kwargs):
        if isinstance(pre, (list, tuple)):
            raise ValueError("PredictRes does not support multi-pre")
        pre_dim = pre.dim if isinstance(pre, Layer) else pre.dim
        post_dim = post.dim if isinstance(post, Layer) else post.dim
        if pre_dim != post_dim:
            raise ValueError(
                f"PredictRes requires pre.dim == post.dim, "
                f"got pre.dim={pre_dim}, post.dim={post_dim}"
            )
        self.is_res = True
        super().__init__(pre, post, **kwargs)


class PredictConv(Predict):
    """
    Convolutional predictive coding connection.

    Convenience wrapper for ``Predict(..., transformation='conv', ...)``.
    """

    def __init__(self, pre, post, kernel_size, input_shape,
                 stride=1, padding=0, **kwargs):
        super().__init__(pre, post, transformation='conv',
                         kernel_size=kernel_size, input_shape=input_shape,
                         stride=stride, padding=padding, **kwargs)


class PredictTransConv(Predict):
    """
    Transposed convolutional predictive coding connection.

    Convenience wrapper for ``Predict(..., transformation='transconv', ...)``.
    """

    def __init__(self, pre, post, kernel_size, input_shape,
                 stride=2, padding=0, **kwargs):
        super().__init__(pre, post, transformation='transconv',
                         kernel_size=kernel_size, input_shape=input_shape,
                         stride=stride, padding=padding, **kwargs)


# ============================================================================
# Project
# ============================================================================

class Project:
    """
    Additive non-PC connection.

    Adds a weighted projection to a target node:
        target += W @ f(source)

    Args:
        pre: Source node(s). A NodeRef, Layer (treated as layer.value),
            or list thereof. For multi-pre, values are concatenated.
            All pre nodes must share the same node type (value or error).
        post: Target node. A NodeRef or Layer (treated as layer.value).
        transformation: Transform type ('linear', 'linear-<activation>' for
            ``g(W f(x) + b)`` post-nonlinearity, 'conv', 'transconv',
            'banded{N}', 'masked'). ``'masked'`` requires ``weight_mask``.
        update_rule: Learning rule for weight updates (default: Hebbian).
        order: Execution order (lower = earlier).
        init_weight: Optional initial weight matrix. The connection still learns
            from this starting point (unlike Predict, learning is not disabled).
        init_bias: Optional initial bias vector (only used if ``use_bias`` is True).
        use_bias: Whether to add a bias term: ``target += W @ f(pre) + b``.
            Defaults to the network-level ``use_bias`` config setting.
        weight_mask: Required when ``transformation='masked'`` (shape
            ``(post_dim, pre_dim)``); optional for ``'conv'`` / ``'transconv'``
            (shape ``(out_channels, in_channels, kH, kW)``). Element-wise
            multiplied into ``W`` at init and after every weight update.

    Example:
        Project(l4.value, l2.value, update_rule=Hebbian(learning_rate=1e-4))
        Project([l1, l2], l3.value)  # multi-pre, Layer input
    """

    def __init__(
        self,
        pre: Union[NodeRef, Layer, List[Union[NodeRef, Layer]]],
        post: Union[NodeRef, Layer],
        update_rule: Optional[LearningRule] = None,
        order: Optional[int] = None,
        init_weight: Optional[np.ndarray] = None,
        init_bias: Optional[np.ndarray] = None,
        use_bias=_DEFAULT,
        label: Optional[str] = None,
        transformation: str = 'linear',
        kernel_size=None,
        input_shape=None,
        stride=None,
        padding=None,
        weight_mask=None,
    ):
        from .network import _get_current_network
        net = _get_current_network()
        defaults = net._defaults

        self.label = label
        self.use_bias = use_bias if use_bias is not _DEFAULT else defaults['use_bias']

        # Normalize pre to list of NodeRefs (Layer -> layer.value)
        self._pre_list = _normalize_pre_noderefs(pre)
        # Normalize post (Layer -> layer.value)
        self.post = post.value if isinstance(post, Layer) else post
        # Keep .pre for backward compat (single-pre returns the NodeRef)
        self.pre = self._pre_list[0] if len(self._pre_list) == 1 else self._pre_list[0]

        # Extract slice bounds
        self.pre_slices = tuple(p.slice_bounds for p in self._pre_list)
        self.post_slice = self.post.slice_bounds

        # Validate: all pre must have the same node type
        node_types = set(p.node_type_id for p in self._pre_list)
        if len(node_types) > 1:
            raise ValueError(
                "All pre nodes in a Project connection must have the same "
                f"node type, got: {[p.node_type for p in self._pre_list]}")

        # Validate: flow nodes cannot be pre source
        pre_type = self._pre_list[0].node_type_id
        if pre_type in (3, 4):
            raise ValueError(
                "Flow nodes (flow_to_pre, flow_to_post) cannot be used as "
                "pre source for Project connections")

        # Validate: flow nodes cannot be post target for Project (use Modulate)
        post_type = self.post.node_type_id
        if post_type in (3, 4):
            raise ValueError(
                "Flow nodes (flow_to_pre, flow_to_post) can only be targeted "
                "by Modulate connections, not Project")

        if update_rule is not None:
            self.update_rule = update_rule
        else:
            lr = defaults['hebbian_learning_rate']
            self.update_rule = Hebbian(learning_rate=lr)
        self.order = order
        self.weight = init_weight
        self.bias = init_bias

        self._idx: Optional[int] = None
        self._default_order: int = 0
        self.reward_fn_idx: int = -1
        self.loss_fn_idx: int = -1

        # Derived properties for backend
        self.pre_layer_idxs: tuple = tuple(_resolve_owner_idx(p) for p in self._pre_list)
        self.pre_node_type: int = self._pre_list[0].node_type_id
        self.post_layer_idx: Optional[int] = _resolve_owner_idx(self.post)
        self.post_node_type: int = self.post.node_type_id

        # Store dimensions for weight initialization (respects slicing)
        self.pre_dim: int = sum(_resolve_owner_dim(p) for p in self._pre_list)
        self.post_dim: int = _resolve_owner_dim(self.post)

        # Set up transformation
        self.n_bands = 0
        _setup_transform(self, transformation, self.pre_dim, self.post_dim,
                         kernel_size, input_shape, stride, padding,
                         weight_mask=weight_mask)

        # Validate: slicing incompatible with conv/transconv
        has_slicing = any(s is not None for s in self.pre_slices) or self.post_slice is not None
        if has_slicing and (self.is_conv or self.is_transconv):
            raise ValueError(
                "Slicing is not compatible with conv/transconv transforms")

        # Register with current network context
        net._add_project(self)

    def __repr__(self):
        if len(self._pre_list) == 1:
            pre_label = _resolve_owner_label(self._pre_list[0])
        else:
            pre_label = '+'.join(_resolve_owner_label(p) for p in self._pre_list)
        post_label = _resolve_owner_label(self.post)
        return f"Project({pre_label}.{self._pre_list[0].node_type} -> {post_label}.{self.post.node_type})"


# ============================================================================
# Modulate
# ============================================================================

class Modulate:
    """
    Multiplicative non-PC connection (neuromodulation).

    Multiplies a target node by a weighted projection:
        target *= W @ f(source)

    Args:
        pre: Source node(s). A NodeRef, Layer (treated as layer.value),
            or list thereof. For multi-pre, values are concatenated.
            All pre nodes must share the same node type (value or error).
        post: Target node. A NodeRef or Layer (treated as layer.value).
        transformation: Transform type ('linear', 'linear-<activation>' for
            ``g(W f(x) + b)`` post-nonlinearity, 'conv', 'transconv',
            'banded{N}', 'masked'). ``'masked'`` requires ``weight_mask``.
        update_rule: Learning rule for weight updates (default: Hebbian).
        order: Execution order (lower = earlier).
        init_weight: Optional initial weight matrix. The connection still learns
            from this starting point (unlike Predict, learning is not disabled).
        init_bias: Optional initial bias vector (only used if ``use_bias`` is True).
        use_bias: Whether to add a bias term: ``modulation = W @ f(pre) + b``.
            Defaults to the network-level ``use_bias`` config setting.
        weight_mask: Required when ``transformation='masked'`` (shape
            ``(post_dim, pre_dim)``); optional for ``'conv'`` / ``'transconv'``
            (shape ``(out_channels, in_channels, kH, kW)``). Element-wise
            multiplied into ``W`` at init and after every weight update.

    Example:
        Modulate(l3.value, p2.error, update_rule=ThreeFactorHebbian(...))
        Modulate(l3, p2.error)  # Layer input, same as l3.value
    """

    def __init__(
        self,
        pre: Union[NodeRef, Layer, List[Union[NodeRef, Layer]]],
        post: Union[NodeRef, Layer],
        update_rule: Optional[LearningRule] = None,
        order: Optional[int] = None,
        init_weight: Optional[np.ndarray] = None,
        init_bias: Optional[np.ndarray] = None,
        label: Optional[str] = None,
        transformation: str = 'linear',
        kernel_size=None,
        input_shape=None,
        stride=None,
        padding=None,
        weight_mask=None,
        use_bias=_DEFAULT,
    ):
        from .network import _get_current_network
        net = _get_current_network()
        defaults = net._defaults

        self.label = label
        self.use_bias = use_bias if use_bias is not _DEFAULT else defaults['use_bias']

        # Normalize pre to list of NodeRefs (Layer -> layer.value)
        self._pre_list = _normalize_pre_noderefs(pre)
        # Normalize post (Layer -> layer.value)
        self.post = post.value if isinstance(post, Layer) else post
        # Keep .pre for backward compat
        self.pre = self._pre_list[0] if len(self._pre_list) == 1 else self._pre_list[0]

        # Extract slice bounds
        self.pre_slices = tuple(p.slice_bounds for p in self._pre_list)
        self.post_slice = self.post.slice_bounds

        # Validate: all pre must have the same node type
        node_types = set(p.node_type_id for p in self._pre_list)
        if len(node_types) > 1:
            raise ValueError(
                "All pre nodes in a Modulate connection must have the same "
                f"node type, got: {[p.node_type for p in self._pre_list]}")

        # Validate: flow nodes cannot be pre source
        pre_type = self._pre_list[0].node_type_id
        if pre_type in (3, 4):
            raise ValueError(
                "Flow nodes (flow_to_pre, flow_to_post) cannot be used as "
                "pre source for Modulate connections")

        if update_rule is not None:
            self.update_rule = update_rule
        else:
            lr = defaults['hebbian_learning_rate']
            self.update_rule = Hebbian(learning_rate=lr)
        self.order = order
        self.weight = init_weight
        self.bias = init_bias

        self._idx: Optional[int] = None
        self._default_order: int = 0
        self.reward_fn_idx: int = -1
        self.loss_fn_idx: int = -1

        # Derived properties for backend
        self.pre_layer_idxs: tuple = tuple(_resolve_owner_idx(p) for p in self._pre_list)
        self.pre_node_type: int = self._pre_list[0].node_type_id
        self.post_layer_idx: Optional[int] = _resolve_owner_idx(self.post)
        self.post_node_type: int = self.post.node_type_id

        # Store dimensions for weight initialization (respects slicing)
        self.pre_dim: int = sum(_resolve_owner_dim(p) for p in self._pre_list)
        self.post_dim: int = _resolve_owner_dim(self.post)

        # Set up transformation
        self.n_bands = 0
        _setup_transform(self, transformation, self.pre_dim, self.post_dim,
                         kernel_size, input_shape, stride, padding,
                         weight_mask=weight_mask)

        # Validate: slicing incompatible with conv/transconv
        has_slicing = any(s is not None for s in self.pre_slices) or self.post_slice is not None
        if has_slicing and (self.is_conv or self.is_transconv):
            raise ValueError(
                "Slicing is not compatible with conv/transconv transforms")

        # Register with current network context
        net._add_modulate(self)

    def __repr__(self):
        if len(self._pre_list) == 1:
            pre_label = _resolve_owner_label(self._pre_list[0])
        else:
            pre_label = '+'.join(_resolve_owner_label(p) for p in self._pre_list)
        post_label = _resolve_owner_label(self.post)
        return f"Modulate({pre_label}.{self._pre_list[0].node_type} -> {post_label}.{self.post.node_type})"
