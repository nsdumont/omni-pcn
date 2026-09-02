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
- 'masked-<activation>': same as 'masked' but the masked matmul is wrapped in
    a post-nonlinearity, ``g(W f(pre) + b)`` — combines the 'masked' and
    'linear-<activation>' semantics (e.g. ``'masked-sigmoid'``).
- ``sparse=True | 'auto'`` (any of 'masked', 'masked-<activation>',
    'banded{N}', or 'linear' + ``weight_mask``): store the weight as a
    ``SparseWeight`` (CSR/CSC) instead of a dense matrix times a mask.
    ``weight_mask`` may then also be a ``scipy.sparse`` matrix, a
    ``jax.experimental.sparse`` BCOO/BCSR, or a ``(rows, cols)`` tuple of
    index arrays. 'auto' resolves at ``build()`` (sparse iff density <= 5%
    and post*pre >= 2**20); on Metal the dense path is used unless
    ``PCN_SPARSE_ON_METAL=1``. See ``pcn.core.sparse``.

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
from .sparse import mask_to_indices, band_indices, indices_to_dense_mask, is_index_mask
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
                       stride, padding, is_transconv=False, pool=None):
    """Parse convolution parameters and compute output shape + channels.

    Returns a dict with: in_channels, out_channels, kernel_size, stride,
    padding, input_shape, output_shape.

    ``pool``: optional ``(pool_size, pool_stride)`` of (h, w) tuples for a
    fused post-conv spatial pool (VALID, non-overlapping by default). When
    given, ``output_shape`` is the *pooled* spatial extent and ``out_channels``
    is derived against it, so ``post_dim`` must match the pooled feature size.
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

    # Fused spatial pool over the conv output (VALID). The conv produces
    # (H_out, W_out); pooling shrinks that to the post layer's spatial extent.
    if pool is not None:
        (pH, pW), (psH, psW) = pool
        if H_out < pH or W_out < pW:
            raise ValueError(
                f"pool window ({pH}, {pW}) is larger than the conv output "
                f"({H_out}, {W_out}); reduce the pool size or stride.")
        H_out = (H_out - pH) // psH + 1
        W_out = (W_out - pW) // psW + 1

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


def _attach_dense_mask(obj, weight_mask, post_dim, pre_dim, sparse):
    """Mark ``obj`` masked. Without ``sparse`` the mask is validated and
    stored densely (index-format masks are densified); with ``sparse`` only
    the index set is kept (built by the caller) — densified only on fallback."""
    obj.is_masked = True
    if sparse:
        obj.weight_mask = None
        return
    if is_index_mask(weight_mask):
        rows, cols = mask_to_indices(weight_mask, (post_dim, pre_dim))
        obj.weight_mask = indices_to_dense_mask(rows, cols, (post_dim, pre_dim))
        return
    mask = np.asarray(weight_mask, dtype=np.float32)
    if mask.shape != (post_dim, pre_dim):
        raise ValueError(
            f"weight_mask shape {mask.shape} does not match expected "
            f"({post_dim}, {pre_dim}).")
    obj.weight_mask = mask


def _setup_transform(obj, transformation, pre_dim, post_dim,
                     kernel_size=None, input_shape=None,
                     stride=None, padding=None, weight_mask=None,
                     sparse=False):
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
    obj.is_sparse = False          # resolved at build() from sparse_mode
    obj.sparse_mode = False        # False | True | 'auto'
    obj.sparse_indices = None      # (rows, cols) numpy int64, row-major sorted
    obj.post_activation_type_id = 0  # Direct by default
    if sparse not in (False, True, 'auto'):
        raise ValueError(f"sparse= must be False, True or 'auto', got {sparse!r}")
    obj.pool_type = 0                # 0 = none, 1 = max, 2 = avg
    obj.pool_size = ()
    obj.pool_stride = ()

    if transformation == 'linear':
        if weight_mask is not None:
            # 'linear' + weight_mask == 'masked'
            _attach_dense_mask(obj, weight_mask, post_dim, pre_dim, sparse)
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
    elif transformation.startswith('conv-maxpool') or \
            transformation.startswith('conv-avgpool'):
        # 'conv-maxpool' / 'conv-avgpool' [+ integer window, default 2], e.g.
        # 'conv-maxpool' (2x2) or 'conv-maxpool3' (3x3); stride = window
        # (non-overlapping), matching torch's MaxPool2d(k, stride=k).
        if kernel_size is None or input_shape is None:
            raise ValueError(
                f"'{transformation}' transformation requires kernel_size "
                f"and input_shape")
        obj.is_conv = True
        obj.pool_type = 1 if 'maxpool' in transformation else 2
        suffix = transformation.split('pool', 1)[1]
        try:
            p = int(suffix) if suffix else 2
        except ValueError:
            raise ValueError(
                f"Invalid pool window in '{transformation}'. Use e.g. "
                f"'conv-maxpool' or 'conv-maxpool3'.") from None
        obj.pool_size = (p, p)
        obj.pool_stride = (p, p)
        info = _parse_conv_params(
            pre_dim, post_dim, kernel_size, input_shape,
            stride if stride is not None else 1,
            padding if padding is not None else 0,
            is_transconv=False, pool=(obj.pool_size, obj.pool_stride))
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
    elif transformation == 'masked' or transformation.startswith('masked-'):
        if transformation.startswith('masked-'):
            name = transformation[len('masked-'):]
            try:
                act = activation_from_name(name)
            except ValueError as e:
                raise ValueError(
                    f"Invalid transformation '{transformation}': {e}") from None
            obj.post_activation_type_id = act.type_id
        if weight_mask is None:
            raise ValueError(
                f"'{transformation}' transformation requires a weight_mask "
                f"argument of shape ({post_dim}, {pre_dim}).")
        _attach_dense_mask(obj, weight_mask, post_dim, pre_dim, sparse)
    else:
        raise ValueError(
            f"Unknown transformation '{transformation}'. "
            f"Choices: 'linear', 'linear-<activation>', 'conv', "
            f"'conv-maxpool[N]', 'conv-avgpool[N]', 'transconv', "
            f"'banded{{N}}', 'masked', 'masked-<activation>'.")

    if sparse:
        # Sparse (CSR/CSC) storage: the sparsity structure is the mask. Only
        # the index set is kept here; ``PCNetwork.build()`` resolves 'auto' /
        # the platform gate into ``obj.is_sparse`` and densifies on fallback.
        if obj.is_conv or obj.is_transconv:
            raise ValueError(
                "sparse= is not supported for conv/transconv transformations "
                "(kernels are already compact).")
        if obj.is_masked:
            rows, cols = mask_to_indices(weight_mask, (post_dim, pre_dim))
        elif obj.n_bands > 0:
            rows, cols = band_indices(post_dim, pre_dim, obj.n_bands)
        else:
            raise ValueError(
                "sparse= requires a sparsity structure: use transformation="
                "'masked' / 'masked-<activation>' / 'banded{N}', or 'linear' "
                "with a weight_mask.")
        if rows.size == 0:
            raise ValueError("sparse= on an empty weight_mask (no nonzero entries).")
        obj.sparse_indices = (rows, cols)
        obj.sparse_mode = sparse


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
# Delay validation (shared by Predict / Project / Modulate)
# ============================================================================

def _validate_delay(delay, delay_unit):
    """Validate a connection ``delay`` / ``delay_unit`` pair.

    ``delay`` must be a non-negative int (``bool`` rejected); ``delay_unit``
    must be ``'iteration'`` or ``'timestep'``. Returns the delay as a plain int.
    """
    if isinstance(delay, bool) or not isinstance(delay, (int, np.integer)) or int(delay) < 0:
        raise ValueError(f"delay must be a non-negative int, got {delay!r}")
    if delay_unit not in ('iteration', 'timestep'):
        raise ValueError(
            f"delay_unit must be 'iteration' or 'timestep', got {delay_unit!r}")
    return int(delay)


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
        sparse: ``False`` (default) | ``True`` | ``'auto'``. Store a masked /
            banded weight as a ``SparseWeight`` (CSR/CSC) instead of a dense
            matrix times a mask — same math, memory O(nnz). ``'auto'``
            resolves at ``build()`` (density <= 5% and post*pre >= 2**20).
            Falls back to dense on Metal. Not valid for conv transforms.
        weight_mask: Required when ``transformation='masked'`` with shape
            ``(post_dim, pre_dim)``. Optional for ``'conv'`` / ``'transconv'``
            with shape ``(out_channels, in_channels, kH, kW)``. Element-wise
            multiplied into ``W`` at init and after every weight update.
            Entries do not need to be binary.
        order: Execution order (lower = earlier). If None, uses definition order.
        init_weight: Initial weight matrix ``(post_dim, pre_dim)``. By default the
            weights are then *fixed* (not learned) — set ``learn_weights=True`` to
            initialize from ``init_weight`` yet keep learning them (e.g. to seed a
            readout with ``Memory.C(0)`` and refine it).
        learn_weights: Tri-state weight-learning override. ``None`` (default) keeps
            the legacy behaviour — a connection with ``init_weight`` is frozen, one
            without is learned. ``True`` forces learning (init from ``init_weight``
            if given, else random). ``False`` freezes the weights even without an
            ``init_weight``.
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
        delay: Read the pre value(s) at a temporal delay instead of live.
            ``0`` (default) is the standard behaviour — the pre is read live
            inside the energy gradient and is bit-identical to a network built
            without the delay machinery. ``delay=d >= 1`` reads the pre from a
            per-node history ring buffer ``d`` steps back:

            - ``delay_unit='iteration'`` (default) — **sliding**: pre is the
              value ``d`` inference iterations ago; it advances every iteration.
            - ``delay_unit='timestep'`` — **latched**: pre is the end-of-frame
              snapshot from ``d`` input timesteps ago, held constant across all
              ``iters_per_timestep`` iterations of the current frame (the
              tPC/Kalman prior).

            The delayed read is **one-directional**: the buffer is a carry
            constant outside the value gradient, so no error flows back to the
            delayed pre. The first ``delay`` reads return zeros (pre-fill). All
            Predict pres are value nodes, so any ``delay`` is accepted.
            ``delay_unit`` is ignored when ``delay == 0``.
        delay_unit: ``'iteration'`` (sliding) or ``'timestep'`` (latched). See
            ``delay``.

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
        learn_weights: Optional[bool] = None,
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
        precision_clip_min: float = 0.0,
        precision_clip_max: float = 0.0,
        error_activation=_DEFAULT,
        alpha=_DEFAULT,
        n_bands: int = 0,
        transformation: str = 'linear',
        kernel_size=None,
        input_shape=None,
        stride=None,
        padding=None,
        weight_mask=None,
        sparse: Union[bool, str] = False,
        stochastic: bool = True,
        delay: int = 0,
        delay_unit: str = 'iteration',
    ):
        from .network import _get_current_network
        net = _get_current_network()
        defaults = net._defaults

        # Temporal-delay read (Phase 1: value pre nodes only). Predict pres are
        # always layer values, so any non-negative delay is accepted here.
        self.delay = _validate_delay(delay, delay_unit)
        self.delay_unit = delay_unit

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
        self.learn_weights = learn_weights
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

        # Optional hard clamp on the post-activation precision (bounded log-
        # precision). ``precision_clip_max > 0`` enables clamping to
        # ``[precision_clip_min, precision_clip_max]`` — see PredictConnSpec.
        self.precision_clip_min = float(precision_clip_min)
        self.precision_clip_max = float(precision_clip_max)
        if self.precision_clip_max and self.precision_clip_max <= self.precision_clip_min:
            raise ValueError(
                f"precision_clip_max ({self.precision_clip_max}) must exceed "
                f"precision_clip_min ({self.precision_clip_min}) when > 0")

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
                         weight_mask=weight_mask, sparse=sparse)

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
    def perror(self) -> NodeRef:
        """Reference to this connection's precision-weighted error, π ⊙ ε.

        A read-only derived node — the same quantity the energy, value
        gradients, and weight updates consume. Usable as a ``pre`` for
        Project/Modulate (e.g. precision-weighted error highways); it cannot
        be a ``post``. The read uses the carried error and the *effective*
        (post-activation, post-clip) precision of the same iteration the
        errors-routing sees; for connections with unlearned precision the
        (batch, 1) precision broadcasts over the error dims.

        Calibration note: in sum-convention setups (``init_precision=D_post``)
        the magnitude of ``perror`` is ~D× the raw error — rescale any
        consumer's ``init_scale`` accordingly. For *learned* precision on
        wide layers use ``precision_activation='exp'`` and exclude precision
        params from weight decay (see implementation.md).
        """
        return NodeRef(self, 'perror', owner_type='predict')

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

    Like Predict, but adds a residual connection: prediction = W @ f(pre.value) + pre.value

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


class PredictConvPool(Predict):
    """
    Convolutional predictive coding connection with a fused spatial pool.

    Convenience wrapper for
    ``Predict(..., transformation='conv-maxpool'/'conv-avgpool', ...)``.
    Reproduces a VGG-style ``conv -> pool`` block as a single learnable Predict
    edge: the forward prediction is ``pool(conv(f(pre)))`` and the error's
    feedback to the pre value is the pool adjoint (max -> unpool to the argmax,
    avg -> uniform upsample) composed with conv-transpose, both supplied by
    autodiff. ``post.dim`` must equal ``out_channels * (H/pool) * (W/pool)``.

    Args:
        pool: ``'max'`` (default) or ``'avg'``.
        pool_size: pooling window / stride (non-overlapping), default 2.
    """

    def __init__(self, pre, post, kernel_size, input_shape,
                 pool='max', pool_size=2, stride=1, padding=0, **kwargs):
        if pool not in ('max', 'avg'):
            raise ValueError(f"pool must be 'max' or 'avg', got {pool!r}")
        kind = 'maxpool' if pool == 'max' else 'avgpool'
        suffix = '' if pool_size == 2 else str(pool_size)
        super().__init__(pre, post, transformation=f'conv-{kind}{suffix}',
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
            or indexed node (e.g. value[i:j]) or a list of nodes. 
            For multi-pre, values are concatenated.
            All pre nodes must share the same node type (value or error).
        post: Target node. A NodeRef or Layer (treated as layer.value).
        transformation: Transform type ('linear' (default), 'linear-<activation>' for
            ``g(W f(x) + b)`` post-nonlinearity, 'conv', 'transconv',
            'banded{N}', 'masked'). ``'masked'`` requires ``weight_mask``.
        update_rule: Learning rule for weight updates (default: Hebbian).
        order: Execution order (lower = earlier). Matters if using a mix of Project and Modulate conns.
        init_weight: Optional initial weight matrix. 
        init_scale: Scalar multiplier on the default weight init (Gaussian with
            stddev ``sqrt(2 / (fan_in + fan_out))``), i.e. the built weights are
            ``init_scale * sqrt(2/(fan_in+fan_out)) * N(0, 1)``. Ignored when
            ``init_weight`` is given. Default 1.0.
            Use for fixed random projections whose overall gain matters, e.g.
            HEP error highways (``NoLearning`` + small ``init_scale``).
        init_bias: Optional initial bias vector (only used if ``use_bias`` is True).
        use_bias: Whether to add a bias term: ``target += W @ f(pre) + b``.
            Defaults to the network-level ``use_bias`` config setting.
        weight_mask: Required when ``transformation='masked'`` (shape
            ``(post_dim, pre_dim)``); optional for ``'conv'`` / ``'transconv'``
            (shape ``(out_channels, in_channels, kH, kW)``). Element-wise
            multiplied into ``W`` at init and after every weight update.
        advance: When this connection fires during inference. Value-targeting
            only.

            - ``'iteration'`` (default): fires on every inference iteration,
              the historical behavior.
            - ``'timestep'``: fires only on the FIRST iteration of each input
              timestep, i.e. once per input frame instead of once per
              iteration. With a ``(B, T, dim)`` clamped input the loop maps
              iterations to frames via
              ``iters_per_timestep = total_iterations // n_timesteps``; the
              connection fires when ``i % iters_per_timestep == 0``. This lets
              a state operator advance once per frame while the latent relaxes
              for ``iters_per_timestep`` iterations against a held frame. Off
              boundary an additive ``Project`` contributes 0 and a
              multiplicative ``Modulate`` factor becomes the identity (1.0).

            ``advance='timestep'`` is rejected for error- or precision-targeting
            connections: those nodes are re-derived fresh every iteration and do
            not carry, so gating them has no meaning.
        delay: Read the pre value(s) at a temporal delay instead of live.
            ``0`` (default) is bit-identical to today. ``delay=d >= 1`` reads
            the pre from a per-node history ring buffer ``d`` steps back;
            ``delay_unit='iteration'`` slides (``d`` iterations back, advancing
            every iteration) and ``delay_unit='timestep'`` latches (the
            end-of-frame snapshot ``d`` timesteps back, held across the frame).
            The read is one-directional (buffer is a carry constant) and the
            first ``delay`` reads are zeros. Every pre node must be a value node;
            ``delay >= 1`` on an error/precision pre raises ``NotImplementedError``.
            A delayed identity ``Project(pre, post, delay=n, init_weight=I)`` gives
            a plain delayed copy.
        delay_unit: ``'iteration'`` (sliding) or ``'timestep'`` (latched). See
            ``delay``. Ignored when ``delay == 0``.

    Example:
        Project(l4.value, l2.value, update_rule=Hebbian(learning_rate=1e-4))
        Project([l1, l2], l3.value)  # multi-pre, Layer input
        Project(l.value, l.value, init_weight=-np.eye(d), advance='timestep')
        Project(a.value, b.value, delay=2, init_weight=np.eye(d))  # delayed copy
    """

    def __init__(
        self,
        pre: Union[NodeRef, Layer, List[Union[NodeRef, Layer]]],
        post: Union[NodeRef, Layer],
        update_rule: Optional[LearningRule] = None,
        order: Optional[int] = None,
        init_weight: Optional[np.ndarray] = None,
        init_scale: float = 1.0,
        init_bias: Optional[np.ndarray] = None,
        use_bias=_DEFAULT,
        label: Optional[str] = None,
        transformation: str = 'linear',
        kernel_size=None,
        input_shape=None,
        stride=None,
        padding=None,
        weight_mask=None,
        sparse: Union[bool, str] = False,
        advance: str = 'iteration',
        delay: int = 0,
        delay_unit: str = 'iteration',
    ):
        if advance not in ('iteration', 'timestep'):
            raise ValueError(
                "Project(advance=...) must be 'iteration' or 'timestep', "
                f"got {advance!r}")
        self.advance = advance
        self.init_scale = float(init_scale)
        self.delay = _validate_delay(delay, delay_unit)
        self.delay_unit = delay_unit

        from .network import _get_current_network
        net = _get_current_network()
        defaults = net._defaults

        self.label = label
        self.use_bias = use_bias if use_bias is not _DEFAULT else defaults['use_bias']

        # Normalize pre to list of NodeRefs (Layer -> layer.value)
        self._pre_list = _normalize_pre_noderefs(pre)
        # Normalize post (Layer -> layer.value)
        self.post = post.value if isinstance(post, Layer) else post
        if self.post.node_type == 'perror':
            raise ValueError(
                "perror is a read-only derived node (precision * error) and "
                "cannot be a Project/Modulate post; target the connection's "
                ".error or .precision instead")
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

        # Phase 1 restriction: delay is only implemented for value pre nodes.
        if self.delay >= 1 and any(p.node_type_id != 0 for p in self._pre_list):
            raise NotImplementedError("delay on error/precision pres is Phase 2")

        # Validate: flow nodes cannot be post target for Project (use Modulate)
        post_type = self.post.node_type_id
        if post_type in (3, 4):
            raise ValueError(
                "Flow nodes (flow_to_pre, flow_to_post) can only be targeted "
                "by Modulate connections, not Project")

        # Validate: advance='timestep' is only meaningful for value targets.
        if advance == 'timestep' and post_type != 0:
            raise ValueError(
                "Project(advance='timestep') is only supported for "
                "value-targeting connections (post is a layer value), got post "
                f"node type {self.post.node_type!r}. Error- and "
                "precision-targeting routing is re-derived fresh every "
                "iteration and does not carry, so gating it to timestep "
                "boundaries has no effect.")

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
                         weight_mask=weight_mask, sparse=sparse)

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
        advance: When this connection fires during inference. Value-targeting
            only.

            - ``'iteration'`` (default): fires on every inference iteration,
              the historical behavior.
            - ``'timestep'``: fires only on the FIRST iteration of each input
              timestep, i.e. once per input frame instead of once per
              iteration. With a ``(B, T, dim)`` clamped input the loop maps
              iterations to frames via
              ``iters_per_timestep = total_iterations // n_timesteps``; the
              connection fires when ``i % iters_per_timestep == 0``. Off
              boundary the multiplicative factor becomes the identity (1.0),
              so the target passes through unchanged.

            ``advance='timestep'`` is rejected for error-, precision- and
            flow-targeting connections: those nodes are re-derived fresh every
            iteration and do not carry, so gating them has no meaning.
        delay: Read the pre value(s) at a temporal delay instead of live.
            ``0`` (default) is bit-identical to today. ``delay=d >= 1`` reads
            the pre from a per-node history ring buffer ``d`` steps back;
            ``delay_unit='iteration'`` slides and ``delay_unit='timestep'``
            latches (see :class:`Project`). The read is one-directional and the
            first ``delay`` reads are zeros. **Phase 1 restriction:** every pre
            node must be a value node; ``delay >= 1`` on an error/precision pre
            raises ``NotImplementedError``.
        delay_unit: ``'iteration'`` (sliding) or ``'timestep'`` (latched). See
            ``delay``. Ignored when ``delay == 0``.

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
        sparse: Union[bool, str] = False,
        use_bias=_DEFAULT,
        advance: str = 'iteration',
        delay: int = 0,
        delay_unit: str = 'iteration',
    ):
        if advance not in ('iteration', 'timestep'):
            raise ValueError(
                "Modulate(advance=...) must be 'iteration' or 'timestep', "
                f"got {advance!r}")
        self.advance = advance
        self.delay = _validate_delay(delay, delay_unit)
        self.delay_unit = delay_unit

        from .network import _get_current_network
        net = _get_current_network()
        defaults = net._defaults

        self.label = label
        self.use_bias = use_bias if use_bias is not _DEFAULT else defaults['use_bias']

        # Normalize pre to list of NodeRefs (Layer -> layer.value)
        self._pre_list = _normalize_pre_noderefs(pre)
        # Normalize post (Layer -> layer.value)
        self.post = post.value if isinstance(post, Layer) else post
        if self.post.node_type == 'perror':
            raise ValueError(
                "perror is a read-only derived node (precision * error) and "
                "cannot be a Project/Modulate post; target the connection's "
                ".error or .precision instead")
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

        # Phase 1 restriction: delay is only implemented for value pre nodes.
        if self.delay >= 1 and any(p.node_type_id != 0 for p in self._pre_list):
            raise NotImplementedError("delay on error/precision pres is Phase 2")

        # Validate: advance='timestep' is only meaningful for value targets.
        if advance == 'timestep' and self.post.node_type_id != 0:
            raise ValueError(
                "Modulate(advance='timestep') is only supported for "
                "value-targeting connections (post is a layer value), got post "
                f"node type {self.post.node_type!r}. Error-, precision- and "
                "flow-targeting routing is re-derived fresh every iteration "
                "and does not carry, so gating it to timestep boundaries has "
                "no effect.")

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
                         weight_mask=weight_mask, sparse=sparse)

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
