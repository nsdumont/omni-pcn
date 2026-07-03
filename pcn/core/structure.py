"""
Static network structure definitions.

These NamedTuples define the network topology and are treated as static
(compile-time constants) for JAX JIT compilation. NetworkStructure is
hashable and can be used with static_argnums.
"""

from typing import NamedTuple, Tuple, Optional, Union

import jax
import jax.numpy as jnp
from jax import lax

from .activations import ACTIVATIONS


# ============================================================================
# Shared transform function (used by all ConnSpec types)
# ============================================================================

def _apply_transform(pre_act, W, b=None, pre_value=None,
                     is_conv=False, is_transconv=False, is_res=False,
                     in_channels=0, out_channels=0,
                     kernel_size=(), stride=(1, 1), padding='SAME',
                     input_spatial=(), alpha=1.,
                     post_activation_type: int = 0):
    """Apply a weight transform shared by all connection types.

    Computes ``g(W f(x) + b)`` for dense / banded / residual paths and the
    convolutional equivalent for the conv paths. ``g`` is selected by
    ``post_activation_type`` (an index into :data:`ACTIVATIONS`); the
    default ``0`` (``Direct``) is the identity, matching the historical
    behaviour where the transform was strictly linear.

    Args:
        pre_act: (batch, pre_dim) activated pre values.
        W: Weight matrix. Dense: (post_dim, pre_dim).
           Conv: (out_channels, in_channels, kH, kW).
        b: Optional bias. Dense: (post_dim,). Conv: (out_channels,).
        pre_value: Raw pre value for residual connections.
        is_conv, is_transconv, is_res: Transform type flags.
        in_channels, out_channels, kernel_size, stride, padding, input_spatial:
            Conv/transconv parameters.
        alpha: Scaling factor (for depth-mu PC, default 1).
        post_activation_type: ``ACTIVATIONS`` index for the nonlinearity ``g``
            applied to the linear output. ``0`` (Direct) is identity.

    Returns:
        (batch, post_dim) transformed output.
    """
    if is_transconv:
        B = pre_act.shape[0]
        x = pre_act.reshape(B, in_channels, *input_spatial)
        y = lax.conv_transpose(
            x, W, strides=stride, padding=padding,
            dimension_numbers=('NCHW', 'OIHW', 'NCHW'))
        if b is not None:
            y = y + b[None, :, None, None]
        out = y.reshape(B, -1)
    elif is_conv:
        B = pre_act.shape[0]
        x = pre_act.reshape(B, in_channels, *input_spatial)
        y = lax.conv_general_dilated(
            x, W, window_strides=stride, padding=padding,
            dimension_numbers=('NCHW', 'OIHW', 'NCHW'))
        if b is not None:
            y = y + b[None, :, None, None]
        out = y.reshape(B, -1)
    elif is_res:
        out = jnp.dot(pre_act, alpha * W.T) + pre_value
    else:
        out = jnp.dot(pre_act, alpha * W.T)
        if b is not None:
            out = out + b

    if post_activation_type != 0:
        out = ACTIVATIONS[post_activation_type](out)
    return out


class LayerSpec(NamedTuple):
    """
    Specification for a single layer.

    Attributes:
        dim: Dimensionality of the layer
        activation_type: Enum for activation function
            0 = Direct (identity)
            1 = ReLU
            2 = Softmax
            3 = Tanh
            4 = Sigmoid
        label: Unique string identifier for the layer
        dynamics: Rate of value updates
        is_poisson: If True, pre_act is sampled from Poisson(activation(value))
            during inference. The activation_type refers to the inner
            nonlinearity (N in LNP). Value dynamics remain continuous (rates).
        dropout_prob: Float in [0, 1). Bernoulli dropout applied to the
            activated value before downstream predicts consume it. Active
            ONLY during learning (training); inference at test time passes
            through unchanged. 0 disables.
    """
    dim: int
    activation_type: int
    label: str
    dynamics_rate: float
    is_poisson: bool = False
    spatial_structure: str = 'none'
    dropout_prob: float = 0.0
    # Input temperature applied as ``activation(x / T)`` when the backend
    # builds this layer's activation fn. Default 1.0 (no effect). Used by
    # Softmax for temperature-scaled readouts.
    activation_temperature: float = 1.0
    # Number of winners for an NWTA layer activation (N-winners-take-all).
    # 0 means "not an NWTA layer" — the plain activation fn is used. Baked
    # into the backend's per-layer activation closure, like
    # ``activation_temperature``. Carried here so it participates in the
    # static NetworkStructure hash (correct JIT cache keys).
    activation_num_winners: int = 0


class PredictConnSpec(NamedTuple):
    """
    Specification for a Predict connection.

    Standard predictive coding connection where pre layer(s) predict post layer:
        prediction = W @ f(pre.value)
        error = precision * (post.value - prediction)

    For multi-pre connections, pre values are concatenated before the transform.

    Attributes:
        pre_idx: Tuple of layer indices (source/predictor layers).
            Single pre: (idx,). Multi-pre: (idx1, idx2, ...).
        post_idx: Index into layers list (target/predicted layer)
        has_fixed_weights: If True, weights are not updated during learning
    """
    pre_idx: tuple
    post_idx: int
    has_fixed_weights: bool
    learn_precision_weights: bool
    learn_precision_bias: bool
    # Optional custom precision sources (``Predict(precision_input=...)``).
    # Empty tuples (default) mean the precision is keyed on the connection's
    # own pre activation, the historical behaviour. When set, the three
    # tuples are parallel, one entry per source:
    #   precision_input_idx        owner index (layer idx for values, predict
    #                              conn idx for errors/precisions)
    #   precision_input_node_types 0=value, 1=error, 2=precision
    #   precision_input_slices     (start, stop) or None per source
    # Value sources read the *current* iteration's activated values (live in
    # the energy gradient, like the default pre read). Error/precision
    # sources read the *previous* iteration's carried arrays (Jacobi rule),
    # which makes any reference acyclic — including a conn's own error.
    precision_input_idx: tuple = ()
    precision_input_node_types: tuple = ()
    precision_input_slices: tuple = ()
    alpha: float = 1.
    is_conv: bool = False
    is_transconv: bool = False
    in_channels: int = 0
    out_channels: int = 0
    kernel_size: tuple = ()
    stride: tuple = (1, 1)
    padding: Union[str, tuple] = 'SAME'
    input_spatial: tuple = ()
    output_spatial: tuple = ()
    has_bias: bool = True
    is_res: bool = False
    label: str = ''
    n_bands: int = 0
    pre_slices: tuple = ()
    post_slice: tuple = ()
    # Activation type_id (index into ACTIVATIONS) used to map the raw
    # log-precision parameters to a positive precision. Default 9 (Softplus).
    # ``linear`` / ``exp`` / ``softplus`` are the recommended choices because
    # they admit a closed-form inverse for bias initialisation.
    precision_activation_type: int = 9
    # Activation type_id applied to the raw residual ``post_val - prediction``
    # before downstream consumers (energy summation, Project/Modulate, logs).
    # Default 0 (Direct = identity), preserving the standard PC residual.
    error_activation_type: int = 0
    # Activation type_id for the per-connection post-transform g in g(Wf(x)+b).
    # Default 0 (Direct = identity), so the transform stays strictly linear
    # unless ``transformation='linear-<name>'`` is requested.
    post_activation_type: int = 0
    # ``transformation='masked'``: the connection carries a user-supplied
    # ``(post_dim, pre_dim)`` mask multiplied into W at init and after every
    # weight update. The mask itself lives on ``PCNetwork.weight_masks`` —
    # this flag tells the backend whether to apply it.
    is_masked: bool = False
    # When True, per-sample standardise the pre-activation (zero-mean, unit-var
    # over the feature axis) *inside* the precision parameterisation only, with
    # stop_gradient, so ``precision = softplus(W_ρ @ standardize(pre) + b)``.
    # Gives precision=f(pre) an O(1) signal to read when the pre latent has
    # collapsed in magnitude (exp-multimodal-alphanum-gen-05), WITHOUT changing
    # the forward generative dynamics or feeding the stiff normaliser Jacobian
    # into the value inference. Default False (standard raw-pre behaviour).
    precision_input_norm: bool = False
    # When False, this connection's prediction gets NO is_stochastic noise even
    # when the run is stochastic — lets you sample noise for only some Predicts
    # (e.g. drift the image via the input-prediction only). Default True =
    # historical behaviour (every prediction noised when is_stochastic=True).
    stochastic: bool = True

    def apply(self, pre_act, W, b=None, pre_value=None):
        """Apply the connection transform."""
        return _apply_transform(
            pre_act, W, b, pre_value,
            is_conv=self.is_conv, is_transconv=self.is_transconv,
            is_res=self.is_res, in_channels=self.in_channels,
            out_channels=self.out_channels, kernel_size=self.kernel_size,
            stride=self.stride, padding=self.padding,
            input_spatial=self.input_spatial, alpha=self.alpha,
            post_activation_type=self.post_activation_type,
        )

    def prediction(self, pre_act, W, b, pre_value=None):
        """Compute prediction (alias for apply)."""
        return self.apply(pre_act, W, b, pre_value)

    def get_pre(self, values, errors, activation_fns, act_instances=(),
                key=None):
        """Get concatenated activated pre values, respecting slicing.

        ``act_instances`` is an optional per-layer tuple of Activation
        instances. When the instance for a pre-layer is stochastic
        (``needs_key``) and ``key`` is supplied, its noisy ``apply`` is used
        (with a per-layer key folded from ``key``); otherwise the plain
        ``activation_fns[idx]`` is applied. The same fold (per layer index)
        means all connections reading a given layer in one iteration see the
        same noise sample.
        """
        parts = []
        for k, idx in enumerate(self.pre_idx):
            inst = act_instances[idx] if act_instances else None
            if inst is not None and inst.needs_key and key is not None:
                act = inst.apply(values[idx], key=jax.random.fold_in(key, idx))
            else:
                act = activation_fns[idx](values[idx])
            sl = self.pre_slices[k] if self.pre_slices else None
            if sl is not None:
                act = act[:, sl[0]:sl[1]]
            parts.append(act)
        return parts[0] if len(parts) == 1 else jnp.concatenate(parts, axis=-1)

    def get_pre_value(self, values):
        """Get concatenated raw pre values (before activation), respecting slicing."""
        parts = []
        for k, idx in enumerate(self.pre_idx):
            v = values[idx]
            sl = self.pre_slices[k] if self.pre_slices else None
            if sl is not None:
                v = v[:, sl[0]:sl[1]]
            parts.append(v)
        return parts[0] if len(parts) == 1 else jnp.concatenate(parts, axis=-1)

    def get_post(self, values, errors=None):
        """Get post layer value, respecting slicing."""
        v = values[self.post_idx]
        if self.post_slice:
            return v[:, self.post_slice[0]:self.post_slice[1]]
        return v

    def stochastic_prediction(self, pre_act, W, b, precision_weights, precision_bias, key, pre_value=None,
                              precision_input=None, noise_scale=1.0):
        mu = self.prediction(pre_act, W, b, pre_value)
        eps = jax.random.normal(key, mu.shape)
        prec_in = precision_input if precision_input is not None else pre_act
        precision = self.precision_fn(prec_in, precision_weights, precision_bias)
        std = noise_scale / jnp.sqrt(jnp.clip(precision, 1e-8, None))
        return mu + std * eps

    def get_precision_input(self, pre_act, values, prev_errors, prev_precisions,
                            activation_fns, act_instances=(), key=None):
        """Gather the input to the precision function for this connection.

        Default (``precision_input_idx`` empty): returns ``pre_act`` unchanged,
        so the precision stays keyed on the connection's own pre activation
        and the default code path is bit-identical to the historical one.

        With custom sources, gathers per source:
          - value nodes: the *current* activated layer value (live in the
            energy gradient, mirroring the default pre read; same per-layer
            key fold as :meth:`get_pre` for stochastic activations)
          - error / precision nodes: the *previous* iteration's carried array
            (``prev_errors`` / ``prev_precisions``) — Jacobi rule; inherently
            detached from current-iteration value gradients

        and concatenates along the feature axis.
        """
        if not self.precision_input_idx:
            return pre_act
        parts = []
        for k, idx in enumerate(self.precision_input_idx):
            nt = self.precision_input_node_types[k]
            if nt == 0:  # layer value (current iteration, activated)
                inst = act_instances[idx] if act_instances else None
                if inst is not None and inst.needs_key and key is not None:
                    arr = inst.apply(
                        values[idx], key=jax.random.fold_in(key, idx))
                else:
                    arr = activation_fns[idx](values[idx])
            elif nt == 1:  # carried error from the previous iteration
                arr = prev_errors[idx]
            else:  # nt == 2, carried precision from the previous iteration
                arr = prev_precisions[idx]
            sl = (self.precision_input_slices[k]
                  if self.precision_input_slices else None)
            if sl is not None:
                arr = arr[:, sl[0]:sl[1]]
            parts.append(arr)
        return parts[0] if len(parts) == 1 else jnp.concatenate(parts, axis=-1)

    def precision_fn(self, pre_act, precision_weights, precision_bias):
        """Compute per-sample precision as a function of pre-layer activations.

        Args:
            pre_act: (batch, pre_dim) activated pre-layer values
            precision_weights: (1, pre_dim) or (post_dim, pre_dim)
            precision_bias: (1,) or (post_dim,)

        Returns:
            (batch, 1) or (batch, post_dim) precision values
        """
        return self.precision_transform(self.log_precision_fn(pre_act, precision_weights, precision_bias))

    def precision_transform(self, log_prec):
        """Map raw precision parameters to a positive precision via the
        configured activation function (selected by ``precision_activation_type``)."""
        return ACTIVATIONS[self.precision_activation_type](log_prec)

    def error_transform(self, residual):
        """Apply the configured error activation to a raw residual.

        ``residual`` is ``post_val - prediction`` (or one of the per-leg
        variants used by the flow-gating mechanism). The transformed value
        is what every downstream consumer (energy summation, Project /
        Modulate reading errors, logs) sees. Default activation is
        ``Direct`` (identity), so behaviour is unchanged unless the user
        opts in via ``Predict(error_activation=...)``.
        """
        if self.error_activation_type == 0:
            return residual
        return ACTIVATIONS[self.error_activation_type](residual)

    @property
    def precision_param_type(self) -> int:
        """Deprecated alias for ``precision_activation_type``."""
        return self.precision_activation_type

    def log_precision_fn(self, pre_act, precision_weights, precision_bias):
        """Compute per-sample log precision as a function of pre-layer activations.

        Args:
            pre_act: (batch, pre_dim) activated pre-layer values
            precision_weights: (1, pre_dim) or (post_dim, pre_dim)
            precision_bias: (1,) or (post_dim,)

        Returns:
            (batch, 1) or (batch, post_dim) precision values
        """
        if self.precision_input_norm:
            # Per-sample standardise over the feature axis so precision reads an
            # O(1) signal even when the pre latent has collapsed in magnitude.
            # stop_gradient keeps the (stiff) normaliser Jacobian out of the
            # value inference dynamics: precision reacts to the latent but the
            # latent cannot game the precision to lower energy (no runaway).
            # W_ρ still learns (its gradient is ∝ the detached standardised pre).
            mu = jnp.mean(pre_act, axis=-1, keepdims=True)
            var = jnp.mean((pre_act - mu) ** 2, axis=-1, keepdims=True)
            pre_act = jax.lax.stop_gradient((pre_act - mu) * jax.lax.rsqrt(var + 1e-5))
        return jnp.dot(pre_act, precision_weights.T) + precision_bias


# ============================================================================
# Shared helpers for Project/Modulate specs
# ============================================================================

def _pm_get_pre(pre_idx, pre_node_type, values, errors, activation_fns,
                pre_slices=(), precisions=()):
    """Get concatenated activated pre node values for Project/Modulate.

    All pre nodes must share the same node type (enforced at construction).
    """
    parts = []
    for k, idx in enumerate(pre_idx):
        if pre_node_type == 0:  # value nodes
            arr = activation_fns[idx](values[idx])
        elif pre_node_type == 1:  # error nodes (identity activation)
            arr = errors[idx]
        else:  # pre_node_type == 2, precision nodes
            arr = precisions[idx]
        sl = pre_slices[k] if pre_slices else None
        if sl is not None:
            arr = arr[:, sl[0]:sl[1]]
        parts.append(arr)
    return parts[0] if len(parts) == 1 else jnp.concatenate(parts, axis=-1)


def _pm_get_post(post_idx, post_node_type, values, errors, post_slice=(),
                 precisions=()):
    """Get post node value for Project/Modulate."""
    if post_node_type == 0:
        arr = values[post_idx]
    elif post_node_type == 1:
        arr = errors[post_idx]
    else:  # post_node_type == 2, precision
        arr = precisions[post_idx]
    if post_slice:
        return arr[:, post_slice[0]:post_slice[1]]
    return arr


def _pm_apply(pre_act, W, is_conv, is_transconv, in_channels, out_channels,
              kernel_size, stride, padding, input_spatial,
              post_activation_type: int = 0):
    """Apply transform for Project/Modulate (no bias, no residual)."""
    return _apply_transform(
        pre_act, W, b=None, pre_value=None,
        is_conv=is_conv, is_transconv=is_transconv,
        is_res=False, in_channels=in_channels,
        out_channels=out_channels, kernel_size=kernel_size,
        stride=stride, padding=padding,
        input_spatial=input_spatial, alpha=1.,
        post_activation_type=post_activation_type,
    )


class ProjectConnSpec(NamedTuple):
    """
    Specification for a Project (additive) connection.

    Adds a weighted projection to a target node: target += W @ f(source)

    Attributes:
        pre_idx: Tuple of owner indices (into values or errors based on pre_node_type).
            Single pre: (idx,). Multi-pre: (idx1, idx2, ...).
        pre_node_type: Source node type (0=value, 1=error). All pre nodes share this type.
        post_idx: Index into appropriate array based on post_node_type
        post_node_type: Target node type (0=value, 1=error)
        learning_rule_type: Learning rule (0=Hebbian, 1=ThreeFactorHebbian, 2=GradientDescent, 3=Oja)
        learning_rate: Learning rate for weight updates
        reward_fn_idx: Index into reward functions list (-1 if not needed)
        loss_fn_idx: Index into loss functions list (-1 if not needed, for GradientDescent)
    """
    pre_idx: tuple
    pre_node_type: int
    post_idx: int
    post_node_type: int
    learning_rule_type: int
    learning_rate: float
    reward_fn_idx: int
    loss_fn_idx: int = -1
    is_conv: bool = False
    is_transconv: bool = False
    in_channels: int = 0
    out_channels: int = 0
    kernel_size: tuple = ()
    stride: tuple = (1, 1)
    padding: Union[str, tuple] = 'SAME'
    input_spatial: tuple = ()
    output_spatial: tuple = ()
    n_bands: int = 0
    pre_slices: tuple = ()
    post_slice: tuple = ()
    has_bias: bool = False
    # Activation type_id for the post-transform g in g(Wf(x)+b). 0 = Direct.
    post_activation_type: int = 0
    # See PredictConnSpec.is_masked.
    is_masked: bool = False

    def apply(self, pre_act, W):
        """Apply the connection transform (no bias)."""
        return _pm_apply(
            pre_act, W, self.is_conv, self.is_transconv,
            self.in_channels, self.out_channels, self.kernel_size,
            self.stride, self.padding, self.input_spatial,
            post_activation_type=self.post_activation_type,
        )

    def get_pre(self, values, errors, activation_fns, precisions=()):
        """Get concatenated activated pre node values."""
        return _pm_get_pre(self.pre_idx, self.pre_node_type, values, errors,
                           activation_fns, self.pre_slices, precisions)

    def get_post(self, values, errors, precisions=()):
        """Get post node value."""
        return _pm_get_post(self.post_idx, self.post_node_type, values, errors,
                            self.post_slice, precisions)


class ModulateConnSpec(NamedTuple):
    """
    Specification for a Modulate (multiplicative) connection.

    Multiplies a target node by a weighted projection: target *= W @ f(source)
    Applied AFTER additive updates (Predict, Project) by default.

    Attributes:
        pre_idx: Tuple of owner indices. All pre nodes share pre_node_type.
        pre_node_type: Source node type (0=value, 1=error)
        post_idx: Index into appropriate array based on post_node_type
        post_node_type: Target node type (0=value, 1=error)
        learning_rule_type: Learning rule (0=Hebbian, 1=ThreeFactorHebbian, 2=GradientDescent, 3=Oja)
        learning_rate: Learning rate for weight updates
        reward_fn_idx: Index into reward functions list (-1 if not needed)
        loss_fn_idx: Index into loss functions list (-1 if not needed, for GradientDescent)
    """
    pre_idx: tuple
    pre_node_type: int
    post_idx: int
    post_node_type: int
    learning_rule_type: int
    learning_rate: float
    reward_fn_idx: int
    loss_fn_idx: int = -1
    is_conv: bool = False
    is_transconv: bool = False
    in_channels: int = 0
    out_channels: int = 0
    kernel_size: tuple = ()
    stride: tuple = (1, 1)
    padding: Union[str, tuple] = 'SAME'
    input_spatial: tuple = ()
    output_spatial: tuple = ()
    n_bands: int = 0
    pre_slices: tuple = ()
    post_slice: tuple = ()
    has_bias: bool = False
    # Activation type_id for the post-transform g in g(Wf(x)+b). 0 = Direct.
    post_activation_type: int = 0
    # See PredictConnSpec.is_masked.
    is_masked: bool = False

    def apply(self, pre_act, W):
        """Apply the connection transform (no bias)."""
        return _pm_apply(
            pre_act, W, self.is_conv, self.is_transconv,
            self.in_channels, self.out_channels, self.kernel_size,
            self.stride, self.padding, self.input_spatial,
            post_activation_type=self.post_activation_type,
        )

    def get_pre(self, values, errors, activation_fns, precisions=()):
        """Get concatenated activated pre node values."""
        return _pm_get_pre(self.pre_idx, self.pre_node_type, values, errors,
                           activation_fns, self.pre_slices, precisions)

    def get_post(self, values, errors, precisions=()):
        """Get post node value."""
        return _pm_get_post(self.post_idx, self.post_node_type, values, errors,
                            self.post_slice, precisions)


class StructuralAttentionGroup(NamedTuple):
    """Group of predict connections competing for the same target via softmax.

    Attention weights: alpha_k = softmax(-0.5 * prec_k * ||error_k||^2 / temperature)
    """
    conn_indices: Tuple[int, ...]
    temperature: float = 1.0


class NetworkStructure(NamedTuple):
    """
    Complete static structure of the network.

    This is hashable and can be used as a static argument for JAX JIT.

    Attributes:
        layers: Tuple of LayerSpec defining all layers
        predict_conns: Tuple of PredictConnSpec for standard PC connections
        project_conns: Tuple of ProjectConnSpec for additive connections
        modulate_conns: Tuple of ModulateConnSpec for multiplicative connections
        layer_dims: Tuple of layer dimensions for quick access
        predict_error_dims: Tuple of post_dim per predict connection (for error init)
        project_conns_internal: Pre-sorted Project conns targeting errors (post_node_type==1)
            Each entry is (weight_index, ProjectConnSpec).
        project_conns_value: Pre-sorted Project conns targeting values (post_node_type==0)
        modulate_conns_internal: Pre-sorted Modulate conns targeting errors
        modulate_conns_value: Pre-sorted Modulate conns targeting values
        gd_loss_project: Tuples of (weight_idx, loss_fn_idx) for GD Project conns with loss_fn
        gd_loss_modulate: Tuples of (weight_idx, loss_fn_idx) for GD Modulate conns with loss_fn
        loss_fn_sample_keys: Sample dict keys needed by loss functions (sorted tuple of strs)
    """
    layers: Tuple[LayerSpec, ...]
    predict_conns: Tuple[PredictConnSpec, ...]
    project_conns: Tuple[ProjectConnSpec, ...]
    modulate_conns: Tuple[ModulateConnSpec, ...]
    layer_dims: Tuple[int, ...]
    predict_error_dims: Tuple[int, ...]
    project_conns_internal: Tuple[Tuple[int, ProjectConnSpec], ...] = ()
    project_conns_value: Tuple[Tuple[int, ProjectConnSpec], ...] = ()
    modulate_conns_internal: Tuple[Tuple[int, ModulateConnSpec], ...] = ()
    modulate_conns_value: Tuple[Tuple[int, ModulateConnSpec], ...] = ()
    spatial_layers: Tuple[int, ...] = ()
    gd_loss_project: Tuple[Tuple[int, int], ...] = ()
    gd_loss_modulate: Tuple[Tuple[int, int], ...] = ()
    loss_fn_sample_keys: Tuple[str, ...] = ()
    inference_regs: tuple = ()
    train_regs: tuple = ()
    # Mechanism 1: precision-targeting Project/Modulate
    project_conns_precision: Tuple[Tuple[int, ProjectConnSpec], ...] = ()
    modulate_conns_precision: Tuple[Tuple[int, ModulateConnSpec], ...] = ()
    # Mechanism 2: per-leg flow gating
    modulate_conns_flow_pre: Tuple[Tuple[int, ModulateConnSpec], ...] = ()
    modulate_conns_flow_post: Tuple[Tuple[int, ModulateConnSpec], ...] = ()
    predict_has_flow_gates: Tuple[bool, ...] = ()
    # Mechanism 3: structural attention
    structural_attention_groups: tuple = ()
    # Per-connection energy scale: 1/n_predicts_to_pre (pre-computed at build)
    predict_pre_scales: Tuple[float, ...] = ()

    def __hash__(self):
        return hash((
            self.layers,
            self.predict_conns,
            self.project_conns,
            self.modulate_conns,
            self.layer_dims,
            self.predict_error_dims,
            self.project_conns_internal,
            self.project_conns_value,
            self.modulate_conns_internal,
            self.modulate_conns_value,
            self.spatial_layers,
            self.gd_loss_project,
            self.gd_loss_modulate,
            self.loss_fn_sample_keys,
            self.inference_regs,
            self.train_regs,
            self.project_conns_precision,
            self.modulate_conns_precision,
            self.modulate_conns_flow_pre,
            self.modulate_conns_flow_post,
            self.predict_has_flow_gates,
            self.structural_attention_groups,
            self.predict_pre_scales,
        ))
