"""
Activation function classes for PCN layers and transformations.

Each activation has a ``type_id`` used as the index into :data:`ACTIVATIONS`,
the canonical tuple of JIT-friendly JAX functions consumed by the backend.
This module is the *single source of truth* for activation behaviour — the
backend (``pcn.backend.simulation``) and connection-level transforms simply
import :data:`ACTIVATIONS` from here.

Activations are used in three places:

1. **Layer activations** — applied at read time before predictions
   (``Layer.activation``).
2. **Precision parameterizations** — the map from the raw precision
   parameters to a positive precision value
   (``Predict(..., precision_parameterization=...)``). ``'linear'``,
   ``'exp'`` and ``'softplus'`` are the practical choices because they
   admit a closed-form inverse for bias initialization. Other activations
   are accepted but the precision bias is initialised raw.
3. **Per-connection post-transform** — the ``g`` in ``g(W f(x) + b)``
   selected via ``transformation='linear-<name>'`` on Predict / Project /
   Modulate (e.g. ``'linear-relu'``, ``'linear-softplus'``).
"""
import jax
import jax.numpy as jnp


class Activation:
    """Base class for activation functions.

    Subclasses define:
        type_id:    Integer index into :data:`ACTIVATIONS`.
        init_type:  Weight-init scheme suggested for this nonlinearity.
        init_scale: Weight-init scale factor.
        fn:         Static method ``fn(x) -> y`` — the actual JAX op.
        has_memory: Whether this activation reads a previous-iteration value.
            ``False`` for the standard (stateless) activations defined in this
            module; ``True`` for :class:`MemoryActivation` subclasses such as
            :class:`Leaky`.
        needs_key: Whether ``apply`` consumes a JAX PRNG key (i.e. the
            activation is stochastic). ``False`` for the deterministic
            activations; ``True`` for :class:`StochasticActivation` subclasses
            such as :class:`Stochastic`. The backend threads a fresh key in
            when this is set.

    Activation instances are *hashable* (by ``_state()``), so they can be
    passed as JIT-static arguments to the backend without forcing a retrace
    on every call.
    """
    type_id: int = 0
    init_type: str = 'xavier'
    init_scale: float = 1.
    has_memory: bool = False
    needs_key: bool = False

    @staticmethod
    def fn(x):
        return x

    def apply(self, x, prev=None):
        """Apply the activation. ``prev`` is the previous-iteration
        activated value; ignored for stateless activations.

        Memory-aware subclasses override this and *must* receive a non-None
        ``prev`` (the backend supplies zeros for the first iteration).
        """
        return self.fn(x)

    def _state(self):
        """Hashable identity tuple. Override to include parameters."""
        return (type(self).__name__,)

    def __hash__(self):
        return hash(self._state())

    def __eq__(self, other):
        return isinstance(other, Activation) and self._state() == other._state()

    def __repr__(self):
        return f"{self.__class__.__name__}()"


class Direct(Activation):
    """Identity activation: f(x) = x."""
    type_id = 0
    init_type = 'xavier'
    init_scale = 1.

    @staticmethod
    def fn(x):
        return x


class Relu(Activation):
    """f(x) = max(0, x)."""
    type_id = 1
    init_type = 'he'
    init_scale = jnp.sqrt(2)

    @staticmethod
    def fn(x):
        return jnp.maximum(x, 0)


class Softmax(Activation):
    """f(x)_i = exp(x_i / T) / sum_j exp(x_j / T) along the last axis.

    ``temperature`` (T, default 1.0) sharpens (T<1) or softens (T>1) the
    distribution. The temperature is applied as an input scale ``x / T`` when
    the backend builds the per-layer activation functions (the static
    ``fn`` itself is plain softmax); T is carried on the LayerSpec via
    ``activation_temperature``. T<1 on a settling readout makes the
    distribution more confident — but note (exp-gen-05 phase 4) that an
    aggressive T also stiffens the value-inference dynamics.
    """
    type_id = 2
    init_type = 'xavier'
    init_scale = 4.

    def __init__(self, temperature: float = 1.0):
        t = float(temperature)
        if t <= 0:
            raise ValueError(f"Softmax temperature must be > 0, got {t}")
        self.temperature = t

    @staticmethod
    def fn(x):
        return jax.nn.softmax(x, axis=-1)

    def _state(self):
        return ('Softmax', self.temperature)

    def __repr__(self):
        return f"Softmax(temperature={self.temperature})"


class Tanh(Activation):
    """f(x) = tanh(x)."""
    type_id = 3
    init_type = 'xavier'
    init_scale = 1.

    @staticmethod
    def fn(x):
        return jnp.tanh(x)


class Sigmoid(Activation):
    """f(x) = 1 / (1 + exp(-x))."""
    type_id = 4
    init_type = 'xavier'
    init_scale = 4.

    @staticmethod
    def fn(x):
        return jax.nn.sigmoid(x)


class LeakyRelu(Activation):
    """f(x) = x if x > 0 else 0.01 * x."""
    type_id = 5
    init_type = 'he'
    init_scale = jnp.sqrt(2 / (1 + 0.01 ** 2))

    @staticmethod
    def fn(x):
        return jnp.where(x > 0, x, 0.01 * x)


class Gelu(Activation):
    """f(x) = x * Phi(x) (Gaussian Error Linear Unit)."""
    type_id = 6
    init_type = 'he'
    init_scale = 1.

    @staticmethod
    def fn(x):
        return jax.nn.gelu(x)


class Elu(Activation):
    """f(x) = x if x > 0 else exp(x) - 1."""
    type_id = 7
    init_type = 'he'
    init_scale = 1.

    @staticmethod
    def fn(x):
        return jnp.where(x > 0, x, jnp.exp(x) - 1)


class Sin(Activation):
    """f(x) = sin(x). Pairs with the SIREN-style init."""
    type_id = 8
    init_type = 'siren'
    init_scale = jnp.sqrt(6)

    @staticmethod
    def fn(x):
        return jnp.sin(x)


class Softplus(Activation):
    """f(x) = log(1 + exp(x)). Smooth positive output, bounded gradient."""
    type_id = 9
    init_type = 'xavier'
    init_scale = 1.

    @staticmethod
    def fn(x):
        return jax.nn.softplus(x)


class Exp(Activation):
    """f(x) = exp(clip(x, -10, 10)). Positive output; clip avoids overflow."""
    type_id = 10
    init_type = 'xavier'
    init_scale = 1.

    @staticmethod
    def fn(x):
        return jnp.exp(jnp.clip(x, -10., 10.))


def _nwta(x, num_winners):
    """N-winners-take-all over the last (feature) axis.

    Keeps the ``num_winners`` largest entries of each row at their raw value
    and zeros the rest:  ``y_i = x_i if x_i is among the top-N_w else 0``.

    The threshold is the ``num_winners``-th largest value per row; ties at the
    threshold all pass (negligible for continuous activations). Under autodiff
    the gradient is the winners mask (``∂y_i/∂x_i = 1`` for winners, ``0``
    otherwise) — the straight-through behaviour the value-inference dynamics
    expect from a hard competition. ``num_winners >= dim`` is the identity.
    """
    k = int(num_winners)
    d = x.shape[-1]
    if k <= 0 or k >= d:
        return x
    thresh = jax.lax.top_k(x, k)[0][..., -1:]   # (..., 1) k-th largest per row
    return jnp.where(x >= thresh, x, 0.0)


class NWTA(Activation):
    """N-winners-take-all lateral competition (Ororbia/Rao MPC, Eq. 5).

    Only the ``num_winners`` neurons with the highest activations in the layer
    emit a (raw) firing rate; the rest are silenced to zero. This induces a
    fast, parameter-free sparse competition within the layer without explicit
    lateral inhibitory weights, and is the collapse-prevention / sparsity
    mechanism in meta-representational predictive coding.

    The per-layer ``num_winners`` is carried on the :class:`LayerSpec`
    (``activation_num_winners``) and baked into the backend's per-layer
    activation closure — exactly like :class:`Softmax`'s ``temperature``. The
    static :meth:`fn` (used only by the post-transform / precision paths) is
    the identity; the real competition happens in the layer-activation path.

    Args:
        num_winners: Number of neurons kept active per sample (``N_w``).
    """
    type_id = 12
    init_type = 'xavier'
    init_scale = 1.

    def __init__(self, num_winners: int = 15):
        nw = int(num_winners)
        if nw < 1:
            raise ValueError(f"num_winners must be >= 1, got {nw}")
        self.num_winners = nw

    @staticmethod
    def fn(x):
        # Fallback (post-transform / precision paths): identity. The layer
        # path uses the num_winners-aware closure built in the backend.
        return x

    def apply(self, x, prev=None):
        return _nwta(x, self.num_winners)

    def _state(self):
        return ('NWTA', self.num_winners)

    def __repr__(self):
        return f"NWTA(num_winners={self.num_winners})"


class LayerNorm(Activation):
    """Parameter-free layer normalization over the feature (last) axis.

    ``f(x) = (x - mean(x)) / sqrt(var(x) + eps)`` per sample, where the
    statistics are taken across the feature dimension. Output has ~zero mean
    and ~unit per-sample variance regardless of the input scale.

    Added for the multimodal-gen joint-hidden experiment (exp-multimodal-
    alphanum-gen-05): deep PC latents collapse in *magnitude* (~1e-2) even
    when they retain class structure, which starves any precision = f(latent)
    parameterization (it reads the raw activation). A normalizing activation
    on the latent gives the precision an O(1) signal to read without the
    weight-runaway / soft-regulariser failure modes. Stateless and
    scale-invariant (multiplying the value by a constant leaves the output
    unchanged), so it is safe to read at any inference iteration.
    """
    type_id = 11
    init_type = 'xavier'
    init_scale = 1.
    eps = 1e-5

    @staticmethod
    def fn(x):
        mu = jnp.mean(x, axis=-1, keepdims=True)
        var = jnp.mean((x - mu) ** 2, axis=-1, keepdims=True)
        return (x - mu) * jax.lax.rsqrt(var + 1e-5)


class Poisson(Activation):
    """
    Linear-Nonlinear-Poisson (LNP) neuron model.

    Applies an inner activation to get firing rates, then samples spike
    counts from a Poisson distribution parameterized by those rates.
    The linear filter is identity (handled by the weight matrix externally).

    f(x) = Poisson(inner_activation(x))

    The layer's value state tracks rates (continuous). Spike counts are
    sampled during the pre_act computation for downstream predictions.
    Gradients flow through the smooth inner activation, not through sampling.

    Args:
        activation: The nonlinearity (N in LNP). Defaults to Relu().
    """

    def __init__(self, activation: 'Activation' = None):
        if activation is None:
            activation = Relu()
        if isinstance(activation, MemoryActivation):
            raise ValueError(
                "Poisson cannot wrap a memory activation; pass a stateless one")
        self.activation = activation
        # Delegate to the inner activation so weight init and gradients use
        # the smooth nonlinearity; spike sampling happens in the backend.
        self.type_id = activation.type_id
        self.init_type = activation.init_type
        self.init_scale = activation.init_scale
        self.fn = activation.fn

    def _state(self):
        return ('Poisson', self.activation._state())

    def __repr__(self):
        return f"Poisson({self.activation!r})"


# ----------------------------------------------------------------------------
# Memory-aware activations
# ----------------------------------------------------------------------------

class MemoryActivation(Activation):
    """Base class for activations that read a previous-iteration value.

    Subclasses define ``fn(self, x, prev)`` — a 2-arg JAX op. The base
    :meth:`apply` wraps ``prev`` in :func:`jax.lax.stop_gradient` before the
    call so that gradients **never** flow across iterations. This matches
    the intended semantics of biologically-plausible recurrent state:
    learning at iteration ``t`` should not unroll back through iteration
    ``t-1``.

    Backend dispatch: ``has_memory == True`` is checked at trace time. The
    backend supplies the previous-iteration activated value from the
    inference loop carry (zeros on the very first iteration).
    """
    has_memory: bool = True

    def fn(self, x, prev):
        raise NotImplementedError(
            f"{type(self).__name__} must implement fn(x, prev)")

    def apply(self, x, prev=None):
        if prev is None:
            raise ValueError(
                f"{type(self).__name__} requires a non-None prev; the "
                "backend must thread the previous activated value here")
        return self.fn(x, jax.lax.stop_gradient(prev))


class Leaky(MemoryActivation):
    """Convex combination of a base activation with its previous output.

    ``y_t = (1 - leak) * base.fn(x_t) + leak * stop_grad(y_{t-1})``

    ``leak == 0`` recovers the base activation. ``leak == 1`` freezes the
    output at its previous value (mostly a sanity-check setting). The
    suggested weight-init metadata (``init_type``, ``init_scale``) is
    delegated to the base activation, since the linear weights upstream
    see ``base.fn`` as the effective nonlinearity at each step.

    Args:
        base: Stateless :class:`Activation` instance providing the
            "fresh" component of each step. Defaults to :class:`Direct`.
        leak: Float in ``[0, 1]``. The fraction of the previous output
            mixed into the current one.
    """

    def __init__(self, base: 'Activation' = None, leak: float = 0.1):
        if base is None:
            base = Direct()
        if isinstance(base, MemoryActivation):
            raise ValueError(
                "Leaky.base must be a stateless activation, not another "
                "MemoryActivation")
        leak = float(leak)
        if not 0.0 <= leak <= 1.0:
            raise ValueError(f"leak must be in [0, 1], got {leak}")
        self.base = base
        self.leak = leak
        # Mirror weight-init metadata from the base.
        self.type_id = base.type_id
        self.init_type = base.init_type
        self.init_scale = base.init_scale

    def fn(self, x, prev):
        return self.leak * prev + (1.0 - self.leak) * self.base.fn(x)

    def _state(self):
        return ('Leaky', self.base._state(), self.leak)

    def __repr__(self):
        return f"Leaky({self.base!r}, leak={self.leak})"


# ----------------------------------------------------------------------------
# Stochastic (noise-injecting) activations
# ----------------------------------------------------------------------------

def gaussian_noise(key, shape, sigma=1.0):
    """Default noise sampler for :class:`Stochastic`: i.i.d. Gaussian.

    ``noise = sigma * N(0, 1)``. The signature ``(key, shape, **params)`` is
    the contract every custom ``noise_fn`` must follow.
    """
    return sigma * jax.random.normal(key, shape)


class StochasticActivation(Activation):
    """Base class for activations that consume a JAX PRNG key.

    Subclasses define ``fn(self, x, key)`` — a 2-arg JAX op where ``key`` is a
    PRNG key. The base :meth:`apply` forwards the key supplied by the backend.
    When no key is available (e.g. the forward-only logging recompute), the
    backend passes ``key=None`` and :meth:`apply` falls back to the
    deterministic base output (no noise), so logged readouts stay stable.

    Backend dispatch: ``needs_key == True`` is checked at trace time. The
    backend folds a fresh per-iteration / per-node key into ``apply``.
    """
    needs_key: bool = True

    def fn(self, x, key):
        raise NotImplementedError(
            f"{type(self).__name__} must implement fn(x, key)")

    def apply(self, x, key=None):
        if key is None:
            return self.deterministic(x)
        return self.fn(x, key)

    def deterministic(self, x):
        """Noise-free output, used when no key is available. Override."""
        raise NotImplementedError(
            f"{type(self).__name__} must implement deterministic(x)")


class Stochastic(StochasticActivation):
    """Add sampled noise to a base activation's output.

    ``y = base.fn(x) + noise_fn(key, base.fn(x).shape, **noise_params)``

    Mirrors :class:`Leaky` as a wrapper: weight-init metadata (``init_type``,
    ``init_scale``, ``type_id``) is delegated to ``base`` since the upstream
    linear weights see ``base.fn`` as the effective nonlinearity. The noise is
    additive and independent of ``x``, so gradients flow through ``base.fn``
    exactly as for the bare base activation (the noise is a constant offset
    w.r.t. differentiation).

    Args:
        base: Stateless :class:`Activation` providing the deterministic output.
            Defaults to :class:`Direct`.
        noise_fn: Callable ``(key, shape, **noise_params) -> array``. Defaults
            to :func:`gaussian_noise`.
        **noise_params: Keyword parameters forwarded to ``noise_fn`` (e.g.
            ``sigma=1.0`` for the default Gaussian). Must be hashable so the
            instance stays usable as a JIT-static argument.
    """

    def __init__(self, base: 'Activation' = None, noise_fn=None, **noise_params):
        if base is None:
            base = Direct()
        if isinstance(base, (MemoryActivation, StochasticActivation)):
            raise ValueError(
                "Stochastic.base must be a stateless activation, not another "
                "memory-aware or stochastic activation")
        self.base = base
        self.noise_fn = noise_fn if noise_fn is not None else gaussian_noise
        # Stored as a sorted tuple of items so the instance stays hashable.
        self.noise_params = tuple(sorted(noise_params.items()))
        # Mirror weight-init metadata from the base.
        self.type_id = base.type_id
        self.init_type = base.init_type
        self.init_scale = base.init_scale

    def deterministic(self, x):
        return self.base.fn(x)

    def fn(self, x, key):
        y = self.base.fn(x)
        return y + self.noise_fn(key, y.shape, **dict(self.noise_params))

    def _state(self):
        return ('Stochastic', self.base._state(), self.noise_fn,
                self.noise_params)

    def __repr__(self):
        params = dict(self.noise_params)
        return (f"Stochastic({self.base!r}, noise_fn={self.noise_fn.__name__}, "
                f"{params})")


# Ordered list of activation classes — index = type_id.
_ACTIVATION_CLASSES = (
    Direct,      # 0
    Relu,        # 1
    Softmax,     # 2
    Tanh,        # 3
    Sigmoid,     # 4
    LeakyRelu,   # 5
    Gelu,        # 6
    Elu,         # 7
    Sin,         # 8
    Softplus,    # 9
    Exp,         # 10
    LayerNorm,   # 11
    NWTA,        # 12
)

# Canonical activation function tuple, indexed by type_id. Imported by the
# backend and connection-level transforms.
ACTIVATIONS = tuple(cls.fn for cls in _ACTIVATION_CLASSES)


ACTIVATION_REGISTRY = {
    "direct": Direct,
    "linear": Direct,  # alias — used by 'linear' / 'linear-<name>' transforms
    "relu": Relu,
    "softmax": Softmax,
    "tanh": Tanh,
    "sigmoid": Sigmoid,
    "leaky_relu": LeakyRelu,
    "gelu": Gelu,
    "elu": Elu,
    "sin": Sin,
    "softplus": Softplus,
    "exp": Exp,
    "layernorm": LayerNorm,
    "layer_norm": LayerNorm,
    "nwta": NWTA,
    "poisson": Poisson,
    "leaky": Leaky,
    "stochastic": Stochastic,
}


def activation_from_name(name: str) -> Activation:
    """Look up an activation class by name and return an instance.

    Args:
        name: Case-insensitive activation name (e.g. ``"relu"``, ``"softplus"``).

    Returns:
        An instance of the corresponding Activation subclass.

    Raises:
        ValueError: If *name* is not in :data:`ACTIVATION_REGISTRY`.
    """
    cls = ACTIVATION_REGISTRY.get(name.lower())
    if cls is None:
        raise ValueError(
            f"Unknown activation: {name!r}. "
            f"Valid: {list(ACTIVATION_REGISTRY)}"
        )
    return cls()
