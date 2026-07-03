"""
Memory (Legendre Memory Unit / HiPPO) module for PCN.

``Memory`` is a composite built from a layer plus fixed-weight ``Project``
connections. It maintains an online polynomial projection of an input signal's
recent history as a linear recurrence

    m[t+1] = Ā m[t] + B̄ u[t]

where ``Ā, B̄`` are the zero-order-hold discretization of a continuous delay
network ``θ ṁ = A m + B u`` for the chosen orthogonal-polynomial basis
(``'legt'`` = the Legendre Memory Unit; ``'lagt'`` = translated Laguerre).

Implementation: the recurrence is realized with two ``NoLearning`` ``Project``
connections on a ``Direct`` "recurrent" layer, relying on the value-update
semantics documented in CLAUDE.md ("Project/Modulate persistence"):

- value Projects persist into the carried state (integrating drive) and combine
  **Jacobi**-style (all read the frozen pre-update ``v[t]``), so the self-loop
  weight is ``Ā − I`` (the ``−I`` cancels the carry) and the input weight is
  ``B̄``:  ``m += (Ā−I)m + B̄u  ⇒  m = Ā m + B̄ u``.

Because ``Ā`` is contractive, the recurrence is self-stabilizing (no extra leak
needed). The recurrent layer is never a ``pre``/``post`` of a ``Predict``, so it
receives no energy gradient and evolves purely by its recurrence.

Modes (whether energy minimization may perturb the memory):

- ``'deterministic'`` (default): an extra ``Direct`` "output" layer mirrors the
  recurrent layer via a full-leak self-Project (``−I``) plus an identity copy
  (``I``); ``Memory.value`` points at the output layer. Readouts attach to the
  output layer, so the readout's energy gradient lands there and the recurrent
  layer stays a pure LMU. Decode/``C(t)`` use the recurrent layer for exact
  coefficients.
- ``'energy_coupled'``: no output layer; ``Memory.value`` is the recurrent
  layer, so readouts couple energy into the memory (a precision-weighted
  temporal-prior correction).

Timing / discretization:

- ``dt`` is the **per-inference-iteration** step. The recurrence advances once
  per inference iteration, NOT once per input timestep. If you run
  ``total_iterations`` inference iterations for a sequence of ``T`` timesteps
  (so ``iters_per_timestep = total_iterations // T`` advances per input sample),
  set ``dt`` to the per-iteration step — i.e. ``dt = physical_step /
  iters_per_timestep``. For one recurrence step per input sample, run
  ``total_iterations == T`` (``iters_per_timestep == 1``).
- Use ``feedforward_init=False`` in ``sim.train`` / ``sim.test``: the
  feedforward seed otherwise applies one extra recurrence step before the loop
  (an off-by-one in the memory state).
- The ``input`` signal should be linear: a ``Direct``-activated layer/value, or
  a ``Predict`` ``error`` / ``precision`` node (read with identity). A nonlinear
  input activation feeds ``B̄ f(u)`` instead of ``B̄ u``.

Weights are fixed (``NoLearning``) for now; learnable dynamics are a future
extension (swap to a GD-loss rule or expose ``Ā, B̄`` as trainable).
"""

from typing import Optional, Union

import numpy as np
import jax.numpy as jnp
from scipy.linalg import expm

from .layer import Layer, NodeRef
from .activations import Direct
from .learning_rules import NoLearning
from .connections import Project


def _legt_matrices(d: int):
    """Continuous ``A (d,d), B (d,1)`` for the LMU / translated-Legendre basis.

    Realizes ``θ ṁ = A m + B u`` whose state holds the shifted-Legendre
    coefficients of the input over the window ``[t−θ, t]`` (Voelker 2019)::

        A_ij = (2i+1) · {  −1            if i < j
                          (−1)^{i−j+1}  if i ≥ j }
        B_i  = (2i+1) · (−1)^i
    """
    i = np.arange(d)[:, None]
    j = np.arange(d)[None, :]
    exp = i - j + 1
    signs = np.where(exp % 2 == 0, 1.0, -1.0)          # (−1)^{i−j+1}, integer-safe
    A = (2 * i + 1) * np.where(i < j, -1.0, signs)
    B = ((2 * np.arange(d) + 1) * ((-1.0) ** np.arange(d))).reshape(d, 1)
    return A.astype(np.float64), B.astype(np.float64)


def _lagt_matrices(d: int):
    """Continuous ``A,B`` for the translated-Laguerre (LagT) basis (Gu 2020).

    ``A = −(tril(1, k=−1) + ½ I)``, ``B = 1``. The recurrence is exact; the
    closed-form ``C(t)`` decoder (Laguerre basis) is not yet implemented, so
    ``C``/``decode`` raise ``NotImplementedError`` for this basis.
    """
    A = -(np.tril(np.ones((d, d)), -1) + 0.5 * np.eye(d))
    B = np.ones((d, 1))
    return A.astype(np.float64), B.astype(np.float64)


_BASES = {
    'legt': _legt_matrices,
    'lmu': _legt_matrices,   # alias
    'lagt': _lagt_matrices,
}


def _discretize_zoh(A, B, step):
    """Zero-order-hold discretization: ``Ā = expm(A·step)``, ``B̄ = A⁻¹(Ā−I)B``.

    ``step = dt / θ``. Falls back to a least-squares solve if ``A`` is singular.
    """
    d = A.shape[0]
    Abar = expm(A * step)
    rhs = (Abar - np.eye(d)) @ B
    try:
        Bbar = np.linalg.solve(A, rhs)
    except np.linalg.LinAlgError:
        Bbar = np.linalg.lstsq(A, rhs, rcond=None)[0]
    return Abar, Bbar


class Memory:
    """Legendre/Laguerre Memory Unit as a composite of fixed-weight Projects.

    Args:
        input: Source signal — a ``Layer``, ``Layer.value``, or a ``Predict``
            ``error`` / ``precision`` ``NodeRef``. Its dimensionality is the
            number of independent channels; each channel gets its own
            ``dims_per_input``-dim memory.
        dims_per_input: Polynomial order ``d`` (memory dims per input channel).
        memory_type: ``'legt'``/``'lmu'`` (Legendre) or ``'lagt'`` (Laguerre).
        theta: Window length ``θ`` of the continuous delay network.
        dt: Per-inference-iteration timestep (see module docstring on timing).
        mode: ``'deterministic'`` (default; adds an output mirror layer so energy
            only affects the readout) or ``'energy_coupled'`` (readouts couple
            energy into the memory).
        label: Base label for the created layers/connections.

    Attributes:
        recurrent: The ``Layer`` holding the memory state ``m`` (dim ``D``).
        output: The mirror ``Layer`` (``deterministic`` mode only, else ``None``).
        dim: Total memory dimension ``D = input.dim · dims_per_input``.
        A, B: Continuous matrices (per channel, numpy).
        Abar, Bbar: Discretized matrices (per channel, numpy).
        projects: List of created ``Project`` connections.

    Use ``Memory.value`` as the ``pre`` of readout ``Predict`` connections.
    """

    def __init__(
        self,
        input: Union[Layer, NodeRef],
        dims_per_input: int,
        memory_type: str = 'legt',
        theta: float = 1.0,
        dt: float = 1.0,
        mode: str = 'deterministic',
        label: Optional[str] = None,
    ):
        from .network import _get_current_network
        net = _get_current_network()

        mt = memory_type.lower()
        if mt not in _BASES:
            raise ValueError(
                f"Unknown memory_type {memory_type!r}; choices: "
                f"{sorted(_BASES)}")
        if mode not in ('deterministic', 'energy_coupled'):
            raise ValueError(
                f"mode must be 'deterministic' or 'energy_coupled', got {mode!r}")
        if dims_per_input < 1:
            raise ValueError("dims_per_input must be >= 1")
        if theta <= 0 or dt <= 0:
            raise ValueError("theta and dt must be positive")

        # Resolve the input node and its channel count.
        input_node = input.value if isinstance(input, Layer) else input
        if not isinstance(input_node, NodeRef):
            raise TypeError(
                "Memory input must be a Layer or NodeRef, got "
                f"{type(input).__name__}")
        n = input_node.dim

        self.input = input_node
        self.memory_type = mt
        self.theta = float(theta)
        self.dt = float(dt)
        self.mode = mode
        self.input_dim = n
        self.d = int(dims_per_input)
        self.dim = n * self.d
        self.label = label or 'memory'

        # --- dynamics (host-side) -------------------------------------------
        A, B = _BASES[mt](self.d)
        Abar, Bbar = _discretize_zoh(A, B, self.dt / self.theta)
        self.A, self.B = A, B
        self.Abar, self.Bbar = Abar, Bbar

        eye_n = np.eye(n)
        # Block-diagonal over channels: each channel an independent d-dim memory.
        Abar_full = np.kron(eye_n, Abar)            # (D, D)
        Bbar_full = np.kron(eye_n, Bbar)            # (D, n)
        S_full = Abar_full - np.eye(self.dim)       # self-Project weight (Ā−I)
        self._Abar_full = Abar_full
        self._Bbar_full = Bbar_full

        # --- graph ----------------------------------------------------------
        self.projects = []
        self.recurrent = Layer(
            dim=self.dim, activation=Direct(), label=f"{self.label}_rec")

        # Recurrence (Jacobi: order-independent): m += (Ā−I)m + B̄u  ⇒  Ā m + B̄ u
        self.projects.append(Project(
            self.recurrent.value, self.recurrent.value,
            update_rule=NoLearning(),
            init_weight=jnp.asarray(S_full, dtype=jnp.float32),
            label=f"{self.label}_self"))
        self.projects.append(Project(
            self.input, self.recurrent.value,
            update_rule=NoLearning(),
            init_weight=jnp.asarray(Bbar_full, dtype=jnp.float32),
            label=f"{self.label}_input"))

        if mode == 'deterministic':
            # Output mirror: out += −I·out + I·rec  ⇒  out = rec (+ readout energy).
            self.output = Layer(
                dim=self.dim, activation=Direct(), label=f"{self.label}_out")
            eye_D = jnp.eye(self.dim, dtype=jnp.float32)
            self.projects.append(Project(
                self.output.value, self.output.value,
                update_rule=NoLearning(), init_weight=-eye_D,
                label=f"{self.label}_out_leak"))
            self.projects.append(Project(
                self.recurrent.value, self.output.value,
                update_rule=NoLearning(), init_weight=eye_D,
                label=f"{self.label}_out_copy"))
            self._value_layer = self.output
        else:
            self.output = None
            self._value_layer = self.recurrent

    # ------------------------------------------------------------------ #
    #  Node access                                                        #
    # ------------------------------------------------------------------ #

    @property
    def value(self) -> NodeRef:
        """The value node to read downstream (output mirror in deterministic
        mode, recurrent state in energy_coupled mode)."""
        return self._value_layer.value

    def __getitem__(self, key):
        """Slice the readable value node: ``memory[a:b]``."""
        return self.value[key]

    # ------------------------------------------------------------------ #
    #  Decoding                                                           #
    # ------------------------------------------------------------------ #

    def _channel_C(self, t):
        """Per-channel readout vector ``c (d,)`` reconstructing the input
        delayed by time ``t`` (``0 ≤ t ≤ θ``). LegT only."""
        if self.memory_type not in ('legt', 'lmu'):
            raise NotImplementedError(
                "C(t)/decode is only implemented for the Legendre (legt/lmu) "
                f"basis; got memory_type={self.memory_type!r}.")
        from scipy.special import eval_legendre
        r = float(t) / self.theta
        if not 0.0 <= r <= 1.0:
            raise ValueError(
                f"delay t={t} must be in [0, theta={self.theta}]")
        x = 2.0 * r - 1.0  # map window fraction to shifted-Legendre argument
        return np.array([eval_legendre(i, x) for i in range(self.d)],
                        dtype=np.float64)

    def C(self, t) -> jnp.ndarray:
        """Readout matrix ``(input_dim, D)`` reconstructing the input delayed by
        time ``t`` from the memory state: ``û(t_now − t) = C(t) @ m``."""
        c = self._channel_C(t)
        C_full = np.kron(np.eye(self.input_dim), c.reshape(1, self.d))
        return jnp.asarray(C_full, dtype=jnp.float32)

    def decode(self, m: jnp.ndarray, t) -> jnp.ndarray:
        """Reconstruct the delayed input from a memory state.

        Args:
            m: Memory state, shape ``(..., D)`` (e.g. ``(batch, D)``) — read
               from the recurrent layer for exact coefficients.
            t: Delay in ``[0, θ]``.

        Returns:
            ``(..., input_dim)`` reconstruction ``û(t_now − t)``.
        """
        C_full = self.C(t)                 # (input_dim, D)
        return jnp.asarray(m) @ C_full.T   # (..., input_dim)

    def __repr__(self):
        return (f"Memory({self.memory_type}, input_dim={self.input_dim}, "
                f"d={self.d}, D={self.dim}, theta={self.theta}, dt={self.dt}, "
                f"mode={self.mode})")
