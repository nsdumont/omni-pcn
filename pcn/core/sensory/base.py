"""
Fixed sensory front-end transforms and the ``SensoryInput`` layer wrapper.

A ``SensoryTransform`` is a **fixed** (frozen, non-learned), approximately
invertible batched feature map applied to a modality's raw input *once per
batch, outside* the predictive-coding energy-minimization loop. The idea: a PC
network should not relax in raw pixel / raw audio space but in cortical-level
feature space (retina→LGN→V1, cochlea→auditory-nerve→cortex). The transform runs
in ``Simulation.train``/``test`` before ``run_batch``; predictive-coding
connections then attach to the *transformed features*.

Two roles:

- ``SensoryTransform`` — the pure-JAX transform. ``forward`` maps
  ``(B, *in_shape) -> (B, *out_shape)``; ``inverse`` maps back (approximately),
  for pushing a network's generated feature values back to pixel/audio space for
  visualization. All internal state is host-constructed constant JAX arrays with
  no gradients (v1 is fixed). ``Sequential`` chains stages.
- ``SensoryInput`` — a thin :class:`~pcn.core.layer.Layer` subclass whose value
  slot holds the *flattened transformed features* (so Predict/Project/Modulate
  read features automatically). It carries the transform plus ``encode``/``decode``
  helpers that flatten/reshape at the layer boundary.

Concrete modality pipelines live in :mod:`pcn.core.sensory.vision` and
:mod:`pcn.core.sensory.audio`.
"""

from typing import Optional, Sequence, Tuple

import numpy as np
import jax.numpy as jnp

from ..layer import Layer
from ..activations import Direct


Shape = Tuple[int, ...]


def _prod(shape: Sequence[int]) -> int:
    return int(np.prod(shape)) if len(shape) else 1


def to_shaped(x: jnp.ndarray, shape: Shape) -> jnp.ndarray:
    """Reshape a batched array to ``(B, *shape)``.

    Accepts either an already-shaped ``(B, *shape)`` array or a flat
    ``(B, prod(shape))`` array (the network's flat value convention).
    """
    x = jnp.asarray(x)
    if x.ndim == 0:
        raise ValueError("Expected a batched array, got a scalar")
    b = x.shape[0]
    return jnp.reshape(x, (b,) + tuple(int(s) for s in shape))


def to_flat(x: jnp.ndarray) -> jnp.ndarray:
    """Flatten a batched array ``(B, *shape) -> (B, prod(shape))``."""
    x = jnp.asarray(x)
    return jnp.reshape(x, (x.shape[0], -1))


class SensoryTransform:
    """Base class for a fixed, approximately-invertible batched feature map.

    Subclasses set ``in_shape`` and ``out_shape`` (both exclude the batch axis)
    and implement :meth:`_forward` / :meth:`_inverse` operating on shaped
    ``(B, *in_shape)`` / ``(B, *out_shape)`` arrays. The public
    :meth:`forward` / :meth:`inverse` accept *either* flat or shaped input
    (reshaped on entry), so a transform is convenient to call directly and
    composes cleanly.

    v1 contract: all state is constant (no learnable parameters, no gradients).
    """

    in_shape: Shape = ()
    out_shape: Shape = ()

    # -- subclass hooks ---------------------------------------------------- #
    def _forward(self, x: jnp.ndarray) -> jnp.ndarray:
        raise NotImplementedError

    def _inverse(self, y: jnp.ndarray) -> jnp.ndarray:
        raise NotImplementedError

    # -- public API -------------------------------------------------------- #
    def forward(self, x: jnp.ndarray) -> jnp.ndarray:
        """``(B, *in_shape)`` or flat ``(B, in_dim)`` -> ``(B, *out_shape)``."""
        return self._forward(to_shaped(x, self.in_shape))

    def inverse(self, y: jnp.ndarray) -> jnp.ndarray:
        """``(B, *out_shape)`` or flat ``(B, out_dim)`` -> ``(B, *in_shape)`` (approx)."""
        return self._inverse(to_shaped(y, self.out_shape))

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return self.forward(x)

    @property
    def in_dim(self) -> int:
        return _prod(self.in_shape)

    @property
    def out_dim(self) -> int:
        return _prod(self.out_shape)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(in={self.in_shape}, out={self.out_shape})"


class Sequential(SensoryTransform):
    """Compose ``SensoryTransform`` stages: ``forward`` chains left→right,
    ``inverse`` runs the stages in reverse (each stage's own inverse).

    Adjacent shapes must match: ``stages[i].out_shape == stages[i+1].in_shape``.
    """

    def __init__(self, stages: Sequence[SensoryTransform]):
        stages = list(stages)
        if not stages:
            raise ValueError("Sequential requires at least one stage")
        for a, b in zip(stages, stages[1:]):
            if tuple(a.out_shape) != tuple(b.in_shape):
                raise ValueError(
                    f"Shape mismatch between {a!r} (out={a.out_shape}) and "
                    f"{b!r} (in={b.in_shape})")
        self.stages = stages
        self.in_shape = tuple(stages[0].in_shape)
        self.out_shape = tuple(stages[-1].out_shape)

    def _forward(self, x: jnp.ndarray) -> jnp.ndarray:
        for s in self.stages:
            x = s._forward(x)
        return x

    def _inverse(self, y: jnp.ndarray) -> jnp.ndarray:
        for s in reversed(self.stages):
            y = s._inverse(y)
        return y

    def __repr__(self) -> str:
        inner = " -> ".join(type(s).__name__ for s in self.stages)
        return f"Sequential({inner}; in={self.in_shape}, out={self.out_shape})"


class SensoryInput(Layer):
    """A :class:`~pcn.core.layer.Layer` whose value holds fixed transformed
    features of a raw modality input.

    The layer's ``dim`` is the *flattened* feature dimension
    (``transform.out_dim``); its value slot holds the flattened features, so
    Predict/Project/Modulate connections operate on the cortical features with
    no connection-side changes. ``Simulation.train``/``test`` detect a
    ``SensoryInput`` used as a ``data_map`` key and apply :meth:`encode` to the
    raw data once per batch, outside inference.

    Args:
        transform: The fixed feature map (a ``SensoryTransform``; often a
            ``Sequential``).
        activation: Value-read nonlinearity. ``Direct()`` by default — features
            are precomputed, so no extra nonlinearity is applied.
        label: Optional layer label.

    Attributes:
        transform: The ``SensoryTransform``.
        feature_shape: ``transform.out_shape`` (e.g. ``(C, H, W)``) — pass as a
            conv connection's ``input_shape`` when feeding ``PredictConv``.
        raw_shape: ``transform.in_shape`` (the expected raw-input shape).

    Use :meth:`encode` to apply the transform manually and :meth:`decode` for the
    approximate inverse (e.g. reconstructing pixels/audio from a generated
    feature value during generative probing).
    """

    def __init__(self, transform: SensoryTransform, *,
                 activation=None, label: Optional[str] = None):
        if not isinstance(transform, SensoryTransform):
            raise TypeError(
                f"transform must be a SensoryTransform, got {type(transform).__name__}")
        super().__init__(dim=transform.out_dim,
                         activation=Direct() if activation is None else activation,
                         label=label)
        self.transform = transform
        self.feature_shape: Shape = tuple(transform.out_shape)
        self.raw_shape: Shape = tuple(transform.in_shape)

    def encode(self, raw: jnp.ndarray) -> jnp.ndarray:
        """Raw input ``(B, *raw_shape)`` or flat -> flattened features ``(B, dim)``."""
        return to_flat(self.transform.forward(raw))

    def decode(self, features: jnp.ndarray) -> jnp.ndarray:
        """Flattened features ``(B, dim)`` or shaped -> approx raw ``(B, raw_dim)``."""
        return to_flat(self.transform.inverse(features))

    def __repr__(self) -> str:
        label = self.label or f"layer_{self._idx}"
        return (f"{type(self).__name__}(dim={self.dim}, "
                f"feature_shape={self.feature_shape}, raw_shape={self.raw_shape}, "
                f"label='{label}')")
