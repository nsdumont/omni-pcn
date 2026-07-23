"""
Fixed visual front-end (retina → V1): ``VisualInput``.

Default pipeline (all fixed, FFT-circular-conv based, Metal/CUDA-safe):

    DoGCenterSurround   (1,H,W) -> (2,H,W)      ON/OFF retinal ganglion cells
    GaborBank           (2,H,W) -> (2+K,H,W)    V1 simple cells (K oriented Gabors)

For 28×28 gray input with the defaults (4 orientations × 2 scales × 2 phases,
K=16) the output is ``(18, 28, 28)``. Two opt-in stages are available:
``DivisiveNormalization`` (LGN contrast gain) and ``ComplexEnergy`` (phase-invariant
readout, +orientations·scales channels) — both degrade exact invertibility, so they
are off by default.

Invertibility: the ON/OFF split is lossless (``s = ON − OFF``) and the DoG is
inverted by the bank's Wiener least-squares solve, so ``decode`` reconstructs
band-pass image content well (DC is not represented by a zero-DC DoG — expected).
The Gabor channels are kept signed but are redundant for reconstruction in v1; a
joint least-squares inverse that also uses them is a documented future refinement.
"""

from typing import Optional, Sequence, Tuple

import jax.numpy as jnp

from .base import SensoryTransform, Sequential, SensoryInput
from . import _filters


class DoGCenterSurround(SensoryTransform):
    """Retinal center-surround: DoG filter then ON/OFF half-wave split.

    ``(1, H, W) -> (2, H, W)`` where channel 0 is ON (``[s]_+``) and channel 1 is
    OFF (``[−s]_+``) for the signed DoG response ``s``.
    """

    def __init__(self, spatial_shape: Tuple[int, int],
                 sigma_c: float = 0.8, sigma_s: float = 2.4,
                 size: int = 9, balance: float = 1.0):
        h, w = spatial_shape
        self.in_shape = (1, h, w)
        self.out_shape = (2, h, w)
        self.sigma_c, self.sigma_s = sigma_c, sigma_s
        kernel = _filters.dog_kernel(sigma_c, sigma_s, size, balance)
        self._bank = _filters.FFTConvBank([kernel], (h, w))

    def _forward(self, x: jnp.ndarray) -> jnp.ndarray:
        s = self._bank.apply(x[:, 0])                 # (B, 1, H, W)
        s = s[:, 0]                                    # (B, H, W)
        on = jnp.maximum(s, 0.0)
        off = jnp.maximum(-s, 0.0)
        return jnp.stack([on, off], axis=1)            # (B, 2, H, W)

    def _inverse(self, y: jnp.ndarray) -> jnp.ndarray:
        s = y[:, 0] - y[:, 1]                           # (B, H, W)  signed DoG resp.
        x = self._bank.reconstruct(s[:, None])         # (B, H, W)
        return x[:, None]                              # (B, 1, H, W)


class GaborBank(SensoryTransform):
    """V1 simple-cell Gabor bank on the signed retinal response.

    Input carries ON/OFF (2 channels) from ``DoGCenterSurround``; the signed
    response ``s = ON − OFF`` is filtered by ``K = orientations · len(wavelengths)
    · len(phases)`` Gabors. Output is ``concat(ON, OFF, gabor_1..K)`` so the
    lossless ON/OFF pathway is preserved for reconstruction.
    ``(2, H, W) -> (2 + K, H, W)``.
    """

    def __init__(self, spatial_shape: Tuple[int, int],
                 orientations: int = 4,
                 wavelengths: Sequence[float] = (3.0, 6.0),
                 phases: Sequence[float] = (0.0, jnp.pi / 2),
                 gamma: float = 0.5, sigma_ratio: float = 0.56):
        h, w = spatial_shape
        phases = tuple(float(p) for p in phases)
        kernels, index = _filters.gabor_bank_kernels(
            orientations, wavelengths, phases, gamma, sigma_ratio, (h, w))
        self.orientations = orientations
        self.n_scales = len(wavelengths)
        self.n_phases = len(phases)
        self.n_gabor = len(kernels)
        self.gabor_index = index
        self.in_shape = (2, h, w)
        self.out_shape = (2 + self.n_gabor, h, w)
        self._bank = _filters.FFTConvBank(kernels, (h, w))

    def _forward(self, x: jnp.ndarray) -> jnp.ndarray:
        s = x[:, 0] - x[:, 1]                           # (B, H, W)
        g = self._bank.apply(s)                        # (B, K, H, W)
        return jnp.concatenate([x, g], axis=1)         # (B, 2+K, H, W)

    def _inverse(self, y: jnp.ndarray) -> jnp.ndarray:
        # ON/OFF fully determine the signed response; Gabor channels redundant.
        return y[:, :2]                                # (B, 2, H, W)


class DivisiveNormalization(SensoryTransform):
    """Heeger divisive normalization (LGN contrast gain control), opt-in.

    ``R = L / sqrt(σ² + G_pool ⊛ Σ_c L²)`` with a spatial Gaussian pool over a
    channel-summed energy. Shape-preserving ``(C, H, W) -> (C, H, W)``.

    Not exactly invertible from the output alone (the per-sample normalizer is
    not carried), so ``inverse`` is a best-effort identity — keep this stage out
    of the reconstruction path when exact ``decode`` matters.
    """

    def __init__(self, spatial_shape: Tuple[int, int], channels: int,
                 sigma: float = 0.1, pool_sigma: float = 1.0, pool_size: int = 3):
        h, w = spatial_shape
        self.in_shape = (channels, h, w)
        self.out_shape = (channels, h, w)
        self.sigma = float(sigma)
        pool = _filters.gaussian_kernel_2d(pool_sigma, pool_size)
        # single-filter bank used only for its forward circular conv
        self._pool = _filters.FFTConvBank([pool], (h, w))

    def _forward(self, x: jnp.ndarray) -> jnp.ndarray:
        energy = jnp.sum(x * x, axis=1)                # (B, H, W)
        pooled = self._pool.apply(energy)[:, 0]        # (B, H, W)
        denom = jnp.sqrt(self.sigma * self.sigma + jnp.maximum(pooled, 0.0))
        return x / denom[:, None]

    def _inverse(self, y: jnp.ndarray) -> jnp.ndarray:
        return y


class ComplexEnergy(SensoryTransform):
    """Append V1 complex-cell energy channels (phase-invariant), opt-in.

    For each ``(orientation, scale)`` computes ``sqrt(even² + odd²)`` over the
    quadrature phase pair of the Gabor channels and appends the result. Adds
    ``orientations · n_scales`` channels. Non-invertible (phase discarded);
    ``inverse`` drops the appended channels.
    ``(C, H, W) -> (C + orientations·n_scales, H, W)``.
    """

    def __init__(self, spatial_shape: Tuple[int, int], in_channels: int,
                 orientations: int, n_scales: int, n_phases: int = 2,
                 gabor_offset: int = 2):
        if n_phases < 2:
            raise ValueError("ComplexEnergy needs a quadrature pair (n_phases >= 2)")
        h, w = spatial_shape
        self.orientations = orientations
        self.n_scales = n_scales
        self.n_phases = n_phases
        self.gabor_offset = gabor_offset
        self.n_energy = orientations * n_scales
        self.in_shape = (in_channels, h, w)
        self.out_shape = (in_channels + self.n_energy, h, w)

    def _energy_channels(self, x: jnp.ndarray) -> jnp.ndarray:
        outs = []
        for o in range(self.orientations):
            for s in range(self.n_scales):
                base = self.gabor_offset + (o * self.n_scales + s) * self.n_phases
                even = x[:, base]
                odd = x[:, base + 1]
                outs.append(jnp.sqrt(even * even + odd * odd))
        return jnp.stack(outs, axis=1)                 # (B, n_energy, H, W)

    def _forward(self, x: jnp.ndarray) -> jnp.ndarray:
        return jnp.concatenate([x, self._energy_channels(x)], axis=1)

    def _inverse(self, y: jnp.ndarray) -> jnp.ndarray:
        return y[:, :self.in_shape[0]]


class VisualInput(SensoryInput):
    """A ``SensoryInput`` layer with a fixed retina→V1 feature transform.

    Args:
        in_shape: raw image shape, ``(H, W)`` or ``(1, H, W)`` (grayscale/luma).
        orientations, wavelengths, phases, gamma, sigma_ratio: Gabor bank params.
        sigma_c, sigma_s, dog_size, dog_balance: DoG center-surround params.
        contrast_norm: insert ``DivisiveNormalization`` after the Gabor stage.
        complex_energy: append ``ComplexEnergy`` readout channels.
        activation, label: forwarded to ``SensoryInput``.

    Default output feature map: ``(2 + orientations·|wavelengths|·|phases|, H, W)``
    = ``(18, H, W)`` for the defaults.
    """

    def __init__(self, in_shape: Tuple[int, ...] = (1, 28, 28), *,
                 orientations: int = 4,
                 wavelengths: Sequence[float] = (3.0, 6.0),
                 phases: Sequence[float] = (0.0, jnp.pi / 2),
                 gamma: float = 0.5, sigma_ratio: float = 0.56,
                 sigma_c: float = 0.8, sigma_s: float = 2.4,
                 dog_size: int = 9, dog_balance: float = 1.0,
                 contrast_norm: bool = False, complex_energy: bool = False,
                 activation=None, label: Optional[str] = None):
        if len(in_shape) == 2:
            in_shape = (1,) + tuple(in_shape)
        if len(in_shape) != 3 or in_shape[0] != 1:
            raise NotImplementedError(
                "VisualInput v1 supports single-channel (grayscale/luma) input "
                f"(1, H, W); got {in_shape}. Multi-channel color is a future extension.")
        _, h, w = in_shape
        hw = (h, w)

        dog = DoGCenterSurround(hw, sigma_c, sigma_s, dog_size, dog_balance)
        gabor = GaborBank(hw, orientations, wavelengths, phases, gamma, sigma_ratio)
        stages = [dog, gabor]
        if contrast_norm:
            stages.append(DivisiveNormalization(hw, gabor.out_shape[0]))
        if complex_energy:
            stages.append(ComplexEnergy(
                hw, in_channels=stages[-1].out_shape[0],
                orientations=orientations, n_scales=len(wavelengths),
                n_phases=len(phases), gabor_offset=2))

        super().__init__(Sequential(stages), activation=activation, label=label)
