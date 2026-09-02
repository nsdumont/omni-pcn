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
Measured on raw STL-10 the legacy path is lossless up to that single scalar:
``r = 1.000`` and 58 dB after one global gain+offset fit, while the un-rescaled
output sits at 7.7 dB purely because the mean is missing. The Gabor channels are
kept signed but are redundant for reconstruction, and no joint least-squares
inverse can do better: every composite kernel is ``G_k · DoG`` in Fourier, so the
whole bank shares the DoG's DC null.

In the v2 encoder the band-pass form pathway is crossed over with the low-pass
magno/chroma pathway where that is safe — see :class:`ParallelPathways`.
"""

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
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


# =========================================================================== #
#  v2 encoder stages: color, downsampling, built-in normalization             #
#                                                                             #
#  These turn ``VisualInput`` from a full-resolution feature *expansion* into #
#  a downsampling *encoder* that can replace the early conv+pool blocks of a   #
#  PC network (retina→V1→pool→normalize), with native RGB via color-opponent  #
#  channels. See visin_redesign.md.                                           #
# =========================================================================== #

class ColorOpponent(SensoryTransform):
    """Retinal/LGN color-opponent decomposition ``(3,H,W) -> (3,H,W)``.

    Fixed 3×3 per-pixel map to ``(Y, R−G, B−Y)``: a luminance (form) channel plus
    two chromatic-opponent channels. Unlike a per-channel luminance front-end
    (which discards the chromatic mean), this preserves color as low-frequency
    opponent signals. Exactly invertible (the matrix is full rank).
    """

    _M = np.array([[0.299, 0.587, 0.114],    # Y   (luma)
                   [1.0, -1.0, 0.0],          # R−G
                   [-0.5, -0.5, 1.0]],        # B−Y
                  dtype=np.float64)

    def __init__(self, spatial_shape: Tuple[int, int]):
        h, w = spatial_shape
        self.in_shape = (3, h, w)
        self.out_shape = (3, h, w)
        self._Mf = jnp.asarray(self._M.astype(np.float32))
        self._Minv = jnp.asarray(np.linalg.inv(self._M).astype(np.float32))

    def _forward(self, x: jnp.ndarray) -> jnp.ndarray:
        # highest precision: a fixed 3x3 color map should invert exactly, not at
        # TF32 (~2e-3) accuracy; the matmul is tiny so the cost is negligible.
        return jnp.einsum('oc,bchw->bohw', self._Mf, x, precision='highest')

    def _inverse(self, y: jnp.ndarray) -> jnp.ndarray:
        return jnp.einsum('co,bohw->bchw', self._Minv, y, precision='highest')


class GaussianBlur(SensoryTransform):
    """Per-channel spatial low-pass (circular Gaussian conv), shape-preserving.

    Used for the low-frequency pathways (magnocellular luma + chroma), which the
    band-pass DoG/Gabor stages throw away. Low-pass is not invertible from its
    output alone, so ``inverse`` is identity (a mild blur survives ``decode`` —
    documented, like :class:`DivisiveNormalization`).
    """

    def __init__(self, spatial_shape: Tuple[int, int], channels: int,
                 sigma: float = 2.0, size: Optional[int] = None):
        h, w = spatial_shape
        self.in_shape = (channels, h, w)
        self.out_shape = (channels, h, w)
        self._c = channels
        if size is None:
            size = int(2 * round(3 * sigma) + 1)
        size = max(3, min(size, h if h % 2 == 1 else h - 1))
        kernel = _filters.gaussian_kernel_2d(sigma, size)
        self._bank = _filters.FFTConvBank([kernel], (h, w))

    def _forward(self, x: jnp.ndarray) -> jnp.ndarray:
        b, c, h, w = x.shape
        y = self._bank.apply(x.reshape(b * c, h, w))[:, 0]     # (b*c, H, W)
        return y.reshape(b, c, h, w)

    def _inverse(self, y: jnp.ndarray) -> jnp.ndarray:
        return y


class SpatialPool(SensoryTransform):
    """Non-overlapping spatial pooling ``(C,H,W) -> (C, H/p, W/p)``.

    The dimensionality-reducing "pool1" of the encoder — this is what lets the
    fixed features *replace* a conv+pool block instead of expanding the input.
    ``mode='avg'`` (default) inverts to a nearest-neighbour upsample (approximate);
    ``'max'`` is available but its inverse is cruder.
    """

    def __init__(self, spatial_shape: Tuple[int, int], channels: int,
                 pool_size: int = 2, mode: str = 'avg'):
        h, w = spatial_shape
        if h % pool_size or w % pool_size:
            raise ValueError(f"pool_size {pool_size} must divide {(h, w)}")
        if mode not in ('avg', 'max'):
            raise ValueError(f"mode must be 'avg' or 'max', got {mode!r}")
        self.in_shape = (channels, h, w)
        self.out_shape = (channels, h // pool_size, w // pool_size)
        self.p = pool_size
        self.mode = mode

    def _forward(self, x: jnp.ndarray) -> jnp.ndarray:
        b, c, h, w = x.shape
        p = self.p
        g = x.reshape(b, c, h // p, p, w // p, p)
        return g.max(axis=(3, 5)) if self.mode == 'max' else g.mean(axis=(3, 5))

    def _inverse(self, y: jnp.ndarray) -> jnp.ndarray:
        return jnp.repeat(jnp.repeat(y, self.p, axis=2), self.p, axis=3)


class ChannelStandardize(SensoryTransform):
    """Fixed per-channel gain ``x * scale``, shape-preserving and exactly invertible.

    The scale is frozen after :meth:`VisualInput.fit` measures each channel's std
    on a calibration batch (``scale = 1/std``). This folds the +19 pp feature-scale
    fix from exp-CIFAR10-disc-conv-visin *into* the encoder, so callers no longer
    re-derive it. Before fitting, ``scale = 1`` (identity), so ``encode`` returns
    the un-normalized features that :meth:`fit` measures.
    """

    def __init__(self, shape: Tuple[int, ...], scale: Optional[Sequence[float]] = None):
        self.in_shape = tuple(shape)
        self.out_shape = tuple(shape)
        c = int(shape[0])
        self.fitted = scale is not None
        s = np.ones(c, np.float32) if scale is None else np.asarray(scale, np.float32)
        self._scale = jnp.asarray(s.reshape(1, c, *([1] * (len(shape) - 1))))

    def set_scale(self, scale: Sequence[float]) -> None:
        c = self.in_shape[0]
        s = np.asarray(scale, np.float32).reshape(1, c, *([1] * (len(self.in_shape) - 1)))
        self._scale = jnp.asarray(s)
        self.fitted = True

    def _forward(self, x: jnp.ndarray) -> jnp.ndarray:
        return x * self._scale

    def _inverse(self, y: jnp.ndarray) -> jnp.ndarray:
        return y / self._scale


class ChannelSelect(SensoryTransform):
    """Keep a contiguous channel slice ``(C,H,W) -> (length,H,W)``.

    Used to keep only the phase-invariant complex-energy channels (dropping the
    raw quadrature Gabor pairs). Non-invertible (dropped channels are zero-filled
    on ``inverse``).
    """

    def __init__(self, spatial_shape: Tuple[int, int], in_channels: int,
                 start: int, length: int):
        h, w = spatial_shape
        self.in_shape = (in_channels, h, w)
        self.out_shape = (length, h, w)
        self.start = start
        self.length = length

    def _forward(self, x: jnp.ndarray) -> jnp.ndarray:
        return x[:, self.start:self.start + self.length]

    def _inverse(self, y: jnp.ndarray) -> jnp.ndarray:
        b, _, h, w = y.shape
        out = jnp.zeros((b,) + self.in_shape)
        return out.at[:, self.start:self.start + self.length].set(y)


class ParallelPathways(SensoryTransform):
    """Route channel-slices of the input through independent sub-transforms and
    concatenate their outputs ``(C_in,H,W) -> (ΣC_out, H, W)``.

    This builds the retina's parallel streams: a high-detail luma *form* pathway
    (DoG→Gabor→complex energy) beside low-frequency *magno* (blurred luma) and
    *chroma* (blurred opponent) pathways. Every sub-transform must preserve the
    spatial shape ``(H,W)``.

    Pathway dict keys: ``in_start``, ``in_len``, ``transform``, and the optional
    flags ``invertible`` (this pathway's inverse reconstructs its input slice)
    and ``lowpass`` (its ``_forward`` is a low-pass operator, usable as the
    crossover's blur — see below).

    ``inverse`` has two modes:

    - ``crossover=False`` (default): each input slice is reconstructed from the
      pathway flagged ``invertible`` (later pathways win on overlap);
      non-invertible pathways contribute nothing, so ``decode`` recovers a
      blurred-colour approximation — honest for a phase-discarding readout.
    - ``crossover=True``: the band-pass (non-``invertible``) pathway is inverted
      too and crossed over with the ``lowpass`` pathway on the same slice,
      ``x̂ = f − blur(f) + low``. This keeps the form pathway's detail *and* the
      low frequencies (incl. the DC a zero-mean DoG drops) without counting the
      low band twice. A slice fed only by a band-pass pathway falls back to that
      pathway's inverse alone (DC-free, but far better than zeros).

    Only enable ``crossover`` when the band-pass pathway's inverse is actually
    usable: it must not be phase-discarding (a ``ChannelSelect`` down to complex
    energy inverts to zeros) and nothing downstream may resample it (pooling then
    nearest-upsampling band-pass channels is blocky, measurably *worse* than the
    plain low-pass answer). :class:`VisualInput` applies both guards for you.
    """

    def __init__(self, spatial_shape: Tuple[int, int], in_channels: int,
                 pathways: List[Dict], crossover: bool = False):
        h, w = spatial_shape
        self.in_shape = (in_channels, h, w)
        self.out_shape = (sum(p['transform'].out_shape[0] for p in pathways), h, w)
        self.pathways = pathways
        self.crossover = bool(crossover)

    def _forward(self, x: jnp.ndarray) -> jnp.ndarray:
        outs = []
        for p in self.pathways:
            xs = x[:, p['in_start']:p['in_start'] + p['in_len']]
            outs.append(p['transform']._forward(xs))
        return jnp.concatenate(outs, axis=1)

    def _inverse(self, y: jnp.ndarray) -> jnp.ndarray:
        # Collect each pathway's contribution keyed by the input slice it feeds,
        # keeping band-pass ("form") and low-pass contributions apart. `order`
        # preserves first-appearance pathway order so overlapping-but-different
        # slices resolve the same way they always have (later pathway wins).
        band: Dict[Tuple[int, int], jnp.ndarray] = {}
        low: Dict[Tuple[int, int], jnp.ndarray] = {}
        lowop: Dict[Tuple[int, int], SensoryTransform] = {}
        order: List[Tuple[int, int]] = []
        off = 0
        for p in self.pathways:
            co = p['transform'].out_shape[0]
            ys = y[:, off:off + co]
            off += co
            key = (p['in_start'], p['in_len'])
            if p.get('invertible', False):
                low[key] = p['transform']._inverse(ys)      # later pathway wins
                if p.get('lowpass', False):
                    lowop[key] = p['transform']
            elif self.crossover:
                xs = p['transform']._inverse(ys)
                band[key] = band[key] + xs if key in band else xs
            else:
                continue
            if key not in order:
                order.append(key)

        recon = jnp.zeros((y.shape[0],) + self.in_shape)
        for key in order:
            if key in band and key in lowop:
                # keep the band-pass detail, take the low frequencies (incl. DC)
                # from the blur pathway -- no double-counted low band
                f = band[key]
                xs = f - lowop[key]._forward(f) + low[key]
            elif key in band and key not in low:
                xs = band[key]                              # DC-free, still useful
            elif key in low:
                xs = low[key]
            else:
                continue
            recon = recon.at[:, key[0]:key[0] + key[1]].set(xs)
        return recon


class VisualInput(SensoryInput):
    """A ``SensoryInput`` layer with a fixed retina→V1 feature transform.

    Two modes share this class:

    **Legacy full-resolution feature map** (default, single-channel input). DoG
    ON/OFF center-surround → Gabor simple-cell bank, at full resolution:
    ``(2 + orientations·|wavelengths|·|phases|, H, W)`` = ``(18, H, W)`` for the
    defaults. This *expands* the input (see exp-CIFAR10-disc-conv-visin: as an
    added stage in front of a learned conv net it costs accuracy).

    **v2 downsampling encoder** (opt-in — enabled by any of ``color='opponent'``,
    ``downsample > 1``, ``complex_cells=True``, ``keep_lowpass=True``, or an
    explicit ``normalize``). A retina→V1→pool→normalize pipeline meant to *replace*
    the early conv+pool blocks of a PC network so the learned part can be
    shallower:

        ColorOpponent (Y,R−G,B−Y) → parallel pathways → [DivisiveNorm] →
        SpatialPool(/downsample) → ChannelStandardize

    Parallel pathways: a luma *form* pathway (DoG → Gabor → optional phase-invariant
    ComplexEnergy), plus low-frequency *magno* (blurred luma) and *chroma* (blurred
    opponent) pathways so mean luminance/color survive. The pooled output
    ``(C, H/downsample, W/downsample)`` is *smaller* than the raw image. Call
    :meth:`fit` on a calibration batch to freeze the per-channel normalization.

    ``decode`` quality depends on the pathway settings. With ``downsample=1`` and
    ``complex_cells=False`` the form pathway is inverted and crossed over with the
    blur pathway (see :class:`ParallelPathways`), which recovers the image almost
    exactly (measured on raw STL-10: ~89 dB gray, ~35 dB RGB — the RGB cap is the
    deliberately blurred chroma pathway). With ``complex_cells=True`` phase is
    genuinely discarded and with ``downsample > 1`` the band-pass channels are
    resampled, so both fall back to a low-pass-only reconstruction (~22 dB): a
    blurred-colour approximation, honest for a complex-cell readout. ``decode``
    is therefore a readout of the *reconstructible* pathways, not a measure of how
    much the features retain.

    Args:
        in_shape: raw image shape, ``(H, W)`` / ``(1, H, W)`` (gray) or
            ``(3, H, W)`` (RGB — v2 encoder).
        color: ``'gray'`` (luma only), ``'opponent'`` (Y, R−G, B−Y). Default:
            ``'opponent'`` for 3-channel input, ``'gray'`` for 1-channel.
        downsample: integer spatial pooling factor for the encoder output
            (1 = legacy full resolution).
        complex_cells: use phase-invariant complex-energy channels in the form
            pathway (replaces the raw quadrature Gabor pairs).
        keep_lowpass: add a blurred-luma (magnocellular) low-frequency channel,
            restoring the DC the band-pass DoG drops.
        lowpass_sigma: Gaussian sigma for the magno/chroma low-pass pathways.
        normalize: ``'std'`` (per-channel gain via :meth:`fit`), ``None``, or
            ``'auto'`` (``'std'`` in encoder mode, ``None`` in legacy mode).
        orientations, wavelengths, phases, gamma, sigma_ratio: Gabor bank params.
        sigma_c, sigma_s, dog_size, dog_balance: DoG center-surround params.
        contrast_norm: insert ``DivisiveNormalization`` (LGN contrast gain).
        complex_energy: legacy — *append* ``ComplexEnergy`` channels (full-res mode).
        activation, label: forwarded to ``SensoryInput``.
    """

    def __init__(self, in_shape: Tuple[int, ...] = (1, 28, 28), *,
                 color: Optional[str] = None,
                 downsample: int = 1,
                 complex_cells: bool = False,
                 keep_lowpass: bool = False,
                 lowpass_sigma: float = 2.0,
                 normalize: Optional[str] = 'auto',
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
        if len(in_shape) != 3 or in_shape[0] not in (1, 3):
            raise ValueError(
                f"in_shape must be (H,W), (1,H,W) or (3,H,W); got {in_shape}")
        n_in, h, w = in_shape
        hw = (h, w)
        if color is None:
            color = 'opponent' if n_in == 3 else 'gray'
        if color not in ('gray', 'opponent'):
            raise ValueError(f"color must be 'gray' or 'opponent', got {color!r}")

        encoder = (downsample > 1 or complex_cells or keep_lowpass
                   or color == 'opponent' or n_in == 3
                   or normalize not in (None, 'auto'))
        if normalize == 'auto':
            normalize = 'std' if encoder else None
        self._standardize: Optional[ChannelStandardize] = None

        if not encoder:
            # ---- Legacy full-resolution feature map (unchanged v1 behaviour) --
            dog = DoGCenterSurround(hw, sigma_c, sigma_s, dog_size, dog_balance)
            gabor = GaborBank(hw, orientations, wavelengths, phases, gamma, sigma_ratio)
            stages: List[SensoryTransform] = [dog, gabor]
            if contrast_norm:
                stages.append(DivisiveNormalization(hw, gabor.out_shape[0]))
            if complex_energy:
                stages.append(ComplexEnergy(
                    hw, in_channels=stages[-1].out_shape[0],
                    orientations=orientations, n_scales=len(wavelengths),
                    n_phases=len(phases), gabor_offset=2))
            super().__init__(Sequential(stages), activation=activation, label=label)
            return

        # ---- v2 downsampling encoder -------------------------------------- #
        stages = []
        if color == 'opponent':
            if n_in != 3:
                raise ValueError("color='opponent' requires a (3,H,W) input")
            stages.append(ColorOpponent(hw))
            pathway_in = 3
        else:                                            # gray
            pathway_in = 1
            if n_in == 3:
                raise ValueError("color='gray' requires a (1,H,W) input")

        # Luma *form* pathway: DoG -> Gabor (-> complex energy), on channel 0.
        form_stages: List[SensoryTransform] = [
            DoGCenterSurround(hw, sigma_c, sigma_s, dog_size, dog_balance),
            GaborBank(hw, orientations, wavelengths, phases, gamma, sigma_ratio)]
        if complex_cells:
            ce = ComplexEnergy(hw, in_channels=form_stages[-1].out_shape[0],
                               orientations=orientations, n_scales=len(wavelengths),
                               n_phases=len(phases), gabor_offset=2)
            form_stages.append(ce)
            # keep only the phase-invariant energy channels (drop raw phase pairs)
            form_stages.append(ChannelSelect(
                hw, ce.out_shape[0], ce.out_shape[0] - ce.n_energy, ce.n_energy))
        form = Sequential(form_stages)
        pathways: List[Dict] = [
            dict(in_start=0, in_len=1, transform=form, invertible=False)]

        if keep_lowpass:                                 # magnocellular luma DC
            pathways.append(dict(in_start=0, in_len=1, invertible=True, lowpass=True,
                                 transform=GaussianBlur(hw, 1, lowpass_sigma)))
        if color == 'opponent':                          # chroma (low bandwidth)
            pathways.append(dict(in_start=1, in_len=2, invertible=True, lowpass=True,
                                 transform=GaussianBlur(hw, 2, lowpass_sigma)))

        # Invert the band-pass form pathway too (crossing it over with the blur
        # pathway) only where its inverse survives: `complex_cells` discards phase
        # so the form inverse is identically zero, and `downsample > 1` puts a
        # pool/nearest-upsample in front of it, which is measurably worse than
        # just taking the low-pass answer. See ParallelPathways.
        crossover = (downsample == 1) and not complex_cells
        stages.append(ParallelPathways(hw, pathway_in, pathways, crossover=crossover))
        c = stages[-1].out_shape[0]
        if contrast_norm:
            stages.append(DivisiveNormalization(hw, c))
        if downsample > 1:
            stages.append(SpatialPool(hw, c, downsample))
        hn, wn = h // downsample, w // downsample
        if normalize == 'std':
            self._standardize = ChannelStandardize((c, hn, wn))
            stages.append(self._standardize)
        elif normalize is not None:
            raise ValueError(f"normalize must be 'std', None or 'auto', got {normalize!r}")

        super().__init__(Sequential(stages), activation=activation, label=label)

    def fit(self, raw: jnp.ndarray) -> "VisualInput":
        """Freeze the built-in per-channel normalization from a calibration batch.

        Measures each feature channel's std on ``raw`` (un-augmented images are
        best) and sets the :class:`ChannelStandardize` gain to ``1/std``. No-op if
        the encoder was built with ``normalize=None`` or in legacy mode. Returns
        ``self`` for chaining.
        """
        if self._standardize is None:
            return self
        feats = self.encode(raw)                          # scale=1 -> pre-norm feats
        c = self.feature_shape[0]
        f = np.asarray(feats).reshape(feats.shape[0], c, -1)
        std = f.std(axis=(0, 2))
        self._standardize.set_scale(1.0 / (std + 1e-6))
        return self
