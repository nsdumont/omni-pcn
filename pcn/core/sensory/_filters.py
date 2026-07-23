"""
Pure filter-construction and FFT-convolution helpers for the fixed sensory
front-ends. Everything here is host-side kernel construction (numpy) plus small
JAX ops (FFT convolution) — no learnable state.

Circular (FFT) convolution is used throughout so that a filter bank and its
Wiener least-squares inverse form an exact adjoint pair:

    y_k = F_k ⊛ x                         (per-filter circular conv)
    x̂  = IFFT( Σ_k conj(F_k)·Y_k / (Σ_k |F_k|² + ε) )     (least-squares inverse)

This is the standard "matched filtering ÷ summed filter power" dual-frame
reconstruction (cf. NSL cortical inverse, steerable-pyramid tight frames). It is
exact for signal content the bank spans; band-pass banks (DoG, Gabor) lose the DC
component, which is expected.
"""

from typing import List, Sequence, Tuple

import numpy as np
import jax.numpy as jnp


# --------------------------------------------------------------------------- #
#  Spatial kernels                                                            #
# --------------------------------------------------------------------------- #

def gaussian_kernel_2d(sigma: float, size: int) -> np.ndarray:
    """Normalized isotropic 2-D Gaussian, shape ``(size, size)`` (sums to 1)."""
    r = (size - 1) / 2.0
    y, x = np.mgrid[-r:r + 1, -r:r + 1]
    g = np.exp(-(x * x + y * y) / (2.0 * sigma * sigma))
    return (g / g.sum()).astype(np.float64)


def dog_kernel(sigma_c: float, sigma_s: float, size: int,
               balance: float = 1.0) -> np.ndarray:
    """Difference-of-Gaussians (center − balance·surround).

    ``balance=1`` gives a zero-DC (purely band-pass) filter. ``size`` should be
    odd and large enough to contain the surround (≳ 6·sigma_s).
    """
    center = gaussian_kernel_2d(sigma_c, size)
    surround = gaussian_kernel_2d(sigma_s, size)
    k = center - balance * surround
    return k.astype(np.float64)


def gabor_kernel(size: int, wavelength: float, theta: float, phase: float,
                 sigma: float, gamma: float = 0.5) -> np.ndarray:
    """Real 2-D Gabor (Daugman parameterization).

    Args:
        size: odd kernel side length.
        wavelength: carrier wavelength ``λ`` in pixels (spatial freq ``1/λ``).
        theta: orientation of the carrier normal (radians).
        phase: carrier phase ``ψ`` (0 → even/cosine, π/2 → odd/sine).
        sigma: Gaussian envelope std along the carrier axis.
        gamma: spatial aspect ratio (envelope elongation).
    """
    r = (size - 1) / 2.0
    y, x = np.mgrid[-r:r + 1, -r:r + 1]
    xr = x * np.cos(theta) + y * np.sin(theta)
    yr = -x * np.sin(theta) + y * np.cos(theta)
    env = np.exp(-(xr * xr + (gamma * gamma) * yr * yr) / (2.0 * sigma * sigma))
    carrier = np.cos(2.0 * np.pi * xr / wavelength + phase)
    k = env * carrier
    # Remove DC so the filter is a proper band-pass (zero mean).
    k = k - k.mean()
    return k.astype(np.float64)


# --------------------------------------------------------------------------- #
#  FFT convolution bank (forward + least-squares inverse)                     #
# --------------------------------------------------------------------------- #

def _embed_kernel_ft(kernel: np.ndarray, hw: Tuple[int, int]) -> np.ndarray:
    """FFT of a spatial ``kernel`` embedded (centered) into an ``(H, W)`` grid.

    The kernel center is placed at the grid center and then ``ifftshift``-ed to
    the origin, so ``IFFT(F·FFT(x))`` is a zero-shift (centered) circular
    convolution.
    """
    h, w = hw
    kh, kw = kernel.shape
    if kh > h or kw > w:
        raise ValueError(
            f"kernel {kernel.shape} does not fit in spatial shape {hw}")
    full = np.zeros((h, w), dtype=np.float64)
    top = h // 2 - kh // 2
    left = w // 2 - kw // 2
    full[top:top + kh, left:left + kw] = kernel
    full = np.fft.ifftshift(full)
    return np.fft.fft2(full)


class FFTConvBank:
    """A bank of ``K`` fixed 2-D filters as circular convolutions, with a
    Wiener least-squares inverse.

    Args:
        kernels: sequence of 2-D real kernels (each ``(kh, kw)``, sizes may
            differ; each is embedded into the ``(H, W)`` grid).
        spatial_shape: ``(H, W)`` of the signal the bank operates on.
        eps: Tikhonov term added to the summed filter power in the inverse.
    """

    def __init__(self, kernels: Sequence[np.ndarray],
                 spatial_shape: Tuple[int, int], eps: float = 1e-6):
        h, w = int(spatial_shape[0]), int(spatial_shape[1])
        self.spatial_shape = (h, w)
        self.n = len(kernels)
        F = np.stack([_embed_kernel_ft(np.asarray(k, dtype=np.float64), (h, w))
                      for k in kernels], axis=0)          # (K, H, W) complex
        power = np.sum(np.abs(F) ** 2, axis=0) + float(eps)  # (H, W)
        self._F = jnp.asarray(F)                            # (K, H, W)
        self._Fconj = jnp.asarray(np.conj(F))               # (K, H, W)
        self._inv_power = jnp.asarray(1.0 / power)          # (H, W)

    def apply(self, x: jnp.ndarray) -> jnp.ndarray:
        """``x``: ``(B, H, W)`` -> filtered responses ``(B, K, H, W)`` (real)."""
        xf = jnp.fft.fft2(x)                                # (B, H, W)
        yf = self._F[None] * xf[:, None]                    # (B, K, H, W)
        return jnp.real(jnp.fft.ifft2(yf))

    def reconstruct(self, y: jnp.ndarray) -> jnp.ndarray:
        """Least-squares inverse. ``y``: ``(B, K, H, W)`` -> ``(B, H, W)``."""
        yf = jnp.fft.fft2(y)                                # (B, K, H, W)
        xf = jnp.sum(self._Fconj[None] * yf, axis=1) * self._inv_power
        return jnp.real(jnp.fft.ifft2(xf))                  # (B, H, W)


# --------------------------------------------------------------------------- #
#  Gabor bank spec                                                            #
# --------------------------------------------------------------------------- #

def gabor_bank_kernels(orientations: int, wavelengths: Sequence[float],
                       phases: Sequence[float], gamma: float,
                       sigma_ratio: float, spatial_shape: Tuple[int, int]
                       ) -> Tuple[List[np.ndarray], List[Tuple[int, int, int]]]:
    """Build a Gabor bank as (kernels, index-tuples).

    Channel order is ``(orientation, wavelength, phase)``. ``sigma = sigma_ratio *
    wavelength`` (≈0.56·λ for a 1-octave bandwidth). Kernel size is chosen per
    filter (~6·sigma, odd) and capped to fit the spatial grid.
    """
    h, w = spatial_shape
    cap = min(h, w)
    thetas = [np.pi * o / orientations for o in range(orientations)]
    kernels: List[np.ndarray] = []
    index: List[Tuple[int, int, int]] = []
    for oi, theta in enumerate(thetas):
        for wi, lam in enumerate(wavelengths):
            sigma = sigma_ratio * lam
            size = int(2 * round(3 * sigma) + 1)
            size = max(3, min(size, cap if cap % 2 == 1 else cap - 1))
            for pi, phase in enumerate(phases):
                kernels.append(gabor_kernel(size, lam, theta, phase, sigma, gamma))
                index.append((oi, wi, pi))
    return kernels, index
