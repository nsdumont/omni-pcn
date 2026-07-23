"""
Fixed auditory front-end (cochlea → cortex): ``AuditoryInput``.

Per the design decision, the wrapper consumes a **raw fixed-length waveform**
(``in_shape = (n_samples,)``); the STFT→mel-power computation lives inside the
transform (Stage 1), so the network never sees anything but cortical features.

Default pipeline (variant "a", all fixed, conv/FFT/elementwise — Metal/CUDA-safe):

    MelPower           (n_samples,)      -> (n_mels, n_frames)   STFT → mel power
    PowerCompression   (n_mels,n_frames) -> (n_mels, n_frames)   y = (E+ε)^α
    LateralInhibition  (n_mels,n_frames) -> (n_mels, n_frames)   [1,-1] over freq, ReLU
    LeakyIntegrator    (n_mels,n_frames) -> (n_mels, n_frames)   causal EMA over time

An opt-in ``STRFBank`` cortical modulation stage (2-D rate×scale filters) can be
appended.

Invertibility: compression (``E = y^{1/α}``) and the leaky integrator (exact
recurrence) invert exactly; lateral inhibition inverts by reverse-cumsum over
frequency (approximate through the ReLU); mel power inverts via the mel
pseudo-inverse followed by Griffin–Lim to a waveform.
"""

from typing import Optional, Sequence, Tuple

import numpy as np
import jax
import jax.numpy as jnp

from .base import SensoryTransform, Sequential, SensoryInput
from . import _filters


# --------------------------------------------------------------------------- #
#  STFT / mel helpers                                                         #
# --------------------------------------------------------------------------- #

def _hann(n: int) -> np.ndarray:
    return np.hanning(n + 1)[:-1].astype(np.float64) if n > 1 else np.ones(1)


def _mel_filterbank(sr: int, n_fft: int, n_mels: int,
                    fmin: float, fmax: float) -> np.ndarray:
    """Standard HTK triangular mel filterbank, shape ``(n_mels, n_fft//2+1)``."""
    n_freq = n_fft // 2 + 1

    def hz_to_mel(f):
        return 2595.0 * np.log10(1.0 + f / 700.0)

    def mel_to_hz(m):
        return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

    mel_pts = np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2)
    hz_pts = mel_to_hz(mel_pts)
    bins = np.floor((n_fft + 1) * hz_pts / sr).astype(int)
    bins = np.clip(bins, 0, n_freq - 1)
    fb = np.zeros((n_mels, n_freq), dtype=np.float64)
    for m in range(1, n_mels + 1):
        left, center, right = bins[m - 1], bins[m], bins[m + 1]
        if center > left:
            fb[m - 1, left:center] = (np.arange(left, center) - left) / (center - left)
        if right > center:
            fb[m - 1, center:right] = (right - np.arange(center, right)) / (right - center)
    return fb


def _frame_indices(n_samples: int, n_fft: int, hop: int) -> int:
    if n_samples < n_fft:
        raise ValueError(f"n_samples ({n_samples}) must be >= n_fft ({n_fft})")
    return 1 + (n_samples - n_fft) // hop


def _stft(x: jnp.ndarray, n_fft: int, hop: int, window: jnp.ndarray) -> jnp.ndarray:
    """``x``: ``(B, n_samples)`` -> complex STFT ``(B, n_frames, n_freq)``."""
    n_samples = x.shape[-1]
    n_frames = _frame_indices(n_samples, n_fft, hop)
    idx = jnp.arange(n_fft)[None, :] + hop * jnp.arange(n_frames)[:, None]  # (F, n_fft)
    frames = x[:, idx] * window                                            # (B, F, n_fft)
    return jnp.fft.rfft(frames, n=n_fft, axis=-1)                          # (B, F, n_freq)


def _istft(S: jnp.ndarray, n_fft: int, hop: int, window: jnp.ndarray,
           length: int) -> jnp.ndarray:
    """Inverse STFT (overlap-add). ``S``: ``(B, n_frames, n_freq)`` -> ``(B, length)``."""
    frames = jnp.fft.irfft(S, n=n_fft, axis=-1) * window        # (B, F, n_fft)
    b, n_frames, _ = frames.shape
    out_len = (n_frames - 1) * hop + n_fft
    out = jnp.zeros((b, out_len))
    wsum = jnp.zeros((out_len,))
    win_sq = window * window
    for f in range(n_frames):
        start = f * hop
        out = out.at[:, start:start + n_fft].add(frames[:, f])
        wsum = wsum.at[start:start + n_fft].add(win_sq)
    out = out / jnp.maximum(wsum, 1e-8)[None]
    if length <= out_len:
        return out[:, :length]
    return jnp.pad(out, ((0, 0), (0, length - out_len)))


def _griffin_lim(mag: jnp.ndarray, n_fft: int, hop: int, window: jnp.ndarray,
                 length: int, n_iters: int) -> jnp.ndarray:
    """Reconstruct a waveform from a magnitude spectrogram (zero-phase init).

    ``mag``: ``(B, n_frames, n_freq)`` -> ``(B, length)``.
    """
    S = mag.astype(jnp.complex64)
    for _ in range(n_iters):
        x = _istft(S, n_fft, hop, window, length)
        S_est = _stft(x, n_fft, hop, window)
        phase = S_est / (jnp.abs(S_est) + 1e-8)
        S = mag * phase
    return _istft(S, n_fft, hop, window, length)


# --------------------------------------------------------------------------- #
#  Stages                                                                     #
# --------------------------------------------------------------------------- #

class MelPower(SensoryTransform):
    """Raw waveform -> linear mel-power spectrogram (STFT → |·|² → mel).

    ``(n_samples,) -> (n_mels, n_frames)``. Inverse: mel pseudo-inverse to a
    linear power spectrogram, then Griffin–Lim to a waveform.
    """

    def __init__(self, n_samples: int, sr: int = 16000, n_fft: int = 1024,
                 hop: int = 512, n_mels: int = 64, fmin: float = 0.0,
                 fmax: Optional[float] = None, griffin_lim_iters: int = 24):
        fmax = float(fmax) if fmax is not None else sr / 2.0
        self.n_samples = int(n_samples)
        self.sr, self.n_fft, self.hop, self.n_mels = sr, n_fft, hop, n_mels
        self.gl_iters = int(griffin_lim_iters)
        self.n_frames = _frame_indices(self.n_samples, n_fft, hop)
        self.in_shape = (self.n_samples,)
        self.out_shape = (n_mels, self.n_frames)

        window = _hann(n_fft)
        mel_fb = _mel_filterbank(sr, n_fft, n_mels, fmin, fmax)          # (n_mels, n_freq)
        mel_pinv = np.linalg.pinv(mel_fb)                                # (n_freq, n_mels)
        self._window = jnp.asarray(window)
        self._mel_fb = jnp.asarray(mel_fb)
        self._mel_pinv = jnp.asarray(mel_pinv)

    def _forward(self, x: jnp.ndarray) -> jnp.ndarray:
        S = _stft(x, self.n_fft, self.hop, self._window)                # (B, F, n_freq)
        power = jnp.abs(S) ** 2                                          # (B, F, n_freq)
        mel = jnp.einsum('mk,bfk->bmf', self._mel_fb, power)            # (B, n_mels, F)
        return mel

    def _inverse(self, y: jnp.ndarray) -> jnp.ndarray:
        # y: (B, n_mels, n_frames) mel power -> linear power -> magnitude -> wav
        power = jnp.einsum('km,bmf->bfk', self._mel_pinv, y)            # (B, F, n_freq)
        mag = jnp.sqrt(jnp.maximum(power, 0.0))
        return _griffin_lim(mag, self.n_fft, self.hop, self._window,
                            self.n_samples, self.gl_iters)


class PowerCompression(SensoryTransform):
    """Power-law (loudness) compression ``y = (x+ε)^α``, invertible.

    Shape-preserving; ``inverse`` is ``x = y^{1/α} − ε`` (clipped at 0).
    """

    def __init__(self, shape: Tuple[int, ...], alpha: float = 1.0 / 3.0,
                 eps: float = 1e-6):
        self.in_shape = tuple(shape)
        self.out_shape = tuple(shape)
        self.alpha = float(alpha)
        self.eps = float(eps)

    def _forward(self, x: jnp.ndarray) -> jnp.ndarray:
        return jnp.power(jnp.maximum(x, 0.0) + self.eps, self.alpha)

    def _inverse(self, y: jnp.ndarray) -> jnp.ndarray:
        return jnp.maximum(jnp.power(jnp.maximum(y, 0.0), 1.0 / self.alpha) - self.eps, 0.0)


class LateralInhibition(SensoryTransform):
    """Across-frequency lateral inhibition: first-difference ``[1,-1]`` over the
    frequency axis followed by ReLU (positive spectral-change detector).

    Shape-preserving ``(F, T) -> (F, T)`` (boundary row keeps its own value).
    Inverse: reverse-cumsum over frequency (approximate through the ReLU).
    """

    def __init__(self, shape: Tuple[int, int]):
        self.in_shape = tuple(shape)
        self.out_shape = tuple(shape)

    def _forward(self, x: jnp.ndarray) -> jnp.ndarray:
        # x: (B, F, T); s[f] = x[f] - x[f+1], with x[F] := 0
        up = jnp.concatenate([x[:, 1:], jnp.zeros_like(x[:, :1])], axis=1)
        return jnp.maximum(x - up, 0.0)

    def _inverse(self, y: jnp.ndarray) -> jnp.ndarray:
        # telescoping: x[f] = sum_{k>=f} s[k]  ->  reverse-cumsum over freq
        flipped = jnp.flip(y, axis=1)
        return jnp.flip(jnp.cumsum(flipped, axis=1), axis=1)


class LeakyIntegrator(SensoryTransform):
    """Short-term temporal integration: causal EMA over time,
    ``y[t] = x[t] + a·y[t-1]`` with ``a = exp(-1/tau)``.

    Shape-preserving ``(F, T) -> (F, T)``. Inverse is the exact recurrence
    ``x[t] = y[t] - a·y[t-1]``.
    """

    def __init__(self, shape: Tuple[int, int], tau: float = 2.0):
        self.in_shape = tuple(shape)
        self.out_shape = tuple(shape)
        self.tau = float(tau)
        self.a = float(np.exp(-1.0 / max(tau, 1e-6)))

    def _forward(self, x: jnp.ndarray) -> jnp.ndarray:
        # x: (B, F, T); scan over time
        a = self.a
        xt = jnp.moveaxis(x, 2, 0)                     # (T, B, F)

        def step(carry, cur):
            c = cur + a * carry
            return c, c

        init = jnp.zeros(xt.shape[1:])
        _, ys = jax.lax.scan(step, init, xt)           # (T, B, F)
        return jnp.moveaxis(ys, 0, 2)                  # (B, F, T)

    def _inverse(self, y: jnp.ndarray) -> jnp.ndarray:
        yprev = jnp.concatenate([jnp.zeros_like(y[:, :, :1]), y[:, :, :-1]], axis=2)
        return y - self.a * yprev


class STRFBank(SensoryTransform):
    """Cortical spectrotemporal modulation filters (opt-in).

    A bank of fixed 2-D modulation filters over the (frequency, time)
    spectrogram, built from the NSL seeds — spectral ``(1-x²)e^{-x²/2}`` and
    temporal ``t²e^{-3.5t}sin(2πt)`` — dilated over ``scales`` (cyc/oct) and
    ``rates`` (Hz), with up/down direction via time-reversal. Linear (signed)
    responses, invertible via the bank's Wiener least-squares solve.
    ``(F, T) -> (n_filters, F, T)`` with ``n_filters = |scales|·|rates|·2``.

    v1 note: an approximation of the full NSL STRF (separable seed products,
    signed responses instead of analytic-magnitude); adequate as a modulation
    feature stage and kept invertible.
    """

    def __init__(self, shape: Tuple[int, int],
                 scales: Sequence[float] = (0.5, 1.0, 2.0, 4.0),
                 rates: Sequence[float] = (2.0, 4.0, 8.0, 16.0),
                 freq_size: int = 11, time_size: int = 11):
        f, t = int(shape[0]), int(shape[1])
        self.in_shape = (f, t)
        fs = min(freq_size, f if f % 2 == 1 else f - 1)
        ts = min(time_size, t if t % 2 == 1 else t - 1)
        kernels = self._build_kernels(scales, rates, fs, ts)
        self.n_filters = len(kernels)
        self.out_shape = (self.n_filters, f, t)
        self._bank = _filters.FFTConvBank(kernels, (f, t))

    @staticmethod
    def _build_kernels(scales, rates, fs, ts):
        # spectral seed over a normalized freq window, temporal seed over time
        xr = np.linspace(-2.0, 2.0, fs)
        tt = np.linspace(0.0, 2.0, ts)
        kernels = []
        for Om in scales:
            hs = (1.0 - (Om * xr) ** 2) * np.exp(-(Om * xr) ** 2 / 2.0)
            hs = hs - hs.mean()
            for w in rates:
                ht = (w * tt) ** 2 * np.exp(-3.5 * w * tt) * np.sin(2.0 * np.pi * w * tt / max(rates))
                ht = ht - ht.mean()
                down = np.outer(hs, ht)
                up = np.outer(hs, ht[::-1])
                for k in (down, up):
                    n = np.linalg.norm(k)
                    kernels.append(k / n if n > 0 else k)
        return kernels

    def _forward(self, x: jnp.ndarray) -> jnp.ndarray:
        return self._bank.apply(x)                     # (B, n_filters, F, T)

    def _inverse(self, y: jnp.ndarray) -> jnp.ndarray:
        return self._bank.reconstruct(y)               # (B, F, T)


# --------------------------------------------------------------------------- #
#  AuditoryInput                                                              #
# --------------------------------------------------------------------------- #

class AuditoryInput(SensoryInput):
    """A ``SensoryInput`` layer with a fixed cochlea→cortex feature transform.

    Args:
        n_samples: raw waveform length (fixed; zero-pad/truncate upstream).
        sr, n_fft, hop, n_mels, fmin, fmax: STFT / mel parameters.
        alpha: power-law compression exponent (1/3 = cube-root loudness).
        tau: leaky-integrator time constant (in frames).
        lateral_inhibition: include the across-frequency ``[1,-1]``+ReLU stage.
        strf: append the opt-in cortical ``STRFBank`` stage.
        griffin_lim_iters: iterations for the waveform inverse.
        activation, label: forwarded to ``SensoryInput``.

    Default output feature map: ``(n_mels, n_frames)`` (no STRF).
    """

    def __init__(self, n_samples: int, *, sr: int = 16000, n_fft: int = 1024,
                 hop: int = 512, n_mels: int = 64, fmin: float = 0.0,
                 fmax: Optional[float] = None, alpha: float = 1.0 / 3.0,
                 tau: float = 2.0, lateral_inhibition: bool = True,
                 strf: bool = False, griffin_lim_iters: int = 24,
                 activation=None, label: Optional[str] = None):
        mel = MelPower(n_samples, sr, n_fft, hop, n_mels, fmin, fmax, griffin_lim_iters)
        shape = mel.out_shape
        stages = [mel, PowerCompression(shape, alpha)]
        if lateral_inhibition:
            stages.append(LateralInhibition(shape))
        stages.append(LeakyIntegrator(shape, tau))
        if strf:
            stages.append(STRFBank(shape))
        super().__init__(Sequential(stages), activation=activation, label=label)
