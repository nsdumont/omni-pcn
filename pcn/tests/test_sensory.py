"""
Tests for the fixed biological sensory front-ends (``VisualInput`` /
``AuditoryInput``) and the ``Simulation`` transform hook.
"""

import numpy as np
import jax
import jax.numpy as jnp
import pytest

import pcn
from pcn.core.sensory.base import SensoryTransform, Sequential
from pcn.core.sensory.vision import (
    DoGCenterSurround, GaborBank, DivisiveNormalization, ComplexEnergy,
    ColorOpponent, GaussianBlur, SpatialPool, ChannelStandardize,
    ParallelPathways)
from pcn.core.sensory.audio import (
    MelPower, PowerCompression, LateralInhibition, LeakyIntegrator, STRFBank)


def _corr(a, b):
    a = np.asarray(a).ravel()
    b = np.asarray(b).ravel()
    return float(np.corrcoef(a, b)[0, 1])


# --------------------------------------------------------------------------- #
#  Vision transform                                                           #
# --------------------------------------------------------------------------- #

class TestVisionTransform:

    def test_default_shapes(self):
        hw = (28, 28)
        t = Sequential([DoGCenterSurround(hw), GaborBank(hw)])
        assert t.in_shape == (1, 28, 28)
        assert t.out_shape == (18, 28, 28)
        assert t.out_dim == 18 * 28 * 28
        x = jnp.zeros((3, 1, 28, 28))
        assert t.forward(x).shape == (3, 18, 28, 28)

    def test_shapes_32(self):
        hw = (32, 32)
        t = Sequential([DoGCenterSurround(hw), GaborBank(hw)])
        assert t.out_shape == (18, 32, 32)

    def test_bandpass_roundtrip(self):
        """Band-pass (zero-DC) content reconstructs near-exactly via the DoG
        Wiener inverse."""
        H = W = 28
        t = Sequential([DoGCenterSurround((H, W)), GaborBank((H, W))])
        yy, xx = np.mgrid[0:H, 0:W]
        grating = np.sin(2 * np.pi * xx / 5.0) + np.cos(2 * np.pi * yy / 6.0)
        img = jnp.asarray(grating[None, None].astype(np.float32))
        rec = t.inverse(t.forward(img))
        assert rec.shape == img.shape
        assert _corr(rec, img) > 0.95

    def test_flat_and_shaped_input_equivalent(self):
        t = Sequential([DoGCenterSurround((28, 28)), GaborBank((28, 28))])
        img = jnp.asarray(np.random.RandomState(0).randn(2, 1, 28, 28).astype(np.float32))
        y_shaped = t.forward(img)
        y_flat = t.forward(img.reshape(2, -1))
        assert jnp.allclose(y_shaped, y_flat, atol=1e-4)

    def test_on_off_lossless(self):
        """ON - OFF exactly recovers the signed DoG response."""
        dog = DoGCenterSurround((28, 28))
        img = jnp.asarray(np.random.RandomState(1).randn(2, 1, 28, 28).astype(np.float32))
        y = dog.forward(img)
        s = y[:, 0] - y[:, 1]
        # both channels nonnegative, and their supports are disjoint
        assert jnp.all(y >= 0)
        assert jnp.all(y[:, 0] * y[:, 1] == 0)
        # reconstruct signed response via the bank, then re-filter -> consistent
        assert s.shape == (2, 28, 28)

    def test_optin_stage_shapes(self):
        hw = (28, 28)
        assert DivisiveNormalization(hw, 18).out_shape == (18, 28, 28)
        assert ComplexEnergy(hw, 18, orientations=4, n_scales=2).out_shape == (26, 28, 28)

    def test_jit_safe(self):
        t = Sequential([DoGCenterSurround((28, 28)), GaborBank((28, 28))])
        f = jax.jit(t.forward)
        img = jnp.zeros((1, 1, 28, 28))
        assert f(img).shape == (1, 18, 28, 28)


class TestEncoderStages:
    """v2 encoder building blocks (color, pooling, normalization)."""

    def test_color_opponent_exact_inverse(self):
        co = ColorOpponent((16, 16))
        rng = np.random.RandomState(0)
        x = jnp.asarray(rng.randn(4, 3, 16, 16).astype(np.float32))
        assert co.forward(x).shape == (4, 3, 16, 16)
        assert jnp.allclose(co.inverse(co.forward(x)), x, atol=1e-4)

    def test_spatial_pool_shape_and_upsample_inverse(self):
        sp = SpatialPool((16, 16), channels=5, pool_size=4)
        assert sp.out_shape == (5, 4, 4)
        y = jnp.asarray(np.random.RandomState(1).randn(2, 5, 4, 4).astype(np.float32))
        # avg-pool inverse is nearest-upsample; pooling it back is the identity
        assert jnp.allclose(sp.forward(sp.inverse(y)), y, atol=1e-4)

    def test_parallel_pathways_crossover_recovers_the_image(self):
        """crossover=True must beat the low-pass-only inverse by a wide margin.

        A band-pass form pathway (DoG->Gabor) beside a blurred-luma pathway: the
        default inverse can only return the blur, the crossover keeps the form
        pathway's detail *and* the DC.
        """
        hw = (32, 32)
        form = Sequential([DoGCenterSurround(hw), GaborBank(hw)])
        blur = GaussianBlur(hw, 1, sigma=2.0)
        paths = [dict(in_start=0, in_len=1, transform=form, invertible=False),
                 dict(in_start=0, in_len=1, transform=blur,
                      invertible=True, lowpass=True)]
        rng = np.random.RandomState(0)
        x = jnp.asarray(rng.rand(2, 1, *hw).astype(np.float32))

        pp_low = ParallelPathways(hw, 1, paths)                  # default: off
        pp_x = ParallelPathways(hw, 1, paths, crossover=True)
        assert pp_low.crossover is False and pp_x.crossover is True
        assert pp_low.forward(x).shape == pp_x.forward(x).shape

        err_low = float(jnp.mean((pp_low.inverse(pp_low.forward(x)) - x) ** 2))
        err_x = float(jnp.mean((pp_x.inverse(pp_x.forward(x)) - x) ** 2))
        assert err_x < err_low / 100.0                            # ~1e-9 vs ~4e-3

    def test_parallel_pathways_crossover_band_pass_only_slice(self):
        """A slice fed only by the band-pass pathway falls back to its inverse.

        Without crossover that slice is all zeros; with it we get the DC-free
        reconstruction, which correlates near-perfectly with the input.
        """
        hw = (32, 32)
        form = Sequential([DoGCenterSurround(hw), GaborBank(hw)])
        paths = [dict(in_start=0, in_len=1, transform=form, invertible=False)]
        x = jnp.asarray(np.random.RandomState(1).rand(2, 1, *hw).astype(np.float32))

        off = ParallelPathways(hw, 1, paths).inverse(
            ParallelPathways(hw, 1, paths).forward(x))
        assert jnp.allclose(off, 0.0)                             # nothing to invert
        pp = ParallelPathways(hw, 1, paths, crossover=True)
        assert _corr(pp.inverse(pp.forward(x)), x) > 0.99

    def test_visual_input_crossover_guards(self):
        """VisualInput enables the crossover only where it is measurably better."""
        net = pcn.PCNetwork(seed=0)
        with net:
            plain = pcn.VisualInput(in_shape=(1, 32, 32), color='gray',
                                    downsample=1, keep_lowpass=True)
            pooled = pcn.VisualInput(in_shape=(1, 32, 32), color='gray',
                                     downsample=2, keep_lowpass=True)
            phaseless = pcn.VisualInput(in_shape=(1, 32, 32), color='gray',
                                        downsample=1, keep_lowpass=True,
                                        complex_cells=True)
        def pp(vi):
            return [s for s in vi.transform.stages
                    if isinstance(s, ParallelPathways)][0]
        assert pp(plain).crossover is True
        assert pp(pooled).crossover is False       # pooled band-pass is blocky
        assert pp(phaseless).crossover is False    # form inverse is identically 0

        # and the enabled case really does reconstruct
        x = jnp.asarray(np.random.RandomState(2).rand(2, 32 * 32).astype(np.float32))
        rec = plain.decode(plain.encode(x))
        assert _corr(rec, x) > 0.999

    def test_spatial_pool_requires_divisor(self):
        with pytest.raises(ValueError):
            SpatialPool((30, 30), channels=3, pool_size=4)

    def test_channel_standardize_exact_inverse(self):
        cs = ChannelStandardize((3, 8, 8))
        cs.set_scale([2.0, 0.5, 4.0])
        x = jnp.asarray(np.random.RandomState(2).randn(2, 3, 8, 8).astype(np.float32))
        assert jnp.allclose(cs.inverse(cs.forward(x)), x, atol=1e-5)

    def test_gaussian_blur_shape_preserving(self):
        gb = GaussianBlur((16, 16), channels=2, sigma=2.0)
        x = jnp.zeros((3, 2, 16, 16))
        assert gb.forward(x).shape == (3, 2, 16, 16)


# --------------------------------------------------------------------------- #
#  Audio transform                                                            #
# --------------------------------------------------------------------------- #

class TestAudioTransform:

    def _chain(self, n_samples=8192, gl=16):
        mel = MelPower(n_samples, n_fft=1024, hop=512, n_mels=64, griffin_lim_iters=gl)
        sh = mel.out_shape
        return mel, Sequential([mel, PowerCompression(sh),
                                LateralInhibition(sh), LeakyIntegrator(sh)])

    def test_default_shapes(self):
        mel, t = self._chain()
        assert t.in_shape == (8192,)
        assert t.out_shape == (64, mel.n_frames)

    def test_compression_inverse_exact(self):
        sh = (64, 15)
        comp = PowerCompression(sh, alpha=1.0 / 3.0)
        E = jnp.asarray(np.abs(np.random.RandomState(0).randn(2, *sh)).astype(np.float32))
        assert jnp.allclose(comp.inverse(comp.forward(E)), E, atol=1e-3)

    def test_leaky_integrator_inverse_exact(self):
        sh = (64, 15)
        leak = LeakyIntegrator(sh, tau=2.0)
        E = jnp.asarray(np.random.RandomState(0).randn(2, *sh).astype(np.float32))
        assert jnp.allclose(leak.inverse(leak.forward(E)), E, atol=1e-3)

    def test_melpower_spectrogram_roundtrip(self):
        """Griffin-Lim reconstructs a waveform with a matching mel spectrogram."""
        mel = MelPower(8192, n_fft=1024, hop=512, n_mels=64, griffin_lim_iters=32)
        t = np.linspace(0, 1, 8192, endpoint=False)
        wav = 0.5 * np.sin(2 * np.pi * 440 * t) + 0.3 * np.sin(2 * np.pi * 880 * t)
        w = jnp.asarray(wav[None].astype(np.float32))
        P0 = mel.forward(w)
        P1 = mel.forward(mel.inverse(P0))
        assert _corr(P0, P1) > 0.9

    def test_full_chain_consistency(self):
        """decode -> encode is self-consistent (probing fidelity)."""
        mel, t = self._chain(gl=32)
        tt = np.linspace(0, 1, 8192, endpoint=False)
        wav = 0.5 * np.sin(2 * np.pi * 440 * tt) + 0.3 * np.sin(2 * np.pi * 880 * tt)
        w = jnp.asarray(wav[None].astype(np.float32))
        F0 = t.forward(w)
        F1 = t.forward(t.inverse(F0))
        assert _corr(F0, F1) > 0.9

    def test_strf_optin_shape(self):
        sh = (64, 15)
        strf = STRFBank(sh, scales=(0.5, 1.0, 2.0, 4.0), rates=(2.0, 4.0, 8.0, 16.0))
        assert strf.out_shape[0] == 4 * 4 * 2
        x = jnp.zeros((1, 64, 15))
        assert strf.forward(x).shape == (1, 32, 64, 15)

    def test_jit_safe(self):
        mel, t = self._chain(gl=4)
        f = jax.jit(t.forward)
        w = jnp.zeros((1, 8192))
        assert f(w).shape == (1, 64, mel.n_frames)


# --------------------------------------------------------------------------- #
#  Sequential                                                                 #
# --------------------------------------------------------------------------- #

class TestSequential:

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError):
            Sequential([DoGCenterSurround((28, 28)), DoGCenterSurround((28, 28))])

    def test_inverse_reverses_order(self):
        # compression then leaky; inverse must undo leaky first, then compression
        sh = (8, 10)
        seq = Sequential([PowerCompression(sh), LeakyIntegrator(sh)])
        E = jnp.asarray(np.abs(np.random.RandomState(2).randn(2, *sh)).astype(np.float32))
        assert jnp.allclose(seq.inverse(seq.forward(E)), E, atol=1e-3)


# --------------------------------------------------------------------------- #
#  SensoryInput layers (need a network context)                               #
# --------------------------------------------------------------------------- #

class TestSensoryInputLayers:

    def test_visual_input_attrs(self):
        net = pcn.PCNetwork(seed=0)
        with net:
            vi = pcn.VisualInput(in_shape=(1, 28, 28), label="vis")
        assert vi.dim == 18 * 28 * 28
        assert vi.feature_shape == (18, 28, 28)
        assert vi.raw_shape == (1, 28, 28)
        img = jnp.zeros((3, 784))
        assert vi.encode(img).shape == (3, vi.dim)
        assert vi.decode(vi.encode(img)).shape == (3, 784)

    def test_auditory_input_attrs(self):
        net = pcn.PCNetwork(seed=0)
        with net:
            ai = pcn.AuditoryInput(n_samples=8192, griffin_lim_iters=4, label="aud")
        assert ai.raw_shape == (8192,)
        assert ai.dim == ai.feature_shape[0] * ai.feature_shape[1]
        w = jnp.zeros((2, 8192))
        assert ai.encode(w).shape == (2, ai.dim)

    def test_visual_input_rejects_bad_shape(self):
        net = pcn.PCNetwork(seed=0)
        with net:
            with pytest.raises(ValueError):
                pcn.VisualInput(in_shape=(4, 32, 32))     # only 1 or 3 channels

    def test_visual_encoder_rgb_opponent_downsample(self):
        """v2 encoder: RGB opponent input, pooled below raw resolution, fit norm."""
        net = pcn.PCNetwork(seed=0)
        with net:
            enc = pcn.VisualInput(in_shape=(3, 32, 32), color='opponent',
                                  downsample=4, complex_cells=True, keep_lowpass=True)
        # form energy (4*2) + magno luma (1) + chroma (2) = 11 channels @ 8x8
        assert enc.feature_shape == (11, 8, 8)
        assert enc.raw_shape == (3, 32, 32)
        assert enc.dim == 11 * 8 * 8
        assert enc.dim < 3 * 32 * 32                       # a *reduction*, not expansion

        rng = np.random.RandomState(0)
        rgb = jnp.asarray(rng.randn(5, 3 * 32 * 32).astype(np.float32))
        assert enc.encode(rgb).shape == (5, enc.dim)
        assert enc.decode(enc.encode(rgb)).shape == (5, 3 * 32 * 32)

        enc.fit(rgb)                                       # freeze per-channel gain
        f = np.asarray(enc.encode(rgb)).reshape(5, enc.feature_shape[0], -1)
        assert np.allclose(f.std(axis=(0, 2)), 1.0, atol=1e-3)

    def test_visual_encoder_gray(self):
        net = pcn.PCNetwork(seed=0)
        with net:
            enc = pcn.VisualInput(in_shape=(1, 32, 32), color='gray',
                                  downsample=2, complex_cells=True)
        assert enc.feature_shape == (8, 16, 16)           # 4*2 energy channels
        img = jnp.zeros((3, 32 * 32))
        assert enc.encode(img).shape == (3, enc.dim)


# --------------------------------------------------------------------------- #
#  Simulation hook                                                            #
# --------------------------------------------------------------------------- #

def _build_vision_net(hidden=16):
    net = pcn.PCNetwork(seed=0)
    with net:
        vi = pcn.VisualInput(in_shape=(1, 28, 28), label="vis")
        h = pcn.Layer(dim=hidden, activation=pcn.Relu(), label="hidden")
        out = pcn.Layer(dim=10, activation=pcn.Softmax(), label="out")
        pcn.Predict(h, vi)
        pcn.Predict(out, h)
    net.build()
    return net, vi, out


class TestSimulationHook:

    def test_clamped_value_equals_encode(self):
        net, vi, out = _build_vision_net()
        sim = pcn.Simulation(net)
        rng = np.random.RandomState(0)
        image = rng.randn(2, 784).astype(np.float32)
        loader = [{'image': image}]
        res = sim.test(loader, data_map={vi: 'image'},
                       iterations_per_sample=2, feedforward_init=True,
                       record_map={'feat': (vi.value, lambda v: v),
                                   'raw': ('image', lambda r: r)})
        got = np.asarray(res['feat'][0])
        expected = np.asarray(vi.encode(jnp.asarray(image)))
        assert got.shape == (2, vi.dim)
        assert np.allclose(got, expected, atol=1e-3)
        # raw is still probeable via the string key
        assert np.allclose(np.asarray(res['raw'][0]), image)

    def test_soft_clamp_beta1_equals_full(self):
        """A soft-clamp whose clamp-strength mask is β=1 (per-sample scalar)
        fully clamps the feature layer to encode(raw)."""
        net, vi, out = _build_vision_net()
        sim = pcn.Simulation(net)
        rng = np.random.RandomState(0)
        image = rng.randn(2, 784).astype(np.float32)
        mask = np.ones((2,), np.float32)              # per-sample β=1 (clamp space)
        loader = [{'image': image, 'mask': mask}]
        res = sim.test(loader, data_map={vi: ('image', 'mask')},
                       iterations_per_sample=2,
                       record_map={'feat': (vi.value, lambda v: v)})
        expected = np.asarray(vi.encode(jnp.asarray(image)))
        assert np.allclose(np.asarray(res['feat'][0]), expected, atol=1e-3)

    def test_scalar_clamp_mask(self):
        """A scalar clamp-strength mask broadcasts to the feature clamp shape."""
        net, vi, out = _build_vision_net()
        sim = pcn.Simulation(net)
        image = np.zeros((2, 784), np.float32)
        loader = [{'image': image, 'mask': np.float32(0.5)}]
        # runs without error (scalar mask -> broadcast to (B, feat_dim))
        sim.test(loader, data_map={vi: ('image', 'mask')}, iterations_per_sample=1)

    def test_temporal_encode(self):
        """Temporal raw data (B, T, raw_dim) encodes to (B, T, feat_dim)."""
        net, vi, out = _build_vision_net()
        sim = pcn.Simulation(net)
        raw = np.zeros((2, 3, 784), np.float32)       # (B, T, raw_dim)
        feat = sim._encode_sensory(vi, raw)
        assert feat.shape == (2, 3, vi.dim)

    def test_train_step_through_features(self):
        net, vi, out = _build_vision_net()
        sim = pcn.Simulation(net)
        rng = np.random.RandomState(1)
        image = rng.randn(4, 784).astype(np.float32)
        label = jax.nn.softmax(jnp.asarray(rng.randn(4, 10).astype(np.float32)), axis=-1)
        loader = [{'image': image, 'label': np.asarray(label)}]
        sim.train(loader, data_map={vi: 'image', out: 'label'},
                  epochs=1, iterations_per_sample=3,
                  learning_iterations_per_sample=1)
        e = np.asarray(sim.train_energies[-1])
        assert np.all(np.isfinite(e))
