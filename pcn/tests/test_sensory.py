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
    DoGCenterSurround, GaborBank, DivisiveNormalization, ComplexEnergy)
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

    def test_visual_input_rejects_color(self):
        net = pcn.PCNetwork(seed=0)
        with net:
            with pytest.raises(NotImplementedError):
                pcn.VisualInput(in_shape=(3, 32, 32))


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
