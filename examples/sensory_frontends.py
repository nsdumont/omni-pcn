"""Fixed biological sensory front-ends: ``VisualInput`` and ``AuditoryInput``.

A predictive-coding network should relax over cortical-level features, not raw
pixels/audio. ``VisualInput`` (retina DoG center-surround → V1 Gabor simple
cells) and ``AuditoryInput`` (cochlear mel-power → compression → lateral
inhibition → temporal integration) apply a fixed, approximately invertible
transform to the raw input *outside* the energy loop. Predict/Project/Modulate
connections then attach to the transformed features; ``decode`` maps generated
feature values back to pixel/audio space for probing.

This example (CPU-friendly, synthetic data — no downloads):

1. Encodes/decodes a synthetic image and waveform, reporting round-trip fidelity.
2. Builds a tiny generative PC net over ``VisualInput`` features, settles it on a
   clamped image, and decodes the input layer's settled value back to pixels.

Run:  uv run python examples/sensory_frontends.py
"""
import numpy as np
import jax
import jax.numpy as jnp

import pcn


def _corr(a, b):
    a = np.asarray(a).ravel()
    b = np.asarray(b).ravel()
    return float(np.corrcoef(a, b)[0, 1])


def vision_roundtrip():
    print("== VisualInput (retina DoG ON/OFF -> V1 Gabor bank) ==")
    net = pcn.PCNetwork(seed=0)
    with net:
        vi = pcn.VisualInput(in_shape=(1, 28, 28), label="vis")
    print(f"  raw {vi.raw_shape} -> features {vi.feature_shape} (dim {vi.dim})")

    # A band-pass test image (an oriented grating) — DoG passes it.
    H = W = 28
    yy, xx = np.mgrid[0:H, 0:W]
    img = (np.sin(2 * np.pi * xx / 5.0) + np.cos(2 * np.pi * yy / 6.0))
    img = jnp.asarray(img[None, None].astype(np.float32))       # (1,1,28,28)

    feats = vi.encode(img)                                       # (1, 18*28*28)
    recon = vi.decode(feats).reshape(H, W)
    print(f"  encode -> {feats.shape}, decode -> {recon.shape}, "
          f"round-trip corr = {_corr(recon, img):.3f}\n")


def audio_roundtrip():
    print("== AuditoryInput (mel power -> compression -> lateral inhib -> EMA) ==")
    n_samples = 8192
    net = pcn.PCNetwork(seed=1)
    with net:
        ai = pcn.AuditoryInput(n_samples=n_samples, n_mels=64,
                               griffin_lim_iters=32, label="aud")
    print(f"  raw {ai.raw_shape} -> features {ai.feature_shape} (dim {ai.dim})")

    t = np.linspace(0, 1, n_samples, endpoint=False)
    wav = 0.5 * np.sin(2 * np.pi * 440 * t) + 0.3 * np.sin(2 * np.pi * 880 * t)
    w = jnp.asarray(wav[None].astype(np.float32))

    feats = ai.encode(w)
    wav_rec = ai.decode(feats)
    # Griffin-Lim recovers magnitude with arbitrary phase, so compare *features*
    # of the reconstruction (self-consistency), not raw samples.
    feats_rec = ai.encode(wav_rec)
    print(f"  encode -> {feats.shape}, decode -> {wav_rec.shape}")
    print(f"  feature self-consistency corr = {_corr(feats, feats_rec):.3f}\n")


def generative_probe():
    print("== Generative PC net over VisualInput features ==")
    net = pcn.PCNetwork(seed=2)
    with net:
        vi = pcn.VisualInput(in_shape=(1, 28, 28), label="vis")
        h = pcn.Layer(dim=32, activation=pcn.Relu(), label="hidden")
        out = pcn.Layer(dim=10, activation=pcn.Softmax(), label="out")
        pcn.Predict(h, vi)      # hidden predicts the V1 features
        pcn.Predict(out, h)
    net.build()

    sim = pcn.Simulation(net)
    rng = np.random.RandomState(0)
    H = W = 28
    yy, xx = np.mgrid[0:H, 0:W]
    img = np.sin(2 * np.pi * xx / 6.0).astype(np.float32).reshape(1, -1)
    loader = [{'image': np.repeat(img, 4, axis=0)}]

    # Settle the network with the image clamped; read the input layer's value
    # (the V1 features) and decode it back to pixels.
    res = sim.test(loader, data_map={vi: 'image'}, iterations_per_sample=5,
                   record_map={'vis_features': (vi.value, lambda v: v)})
    settled = jnp.asarray(res['vis_features'][0])               # (4, dim)
    pixels = vi.decode(settled)[0].reshape(H, W)
    ref = img.reshape(H, W)
    print(f"  settled feature value -> decode -> pixels {pixels.shape}, "
          f"corr with clamped image = {_corr(pixels, ref):.3f}")
    print("  (features feed Predict connections; decode recovers the image "
          "for visualization)\n")


if __name__ == "__main__":
    vision_roundtrip()
    audio_roundtrip()
    generative_probe()
    print("Done.")
