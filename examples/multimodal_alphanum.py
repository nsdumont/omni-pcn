"""Multimodal (image + audio) bidirectional PCN on handwritten + spoken alphanumerics.

36-way recognition of digits 0-9 and letters A-Z from two modalities that share one
label: EMNIST glyphs (784 px) and log-mel spectrograms of the spoken character name
(8 mel channels x 33 frames = 264 dims; neural-TTS clips in 42 voices). One network
classifies from both modalities, from either alone, and generates one modality from
the other -- all by relaxing whatever is left unclamped:

    image(784) <-> h_img(512) --+
                                +--> label(36)   one summed Predict([h_img, h_aud], label)
    audio(264) <-> h_aud(512) --+
    label --> h_img, h_aud     weak top-down edges (init log-precision -2)
    image --> image            diagonal-masked lateral: each pixel predicted by the others

Three ingredients make *one-pass* cross-modal generation work (clamp the audio, relax,
read the image -- the label stays free; there is no argmax/commit step):

1. Joint label fusion. With a single summed label prediction, clamping only the audio
   leaves a shared error that asks the free image stream to supply *the residual of
   the sum* -- a class-consistent target that the top-down edges then render.
2. A sharp label softmax (temperature 0.25): a clamped one-hot produces a peaked
   drive; plain softmax(one-hot) is nearly uniform.
3. Soft-clamp modality dropout in training: with prob 0.5 one modality is only
   *nudged* -- ``data_map={l_audio: ('audio', 'audio_mask')}`` with mask 0.5 gives
   v = 0.5*data + 0.5*v_inference -- while the other modality and the label are
   clamped, so the network learns to predict a modality it is not being shown.

The audio test split holds out (class, voice) combinations: every voice and every
class is seen in training, just not together.

After 8 epochs (single seed, ~4 min on an RTX 5090): classification ~82% with both
modalities, ~53% image only, ~47% audio only; one-pass generation scored by nearest
class-mean of the generated modality ~33% in both directions (chance 2.8%). Saves a
grid of generated glyphs and spectrograms next to this script.

Downloads EMNIST (~560 MB) to ./data on first run. If ./data/spoken_alphanum is
missing, runs generate_alphanum_audio.py to synthesize it (edge-tts, online, ~5 min;
needs ``uv sync --group audio``).
"""
import re
import subprocess
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import scipy.io.wavfile
import scipy.signal
from torchvision import datasets

import pcn

N_CLASSES = 36
CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
IMAGE_DIM = 784
N_MELS, N_FRAMES = 8, 33     # log-mel channels x STFT frames of a 1 s clip
AUDIO_DIM = N_MELS * N_FRAMES
HIDDEN = 512
BATCH_SIZE = 64
N_EPOCHS = 8
EPOCH_SAMPLES = 10000       # random same-class (image, audio) pairs drawn per epoch
TEST_SAMPLES = 2500
N_ITERS = 24                #  iterations per batch during learning
N_EVAL_ITERS = 48            # relaxation for classification
N_GEN_ITERS = 100            # longer relaxation to generate 
GEN_LOG_PRECISION = -2       # top-down edges start weak
LATERAL_LOG_PRECISION = -2  
LABEL_TEMP = 0.25
SOFTDROP_P, SOFTDROP_BETA = 0.5, 0.5
HOLDOUT_FRAC = 0.2           # voices held out per class for the audio test split
GEN_BATCHES = 16             # test batches scored for generation
EMNIST_MEAN, EMNIST_STD = 0.1307, 0.3081
DATA_DIR = Path("./data")
AUDIO_DIR = DATA_DIR / "spoken_alphanum"


# --------------------------------------------------------------------------- data
def ensure_audio(audio_dir):
    """Synthesize the spoken audio with generate_alphanum_audio.py if it is missing."""
    if any(audio_dir.rglob("*.wav")):
        return
    print(f"Spoken-alphanumeric audio not found in {audio_dir} -- creating it now with "
          "generate_alphanum_audio.py (edge-tts, online; ~5 min) ...", flush=True)
    script = Path(__file__).with_name("generate_alphanum_audio.py")
    subprocess.run([sys.executable, str(script), "--out", str(audio_dir)], check=True)


def load_emnist(train):
    """EMNIST 'byclass' restricted to its first 36 classes (digits + uppercase)."""
    ds = datasets.EMNIST(root=DATA_DIR, split="byclass", train=train, download=True)
    labels = ds.targets.numpy()
    images = (ds.data.numpy().astype(np.float32) / 255.0 - EMNIST_MEAN) / EMNIST_STD
    images = images.transpose(0, 2, 1).reshape(-1, IMAGE_DIM)   # EMNIST is stored transposed
    keep = labels < N_CLASSES
    return images[keep], labels[keep]


def mel_filterbank(n_mels, n_fft, sr):
    """Triangular mel filterbank over 0..sr/2, shape (n_mels, n_fft // 2 + 1)."""
    mel_pts = np.linspace(0, 2595 * np.log10(1 + sr / 2 / 700), n_mels + 2)
    bins = np.floor((n_fft + 1) * 700 * (10 ** (mel_pts / 2595) - 1) / sr).astype(int)
    fb = np.zeros((n_mels, n_fft // 2 + 1))
    for i, (lo, c, hi) in enumerate(zip(bins[:-2], bins[1:-1], bins[2:], strict=True)):
        fb[i, lo:c] = (np.arange(lo, c) - lo) / max(c - lo, 1)
        fb[i, c:hi] = (hi - np.arange(c, hi)) / max(hi - c, 1)
    return fb


def load_audio(audio_dir, n_fft=1024, hop=512):
    """Every clip as a flat log-mel spectrogram (N, AUDIO_DIM), with class and speaker."""
    feats, labels, speakers = [], [], []
    fb = None
    for path in sorted(audio_dir.rglob("*.wav")):
        sr, wav = scipy.io.wavfile.read(path)
        fb = mel_filterbank(N_MELS, n_fft, sr) if fb is None else fb
        _, _, stft = scipy.signal.stft(wav.astype(np.float32) / 32768.0, fs=sr,
                                       nperseg=n_fft, noverlap=n_fft - hop)
        spec = np.log(fb @ np.abs(stft) ** 2 + 1e-9)[:, :N_FRAMES]
        feats.append(np.pad(spec, ((0, 0), (0, N_FRAMES - spec.shape[1]))).ravel())
        labels.append(int(path.parent.name[:2]))
        speakers.append(int(re.search(r"_v(\d+)\.wav$", path.name).group(1)))
    return np.array(feats, dtype=np.float32), np.array(labels), np.array(speakers)


def split_audio(feats, labels, speakers, rng):
    """Hold out HOLDOUT_FRAC of each class's voices; z-score with train statistics."""
    test = np.zeros(len(labels), dtype=bool)
    for c in range(N_CLASSES):
        voices = np.unique(speakers[labels == c])
        held = rng.choice(voices, max(1, round(len(voices) * HOLDOUT_FRAC)), replace=False)
        test |= (labels == c) & np.isin(speakers, held)
    feats = (feats - feats[~test].mean(0)) / (feats[~test].std(0) + 1e-9)
    return (feats[~test], labels[~test]), (feats[test], labels[test])


def make_batches(images, audio, n, rng):
    """`n` random same-class (image, audio) pairs -- class drawn uniformly -- as dict batches."""
    (x_img, y_img), (x_aud, y_aud) = images, audio
    img_idx = [np.flatnonzero(y_img == c) for c in range(N_CLASSES)]
    aud_idx = [np.flatnonzero(y_aud == c) for c in range(N_CLASSES)]
    classes = rng.integers(N_CLASSES, size=n)
    i_img = np.array([rng.choice(img_idx[c]) for c in classes])
    i_aud = np.array([rng.choice(aud_idx[c]) for c in classes])
    batches = []
    for s in range(0, n - BATCH_SIZE + 1, BATCH_SIZE):
        sl = slice(s, s + BATCH_SIZE)
        batches.append({"image": x_img[i_img[sl]], "audio": x_aud[i_aud[sl]],
                        "label": np.eye(N_CLASSES, dtype=np.float32)[classes[sl]],
                        "label_idx": classes[sl]})
    return batches


def batch_accuracy(output_values, labels):
    """Fraction of a batch whose argmax output matches the one-hot label."""
    return float(jnp.mean(jnp.argmax(output_values, -1) == jnp.argmax(labels, -1)))


# ------------------------------------------------------------------------ network
def build_network():
    """Two-stream bidirectional PCN with joint label fusion; returns net, layers, gen edges."""
    net = pcn.PCNetwork(seed=10)
    net.config(use_bias=True, learn_precision_weights=False, learn_precision_bias=False)
    with net:
        l_img = pcn.Layer(dim=IMAGE_DIM, activation=pcn.Direct(), label="image")
        l_aud = pcn.Layer(dim=AUDIO_DIM, activation=pcn.Direct(), label="audio")
        l_h_img = pcn.Layer(dim=HIDDEN, activation=pcn.LeakyRelu(), label="hidden_img")
        l_h_aud = pcn.Layer(dim=HIDDEN, activation=pcn.LeakyRelu(), label="hidden_aud")
        l_label = pcn.Layer(dim=N_CLASSES, activation=pcn.Softmax(temperature=LABEL_TEMP),
                            label="label")
        # Bottom-up (discriminative) pathway
        pcn.Predict(l_img, l_h_img)
        pcn.Predict(l_aud, l_h_aud)

        # pcn.Predict([l_h_img, l_h_aud], l_label) 
        # OR
        pcn.Predict(l_h_img, l_label)
        pcn.Predict(l_h_aud, l_label)

        # Top-down (generative) pathway, initially weak. The label->hidden edges learn a
        # precision bias, so the network can tune how much to trust the label route.
        gen_img = pcn.Predict(l_h_img, l_img, init_log_precision=GEN_LOG_PRECISION)
        gen_aud = pcn.Predict(l_h_aud, l_aud, init_log_precision=GEN_LOG_PRECISION)
        pcn.Predict(l_label, l_h_img, init_log_precision=GEN_LOG_PRECISION,learn_precision_bias=True)
        pcn.Predict(l_label, l_h_aud, init_log_precision=GEN_LOG_PRECISION,learn_precision_bias=True)

        
        # Within-image prediction: pixels predict nearby ones (but not themselves)
        pcn.Predict(l_img, l_img, transformation="masked",
                   weight_mask=np.sum([np.diag(np.ones(IMAGE_DIM-np.abs(k)), k=k) for k in [-3,-2,-1,1,2,3]],axis=0) < 0.5,
                    init_log_precision=LATERAL_LOG_PRECISION)
    net.build()
    layers = {"image": l_img, "audio": l_aud, "label": l_label}
    return net, layers, {"image": gen_img, "audio": gen_aud}


# ----------------------------------------------------------------- train / eval
def train_epoch(sim, batches, layers, rng, values_opt, params_opt):
    """One pass over the batches with soft-clamp modality dropout."""
    full = {layers["image"]: "image", layers["audio"]: "audio", layers["label"]: "label"}
    for batch in batches:
        data_map = full
        if rng.random() < SOFTDROP_P:   # nudge one modality instead of clamping it
            key = "image" if rng.random() < 0.5 else "audio"
            batch = {**batch, key + "_mask": np.full_like(batch[key], SOFTDROP_BETA)}
            data_map = {**full, layers[key]: (key, key + "_mask")}
        sim.train([batch], data_map=data_map, epochs=1, iterations_per_sample=0,
                  learning_iterations_per_sample=N_ITERS, verbose=False,
                  params_optimizer=params_opt, values_optimizer=values_opt)


def classify(sim, batches, layers, values_opt):
    """Accuracy with both modalities clamped, image only, audio only (label always free)."""
    record = {"acc": ((layers["label"].value, "label"), batch_accuracy)}
    conditions = {"multi": {layers["image"]: "image", layers["audio"]: "audio"},
                  "img": {layers["image"]: "image"}, "aud": {layers["audio"]: "audio"}}
    return {name: 100 * np.mean(sim.test(batches, data_map=dmap, record_map=record,
                                         iterations_per_sample=N_EVAL_ITERS, verbose=False,
                                         values_optimizer=values_opt)["acc"])
            for name, dmap in conditions.items()}


def generate(sim, batches, layers, gen_conns, values_opt):
    """One-pass cross-modal generation: clamp one modality, relax, read the other.

    The label stays free. Scores the settled leaf value by nearest class-mean; returns the
    top-down *prediction* of the leaf (value - error of its generative edge) for display.
    """
    batches = batches[:GEN_BATCHES]
    labels = np.concatenate([b["label_idx"] for b in batches])
    results = {}
    for src, tgt in (("audio", "image"), ("image", "audio")):
        record = {"value": (layers[tgt].value, np.asarray),
                  "pred": ((layers[tgt].value, gen_conns[tgt].error),
                           lambda v, e: np.asarray(v - e))}
        r = sim.test(batches, data_map={layers[src]: src}, record_map=record,
                     iterations_per_sample=N_GEN_ITERS, verbose=False, values_optimizer=values_opt)
        data = np.concatenate([b[tgt] for b in batches])
        means = np.stack([data[labels == c].mean(0) for c in range(N_CLASSES)])
        nearest = ((np.concatenate(r["value"])[:, None] - means[None]) ** 2).sum(-1).argmin(1)
        results[tgt] = (100 * np.mean(nearest == labels), np.concatenate(r["pred"]), means)
    return results, labels


def save_figure(results, labels, path):
    """Grid: generated glyph (audio->image), generated and class-mean spectrogram (image->audio)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    show = list(range(6)) + list(range(10, 16))          # 0-5, A-F
    fig, axes = plt.subplots(3, len(show), figsize=(1.2 * len(show), 4.2))
    for j, c in enumerate(show):
        i = np.flatnonzero(labels == c)[0]
        glyph = np.clip(results["image"][1][i] * EMNIST_STD + EMNIST_MEAN, 0, 1)
        axes[0, j].imshow(glyph.reshape(28, 28), cmap="gray", vmin=0, vmax=1)
        axes[0, j].set_title(CHARS[c])
        for row, spec in ((1, results["audio"][1][i]), (2, results["audio"][2][c])):
            axes[row, j].imshow(spec.reshape(N_MELS, N_FRAMES), aspect="auto", origin="lower",
                                cmap="magma", vmin=-2, vmax=2)
    names = ["audio -> image", "image -> audio", "class-mean audio"]
    for ax, name in zip(axes[:, 0], names, strict=True):
        ax.set_ylabel(name, fontsize=8)
    for ax in axes.ravel():
        ax.set_xticks([]), ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved generation examples to {path}")


def main():
    """Train, evaluate the three classification conditions, then generate + plot."""
    print(f"JAX devices: {jax.devices()}")
    ensure_audio(AUDIO_DIR)
    img_train, img_test = load_emnist(train=True), load_emnist(train=False)
    feats, labels, speakers = load_audio(AUDIO_DIR)
    aud_train, aud_test = split_audio(feats, labels, speakers, np.random.RandomState(49))
    print(f"EMNIST {len(img_train[0])}/{len(img_test[0])} train/test glyphs, audio "
          f"{len(aud_train[0])}/{len(aud_test[0])} clips in {len(np.unique(speakers))} voices")
    rng = np.random.default_rng(42)
    test_batches = make_batches(img_test, aud_test, TEST_SAMPLES, np.random.default_rng(43))

    net, layers, gen_conns = build_network()
    total_steps = N_EPOCHS * (EPOCH_SAMPLES // BATCH_SIZE) * N_ITERS
    values_opt = optax.adam(0.05)
    params_opt = optax.adamw(optax.cosine_decay_schedule(5e-4, total_steps, alpha=0.05),
                             weight_decay=1e-3)
    precision_opt = optax.chain(optax.add_decayed_weights(0.1),
                                optax.sgd(optax.linear_schedule(0.0, 1e-3, total_steps // 4)))
    params_opt = net.multi_transform({"precision": precision_opt}, default_optim=params_opt)
    sim = pcn.Simulation(net)

    t0 = time.perf_counter()
    for epoch in range(N_EPOCHS):
        train_epoch(sim, make_batches(img_train, aud_train, EPOCH_SAMPLES, rng), layers, rng,
                    values_opt, params_opt)
        acc = classify(sim, test_batches, layers, values_opt)
        print(f"Epoch {epoch + 1}/{N_EPOCHS} | accuracy: both {acc['multi']:.1f}%  "
              f"image-only {acc['img']:.1f}%  audio-only {acc['aud']:.1f}% | "
              f"{time.perf_counter() - t0:.0f}s")

    results, gen_labels = generate(sim, test_batches, layers, gen_conns, values_opt)
    print(f"One-pass generation, nearest class-mean accuracy (chance {100 / N_CLASSES:.1f}%): "
          f"audio->image {results['image'][0]:.1f}%  image->audio {results['audio'][0]:.1f}%")
    save_figure(results, gen_labels, Path(__file__).parent / "multimodal_alphanum_generation.png")


if __name__ == "__main__":
    main()
