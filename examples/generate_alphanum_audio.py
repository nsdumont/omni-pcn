"""Synthesize the spoken-alphanumeric audio dataset used by multimodal_alphanum.py.

36 classes -- digits 0-9 (class 0-9) and letters A-Z (class 10-35) -- each spoken by
name ("B" -> "bee", "0" -> "zero") with Microsoft Edge neural TTS (``edge-tts``,
online) in every English voice of its catalogue (~42), plus two pitch-shifted +
noise-augmented variants per clip. Clips are 16 kHz / 1 s mono WAV::

    <out>/{class:02d}_{char}/{variant:04d}_v{speaker:02d}.wav

The ``_v<NN>`` speaker tag is what the loader's train/test split keys on.
``multimodal_alphanum.py`` runs this automatically when the folder is missing; it
can also be run directly (re-runs skip clips that already exist)::

    uv run python examples/generate_alphanum_audio.py [--out DIR] [--max-voices N]

Requires the ``audio`` dependency group (``uv sync --group audio``: edge-tts and
imageio-ffmpeg, which bundles the ffmpeg binary used to decode the MP3 stream) and
network access. About 5 min for the full 42 x 36 x 3 = 4536 clips.
"""
import argparse
import asyncio
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import scipy.io.wavfile as wavfile
import scipy.signal

try:
    import edge_tts
    import imageio_ffmpeg
except ImportError as err:
    print(f"Missing dependency '{err.name}' -- run `uv sync --group audio` "
          "(installs edge-tts + imageio-ffmpeg) and retry.")
    sys.exit(1)

SR = 16000                    # sample rate of the saved clips
N_SAMPLES = SR                # fixed 1 s clip length
DIGIT_NAMES = ["zero", "one", "two", "three", "four",
               "five", "six", "seven", "eight", "nine"]
LETTER_NAMES = {
    "A": "ay", "B": "bee", "C": "see", "D": "dee", "E": "ee", "F": "eff",
    "G": "gee", "H": "aitch", "I": "eye", "J": "jay", "K": "kay", "L": "el",
    "M": "em", "N": "en", "O": "oh", "P": "pee", "Q": "cue", "R": "ar",
    "S": "ess", "T": "tee", "U": "you", "V": "vee", "W": "double you",
    "X": "ex", "Y": "why", "Z": "zee",
}


def class_meta(ci):
    """``(char, spoken_text)`` for class index ``ci`` in 0..35."""
    if ci < 10:
        return str(ci), DIGIT_NAMES[ci]
    ch = chr(ord("A") + ci - 10)
    return ch, LETTER_NAMES[ch]


async def list_voices(max_voices):
    """Deterministic (sorted) list of English voices; speaker id == position."""
    voices = await edge_tts.list_voices()
    names = sorted(v["ShortName"] for v in voices
                   if v["ShortName"].startswith("en-") and "Multilingual" not in v["ShortName"])
    return names[:max_voices] if max_voices else names


async def synth_mp3(text, voice, out_mp3, retries=3):
    """Synthesize ``text`` in ``voice`` to an MP3 file, retrying on network errors."""
    for attempt in range(retries):
        try:
            await edge_tts.Communicate(text, voice).save(out_mp3)
            if os.path.getsize(out_mp3) > 0:
                return
        except Exception:  # network hiccups / rate limits
            if attempt == retries - 1:
                raise
            await asyncio.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"empty synthesis for {voice!r}: {text!r}")


def mp3_to_array(mp3_path):
    """Decode MP3 -> 16 kHz mono float32 in [-1, 1] with (bundled) ffmpeg."""
    ffmpeg = shutil.which("ffmpeg") or imageio_ffmpeg.get_ffmpeg_exe()
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        wav_path = tmp.name
    try:
        subprocess.run([ffmpeg, "-y", "-i", mp3_path, "-ac", "1", "-ar", str(SR), wav_path],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        _, data = wavfile.read(wav_path)
    finally:
        os.unlink(wav_path)
    return data.astype(np.float32) / (32768.0 if data.dtype == np.int16 else 1.0)


def fit_length(x, n=N_SAMPLES, threshold=0.02, margin=800):
    """Trim leading/trailing silence, then center the speech in exactly ``n`` samples.

    TTS renders carry variable silence around the word; centering the active span
    keeps every clip aligned in time (a flat log-mel input has no shift invariance).
    """
    active = np.flatnonzero(np.abs(x) > threshold * np.max(np.abs(x)))
    if len(active):
        x = x[max(0, active[0] - margin):active[-1] + margin]
    if len(x) >= n:
        start = (len(x) - n) // 2
        return x[start:start + n]
    out = np.zeros(n, dtype=np.float32)
    off = (n - len(x)) // 2
    out[off:off + len(x)] = x
    return out


def augment(base, variant, rng):
    """Variant 0 = identity; odd = pitch up + noise, even = pitch down + noise."""
    if variant == 0:
        return base
    steps = 1.5 * ((variant + 1) // 2) * (1 if variant % 2 else -1)
    n = max(1, int(round(len(base) / 2.0 ** (steps / 12.0))))   # resample = pitch/tempo shift
    y = fit_length(scipy.signal.resample(base, n).astype(np.float32))
    y = y + 0.005 * float(np.std(y) + 1e-6) * rng.standard_normal(len(y)).astype(np.float32)
    return y / float(np.max(np.abs(y)) + 1e-9)


def save_wav(path, x):
    """Write a peak-normalized 1 s int16 WAV."""
    x = fit_length(x)
    x = np.clip(x / float(np.max(np.abs(x)) + 1e-9), -1.0, 1.0)
    wavfile.write(path, SR, (x * 32767).astype(np.int16))


async def gen_voice(speaker, voice, classes, out_dir, n_augment):
    """All clips for one voice; returns the number of new files written."""
    rng = np.random.RandomState(1000 + speaker)
    made = 0
    for ci in classes:
        char, text = class_meta(ci)
        cls_dir = out_dir / f"{ci:02d}_{char}"
        cls_dir.mkdir(parents=True, exist_ok=True)
        targets = [cls_dir / f"{v:04d}_v{speaker:02d}.wav" for v in range(n_augment + 1)]
        if all(t.exists() for t in targets):
            continue
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            mp3_path = tmp.name
        try:
            await synth_mp3(text, voice, mp3_path)
            base = fit_length(mp3_to_array(mp3_path))
        finally:
            os.unlink(mp3_path)
        for variant, target in enumerate(targets):
            save_wav(target, augment(base, variant, rng))
            made += 1
    return made


async def run(out_dir, max_voices, classes, n_augment, concurrency):
    """Synthesize every (voice, class) with a bounded number of voices in flight."""
    voices = await list_voices(max_voices)
    n_clips = len(voices) * len(classes) * (n_augment + 1)
    print(f"[generate] {len(voices)} voices x {len(classes)} classes x {n_augment + 1} "
          f"variants = {n_clips} clips -> {out_dir}", flush=True)
    sem = asyncio.Semaphore(concurrency)

    async def bounded(speaker, voice):
        async with sem:
            n = await gen_voice(speaker, voice, classes, out_dir, n_augment)
            print(f"  [done] v{speaker:02d} {voice}: {n} new clips", flush=True)

    await asyncio.gather(*[bounded(s, v) for s, v in enumerate(voices)])


def main():
    """CLI entry point."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="./data/spoken_alphanum", help="output directory")
    p.add_argument("--max-voices", type=int, default=None, help="cap the voice count")
    p.add_argument("--classes", default=None,
                   help="comma-separated class indices 0..35 (default: all 36)")
    p.add_argument("--augment", type=int, default=2,
                   help="augmented variants per clip (default 2 -> 3 clips per voice/class)")
    p.add_argument("--concurrency", type=int, default=4,
                   help="voices synthesized in parallel (keep modest to avoid rate limits)")
    args = p.parse_args()
    classes = list(range(36)) if args.classes is None else [int(c) for c in args.classes.split(",")]
    asyncio.run(run(Path(args.out), args.max_voices, classes, args.augment, args.concurrency))


if __name__ == "__main__":
    main()
