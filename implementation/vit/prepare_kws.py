# GOOGLE SPEECH COMMANDS V2 (WARDEN, 2018): ~105k ONE-SECOND CLIPS AT 16 kHz, 35 KEYWORDS.
# OFFICIAL SPLIT FROM validation_list.txt AND testing_list.txt (SPEAKERS DO NOT OVERLAP). _background_noise_ IS NOT USED.
# FEATURES: 40 LOG-MEL BANDS, 30 ms WINDOW, 10 ms HOP -> 98 FRAMES PER CLIP (AS IN KWT, BERG ET AL., 2021).
# WRITES data/kws.npz (FLOAT16 FEATURES, SHAPE (N, 98, 40)). NEEDS ONLY NUMPY.
# Usage, from implementation/vit:
#   python prepare_kws.py --download   (on a node with internet; fetches ~2.3 GB into data/)
#   python prepare_kws.py              (extracts, if needed, and builds data/kws.npz)

import argparse, os, tarfile, urllib.request, wave
import numpy as np

URL = "http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz"
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
TAR = os.path.join(DATA, "speech_commands_v0.02.tar.gz")
ROOT = os.path.join(DATA, "speech_commands_v2")
SR, N_FFT, HOP, N_MELS, LEN = 16000, 480, 160, 40, 16000

p = argparse.ArgumentParser()
p.add_argument("--download", action="store_true")
a = p.parse_args()
os.makedirs(DATA, exist_ok=True)

if a.download:
    if not os.path.exists(TAR):
        print("downloading", URL, flush=True)
        urllib.request.urlretrieve(URL, TAR)
    raise SystemExit(f"downloaded {os.path.getsize(TAR) / 1e9:.2f} GB to {TAR}")

if not os.path.isdir(ROOT):
    print("extracting", TAR, flush=True)
    with tarfile.open(TAR) as t:
        t.extractall(ROOT)

def mel_filterbank():
    # TRIANGULAR HTK-STYLE MEL FILTERS OVER THE N_FFT//2 + 1 FREQUENCY BINS
    mel = lambda f: 2595 * np.log10(1 + f / 700)
    inv = lambda m: 700 * (10 ** (m / 2595) - 1)
    pts = inv(np.linspace(mel(0), mel(SR / 2), N_MELS + 2))
    freqs = np.linspace(0, SR / 2, N_FFT // 2 + 1)
    fb = np.zeros((N_MELS, len(freqs)))
    for i in range(N_MELS):
        lo, c, hi = pts[i], pts[i + 1], pts[i + 2]
        fb[i] = np.clip(np.minimum((freqs - lo) / (c - lo), (hi - freqs) / (hi - c)), 0, None)
    return fb

FB, WIN = mel_filterbank(), np.hanning(N_FFT + 1)[:-1]

def features(path):
    with wave.open(path) as w:
        assert w.getframerate() == SR and w.getsampwidth() == 2 and w.getnchannels() == 1
        x = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768
    x = np.pad(x[:LEN], (0, max(0, LEN - len(x))))  # ZERO-PAD SHORT CLIPS TO 1 s
    idx = np.arange(N_FFT)[None] + HOP * np.arange(1 + (LEN - N_FFT) // HOP)[:, None]  # (98, 480)
    power = np.abs(np.fft.rfft(x[idx] * WIN, axis=1)) ** 2
    return np.log(power @ FB.T + 1e-6).astype(np.float16)  # (98, 40)

classes = sorted(d for d in os.listdir(ROOT) if os.path.isdir(os.path.join(ROOT, d)) and not d.startswith("_"))
read_list = lambda f: set(open(os.path.join(ROOT, f)).read().split())
val, test = read_list("validation_list.txt"), read_list("testing_list.txt")
split = {"train": ([], []), "val": ([], []), "test": ([], [])}
for ci, c in enumerate(classes):
    for fn in sorted(os.listdir(os.path.join(ROOT, c))):
        rel = f"{c}/{fn}"
        s = "test" if rel in test else ("val" if rel in val else "train")
        split[s][0].append(features(os.path.join(ROOT, rel)))
        split[s][1].append(ci)
    print(c, flush=True)

out = {}
for s, (X, y) in split.items():
    out[f"X_{s}"], out[f"y_{s}"] = np.stack(X), np.array(y, np.int64)
np.savez(os.path.join(DATA, "kws.npz"), classes=np.array(classes), **out)
print(len(classes), "classes;", {s: len(v[1]) for s, v in split.items()})
