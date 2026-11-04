# MIT-BIH ARRHYTHMIA DATABASE (PHYSIONET), INTER-PATIENT SPLIT OF DE CHAZAL ET AL. (2004)
# TRAIN ON DS1 PATIENTS, TEST ON DS2 PATIENTS: NO PATIENT APPEARS IN BOTH, SO NO LEAKAGE.
# EACH SAMPLE: ONE HEARTBEAT, LEAD MLII, 256 SAMPLES AROUND THE R-PEAK (360 HZ), PLUS 3 RR-INTERVAL FEATURES.
# CLASSES (AAMI): N NORMAL, S SUPRAVENTRICULAR, V VENTRICULAR, F FUSION. PACED RECORDS AND Q BEATS DROPPED.

import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "data", "mitdb")
CACHE = os.path.join(HERE, "data", "mitbih_beats.npz")

DS1 = [101, 106, 108, 109, 112, 114, 115, 116, 118, 119, 122, 124, 201, 203, 205, 207, 208, 209, 215, 220, 223, 230]
DS2 = [100, 103, 105, 111, 113, 117, 121, 123, 200, 202, 210, 212, 213, 214, 219, 221, 222, 228, 231, 232, 233, 234]

CLASSES = ["N", "S", "V", "F"]
AAMI = {s: 0 for s in "NLRej"} | {s: 1 for s in "AaJS"} | {s: 2 for s in "VE"} | {"F": 3}
BEFORE, AFTER = 90, 166  # 0.25 S BEFORE AND 0.46 S AFTER THE R-PEAK


def download():
    import wfdb
    if not os.path.exists(os.path.join(RAW, "100.dat")):
        os.makedirs(RAW, exist_ok=True)
        wfdb.dl_database("mitdb", RAW, records=[str(r) for r in DS1 + DS2])


def remove_baseline(sig, fs=360):
    # DE CHAZAL: 200 MS THEN 600 MS MEDIAN FILTERS ESTIMATE THE BASELINE WANDER
    from scipy.signal import medfilt
    base = medfilt(medfilt(sig, int(0.2 * fs) | 1), int(0.6 * fs) | 1)
    return sig - base


def beats(record):
    import wfdb
    rec = wfdb.rdrecord(os.path.join(RAW, str(record)))
    ann = wfdb.rdann(os.path.join(RAW, str(record)), "atr")
    lead = rec.sig_name.index("MLII")
    sig = remove_baseline(rec.p_signal[:, lead])
    peaks, symbols = np.array(ann.sample), ann.symbol
    keep = [i for i, s in enumerate(symbols) if s in AAMI]
    peaks_all = peaks[keep]
    rr = np.diff(peaks_all) / 360.0
    X, R, y = [], [], []
    for j in range(1, len(keep) - 1):
        p = peaks_all[j]
        if p - BEFORE < 0 or p + AFTER > len(sig):
            continue
        pre, post = rr[j - 1], rr[j]
        local = rr[max(0, j - 10):j].mean()
        X.append(sig[p - BEFORE:p + AFTER])
        R.append([pre, post, pre / local])
        y.append(AAMI[symbols[keep[j]]])
    return np.array(X, np.float32), np.array(R, np.float32), np.array(y, np.int64)


def build():
    download()
    out = {}
    for name, records in [("train", DS1), ("test", DS2)]:
        parts = [beats(r) for r in records]
        out[f"X_{name}"] = np.concatenate([p[0] for p in parts])
        out[f"R_{name}"] = np.concatenate([p[1] for p in parts])
        out[f"y_{name}"] = np.concatenate([p[2] for p in parts])
    np.savez_compressed(CACHE, **out)


def load():
    if not os.path.exists(CACHE):
        build()
    d = dict(np.load(CACHE))
    # NORMALISE WITH TRAINING STATISTICS ONLY; NO PER-BEAT SCALING, SO THE R-PEAK STAYS AN OUTLIER
    mu, sd = d["X_train"].mean(), d["X_train"].std()
    rmu, rsd = d["R_train"].mean(0), d["R_train"].std(0)
    for s in ["train", "test"]:
        d[f"X_{s}"] = (d[f"X_{s}"] - mu) / sd
        d[f"R_{s}"] = (d[f"R_{s}"] - rmu) / rsd
    return d


if __name__ == "__main__":
    d = load()
    for s in ["train", "test"]:
        counts = np.bincount(d[f"y_{s}"], minlength=4)
        print(s, d[f"X_{s}"].shape, dict(zip(CLASSES, counts.tolist())))
    x = d["X_train"]
    print("peak / median |x| per beat:", float(np.median(np.abs(x).max(1) / np.median(np.abs(x), 1))))
