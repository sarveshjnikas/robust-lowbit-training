# EUROSAT RGB (HELBER ET AL., 2019): 27,000 SENTINEL-2 IMAGES, 64x64, 10 LAND-USE CLASSES.
# FIXED STRATIFIED 80/20 SPLIT (SEED 0), THE SAME FOR EVERY RUN. WRITES data/eurosat.npz (uint8 IMAGES).

import os, zipfile, io
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
z = zipfile.ZipFile(os.path.join(HERE, "data", "EuroSAT_RGB.zip"))
files = sorted(n for n in z.namelist() if n.lower().endswith(".jpg"))
classes = sorted({n.split("/")[-2] for n in files})
X = np.stack([np.array(Image.open(io.BytesIO(z.read(n))).convert("RGB")) for n in files])
y = np.array([classes.index(n.split("/")[-2]) for n in files], np.int64)

rng = np.random.default_rng(0)
test = np.zeros(len(y), bool)
for c in range(len(classes)):
    idx = np.flatnonzero(y == c)
    test[rng.choice(idx, size=len(idx) // 5, replace=False)] = True
np.savez(os.path.join(HERE, "data", "eurosat.npz"), X_train=X[~test], y_train=y[~test], X_test=X[test], y_test=y[test],
         classes=np.array(classes))
print(X.shape, classes, "train", (~test).sum(), "test", test.sum())
