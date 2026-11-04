# EUROSAT CURVES FOR THE SLIDES, AT 1 BIT: (LEFT) TEST ACCURACY AND (RIGHT) TRAINING AND HELD-OUT LOSS PER EPOCH,
# BOTH THE MEAN OF 3 SEEDS. THE FINAL ACCURACY AT 1, 2 AND 4 BITS IS A TABLE ON THE SLIDE.
#   python plot_eurosat.py  -> plots/eurosat_results.png
import glob, json, os
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 15})  # LARGE TEXT: THE FIGURE IS SHRUNK TO SLIDE WIDTH

HERE = os.path.dirname(os.path.abspath(__file__))
C = {"ste": "#c0392b", "g": "#1f5fbf", "float": "#555555"}
LABEL = {"ste": "STE", "g": "$g$", "float": "float"}

runs = defaultdict(list)
for f in glob.glob(os.path.join(HERE, "runs", "eurosat_*.json")):
    r = json.load(open(f))
    runs[(r["method"], r["bits"])].append(r)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
ax = axes[0]
for m in ["float", "g", "ste"]:
    ax.plot(np.mean([[100 * e["acc"] for e in r["epoch"]] for r in runs[(m, 1)]], axis=0), color=C[m], lw=2.2,
            label=LABEL[m] + ("" if m == "float" else ", 1 bit"))
ax.set_xlabel("epoch")
ax.set_ylabel("test accuracy (%)")
ax.set_title("Test accuracy at 1 bit", fontsize=17)
ax.grid(alpha=0.3)
ax.legend(loc="lower right", fontsize=13)

ax = axes[1]
# ONE CURVE PER KIND: MEAN OVER THE 3 SEEDS
for m in ["float", "g", "ste"]:
    rs = runs[(m, 1)]
    for key, ls, kind in [("train_loss", "-", "train"), ("test_ce", "--", "validation")]:
        ax.plot(np.mean([[e[key] for e in r["epoch"]] for r in rs], axis=0), color=C[m], lw=2.2, ls=ls,
                label=f"{LABEL[m]}, {kind}")
ax.set_xlabel("epoch")
ax.set_ylabel("cross-entropy loss")
ax.set_title("Loss at 1 bit", fontsize=17)
ax.grid(alpha=0.3)
ax.legend(loc="upper right", fontsize=12, ncol=2)
fig.tight_layout()
out = os.path.join(HERE, "plots", "eurosat_results.png")
fig.savefig(out, dpi=150, facecolor="none")  # CLEAR AROUND THE AXES, WHITE INSIDE, SO THE SLIDE'S BLOCK SHOWS THROUGH
print("wrote", out)
