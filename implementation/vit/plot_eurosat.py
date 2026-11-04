# EUROSAT RESULTS FOR THE SLIDES: (LEFT) FINAL TEST ACCURACY AGAINST BITS, MEAN +- STD OVER 3 SEEDS;
# (MIDDLE) TEST ACCURACY AND (RIGHT) TRAINING AND HELD-OUT LOSS PER EPOCH AT 1 BIT, BOTH THE MEAN OF 3 SEEDS. FINAL EPOCH ONLY: NO EPOCH IS PICKED BY LOOKING AT THE TEST SET.
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
acc = lambda r: 100 * r["epoch"][-1]["acc"]

fig, axes = plt.subplots(1, 3, figsize=(18, 5.0))
fig.suptitle("Satellite images at 1, 2 and 4 bits: the gain grows as bits drop", fontsize=22, fontweight="bold")
ax = axes[0]
for m in ["ste", "g"]:
    vals = [[acc(r) for r in runs[(m, b)]] for b in (1, 2, 4)]
    ax.errorbar([1, 2, 4], [np.mean(v) for v in vals], yerr=[np.std(v) for v in vals], color=C[m], lw=2.4,
                marker="o", ms=7, capsize=4, label=LABEL[m])
ax.axhline(np.mean([acc(r) for r in runs[("float", 1)]]), color=C["float"], ls="--", lw=1.8, label="float")
ax.set_xscale("log", base=2)
ax.set_xticks([1, 2, 4], ["1", "2", "4"])
ax.set_xlabel("bits for weights and activations")
ax.set_ylabel("test accuracy (%)")
ax.set_title("Final accuracy", fontsize=17)
ax.grid(alpha=0.3)
ax.legend(loc="lower right", fontsize=13)

ax = axes[1]
for m in ["float", "g", "ste"]:
    ax.plot(np.mean([[100 * e["acc"] for e in r["epoch"]] for r in runs[(m, 1)]], axis=0), color=C[m], lw=2.2,
            label=LABEL[m] + ("" if m == "float" else ", 1 bit"))
ax.set_xlabel("epoch")
ax.set_title("Test accuracy at 1 bit", fontsize=17)
ax.grid(alpha=0.3)
ax.legend(loc="lower right", fontsize=13)

ax = axes[2]
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
fig.tight_layout(rect=(0, 0, 1, 0.93))
out = os.path.join(HERE, "plots", "eurosat_results.png")
fig.savefig(out, dpi=150, facecolor="none")  # CLEAR AROUND THE AXES, WHITE INSIDE, SO THE SLIDE'S BLOCK SHOWS THROUGH
print("wrote", out)
