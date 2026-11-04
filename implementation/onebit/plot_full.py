# LOSS OF THE PAPER-EXACT RUNS (repro_padum.pbs: BATCH 64, CONTEXT 256, 5000 STEPS), STE AGAINST RIDGE g, AFFINE AND SIGN.
# SOLID: SMOOTHED TRAINING LOSS. DASHED WITH MARKERS: VALIDATION LOSS (LOGGED EVERY 250 STEPS).
#   python plot_full.py  -> plots/loss_full.png (CLEAR AROUND THE AXES, FOR THE SLIDES)
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
R = lambda name: json.load(open(os.path.join(HERE, "runs", f"{name}.json")))
C_STE, C_G = "#c0392b", "#1f5fbf"
RUNS = [("1-bit, affine", "full_ste_affine", "full_ridge_affine"),
        ("1-bit, sign", "full_ste_linear", "full_ridge_linear")]

def ema(y, beta=0.98):
    # EXPONENTIAL MOVING AVERAGE WITH BIAS CORRECTION: PER-STEP LOSS IS NOISY
    out, m = [], 0.0
    for i, v in enumerate(y):
        m = beta * m + (1 - beta) * v
        out.append(m / (1 - beta ** (i + 1)))
    return np.array(out)

fig, axes = plt.subplots(1, 2, figsize=(14, 4.4))
fig.suptitle("Paper's exact setting, 5000 steps: STE stalls, $g$ keeps learning", fontsize=19, fontweight="bold")
for ax, (title, s, g) in zip(axes, RUNS):
    for name, color, who in [(s, C_STE, "STE"), (g, C_G, "$g$")]:
        r = R(name)
        ax.plot(ema(r["loss"]), color=color, lw=2.4, label=f"{who}, train")
        ev = [e for e in r["eval"] if e["step"] > 0]
        ax.plot([e["step"] for e in ev], [e["val"] for e in ev], color=color, lw=1.8, ls="--", marker="o", ms=4,
                label=f"{who}, validation")
    ax.set_title(title, fontsize=13)
    ax.set_xlim(0, 5000)
    ax.set_ylim(1.5, 3.2)
    ax.set_xlabel("step")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=10)
axes[0].set_ylabel("loss")
fig.tight_layout(rect=(0, 0, 1, 0.93))
out = os.path.join(HERE, "plots", "loss_full.png")
fig.savefig(out, dpi=150, facecolor="none")  # CLEAR AROUND THE AXES, WHITE INSIDE, SO THE SLIDE'S BLOCK SHOWS THROUGH
print("wrote", out)
