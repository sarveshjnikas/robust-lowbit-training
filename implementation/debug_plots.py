# PLOT FOR THE STE DEBUGGING: RESIDUAL-STREAM SIZE AND LOSS, STE VS RIDGE AT 1 BIT. READS runs/debug/*.json
import json
import numpy as np
import matplotlib.pyplot as plt

R = lambda name: json.load(open(f"runs/debug/{name}.json"))

def smooth(y, w=50):
    return np.convolve(y, np.ones(w) / w, mode="valid")

fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
for run, label, c in [("s0", "lr 1e-3", "C0"), ("s2", "lr 1e-2", "C3")]:
    for m, ls in [("ste", "-"), ("ridge", "--")]:
        d = R(f"{run}_{m}" if run == "s0" else f"{run}_{m}_lr1e-2")
        it, rms = zip(*d["resid_rms"])
        ax[0].plot(it, rms, ls=ls, c=c, label=f"{'STE' if m == 'ste' else 'ridge $g$'}, {label}")
        y = smooth(d["step_loss"])
        ax[1].plot(np.arange(len(y)), y, ls=ls, c=c)
ax[0].set_yscale("log"); ax[0].set_xlabel("step"); ax[0].set_ylabel("residual stream RMS (log)")
ax[0].set_title("Inside the network"); ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)
ax[1].set_xlabel("step"); ax[1].set_ylabel("training loss"); ax[1].set_ylim(2.0, 3.2)
ax[1].set_title("The loss"); ax[1].grid(alpha=0.3)
plt.tight_layout(); plt.savefig("plots/ste_blowup.png", dpi=200)
