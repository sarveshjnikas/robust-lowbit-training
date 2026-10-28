# PLOTS FOR THE TOY IMPLEMENTATION SLIDE. READS runs/*.json, WRITES plots/*.png
import json, os
import numpy as np
import matplotlib.pyplot as plt

R = lambda name: json.load(open(f"runs/{name}.json"))
os.makedirs("plots", exist_ok=True)

def smooth(y, w=50):
    # MOVING AVERAGE: PER-STEP LOSS IS NOISY
    return np.convolve(y, np.ones(w) / w, mode="valid")

# RESULT 1: TRAINING LOSS VS STEP AT 1 BIT (OUR VERSION OF FIG. 1a)
plt.figure(figsize=(5, 3.2))
for name, label in [("ste_k1", "STE"), ("ridge_k1", "ridge $g$"), ("float", "float")]:
    y = smooth(R(name)["step_loss"])
    plt.plot(np.arange(len(y)), y, label=label, ls="--" if name == "float" else "-")
plt.xlabel("step"); plt.ylabel("training loss"); plt.ylim(1.2, 3.0)
plt.title("1-bit weights and activations"); plt.legend(); plt.grid(alpha=0.3)
plt.tight_layout(); plt.savefig("plots/loss_k1.png", dpi=200)

# RESULT 2: FINAL VALIDATION LOSS VS BITS
bits = [1, 2, 4]
val = lambda name: R(name)["eval"][-1]["val"]
plt.figure(figsize=(5, 3.2))
plt.plot(bits, [val(f"ste_k{k}") for k in bits], "o-", label="STE")
plt.plot(bits, [val(f"ridge_k{k}") for k in bits], "s-", label="ridge $g$")
plt.axhline(val("float"), ls="--", c="gray", label="float")
plt.xticks(bits); plt.xlabel("bits $k$"); plt.ylabel("validation loss")
plt.legend(); plt.grid(alpha=0.3)
plt.tight_layout(); plt.savefig("plots/val_vs_bits.png", dpi=200)
