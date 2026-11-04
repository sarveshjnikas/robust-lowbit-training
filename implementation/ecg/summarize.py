# READS runs/*.json, PRINTS FINAL-EPOCH METRICS (MEAN +- STD OVER SEEDS) AND WRITES TWO PLOTS TO plots/
# FINAL EPOCH ONLY: NO EPOCH IS PICKED BY LOOKING AT THE TEST SET.

import glob, json, os
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
COLORS = {"float": "#555555", "ste": "#d62728", "g": "#1f77b4"}
LABELS = {"float": "float", "ste": "STE (1 bit)", "g": "with g (1 bit)"}

runs = defaultdict(list)       # 1 BIT (AND FLOAT), KEYED BY (METHOD, LR)
bits_runs = defaultdict(list)  # ALL BITS, KEYED BY (METHOD, BITS, LR)
for f in glob.glob(os.path.join(HERE, "runs", "*.json")):
    r = json.load(open(f))
    if r["bits"] == 1 or r["method"] == "float":
        runs[(r["method"], r["lr"])].append(r)
    bits_runs[(r["method"], r["bits"], r["lr"])].append(r)

def final(r, key, i=None):
    if r["diverged"] or not r["epoch"]:
        return np.nan
    v = r["epoch"][-1][key]
    return v if i is None else v[i]

print(f"{'method':6} {'lr':>6} {'n':>2} {'div':>3}  {'acc':>11}  {'macro F1':>11}  {'sens S':>11}  {'sens V':>11}  {'sens F':>11}")
for (m, lr) in sorted(runs, key=lambda k: (list(COLORS).index(k[0]), k[1])):
    rs = runs[(m, lr)]
    cell = lambda vals: f"{np.nanmean(vals):.3f}±{np.nanstd(vals):.3f}"
    print(f"{m:6} {lr:6g} {len(rs):2d} {sum(r['diverged'] for r in rs):3d}  "
          f"{cell([final(r, 'acc') for r in rs])}  {cell([final(r, 'macro_f1') for r in rs])}  "
          f"{cell([final(r, 'sens', 1) for r in rs])}  {cell([final(r, 'sens', 2) for r in rs])}  {cell([final(r, 'sens', 3) for r in rs])}")

os.makedirs(os.path.join(HERE, "plots"), exist_ok=True)

# PLOT 1: FINAL MACRO F1 AGAINST LEARNING RATE
fig, ax = plt.subplots(figsize=(6, 4))
for m in COLORS:
    lrs = sorted(lr for (mm, lr) in runs if mm == m)
    if not lrs:
        continue
    vals = [[final(r, "macro_f1") for r in runs[(m, lr)]] for lr in lrs]
    ax.errorbar(lrs, [np.nanmean(v) for v in vals], yerr=[np.nanstd(v) for v in vals],
                marker="o", capsize=4, color=COLORS[m], label=LABELS[m])
ax.set_xscale("log")
ax.set_xlabel("learning rate")
ax.set_ylabel("macro F1 on unseen patients")
ax.set_title("1-bit heartbeat classifier, MIT-BIH")
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(HERE, "plots", "f1_vs_lr.png"), dpi=200)

# PLOT 2: TRAINING LOSS PER EPOCH, ONE PANEL PER LEARNING RATE, SEED 0
lrs = sorted({lr for (_, lr) in runs})
fig, axes = plt.subplots(1, len(lrs), figsize=(4 * len(lrs), 3.5), sharey=True)
for ax, lr in zip(np.atleast_1d(axes), lrs):
    for m in COLORS:
        for r in runs.get((m, lr), []):
            ax.plot([e["train_loss"] for e in r["epoch"]], color=COLORS[m], alpha=0.8 if r["seed"] == 0 else 0.3,
                    label=LABELS[m] if r["seed"] == 0 else None)
    ax.set_title(f"lr {lr:g}")
    ax.set_xlabel("epoch")
np.atleast_1d(axes)[0].set_ylabel("training loss")
np.atleast_1d(axes)[0].legend()
fig.tight_layout()
fig.savefig(os.path.join(HERE, "plots", "loss_by_lr.png"), dpi=200)

# TABLE AND PLOT 3: STE VS. g AGAINST BITS, AT THE LEARNING RATE THAT IS BEST FOR FLOAT (PAPER: HYPERPARAMETERS MATCH THE BASELINE)
float_lrs = sorted(lr for (m, lr) in runs if m == "float")
if float_lrs:
    best = max(float_lrs, key=lambda lr: np.nanmean([final(r, "macro_f1") for r in runs[("float", lr)]]))
    fl = np.nanmean([final(r, "macro_f1") for r in runs[("float", best)]])
    print(f"\nAT LR {best:g} (BEST FOR FLOAT, macro F1 {fl:.3f}):")
    print(f"{'bits':>4}  {'STE':>11}  {'with g':>11}  {'gain':>6}   sens S / V / F (STE | g)")
    fig, ax = plt.subplots(figsize=(5, 4))
    for m in ["ste", "g"]:
        bs = sorted(b for (mm, b, lr) in bits_runs if mm == m and lr == best)
        vals = [[final(r, "macro_f1") for r in bits_runs[(m, b, best)]] for b in bs]
        ax.errorbar(bs, [np.nanmean(v) for v in vals], yerr=[np.nanstd(v) for v in vals],
                    marker="o", capsize=4, color=COLORS[m], label=LABELS[m].split(" (")[0])
    ax.axhline(fl, color=COLORS["float"], ls="--", label="float")
    for b in sorted({b for (m, b, lr) in bits_runs if m != "float" and lr == best}):
        st, gg = bits_runs.get(("ste", b, best), []), bits_runs.get(("g", b, best), [])
        mf = lambda rs: np.nanmean([final(r, "macro_f1") for r in rs])
        sens = lambda rs: " ".join(f"{np.nanmean([final(r, 'sens', i) for r in rs]):.2f}" for i in (1, 2, 3))
        print(f"{b:4d}  {mf(st):11.3f}  {mf(gg):11.3f}  {mf(gg) - mf(st):+6.3f}   {sens(st)} | {sens(gg)}")
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 2, 4], ["1", "2", "4"])
    ax.set_xlabel("bits for weights and activations")
    ax.set_ylabel("macro F1 on unseen patients")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "plots", "f1_vs_bits.png"), dpi=200)
