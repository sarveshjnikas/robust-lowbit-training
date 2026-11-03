# ANIMATION OF THE LOSS ONLY, STE AGAINST RIDGE g, IN ALL THREE 1-BIT RUNS SIDE BY SIDE.
# SOLID: SMOOTHED TRAINING LOSS. DASHED WITH MARKERS: VALIDATION LOSS (LOGGED EVERY 250 STEPS).
#   python animate_loss.py             -> plots/loss_three.{gif,mp4,png}
#   python animate_loss.py --png-only  -> plots/loss_three.png (CLEAR AROUND THE AXES, FOR THE SLIDES)
# ALL THREE PANELS FINISH TOGETHER, SO THE 2000-STEP SIGN RUN ADVANCES TWICE AS FAST AS THE 1000-STEP ONES
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

HERE = os.path.dirname(os.path.abspath(__file__))
R = lambda name: json.load(open(os.path.join(HERE, "runs", f"{name}.json")))
C_STE, C_G = "#c0392b", "#1f5fbf"
RUNS = [("1-bit", "ste_k1", "ridge_k1"),
        ("1-bit, sign", "ste_lin_k1", "ridge_lin_k1"),
        ("1-bit, 3x learning rate", "ste_k1_lr3e-3", "ridge_k1_lr3e-3")]
N_FRAMES = 90

def ema(y, beta=0.98):
    # EXPONENTIAL MOVING AVERAGE WITH BIAS CORRECTION: PER-STEP LOSS IS NOISY
    out, m = [], 0.0
    for i, v in enumerate(y):
        m = beta * m + (1 - beta) * v
        out.append(m / (1 - beta ** (i + 1)))
    return np.array(out)

curves = []
for title, s, g in RUNS:
    rs, rg = R(s), R(g)
    L_s, L_g = ema(rs["loss"]), ema(rg["loss"])
    T = min(len(L_s), len(L_g))
    # SKIP THE STEP-0 EVAL: AT ABOUT 4.2 IT SITS FAR ABOVE THE AXIS AND ONLY ADDS A STEEP CONNECTING LINE
    V = lambda run: (np.array([e["step"] for e in run["eval"] if e["step"] > 0]),
                     np.array([e["val"] for e in run["eval"] if e["step"] > 0]))
    curves.append((title, L_s[:T], L_g[:T], T, V(rs), V(rg)))

fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
fig.suptitle("Loss at 1 bit: STE stalls, $g$ keeps learning", fontsize=21, fontweight="bold")
lines = []
for ax, (title, L_s, L_g, T, _, _) in zip(axes, curves):
    ax.set_title(title, fontsize=13)
    ax.set_xlim(0, T)
    ax.set_ylim(2.2, 3.6)
    ax.set_xlabel("step")
    ax.grid(alpha=0.3)
    ls, = ax.plot([], [], color=C_STE, lw=2.4, label="STE, train")
    lg, = ax.plot([], [], color=C_G, lw=2.4, label="$g$, train")
    vs, = ax.plot([], [], color=C_STE, lw=1.8, ls="--", marker="o", ms=5, label="STE, validation")
    vg, = ax.plot([], [], color=C_G, lw=1.8, ls="--", marker="o", ms=5, label="$g$, validation")
    ax.legend(loc="upper right", fontsize=10)
    lines.append((ls, lg, vs, vg))
axes[0].set_ylabel("loss")
fig.tight_layout(rect=(0, 0, 1, 0.90))

def update(frame):
    f = (frame + 1) / N_FRAMES
    for (ls, lg, vs, vg), (_, L_s, L_g, T, (xs, ys), (xg, yg)) in zip(lines, curves):
        n = max(2, int(round(f * T)))
        x = np.arange(n)
        ls.set_data(x, L_s[:n])
        lg.set_data(x, L_g[:n])
        vs.set_data(xs[xs <= n], ys[xs <= n])
        vg.set_data(xg[xg <= n], yg[xg <= n])
    return [l for group in lines for l in group]

out = os.path.join(HERE, "plots", "loss_three")
if "--png-only" not in sys.argv:
    anim = FuncAnimation(fig, update, frames=N_FRAMES, blit=False)
    anim.save(out + ".gif", writer=PillowWriter(fps=12))
    try:
        from matplotlib.animation import FFMpegWriter
        import imageio_ffmpeg
        matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
        anim.save(out + ".mp4", writer=FFMpegWriter(fps=12, bitrate=3000), dpi=150)
    except Exception as e:
        print("mp4 skipped:", e)
update(N_FRAMES - 1)
fig.savefig(out + ".png", dpi=150, facecolor="none")  # CLEAR AROUND THE AXES, WHITE INSIDE, SO THE SLIDE'S BLOCK SHOWS THROUGH
print("wrote", out + ".png")
