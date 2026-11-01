# ANIMATION OF A 1-BIT STE RUN AGAINST A 1-BIT RIDGE g RUN: WHEN AND HOW STE GOES WRONG, AND HOW g STAYS STABLE.
#   python animate.py ste_lin_k1 ridge_lin_k1 linear      -> plots/divergence_linear.{gif,mp4,png}
#   python animate.py ste_k1 ridge_k1 affine              -> plots/divergence_affine.{gif,mp4,png}
# PANELS: LOSS, SIZE OF THE RESIDUAL STREAM, INPUT RANGE OF THE QUANTIZED LAYERS (THE 1-BIT STEP), ONE WEIGHT ROW
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

HERE = os.path.dirname(os.path.abspath(__file__))
R = lambda name: json.load(open(os.path.join(HERE, "runs", f"{name}.json")))
STE_NAME, G_NAME, TAG = sys.argv[1:4] if len(sys.argv) > 3 else ("ste_lin_k1", "ridge_lin_k1", "linear")
ste, rdg = R(STE_NAME), R(G_NAME)
C_STE, C_G = "#c0392b", "#1f5fbf"
os.makedirs(os.path.join(HERE, "plots"), exist_ok=True)

def ema(y, beta=0.98):
    # EXPONENTIAL MOVING AVERAGE WITH BIAS CORRECTION: PER-STEP LOSS IS NOISY
    out, m = [], 0.0
    for i, v in enumerate(y):
        m = beta * m + (1 - beta) * v
        out.append(m / (1 - beta ** (i + 1)))
    return np.array(out)

def first_lasting(cond, hold):
    # FIRST INDEX WHERE cond HOLDS FOR hold CONSECUTIVE ENTRIES
    for i in range(len(cond) - hold + 1):
        if cond[i:i + hold].all():
            return i
    return None

T = min(len(ste["loss"]), len(rdg["loss"]))
L_ste, L_g = ema(ste["loss"])[:T], ema(rdg["loss"])[:T]

def stat_series(run, key):
    st = [s for s in run["stats"] if s["step"] < T]
    x = np.array([s["step"] for s in st])
    if key == "resid":
        y = np.array([s["resid_rms"][-1] for s in st])  # RESIDUAL STREAM AFTER THE LAST BLOCK
    else:
        y = np.array([np.mean(list(s["act_range"].values())) for s in st])  # MEAN PER-TOKEN INPUT RANGE, BLOCK 3
    return x, y

X_r1, Y_r1 = stat_series(ste, "resid")
X_r2, Y_r2 = stat_series(rdg, "resid")
X_a1, Y_a1 = stat_series(ste, "act")
X_a2, Y_a2 = stat_series(rdg, "act")

# WHEN THINGS GO WRONG FOR STE
k = min(len(Y_r1), len(Y_r2))
i_blow = first_lasting(Y_r1[:k] > 10 * Y_r2[:k], 4)
step_blow = int(X_r1[i_blow]) if i_blow is not None else None  # RESIDUAL STREAM 10x LARGER THAN WITH g
step_sep = first_lasting(L_ste - L_g > 0.05, 100)  # LOSS CURVES SEPARATE
best = np.minimum.accumulate(L_ste)
step_rise = first_lasting(L_ste > best + 0.10, 100)  # STE LOSS CLIMBS BACK UP
print(f"{TAG}: residual 10x at step {step_blow}, losses separate at {step_sep}, STE loss rises at {step_rise}")
print(f"  final smoothed loss: STE {L_ste[-1]:.3f}  g {L_g[-1]:.3f}")
print(f"  final residual RMS: STE {Y_r1[-1]:.1f}  g {Y_r2[-1]:.2f}  | input range: STE {Y_a1[-1]:.1f}  g {Y_a2[-1]:.2f}")

def snap_at(run, step):
    s = [x for x in run["snaps"] if x["step"] <= step]
    return s[-1] if s else run["snaps"][0]

fig, ax = plt.subplots(2, 2, figsize=(12.8, 7.2), dpi=100)
(a_loss, a_res), (a_act, a_row) = ax
fig.subplots_adjust(left=0.07, right=0.98, top=0.87, bottom=0.08, hspace=0.42, wspace=0.2)
title = fig.suptitle("", fontsize=14, fontweight="bold")
steps = np.arange(T)

a_loss.set(xlim=(0, T), ylim=(min(L_g.min(), L_ste.min()) - 0.05, min(3.6, max(L_ste.max(), L_g.max()) + 0.05)),
           xlabel="step", ylabel="training loss (smoothed)", title="Loss: STE stalls, $g$ keeps learning")
a_res.set(xlim=(0, T), yscale="log", xlabel="step", ylabel="RMS of the residual stream",
          title="Activations: STE's blow up, $g$'s stay bounded")
a_res.set_ylim(min(Y_r1.min(), Y_r2.min()) * 0.5, max(Y_r1.max(), Y_r2.max()) * 2)
a_act.set(xlim=(0, T), xlabel="step", ylabel="per-token input range\n(= the 1-bit step size)",
          title="Input range of the quantized layers keeps growing under STE")
a_act.set_ylim(0, max(Y_a1.max(), Y_a2.max()) * 1.1)
for x in (a_loss, a_res, a_act):
    x.grid(alpha=0.3)

ln = {}
for key, (axx, col, lab) in {"ls": (a_loss, C_STE, "STE"), "lg": (a_loss, C_G, "ridge $g$"),
                             "rs": (a_res, C_STE, "STE"), "rg": (a_res, C_G, "ridge $g$"),
                             "as": (a_act, C_STE, "STE"), "ag": (a_act, C_G, "ridge $g$")}.items():
    ln[key], = axx.plot([], [], color=col, lw=2.2, label=lab)
for x in (a_loss, a_res, a_act):
    x.legend(loc="upper left" if x is a_res else "upper right", fontsize=9)

marks = []  # (axis, step, text) SHOWN ONCE THE ANIMATION REACHES THAT STEP
if step_blow is not None:
    marks.append((a_res, step_blow, f"STE activations\n10x larger (step {step_blow})"))
if step_sep is not None:
    marks.append((a_loss, step_sep, f"losses separate\n(step {step_sep})"))
if step_rise is not None:
    marks.append((a_loss, step_rise, f"STE loss climbs\n(step {step_rise})"))
mark_art = []
for axx, s, txt in marks:
    vl = axx.axvline(s, color=C_STE, ls=":", lw=1.6, visible=False)
    y0, y1 = axx.get_ylim()
    ypos = y0 * (y1 / y0) ** 0.55 if axx.get_yscale() == "log" else y0 + 0.55 * (y1 - y0)
    tx = axx.text(s + T * 0.02, ypos, txt, color=C_STE, fontsize=9, fontweight="bold", visible=False, zorder=5,
                  bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=2))
    mark_art.append((s, vl, tx))

def draw_row(step):
    # ONE WEIGHT ROW (h.2.mlp.c_fc, ROW 0) WITH THE TWO VALUES ITS 1-BIT CODES ARE MAPPED BACK TO
    a_row.cla()
    s1, s2 = snap_at(ste, step), snap_at(rdg, step)
    w1, w2 = np.array(s1["w"]), np.array(s2["w"])
    lim = max(np.abs(w1).max(), np.abs(w2).max(), *map(abs, s1["levels"] + s2["levels"])) * 1.1
    bins = np.linspace(-lim, lim, 50)
    a_row.hist(w1, bins=bins, color=C_STE, alpha=0.35, label="STE weights")
    a_row.hist(w2, bins=bins, color=C_G, alpha=0.35, label="ridge $g$ weights")
    for v in s1["levels"]:
        a_row.axvline(v, color=C_STE, lw=2.5)
    for v in s2["levels"]:
        a_row.axvline(v, color=C_G, lw=2.5, ls="--")
    a_row.set(xlim=(-lim, lim), xlabel="weight value", ylabel="count",
              title="One weight row; lines = its two 1-bit levels")
    a_row.legend(loc="upper right", fontsize=8)

def update(t):
    t = max(1, int(t))
    title.set_text(f"1-bit weights and activations ({TAG}), small GPT on Shakespeare      step {t}")
    ln["ls"].set_data(steps[:t], L_ste[:t]); ln["lg"].set_data(steps[:t], L_g[:t])
    m1, m2 = X_r1 < t, X_r2 < t
    ln["rs"].set_data(X_r1[m1], Y_r1[m1]); ln["rg"].set_data(X_r2[m2], Y_r2[m2])
    m1, m2 = X_a1 < t, X_a2 < t
    ln["as"].set_data(X_a1[m1], Y_a1[m1]); ln["ag"].set_data(X_a2[m2], Y_a2[m2])
    for s, vl, tx in mark_art:
        vl.set_visible(t >= s); tx.set_visible(t >= s)
    draw_row(t)
    return []

stride = max(1, T // 120)
frames = list(range(stride, T + 1, stride)) + [T] * 24  # HOLD THE LAST FRAME
anim = FuncAnimation(fig, update, frames=frames, blit=False)
out = os.path.join(HERE, "plots", f"divergence_{TAG}")
anim.save(out + ".gif", writer=PillowWriter(fps=12))
try:
    # MP4 NEEDS AN ffmpeg BINARY; imageio-ffmpeg SHIPS ONE
    import imageio_ffmpeg
    from matplotlib.animation import FFMpegWriter
    matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
    anim.save(out + ".mp4", writer=FFMpegWriter(fps=12, bitrate=4000), dpi=150)
except ImportError:
    print("imageio-ffmpeg not installed: skipped the MP4")
update(T)
fig.savefig(out + ".png", dpi=150)
print("wrote", out + ".{gif,mp4,png}")
