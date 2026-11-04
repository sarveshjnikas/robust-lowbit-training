# 1-BIT HEARTBEAT CLASSIFIER ON MIT-BIH (INTER-PATIENT), STE VS. RIDGE DEQUANTIZATION g.
# SMALL 1-D CNN: 4 CONV LAYERS + 2 LINEAR LAYERS. EVERY LAYER EXCEPT THE FINAL CLASSIFIER IS QUANTIZED,
# INCLUDING THE FIRST CONV ON THE RAW ECG (HARDWARE DESIGNS OFTEN KEEP THAT ONE IN FULL PRECISION).
# WEIGHTS: ONE BLOCK PER OUTPUT CHANNEL. ACTIVATIONS: ONE BLOCK PER BEAT (THE WHOLE FEATURE MAP).
# ONLY THE DEQUANTIZATION STEP DIFFERS BETWEEN ste AND g.

import argparse, json, math, os, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, recall_score, precision_score
from data import load, CLASSES

p = argparse.ArgumentParser()
p.add_argument("--method", default="g", choices=["float", "ste", "g"])
p.add_argument("--bits", type=int, default=1)
p.add_argument("--lr", type=float, default=1e-3)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--epochs", type=int, default=20)
p.add_argument("--batch", type=int, default=256)
p.add_argument("--bn", type=int, default=1)  # 0 REMOVES BATCHNORM
p.add_argument("--clip", type=float, default=0.0)  # 0 TURNS CLIPPING OFF
p.add_argument("--name", default=None)
a = p.parse_args()

METHOD, BITS, LAM = a.method, a.bits, 1e-2
NAME = a.name or f"{a.method}_k{a.bits}_lr{a.lr:g}_bn{a.bn}_s{a.seed}"
HERE = os.path.dirname(os.path.abspath(__file__))
device = "mps" if torch.backends.mps.is_available() else "cpu"
torch.manual_seed(a.seed)
np.random.seed(a.seed)

# QUANTIZATION (SAME AS onebit/train_1bit.py), APPLIED TO ROWS OF A 2-D VIEW
def quantize(x, k, eps=1e-8):
    # q = f(x) + stopgrad(delta), AFFINE f ONTO {0, ..., 2^k - 1}, ONE BLOCK PER ROW
    lo, hi = x.amin(-1, keepdim=True), x.amax(-1, keepdim=True)
    s_f = (hi - lo + eps) / (2**k - 1)
    f_x = (x - lo) / s_f
    return f_x + (torch.round(f_x) - f_x).detach(), s_f, lo

def g(q, x, lam=LAM):
    # RIDGE DEQUANTIZATION: g(q) = s_g (q - q_bar) + x_bar
    m = lambda t: t.mean(-1, keepdim=True)
    qc, xc = q - m(q), x - m(x)
    s_g = m(xc * qc) / (m(qc * qc) + lam)
    return s_g * qc + m(x)

def deq(x):
    # BLOCK = EVERYTHING BUT THE FIRST DIMENSION (ONE BEAT, OR ONE OUTPUT CHANNEL)
    if METHOD == "float":
        return x
    rows = x.flatten(1)
    q, s_f, lo = quantize(rows, BITS)
    out = s_f * q + lo if METHOD == "ste" else g(q, rows)
    return out.view_as(x)

class QConv1d(nn.Conv1d):
    def forward(self, x):
        return F.conv1d(deq(x), deq(self.weight), self.bias, self.stride, self.padding)

class QLinear(nn.Linear):
    def forward(self, x):
        return F.linear(deq(x), deq(self.weight), self.bias)

# MODEL
def conv_block(c_in, c_out, k, pool):
    layers = [QConv1d(c_in, c_out, k, padding=k // 2, bias=not a.bn)]
    layers += [nn.BatchNorm1d(c_out)] if a.bn else []
    layers += [nn.ReLU()] + ([nn.MaxPool1d(2)] if pool else [])
    return nn.Sequential(*layers)

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(conv_block(1, 16, 7, True), conv_block(16, 32, 5, True),
                                      conv_block(32, 64, 5, True), conv_block(64, 64, 3, False))
        self.fc1 = QLinear(64 + 3, 32)
        self.fc2 = nn.Linear(32, len(CLASSES))  # FULL PRECISION CLASSIFIER

    def forward(self, x, rr):
        h = self.features(x.unsqueeze(1))
        self.feat_scale = h.detach().abs().mean()  # SIZE OF THE LAST FEATURE MAP, LOGGED EACH EPOCH
        h = torch.cat([h.mean(-1), rr], dim=1)
        return self.fc2(F.relu(self.fc1(h)))

# DATA
d = load()
Xtr, Rtr, ytr = (torch.tensor(d[k]) for k in ["X_train", "R_train", "y_train"])
Xte, Rte, yte = (torch.tensor(d[k]) for k in ["X_test", "R_test", "y_test"])
counts = torch.bincount(ytr, minlength=len(CLASSES)).float()
class_w = (counts.sum() / (len(CLASSES) * counts)).sqrt().to(device)  # SOFTENED INVERSE FREQUENCY

model = Net().to(device)
opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
steps_per_epoch = math.ceil(len(ytr) / a.batch)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.epochs * steps_per_epoch, pct_start=0.1)

@torch.no_grad()
def evaluate():
    model.eval()
    preds = []
    for i in range(0, len(yte), 2048):
        preds.append(model(Xte[i:i + 2048].to(device), Rte[i:i + 2048].to(device)).argmax(1).cpu())
    model.train()
    y, yp = yte.numpy(), torch.cat(preds).numpy()
    labels = list(range(len(CLASSES)))
    return {"acc": float((y == yp).mean()),
            "macro_f1": float(f1_score(y, yp, labels=labels, average="macro", zero_division=0)),
            "sens": recall_score(y, yp, labels=labels, average=None, zero_division=0).tolist(),
            "ppv": precision_score(y, yp, labels=labels, average=None, zero_division=0).tolist()}

log = {"name": NAME, "method": METHOD, "bits": BITS, "lr": a.lr, "seed": a.seed, "bn": a.bn, "clip": a.clip,
       "epochs": a.epochs, "batch": a.batch, "loss": [], "grad_norm": [], "epoch": []}
t0, diverged = time.time(), False
for ep in range(a.epochs):
    perm = torch.randperm(len(ytr))
    for i in range(0, len(ytr), a.batch):
        idx = perm[i:i + a.batch]
        loss = F.cross_entropy(model(Xtr[idx].to(device), Rtr[idx].to(device)), ytr[idx].to(device), weight=class_w)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), a.clip if a.clip > 0 else float("inf"))
        opt.step()
        sched.step()
        log["loss"].append(loss.item())
        log["grad_norm"].append(gn.item())
        if not math.isfinite(loss.item()):
            diverged = True
            break
    if diverged:
        print(f"{NAME}: loss is not finite at epoch {ep}")
        break
    m = evaluate() | {"train_loss": float(np.mean(log["loss"][-steps_per_epoch:])), "feat_scale": model.feat_scale.item()}
    log["epoch"].append(m)
    print(f"{NAME} ep {ep:2d}  loss {m['train_loss']:.3f}  acc {m['acc']:.3f}  macroF1 {m['macro_f1']:.3f}  "
          f"sens N/S/V/F {' '.join(f'{s:.2f}' for s in m['sens'])}  feat {m['feat_scale']:.2f}  {time.time() - t0:.0f}s", flush=True)

log["diverged"] = diverged
os.makedirs(os.path.join(HERE, "runs"), exist_ok=True)
json.dump(log, open(os.path.join(HERE, "runs", f"{NAME}.json"), "w"))
