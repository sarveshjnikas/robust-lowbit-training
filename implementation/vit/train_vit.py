# SMALL VISION-STYLE TRANSFORMER ON TWO REAL-WORLD TASKS, k-BIT WEIGHTS AND ACTIVATIONS, STE VS. RIDGE g.
#   eurosat: SENTINEL-2 LAND-USE IMAGES (10 CLASSES, 64x64 RGB), 8x8 PATCHES -> 64 TOKENS. FOR ON-BOARD SATELLITE INFERENCE.
#   ecg:     MIT-BIH HEARTBEATS (INTER-PATIENT, 4 AAMI CLASSES), 256 SAMPLES IN 16-SAMPLE PATCHES -> 16 TOKENS, PLUS RR FEATURES.
# QUANTIZATION FOLLOWS THE PAPER'S TRANSFORMER RECIPE (SAME AS onebit/train_1bit.py): EVERY LINEAR LAYER INSIDE THE BLOCKS
# (QKV, ATTENTION OUTPUT, MLP UP, MLP DOWN) HAS k-BIT AFFINE WEIGHTS (ONE BLOCK PER OUTPUT ROW) AND k-BIT INPUTS (ONE BLOCK
# PER TOKEN). PATCH EMBEDDING AND CLASSIFIER HEAD STAY FULL PRECISION. ALL METHODS SHARE ONE SET OF HYPERPARAMETERS.

import argparse, json, math, os, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

p = argparse.ArgumentParser()
p.add_argument("--task", default="eurosat", choices=["eurosat", "ecg"])
p.add_argument("--method", default="g", choices=["float", "ste", "g"])
p.add_argument("--bits", type=int, default=1)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--lr", type=float, default=1e-3)
p.add_argument("--epochs", type=int, default=None)
p.add_argument("--name", default=None)
a = p.parse_args()

METHOD, BITS, LAM = a.method, a.bits, 1e-2
HERE = os.path.dirname(os.path.abspath(__file__))
device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
torch.manual_seed(a.seed)
np.random.seed(a.seed)

# TASK SETTINGS
if a.task == "eurosat":
    n_layer, n_head, n_embd, dropout, batch_size, epochs = 6, 3, 192, 0.1, 128, a.epochs or 40
else:
    n_layer, n_head, n_embd, dropout, batch_size, epochs = 4, 4, 128, 0.1, 256, a.epochs or 20
NAME = a.name or f"{a.task}_{a.method}" + ("" if a.method == "float" else f"_k{a.bits}") + f"_s{a.seed}"

# QUANTIZATION (SAME AS onebit/train_1bit.py)
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

class QLinear(nn.Linear):
    def deq(self, x):
        q, s_f, lo = quantize(x, BITS)
        return s_f * q + lo if METHOD == "ste" else g(q, x)

    def forward(self, x):
        return F.linear(self.deq(x), self.deq(self.weight), self.bias)

def Lin(fan_in, fan_out):
    return nn.Linear(fan_in, fan_out, bias=False) if METHOD == "float" else QLinear(fan_in, fan_out, bias=False)

# MODEL: PRE-NORM TRANSFORMER ENCODER (ViT)
class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.c_attn, self.c_proj = Lin(n_embd, 3 * n_embd), Lin(n_embd, n_embd)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = (t.view(B, T, n_head, C // n_head).transpose(1, 2) for t in self.c_attn(x).split(n_embd, dim=2))
        y = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout if self.training else 0.0)
        return self.drop(self.c_proj(y.transpose(1, 2).contiguous().view(B, T, C)))

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.c_fc, self.c_proj = Lin(n_embd, 4 * n_embd), Lin(4 * n_embd, n_embd)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.c_proj(F.gelu(self.c_fc(x))))

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln_1, self.ln_2 = nn.LayerNorm(n_embd), nn.LayerNorm(n_embd)
        self.attn, self.mlp = Attention(), MLP()

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        return x + self.mlp(self.ln_2(x))

class ViT(nn.Module):
    def __init__(self, patch_dim, n_tokens, n_classes, n_extra=0):
        super().__init__()
        self.embed = nn.Linear(patch_dim, n_embd)  # FULL PRECISION
        self.cls = nn.Parameter(torch.zeros(1, 1, n_embd))
        self.pos = nn.Parameter(torch.randn(1, n_tokens + 1, n_embd) * 0.02)
        self.blocks = nn.Sequential(*[Block() for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd + n_extra, n_classes)  # FULL PRECISION

    def forward(self, patches, extra=None):
        x = self.embed(patches)
        x = torch.cat([self.cls.expand(len(x), -1, -1), x], dim=1) + self.pos
        x = self.ln_f(self.blocks(x))
        h = x[:, 0] if extra is None else torch.cat([x[:, 0], extra], dim=1)
        return self.head(h)

# DATA (WHOLE DATASET ON THE DEVICE)
if a.task == "eurosat":
    d = np.load(os.path.join(HERE, "data", "eurosat.npz"))
    classes = [str(c) for c in d["classes"]]
    Xtr, ytr = torch.tensor(d["X_train"]).to(device), torch.tensor(d["y_train"]).to(device)
    Xte, yte = torch.tensor(d["X_test"]).to(device), torch.tensor(d["y_test"]).to(device)
    mean = Xtr.float().mean((0, 1, 2)) / 255
    std = (Xtr.float() / 255).std((0, 1, 2))

    def to_patches(xb, train):
        # (B, 64, 64, 3) uint8 -> (B, 64, 192): 8x8 PATCHES; TRAINING: RANDOM FLIPS AND 90-DEGREE ROTATIONS
        x = ((xb.float() / 255 - mean) / std).permute(0, 3, 1, 2)
        if train:
            if torch.rand(1).item() < 0.5:
                x = x.flip(3)
            x = torch.rot90(x, int(torch.randint(4, (1,))), (2, 3))
        return x.unfold(2, 8, 8).unfold(3, 8, 8).permute(0, 2, 3, 1, 4, 5).reshape(len(x), 64, 192)

    model = ViT(192, 64, len(classes)).to(device)
    batch = lambda X, idx, train: (to_patches(X[idx], train), None)
    class_w = None
else:
    import sys
    sys.path.insert(0, os.path.join(HERE, "..", "ecg"))
    from data import load, CLASSES as classes
    d = load()
    Xtr, Rtr, ytr = (torch.tensor(d[k]).to(device) for k in ["X_train", "R_train", "y_train"])
    Xte, Rte, yte = (torch.tensor(d[k]).to(device) for k in ["X_test", "R_test", "y_test"])
    model = ViT(16, 16, len(classes), n_extra=3).to(device)
    batch = lambda X, idx, train: (X[idx].view(len(idx), 16, 16), (Rtr if train else Rte)[idx])
    counts = torch.bincount(ytr, minlength=len(classes)).float()
    class_w = (counts.sum() / (len(classes) * counts)).sqrt()  # SOFTENED INVERSE FREQUENCY, AS IN ecg/train_ecg.py

decay = [p_ for n, p_ in model.named_parameters() if p_.dim() >= 2 and "pos" not in n]
no_decay = [p_ for n, p_ in model.named_parameters() if not (p_.dim() >= 2 and "pos" not in n)]
opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.05}, {"params": no_decay, "weight_decay": 0.0}], lr=a.lr)
steps_per_epoch = math.ceil(len(ytr) / batch_size)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=epochs * steps_per_epoch, pct_start=0.1)

@torch.no_grad()
def evaluate():
    model.eval()
    logits = []
    for i in range(0, len(yte), 1024):
        idx = torch.arange(i, min(i + 1024, len(yte)), device=device)
        logits.append(model(*batch(Xte, idx, False)))
    model.train()
    logits = torch.cat(logits)
    y, yp = yte.cpu().numpy(), logits.argmax(1).cpu().numpy()
    K = len(classes)
    cm = np.zeros((K, K), int)
    np.add.at(cm, (y, yp), 1)
    sens = cm.diagonal() / np.maximum(cm.sum(1), 1)
    ppv = cm.diagonal() / np.maximum(cm.sum(0), 1)
    f1 = np.where(sens + ppv > 0, 2 * sens * ppv / np.maximum(sens + ppv, 1e-12), 0.0)
    return {"acc": float((y == yp).mean()), "macro_f1": float(f1.mean()), "bal_acc": float(sens.mean()),
            "test_ce": F.cross_entropy(logits, yte).item(), "sens": sens.tolist()}

log = {"name": NAME, "task": a.task, "method": METHOD, "bits": BITS, "lr": a.lr, "seed": a.seed, "epochs": epochs,
       "loss": [], "grad_norm": [], "epoch": []}
t0, diverged = time.time(), False
for ep in range(epochs):
    perm = torch.randperm(len(ytr), device=device)
    for i in range(0, len(ytr), batch_size):
        idx = perm[i:i + batch_size]
        loss = F.cross_entropy(model(*batch(Xtr, idx, True)), ytr[idx], weight=class_w)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        log["loss"].append(loss.item())
        log["grad_norm"].append(gn.item())
        if not math.isfinite(loss.item()):
            diverged = True
            break
    if diverged:
        print(f"{NAME}: loss is not finite at epoch {ep}", flush=True)
        break
    m = evaluate() | {"train_loss": float(np.mean(log["loss"][-steps_per_epoch:]))}
    log["epoch"].append(m)
    print(f"{NAME} ep {ep:2d}  loss {m['train_loss']:.3f}  acc {m['acc']:.4f}  macroF1 {m['macro_f1']:.3f}  "
          f"balacc {m['bal_acc']:.3f}  testCE {m['test_ce']:.3f}  {time.time() - t0:.0f}s", flush=True)

log["diverged"] = diverged
os.makedirs(os.path.join(HERE, "runs"), exist_ok=True)
json.dump(log, open(os.path.join(HERE, "runs", f"{NAME}.json"), "w"))
