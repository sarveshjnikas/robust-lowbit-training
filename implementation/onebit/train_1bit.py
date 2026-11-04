# 1-BIT (A1W1) DIVERGENCE EXPERIMENT: STE VS RIDGE g ON THE PAPER'S NANOGPT SHAKESPEARE SETUP (FIG. 1a)
# MODEL AND RECIPE FOLLOW nanoGPT's train_shakespeare_char CONFIG: 6 LAYERS, 6 HEADS, WIDTH 384 (~10.7M PARAMETERS),
# BLOCK 256, BATCH 64, ADAMW LR 1e-3 WITH WARMUP AND COSINE DECAY, DROPOUT 0.2, GRADIENT CLIP 1.0, 5000 STEPS.
# EVERY LINEAR LAYER INSIDE THE BLOCKS HAS k-BIT AFFINE WEIGHTS (PER OUTPUT CHANNEL) AND k-BIT INPUTS (PER TOKEN).
# EMBEDDINGS AND THE TIED OUTPUT HEAD STAY FULL PRECISION. SELF-CONTAINED: SAME quantize AND g AS quant.py.
# LOGS LOSS, GRADIENT NORM, WEIGHT BIT FLIPS AND WEIGHT / ACTIVATION STATS, WHICH animate.py TURNS INTO THE ANIMATION.
#   python train_1bit.py --method ste
import argparse, json, math, os, time
import torch
import torch.nn as nn
from torch.nn import functional as F

p = argparse.ArgumentParser()
p.add_argument("--method", default="ste", choices=["float", "ste", "ridge"])
p.add_argument("--bits", type=int, default=1)
p.add_argument("--scheme", default="affine", choices=["affine", "linear"])  # linear: 1-BIT SIGN, LEVELS {-s, +s}
p.add_argument("--iters", type=int, default=5000)
p.add_argument("--lr", type=float, default=1e-3)
p.add_argument("--clip", type=float, default=1.0)  # 0 TURNS CLIPPING OFF
p.add_argument("--seed", type=int, default=1337)
p.add_argument("--batch", type=int, default=64)
p.add_argument("--block", type=int, default=256)
p.add_argument("--decay", type=int, default=5000)  # COSINE LENGTH; KEEP 5000 TO MATCH THE PAPER'S SCHEDULE
p.add_argument("--eval_iters", type=int, default=40)
p.add_argument("--name", default=None)
# DEBUG: WHY DOES STE NOT EXPLODE? EACH FLAG CHANGES ONE IMPLEMENTATION DETAIL
p.add_argument("--qhead", action="store_true")       # ALSO QUANTIZE THE (TIED) OUTPUT HEAD
p.add_argument("--per_tensor", action="store_true")  # ONE s_f, b_f PER WHOLE TENSOR INSTEAD OF PER ROW
p.add_argument("--detach_scale", action="store_true")  # NO GRADIENT THROUGH s_f, b_f (MIN / MAX)
a = p.parse_args()
METHOD, BITS, LAM, SCHEME = a.method, a.bits, 1e-2, a.scheme
QHEAD, PER_TENSOR, DETACH_SCALE = a.qhead, a.per_tensor, a.detach_scale
assert SCHEME == "affine" or BITS == 1, "linear scheme is implemented for 1 bit only"
NAME = a.name or (a.method if a.method == "float" else f"{a.method}_k{a.bits}")

# NANOGPT train_shakespeare_char
n_layer, n_head, n_embd, block_size, dropout = 6, 6, 384, a.block, 0.2
batch_size, warmup, decay_iters, min_lr = a.batch, 100, a.decay, 1e-4
eval_interval, eval_iters = 250, a.eval_iters
FLIP_EVERY, STAT_EVERY, SNAP_EVERY = 10, 25, 50
device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
HERE = os.path.dirname(os.path.abspath(__file__))

# DATA: TINY SHAKESPEARE, CHARACTER LEVEL, FIRST 90% TRAIN
text = open(os.path.join(HERE, "..", "data", "tinyshakespeare.txt")).read()
chars = sorted(set(text))
stoi = {c: i for i, c in enumerate(chars)}
data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
n = int(0.9 * len(data))
split = {"train": data[:n], "val": data[n:]}
gen = torch.Generator().manual_seed(a.seed)  # OWN GENERATOR: EVERY METHOD SEES THE SAME BATCHES

def get_batch(name):
    d = split[name]
    ix = torch.randint(len(d) - block_size, (batch_size,), generator=gen)
    x = torch.stack([d[i:i + block_size] for i in ix])
    y = torch.stack([d[i + 1:i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)

# QUANTIZATION (NOTATION OF notes.tex; SAME AS quant.py)
def quantize(x, k, eps=1e-8):
    # q = f(x) + stopgrad(delta), AFFINE f ONTO {0, ..., 2^k - 1}, ONE BLOCK PER ROW
    lo, hi = x.amin(-1, keepdim=True), x.amax(-1, keepdim=True)
    if DETACH_SCALE:
        lo, hi = lo.detach(), hi.detach()
    s_f = (hi - lo + eps) / (2**k - 1)
    f_x = (x - lo) / s_f
    return f_x + (torch.round(f_x) - f_x).detach(), s_f, lo

def quantize_sign(x):
    # 1-BIT LINEAR: f(x) = x / s_f WITH s_f = max|x|, ROUNDED TO THE NEAREST OF {-1, +1}, i.e. ITS SIGN
    s_f = x.abs().amax(-1, keepdim=True) + 1e-8
    if DETACH_SCALE:
        s_f = s_f.detach()
    f_x = x / s_f
    q = torch.where(f_x >= 0, 1.0, -1.0).to(x.dtype)
    return f_x + (q - f_x).detach(), s_f

def g_linear(q, x, lam=LAM):
    # RIDGE WITHOUT SHIFT (PAPER EQ. 4): g(q) = s_g q,  s_g = mean(qx) / (mean(q^2) + lam)
    m = lambda t: t.mean(-1, keepdim=True)
    return m(q * x) / (m(q * q) + lam) * q

def g(q, x, lam=LAM):
    # RIDGE DEQUANTIZATION: g(q) = s_g (q - q_bar) + x_bar
    m = lambda t: t.mean(-1, keepdim=True)
    qc, xc = q - m(q), x - m(x)
    s_g = m(xc * qc) / (m(qc * qc) + lam)
    return s_g * qc + m(x)

def _deq(x):
    # QUANTIZE, THEN DEQUANTIZE WITH STE (x_hat) OR RIDGE (g). ONE BLOCK PER ROW
    if SCHEME == "linear":
        q, s_f = quantize_sign(x)
        return s_f * q if METHOD == "ste" else g_linear(q, x)
    q, s_f, lo = quantize(x, BITS)
    return s_f * q + lo if METHOD == "ste" else g(q, x)

def deq(x):
    if PER_TENSOR:  # DEBUG: ONE BLOCK = THE WHOLE TENSOR
        return _deq(x.reshape(1, -1)).reshape(x.shape)
    return _deq(x)

class QLinear(nn.Linear):
    def forward(self, x):
        return F.linear(deq(x), deq(self.weight), self.bias)

def Lin(fan_in, fan_out):
    return nn.Linear(fan_in, fan_out, bias=False) if METHOD == "float" else QLinear(fan_in, fan_out, bias=False)

# MODEL: nanoGPT (NO BIASES, GELU MLP, TIED OUTPUT HEAD)
class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.c_attn, self.c_proj = Lin(n_embd, 3 * n_embd), Lin(n_embd, n_embd)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = (t.view(B, T, n_head, C // n_head).transpose(1, 2) for t in self.c_attn(x).split(n_embd, dim=2))
        y = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout if self.training else 0.0, is_causal=True)
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
        self.ln_1, self.ln_2 = nn.LayerNorm(n_embd, bias=False), nn.LayerNorm(n_embd, bias=False)
        self.attn, self.mlp = Attention(), MLP()

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        return x + self.mlp(self.ln_2(x))

class GPT(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.wte, self.wpe = nn.Embedding(vocab_size, n_embd), nn.Embedding(block_size, n_embd)
        self.drop = nn.Dropout(dropout)
        self.h = nn.ModuleList([Block() for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd, bias=False)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        self.lm_head.weight = self.wte.weight  # WEIGHT TYING, FULL PRECISION
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Embedding)):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
        for pn, prm in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                nn.init.normal_(prm, mean=0.0, std=0.02 / math.sqrt(2 * n_layer))

    def forward(self, idx, targets):
        x = self.drop(self.wte(idx) + self.wpe(torch.arange(idx.shape[1], device=idx.device)))
        for blk in self.h:
            x = blk(x)
        h = self.ln_f(x)
        logits = F.linear(deq(h), deq(self.lm_head.weight)) if QHEAD and METHOD != "float" else self.lm_head(h)
        return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

torch.manual_seed(a.seed)
model = GPT(len(chars)).to(device)
params = [q for q in model.parameters() if q.requires_grad]
opt = torch.optim.AdamW([{"params": [q for q in params if q.dim() >= 2], "weight_decay": 0.1},
                         {"params": [q for q in params if q.dim() < 2], "weight_decay": 0.0}],
                        lr=a.lr, betas=(0.9, 0.99))

def get_lr(it):
    # LINEAR WARMUP, THEN COSINE DECAY TO min_lr
    if it < warmup:
        return a.lr * (it + 1) / (warmup + 1)
    r = min(1.0, (it - warmup) / (decay_iters - warmup))
    return min_lr + 0.5 * (1.0 + math.cos(math.pi * r)) * (a.lr - min_lr)

@torch.no_grad()
def estimate_loss():
    model.eval()
    out = {s: sum(model(*get_batch(s)).item() for _ in range(eval_iters)) / eval_iters for s in ["train", "val"]}
    model.train()
    return out

# DIAGNOSTICS
qlayers = [(nm, m) for nm, m in model.named_modules() if isinstance(m, nn.Linear) and nm != "lm_head"]
WATCH = "h.2.mlp.c_fc"  # LAYER WHOSE FIRST WEIGHT ROW IS SNAPSHOTTED FOR THE ANIMATION
watch = dict(qlayers)[WATCH]
resid, act_range = {}, {}

def hook_block(i):
    def f(mod, inp, out):
        resid[i] = out.detach().float().pow(2).mean().sqrt().item()  # RMS OF THE RESIDUAL STREAM
    return f

def hook_input(nm):
    def f(mod, inp, out):
        x = inp[0].detach()
        act_range[nm] = (x.amax(-1) - x.amin(-1)).mean().item()  # MEAN PER-TOKEN INPUT RANGE = s_f AT 1 BIT
    return f

@torch.no_grad()
def weight_bits():
    # CURRENT ROUNDED INTEGERS OF EVERY QUANTIZED WEIGHT, AS ONE FLAT TENSOR
    if SCHEME == "linear":
        return torch.cat([(m.weight >= 0).flatten() for _, m in qlayers])
    return torch.cat([torch.round(quantize(m.weight, BITS)[0]).flatten() for _, m in qlayers])

@torch.no_grad()
def levels(w):
    # RECONSTRUCTION LEVELS OF ONE WEIGHT ROW: WHAT EACH INTEGER IS MAPPED BACK TO
    if SCHEME == "linear":
        q, s_f = quantize_sign(w)
        s = (q * w).mean() / ((q * q).mean() + LAM) if METHOD == "ridge" else s_f
        return [-float(s), float(s)]
    q, s_f, lo = quantize(w, BITS)
    ks = torch.arange(2**BITS, device=w.device, dtype=w.dtype)
    if METHOD == "ridge":
        qc, xc = q - q.mean(), w - w.mean()
        s_g = (xc * qc).mean() / ((qc * qc).mean() + LAM)
        return (s_g * (ks - q.mean()) + w.mean()).tolist()
    return (lo + s_f * ks).tolist()

log = {"name": NAME, "method": METHOD, "bits": BITS, "scheme": SCHEME, "lr": a.lr, "clip": a.clip, "seed": a.seed,
       "batch": batch_size, "block": block_size, "decay": decay_iters,
       "loss": [], "grad_norm": [], "lr_t": [], "eval": [], "flips": [], "stats": [], "snaps": [], "diverged_at": None}

def save():
    os.makedirs(os.path.join(HERE, "runs"), exist_ok=True)
    with open(os.path.join(HERE, "runs", f"{NAME}.json"), "w") as fh:
        json.dump(log, fh)

prev_bits = weight_bits() if METHOD != "float" else None
t0 = time.time()
for it in range(a.iters + 1):
    for gp in opt.param_groups:
        gp["lr"] = get_lr(it)
    if it % eval_interval == 0 or it == a.iters:
        e = estimate_loss()
        log["eval"].append({"step": it, **e})
        print(f"{NAME} step {it}: train {e['train']:.4f} val {e['val']:.4f} ({time.time() - t0:.0f}s)", flush=True)
        save()
    if it == a.iters:
        break
    hooks = []
    if it % STAT_EVERY == 0:
        hooks = [blk.register_forward_hook(hook_block(i)) for i, blk in enumerate(model.h)]
        hooks += [m.register_forward_hook(hook_input(nm)) for nm, m in qlayers if nm.startswith("h.2.")]
    loss = model(*get_batch("train"))
    for hk in hooks:
        hk.remove()
    if not math.isfinite(loss.item()):
        log["diverged_at"] = it
        print(f"{NAME}: loss is {loss.item()} at step {it}, stopping", flush=True)
        break
    opt.zero_grad(set_to_none=True)
    loss.backward()
    gn = torch.nn.utils.clip_grad_norm_(model.parameters(), a.clip if a.clip > 0 else float("inf")).item()
    opt.step()
    log["loss"].append(loss.item())
    log["grad_norm"].append(gn)
    log["lr_t"].append(get_lr(it))
    if METHOD != "float" and (it + 1) % FLIP_EVERY == 0:
        cur = weight_bits()
        log["flips"].append({"step": it + 1, "frac": (cur != prev_bits).float().mean().item()})
        prev_bits = cur
    if it % STAT_EVERY == 0:
        with torch.no_grad():
            wr = {nm: m.weight.pow(2).mean().sqrt().item() for nm, m in qlayers}
            wmax = max(m.weight.abs().max().item() for _, m in qlayers)
        log["stats"].append({"step": it, "resid_rms": [resid[i] for i in range(n_layer)], "act_range": dict(act_range),
                             "w_rms_mean": sum(wr.values()) / len(wr), "w_absmax": wmax})
    if it % SNAP_EVERY == 0 and METHOD != "float":
        w = watch.weight[0].detach()
        log["snaps"].append({"step": it, "w": w.float().cpu().tolist(), "levels": levels(w)})
log["seconds"] = time.time() - t0
save()
print(f"{NAME} done in {log['seconds']:.0f}s", flush=True)
