# TRAIN ONE CONFIGURATION, SAVE LOSS CURVES TO runs/{method}_k{bits}.json
#   python train.py --method ste --bits 1
import argparse, json, math, os, time
import torch
import config

p = argparse.ArgumentParser()
p.add_argument("--method", default=config.METHOD, choices=["float", "ste", "ridge"])
p.add_argument("--bits", type=int, default=config.BITS)
p.add_argument("--iters", type=int, default=config.max_iters)
a = p.parse_args()
config.METHOD, config.BITS, config.max_iters = a.method, a.bits, a.iters  # SET BEFORE THE MODEL IS BUILT

from data import vocab_size, get_batch
from model import GPT

torch.manual_seed(config.seed)
model = GPT(vocab_size).to(config.device)
opt = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

@torch.no_grad()
def estimate_loss():
    model.eval()
    out = {}
    for split in ["train", "val"]:
        out[split] = sum(model(*get_batch(split))[1].item() for _ in range(config.eval_iters)) / config.eval_iters
    model.train()
    return out

name = "float" if a.method == "float" else f"{a.method}_k{a.bits}"
log = {"method": a.method, "bits": a.bits, "step_loss": [], "eval": [], "diverged": False}
t0 = time.time()
for it in range(config.max_iters):
    if it % config.eval_interval == 0 or it == config.max_iters - 1:
        e = estimate_loss()
        log["eval"].append({"step": it, **e})
        print(f"{name} step {it}: train {e['train']:.4f}  val {e['val']:.4f}  ({time.time() - t0:.0f}s)")
    xb, yb = get_batch("train")
    _, loss = model(xb, yb)
    if not math.isfinite(loss.item()):  # NaN OR INF: RECORD AND STOP
        log["diverged"] = True
        print(f"{name} diverged at step {it}")
        break
    log["step_loss"].append(loss.item())
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()

log["seconds"] = time.time() - t0
os.makedirs("runs", exist_ok=True)
with open(f"runs/{name}.json", "w") as f:
    json.dump(log, f)
