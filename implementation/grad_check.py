# GRADIENT CHECK: STE'S LOCAL GRADIENT IGNORES THE ROUNDING ERROR delta, RIDGE'S DOES NOT.
# TWO INPUTS x, x2 WITH THE SAME q (SAME ROUNDED INTEGERS) BUT DIFFERENT delta.
# SAME UPSTREAM GRADIENT w FOR BOTH. COMPARE THE GRADIENTS THAT REACH x.
import torch
from quant import quantize, x_hat, g

torch.manual_seed(0)
k, N, lam = 1, 64, 1e-2

def grad(x, method):
    # dL/dx FOR L = <w, deq(x)>, I.E. UPSTREAM GRADIENT w PUSHED THROUGH THE DEQUANTIZER
    x = x.clone().requires_grad_(True)
    q, s_f, b_f = quantize(x, k)
    out = x_hat(q, s_f, b_f) if method == "ste" else g(q, x, lam)
    (w * out).sum().backward()
    return x.grad, q.detach(), (q - (x - b_f) / s_f).detach()

x = torch.randn(1, N)
w = torch.randn(1, N)

# NUDGE EVERY ENTRY EXCEPT THE MIN AND MAX (SO s_f, b_f STAY FIXED) A LITTLE
mid = torch.ones(1, N, dtype=torch.bool)
mid[0, x.argmin()] = mid[0, x.argmax()] = False
x2 = x + 0.3 * torch.randn(1, N) * mid
x2 = torch.where(quantize(x2, k)[0].round() == quantize(x, k)[0].round(), x2, x)  # UNDO NUDGES THAT FLIP q

for method in ["ste", "ridge"]:
    g1, q1, d1 = grad(x, method)
    g2, q2, d2 = grad(x2, method)
    assert torch.equal(q1.round(), q2.round()), "q CHANGED: USE A SMALLER NUDGE"
    print(f"{method:5s}  delta changed by {(d1 - d2).abs().max():.3f}   "
          f"gradient changed by {(g1 - g2)[mid].abs().max():.4f}")
