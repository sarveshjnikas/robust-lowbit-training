# JACOBIAN CHECK: SPLIT RIDGE'S LOCAL JACOBIAN INTO EXACT, GUESS AND ERROR PARTS; COMPARE WITH AUTOGRAD.
#   J_STE = 0 + I
#   J_g   = P + rho (I - P) + c u r^T
#   P = 11^T/N + u u^T/(N(Var_q + lam)),  u = q - q_bar 1,  r = x - g(q),  rho = s_g/s_f,  c = 1/(s_f N (Var_q + lam))
# SELF-CONTAINED: ONE BLOCK IN FLOAT64. quantize AND g FOLLOW quant.py (PAPER CODE SNIPPETS 1-2).
import torch

torch.set_default_dtype(torch.float64)
torch.manual_seed(0)
lam, N = 1e-2, 64

def quantize(x, k, eps=1e-8):
    # q = f(x) + stopgrad(delta), AFFINE f ONTO {0, ..., 2^k - 1}. s_f, b_f NOT DETACHED, AS IN quant.py
    lo, hi = x.min(), x.max()
    s_f = (hi - lo + eps) / (2**k - 1)
    f_x = (x - lo) / s_f
    return f_x + (torch.round(f_x) - f_x).detach(), s_f, lo

def g(q, x):
    # RIDGE DEQUANTIZATION: g(q) = s_g (q - q_bar) + x_bar
    u = q - q.mean()
    s_g = (u * (x - x.mean())).mean() / ((u * u).mean() + lam)
    return s_g * u + x.mean(), s_g

def ridge_used(x, k):
    # WHAT TRAINING DIFFERENTIATES: STRAIGHT THROUGH round, EXACT THROUGH g
    return g(quantize(x, k)[0], x)[0]

def ridge_true(x, k):
    # DEPLOYED MAP WITH q LOCALLY CONSTANT: ITS JACOBIAN IS THE TRUE (ALMOST EVERYWHERE) ONE
    return g(torch.round(quantize(x, k)[0]).detach(), x)[0]

def ste_used(x, k):
    q, s_f, b_f = quantize(x, k)
    return s_f * q + b_f

for k in [1, 2, 4]:
    x = torch.randn(N)
    keep = torch.ones(N, dtype=torch.bool)
    keep[x.argmin()] = keep[x.argmax()] = False  # s_f, b_f DEPEND ON THESE TWO ENTRIES; COMPARE THE REST
    sub = lambda A: A[keep][:, keep]
    jac = lambda fn: sub(torch.autograd.functional.jacobian(lambda z: fn(z, k), x))

    # CLOSED FORM
    q, s_f, _ = quantize(x, k)
    q = q.detach()
    gq, s_g = g(q, x)
    u, r = q - q.mean(), x - gq
    V = (u * u).mean()
    rho, c = s_g / s_f, 1 / (s_f * N * (V + lam))
    P = torch.ones(N, N) / N + torch.outer(u, u) / (N * (V + lam))
    E = c * torch.outer(u, r)
    J = P + rho * (torch.eye(N) - P) + E

    print(f"k={k}: closed form vs autograd {(jac(ridge_used) - sub(J)).abs().max():.1e} | "
          f"true ridge J vs P {(jac(ridge_true) - sub(P)).abs().max():.1e} | "
          f"STE J vs I {(jac(ste_used) - torch.eye(N - 2)).abs().max():.1e} | rho {rho:.2f}")

    # SHARE OF A RANDOM UPSTREAM GRADIENT G CARRIED BY EACH PART (GRADIENT = J^T G)
    G = torch.randn(N - 2)
    share = lambda A: ((sub(A).T @ G).norm() / G.norm()).item()
    print(f"      exact part {share(P):.1%}   error term {share(E):.1%}")

# FORWARD: RIDGE RECONSTRUCTION ERROR VS STE'S x_hat ON 10k RANDOM BLOCKS (HALF GAUSSIAN, HALF SKEWED LIKE ACTIVATIONS)
for k in [1, 2, 4]:
    worse, ratio = 0, []
    for i in range(10000):
        x = torch.randn(128).exp() if i % 2 else torch.randn(128)
        q, s_f, b_f = quantize(x, k)
        q = q.detach()
        e_ste = ((s_f * q + b_f - x) ** 2).sum()
        e_g = ((g(q, x)[0] - x) ** 2).sum()
        worse += int(e_g > e_ste + 1e-12)
        ratio.append((e_g / e_ste).item())
    print(f"k={k}: ridge error > STE error in {worse}/10000 blocks, median ratio ridge/STE {torch.tensor(ratio).median():.3f}")
