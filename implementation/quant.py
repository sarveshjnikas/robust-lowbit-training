# QUANTIZATION: NOTATION FOLLOWS notes.tex (SECTIONS 2-4)
import torch
import torch.nn as nn
from torch.nn import functional as F

def quantize(x, k, eps=1e-8):
    # STEPS 1-2: q = f(x) + stopgrad(delta), AFFINE f ONTO THE GRID {0, ..., 2^k - 1}  (PAPER EQ. 1)
    # EACH ROW OF x IS ONE BLOCK: AN OUTPUT CHANNEL FOR WEIGHTS, A TOKEN FOR ACTIVATIONS
    x_min = x.amin(-1, keepdim=True)
    x_max = x.amax(-1, keepdim=True)
    s_f = (x_max - x_min + eps) / (2**k - 1)   # GRID STEP IN REAL UNITS
    b_f = x_min                                # REAL VALUE OF INTEGER 0 (q_min = 0)
    f_x = (x - b_f) / s_f                      # f(x)
    delta = (torch.round(f_x) - f_x).detach()  # ROUNDING ERROR, STOPGRAD
    q = f_x + delta                            # FORWARD: round(f(x)). BACKWARD: dq/df(x) = I
    return q, s_f, b_f

def x_hat(q, s_f, b_f):
    # STEP 3, STANDARD (USED WITH STE): x_hat = s_f q + b_f 1
    return s_f * q + b_f

def g(q, x, lam=1e-2):
    # STEP 3, PAPER (EQ. 3): RIDGE FIT OF x FROM q, PER ROW.  g(q) = s_g q + b_g 1
    q_bar = q.mean(-1, keepdim=True)
    x_bar = x.mean(-1, keepdim=True)
    var_q = ((q - q_bar) ** 2).mean(-1, keepdim=True)
    cov_xq = ((x - x_bar) * (q - q_bar)).mean(-1, keepdim=True)
    s_g = cov_xq / (var_q + lam)               # DEPENDS ON q, SO ON delta. NOT DETACHED: GRADIENT FLOWS THROUGH IT
    b_g = x_bar - s_g * q_bar
    return s_g * q + b_g

class QLinear(nn.Linear):
    # LINEAR LAYER WITH k-BIT WEIGHTS AND k-BIT INPUT ACTIVATIONS (A{k}W{k}). BIAS STAYS FULL PRECISION
    def __init__(self, fan_in, fan_out, bias=True, method="ste", k=1, lam=1e-2):
        super().__init__(fan_in, fan_out, bias=bias)
        self.method, self.k, self.lam = method, k, lam

    def deq(self, x):
        # QUANTIZE, THEN STEP 3 WITH THE CHOSEN DEQUANTIZER
        q, s_f, b_f = quantize(x, self.k)
        return x_hat(q, s_f, b_f) if self.method == "ste" else g(q, x, self.lam)

    def forward(self, x):
        return F.linear(self.deq(x), self.deq(self.weight), self.bias)
