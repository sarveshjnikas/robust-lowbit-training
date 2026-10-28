# QUANTIZER: q = f(x) + stopgrad(delta)  (PAPER EQ. 1, CODE SNIPPET 1)
import torch

def quantize(x, bits, eps=1e-8):
    # AFFINE f: MAP [min, max] OF EACH ROW ONTO THE INTEGER GRID {0, ..., 2^k - 1}
    # ROW = ONE OUTPUT CHANNEL FOR WEIGHTS, ONE TOKEN FOR ACTIVATIONS
    lo = x.amin(-1, keepdim=True)              # b_f: REAL VALUE OF INTEGER 0
    hi = x.amax(-1, keepdim=True)
    s = (hi - lo + eps) / (2**bits - 1)        # s_f: GRID STEP IN REAL UNITS
    fx = (x - lo) / s                          # f(x)
    delta = (torch.round(fx) - fx).detach()    # ROUNDING ERROR, NO GRADIENT
    q = fx + delta                             # FORWARD: round(f(x)). BACKWARD: dq/df(x) = I
    return q, s, lo
