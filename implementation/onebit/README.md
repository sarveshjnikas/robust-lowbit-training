# 1-bit divergence experiment

## Question

The paper (Fig. 1a) reports that with 1-bit weights and activations (A1W1), STE training of a small GPT on Shakespeare diverges, while the ridge dequantizer $g$ trains smoothly. Can we see this, and how exactly does STE go wrong?

## Setup

- Model and recipe follow nanoGPT's `train_shakespeare_char` config, the paper's setup. The model has 6 layers, 6 heads and width 384, about 10.7M parameters. It has no biases, a GELU MLP and a tied output head. Training uses AdamW at lr $10^{-3}$ with 100 warmup steps and cosine decay over 5000 steps, dropout 0.2 and gradient clipping at 1.0.
- Every linear layer inside the blocks is quantized to 1 bit. Weights are quantized per output channel and inputs per token. Embeddings and the output head stay in full precision.
- Two quantizers are compared: **affine** (levels $x_{\min}, x_{\max}$) and **linear** (sign, levels $\pm s$). The rounding uses the same straight-through `quantize` as `quant.py` in both methods; only the dequantizer differs (STE's $\hat x$ vs. ridge $g$, $\lambda=0.01$).
- **Deviation from the paper:** each step uses batch 32 and context 128, against the paper's 64 and 256. At full size a step takes 1.8 s (STE) and 2.9 s (ridge) on an M4 laptop, so 5000 steps would take over 6 hours.

## Results

Validation loss at the last step (lower is better):

| run | steps | STE | ridge $g$ |
|---|---|---|---|
| affine, lr $10^{-3}$ | 1000 | 2.580 | **2.349** |
| linear (sign), lr $10^{-3}$ | 2000 | 2.609 | **2.425** |
| affine, lr $3\times10^{-3}$ | 1000 | 2.827 | **2.292** |

RMS of the residual stream after the last block:

| run | step 0 | last step |
|---|---|---|
| affine, lr $10^{-3}$ | STE 17 vs. $g$ 0.03 | STE 617 vs. $g$ 1.8 |
| linear, lr $10^{-3}$ | STE 30 vs. $g$ 0.04 | STE 8853 vs. $g$ 3.6 |
| affine, lr $3\times10^{-3}$ | STE 17 vs. $g$ 0.03 | STE 11599 vs. $g$ 9.4 |

## What we found

1. **STE diverges inside the network, not in the loss.** The residual stream under STE is already about 500x too large at step 0. It keeps growing roughly linearly for the whole run, reaching 2400x larger than with $g$ by step 2000 in the linear run. The input range of the 1-bit layers, which is the 1-bit step size, also keeps growing: from 13 to 31 (affine), from 15 to 74 (linear) and from 15 to 126 (lr $3\times10^{-3}$). With $g$ both stay flat.
2. **When:** the blow-up starts at step 0. The loss curves separate at step 119 (affine), step 28 (linear) and step 123 (lr $3\times10^{-3}$). After that STE's loss stalls while $g$ keeps improving.
3. **How:** at 1 bit, STE maps every weight of a row to its row's $x_{\min}$ or $x_{\max}$, and every activation of a token to that token's extremes. Each layer's output is therefore far too large. STE's gradient never sees the rounding error $\delta$, so nothing pulls the scale back, and the residual stream accumulates the excess. LayerNorm hides the growth from the loss, but each new layer's contribution shrinks relative to the stream, so learning stalls. $g$ maps the codes to fitted levels (roughly the two group means) and keeps the mean of $x$, so outputs stay the right size.
4. **Higher learning rate:** at lr $3\times10^{-3}$, STE's loss reaches its best (about 2.82) near step 75 and then rises (2.96 at step 225). It never recovers below 2.82. $g$ at the same learning rate trains best of all runs, reaching 2.29.
5. **Not reproduced:** the paper's loss explosion to about 6–8 at lr $10^{-3}$. Likely reasons are our smaller batch and context, and shorter runs (1000–2000 steps against 5000). The paper's STE may also have reached a point we did not, where the exploding activations finally break the loss.

## Animations

Each animation in `plots/` has four panels: loss, size of the residual stream, input range of the 1-bit layers, and one weight row with its two 1-bit levels. Markers show when the losses separate.

- `divergence_linear.{mp4,gif,png}`: linear (sign) 1-bit, 2000 steps. The clearest view.
- `divergence_affine.{mp4,gif,png}`: affine 1-bit, 1000 steps. Fig. 1a's setting.
- `divergence_affine_lr3e-3.{mp4,gif,png}`: affine at 3x the learning rate, 1000 steps.

## Files

- `train_1bit.py`: self-contained training script. It logs loss, gradient norm, weight bit flips, residual-stream RMS, layer input ranges and weight-row snapshots to `runs/<name>.json`.
- `animate.py`: turns a pair of runs into the animation and a final-frame PNG. The MP4 needs `imageio-ffmpeg`.
- `runs/`: the logs of the six runs above.

## Run

```bash
cd implementation/onebit
python train_1bit.py --method ste   --scheme linear --iters 2000 --batch 32 --block 128 --eval_iters 20 --name ste_lin_k1
python train_1bit.py --method ridge --scheme linear --iters 2000 --batch 32 --block 128 --eval_iters 20 --name ridge_lin_k1
python animate.py ste_lin_k1 ridge_lin_k1 linear
```
