# Speaker notes (slides_v4)

Short notes on what to say for each slide, followed by backup answers.

## 1. Title

- Ye et al., Google DeepMind, ICLR 2026.
- Pitch: training at 1 bit usually breaks. The paper changes one step, the rebuild after rounding, and training becomes stable.
- Plan: the problem, STE and its blind spot, the new method, why it works, results, our two experiments, our view.

## 2. Why quantization, and the forward pass

- Models have billions of weights at 16 or 32 bits each. Quantization stores each weight in 1 to 4 bits.
- Training keeps float weights x. The forward pass uses the quantized version, so the network learns to live with rounding.
- Walk the pipeline: f rescales onto the integer grid, round gives q, and s_f q + b_f rebuilds x̂.
- Key identity: x̂ = x + s_f δ. Quantization is x plus a rounding error.

## 3. What gets quantized, and where

- Both activations and weights are quantized, block by block.
- Training: rebuild in float, then do a float matmul, so autograd works. The float W is updated.
- Inference: the float copy is dropped. Multiply the integer codes, then apply the scale and shift (Theorem 1). At 1 bit, the multiply is XNOR plus a bit count.
- Both orders compute the same function. Only the cost differs (see `docs/rebuild_order.md`).

## 4. The problem: the true gradient is zero

- Chain rule through the quantizer: the two scalings cancel, so the local gradient is the slope of round.
- Round is a staircase, so its slope is zero almost everywhere.
- Hence ∇ₓL = 0. A small change in x does not change q, so x never moves.

## 5. The standard fix: STE

- Round in the forward pass. In the backward pass, treat the rounding error δ as a constant (stopgrad).
- Then the Jacobian is I, an N×N identity, since x is a block. The gradient passes straight through.
- Blind spot: the forward pass is shifted by s_f δ, but δ never appears in the gradient.
- At 8 bits δ is tiny, so this is harmless. At 1–2 bits the error is as large as the values.

## 6. STE at 1 bit: results from the paper

- Left: small GPT on Shakespeare, 1-bit weights and activations. STE's loss (red) climbs; g (blue) converges.
- Right: affine quantization adds a shift. It should help. Under STE it does not (0.3399 vs 0.3397). With g it does (0.3547 vs 0.3751).
- Under STE, the levels come from extreme values (max|x|, or x_min and x_max). With g, the levels are fitted: they become the group means.
- The paper's explanation: STE's gradient cannot tune the outlier-sensitive shift. The tie itself is an empirical result.

## 7. The new method: a denoising reconstruction g

- Keep f and round. Replace the rebuild s_f q + b_f by g(q) = s_g q + b_g.
- s_g and b_g are fitted to x by ridge regression. g is the best affine rebuild of x from q.
- Closed form: s_g = Cov_xq / (Var_q + λ), b_g = x̄ − s_g q̄.
- λ = 0.01 everywhere. It keeps the division safe; λ = 0 gives NaN (Table 13).
- Backward: round is still straight through, as in STE. But g's slope now depends on δ.

## 8. Why it works: the backward pass sees δ

- Both methods treat round's slope as 1. Only the step after round differs.
- STE: x̂ has the same slope for every δ. The local gradient is I.
- With g, the gradient reaches x by two paths.
- Through q: s_g and the two new terms are built from x − g(q), which is the rounding error after the fit. This is how δ enters the gradient.
- Through x: for a fixed q, g is linear in x. This slope depends only on q. It never compares x with q, so it carries no information about δ.

## 9. Results from the paper

- Gemma 3 1B, trained from scratch. The score is the share of next tokens predicted correctly.
- The gain grows as bits drop: +0.002 at 4 bits, +0.024 at 2 bits, +0.035 at 1 bit.
- Sparsity: a 4B model with 1-bit weights, half pruned, takes 650 MB and scores 0.452. It beats the full-precision 1B (1400 MB, 0.449).
- At 4 bits, image classification and translation match full precision.

## 10. Our implementation 1: the paper's 1-bit GPT

- We reran the paper's exact Shakespeare setting (nanoGPT recipe, 5000 steps), with 1-bit weights and activations. Only the rebuild differs between the runs.
- STE stalls; g keeps learning. Final training loss: 2.41 for STE, 1.65 for g.
- Our g curve is within 0.05 of the paper's at every checkpoint.
- STE did not blow up to about 6 as in the paper. Its residual stream does grow 250–7000×, but LayerNorm hides this from the loss. Each layer's update is swamped, so learning stalls.

## 11. Our implementation 2: a real-world task

- EuroSAT: satellite images, 10 land-use classes. Small vision transformer, 2.7M parameters, 6 layers, 40 epochs.
- Every number is the mean of 3 seeds. Full precision gives 95.4%.
- 4 bits: both methods match full precision. 2 bits: g is +1.2 ahead. 1 bit: STE falls to 65.2, g holds 91.2 (+26.0).
- Same pattern as the paper: the fewer the bits, the bigger the gain.

## 12. Advantages and disadvantages

- Advantages: 1-bit training becomes stable. The reconstruction is closer to x, because g is a least-squares fit. Sparsity fits the same frame: threshold, then quantize, then g.
- Round is still straight through, so "without gradient estimation" is too strong.
- The 4-bit gain is only +0.002, in the most common setting.
- The energy savings are a proxy (bits × bits × ops). There are no hardware measurements.

## 13. Our takeaways

- Convincing: the comparison is clean, the gains are consistent, and our runs agree.
- Open question: δ is detached exactly as in STE, so ∂q/∂f(x) = I. The δ information arrives only after round, through g.
- Takeaway: a good approximation can still hurt optimization. Training needs the rounding error. Fit parameters to the data.

## Backup answers

- **Why is the Jacobian I and not 1?** x is a block in ℝᴺ, so ∂x̂/∂x is N×N. Entry i depends only on xᵢ with slope 1.
- **Are s_f and b_f constants?** No. They depend on x_max and x_min. The extra terms affect only those 2 of N columns. Elsewhere, I is exact.
- **Is g just a rescaled STE?** Partly. The through-q path is scaled by s_g/s_f: 0.29, 0.87 and 0.99 at 1, 2 and 4 bits. At 4 bits g behaves almost like STE, which matches the tiny 4-bit gain.
- **Full Jacobian (our derivation).** J_g = P + ρ(I − P) + c u rᵀ, with u = q − q̄1, r = x − g(q), ρ = s_g/s_f. Only P (the through-x path) is exact. `implementation/jacobian_check.py` checks this against autograd.
- **Paper's gradient analysis.** Appendix A.4 derives only ∂g/∂q, for the linear case. It misses the through-x path P.
- **Why λ = 0.01?** Table 13, 1-bit translation BLEU: λ = 1 gives 5.83, 0.01 gives 21.42, 0.0001 gives 20.08, 0 gives NaN.
- **Theorem 1.** Y = (s_X s_Wᵀ) ⊙ (Q_X Q_W − n q̄_X q̄_Wᵀ) + n x̄ w̄ᵀ. The proof uses mean-centring: the cross terms cancel.
- **Error bars?** The paper reports single runs. Our EuroSAT runs use 3 seeds. The 4-bit gain (+0.2) is close to the seed spread (standard deviation 0.1–0.2).
- **"For the first time"?** LSQ (Esser et al., 2019) already uses δ in the step-size gradient, but not in the upstream gradient.
- **Why does STE not explode in our run?** We tried model size, run length, gradient clipping, a quantized output head, per-tensor scales and detached scales. None reproduced the explosion. STE's residual stream grows, but LayerNorm hides it from the loss.
