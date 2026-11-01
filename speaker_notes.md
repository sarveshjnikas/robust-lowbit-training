# Speaker notes (slides_v2)

What to say on each slide, then backup answers for likely questions. The slides stay sparse; the detail lives here.

## Slide 1 — Title

- The paper is Ye et al., Google DeepMind, ICLR 2026.
- One-line pitch: training networks at 1 bit usually breaks. The paper changes one small step, the reconstruction after rounding, and training becomes stable.
- Plan: the problem, the standard fix (STE) and its blind spot, the proposed fix, why it works, the paper's results, our own 1-bit run, then our view.

## Slide 2 — Why quantization, and the forward pass during training

- Models are huge. Storing each weight in 16 or 32 bits makes them slow, large and energy-hungry. Quantization stores each weight in a few bits.
- During training we keep float weights x, but the forward pass uses the quantized version. That way the network learns to live with rounding.
- Walk the pipeline left to right: f rescales the block onto the integer grid, round gives the integers q, and s_f q + b_f rebuilds x̂.
- The key identity is x̂ = x + s_f δ. Quantization is x plus a rounding error. Everything later builds on this.

## Slide 3 — The problem: the true gradient is zero

- The chain rule through the quantizer: the two scalings cancel, so the local gradient is just the slope of round.
- Round is a staircase. Its slope is zero almost everywhere.
- So the true gradient to x is zero. A small change in x does not change q, so x never moves. Training cannot start.
- If asked: s_f and b_f are held fixed here. They depend only on the block's largest and smallest entries.

## Slide 4 — The standard fix: Straight-Through Estimator (STE)

- STE keeps rounding in the forward pass but pretends round's slope is 1 in the backward pass. In code: δ is detached with stopgrad.
- Forward: x̂ = x + s_f δ. Backward: the Jacobian is the identity, so the gradient passes straight through.
- The blind spot: the forward pass is shifted by s_f δ, but δ never appears in the gradient. Earlier layers get no signal about rounding.
- At 8 bits this is harmless because δ is tiny. At 1–2 bits the rounding error is as large as the values themselves.

## Slide 5 — STE at 1 bit: results from the paper

- Left (Fig. 1a): small GPT on Shakespeare, 1-bit weights and activations. STE's loss (red) climbs and swings between about 5 and 8. The proposed method (blue) converges smoothly.
- Right (Fig. 1b): affine quantization should help, since it adds a shift. Under STE it gives nothing (0.3399 linear vs 0.3397 affine). With the proposed method it helps (0.3547 vs 0.3751).
- Paper's explanation: STE cannot learn the sensitive shift term because its gradient ignores the rounding error.

## Slide 6 — The proposed approach: a denoising reconstruction g

- Keep f and round. Only the rebuild changes: x̂ = s_f q + b_f becomes g(q) = s_g q + b_g.
- s_g and b_g are not the grid's scale and shift. They are fitted to x by ridge regression, so g is the best affine rebuild of x from q.
- The closed form comes from setting both derivatives to zero: s_g = Cov_xq/(Var_q + λ), b_g = x̄ − s_g q̄.
- λ is the denoising knob (the paper's term, Sec. 3.3.1). As λ grows, s_g shrinks to 0 and g falls back to the mean x̄, ignoring the noisy q. λ also keeps the division safe when Var_q = 0; with λ = 0 training gives NaN (Table 13).
- Cov_xq is the fit term: how well q lines up with x. If q tracks x poorly, Cov_xq is small and s_g shrinks too.
- 2-bit example if useful: x = (0.9, −0.3, 0.1, 0.6) gives q = (3, 0, 1, 2). The standard rebuild has squared error 0.010; g has 0.007.
- Bottom block: the forward pass feeds g(q) to the next layer. The backward pass still goes straight through round, but g's slope now depends on δ.

## Slide 7 — Why it works: the backward pass sees δ

- Both methods use the same surrogate for round's slope, so ∂q/∂x = I/s_f. Only the step after round differs. In STE, s_f · (1/s_f) = I, so the gradient passes straight through. With g, the 1/s_f is the factor in front of the through-q term.
- STE: x̂ = s_f q + b_f has slope s_f I for every δ, so the gradient is simply passed through.
- With g, x reaches the loss by two routes, so the chain rule gives two terms.
- Through q: this route passes round, so it uses the surrogate. s_g and b_g are computed from q = f(x) + δ and from x, so every term of ∂g/∂q changes with the rounding error. s_g I has STE's shape, but its scale carries δ: s_g = s_f (Var_q − Cov(q, δ))/(Var_q + λ). The two product-rule terms are new: ∂s_g/∂q = (r − s_g u)/(N(Var_q + λ)) contains the residual r = x − g(q) = −s_f(δ − δ̄1) + (s_f − s_g)u, which is the rounding error after the fit. This is how δ enters the gradient.
- Through x: g also reads x directly, through x̄ and Cov_xq. With q fixed, g is linear in x, so this slope is exact and needs no surrogate. It depends only on q, so it carries no information about δ.
- The surrogate path starts at s_g/s_f instead of 1: 0.29, 0.87 and 0.99 at 1, 2 and 4 bits (random block of 64). At 4 bits g behaves almost like STE, which matches the paper's tiny 4-bit gain.

## Slide 8 — Results from the paper

- Gemma 3 1B trained from scratch. The score is the share of next tokens predicted correctly.
- The gain grows as bits drop: +0.002 at 4 bits, +0.023 at 2 bits, +0.035 at 1 bit.
- At 1.5 bits (−1, 0, 1), STE fails to converge (NaN); g still trains to 0.330.
- Bigger model, fewer bits: a 4B model with 1-bit weights and 2:4 sparsity takes 650 MB and scores 0.452. That beats the full-precision 1B model at 1400 MB and 0.449.
- Other tasks at 4 bits match full precision: ResNet-50 76.45% vs 76.41%; English–German translation 29.71 vs 29.49 BLEU.

## Slide 9 — Our implementation: training at 1 bit

- We reran the paper's Shakespeare setting ourselves: nanoGPT recipe, 6 layers, width 384, about 10.7M parameters. Every linear layer inside the blocks uses 1-bit weights and activations. Only the reconstruction differs between the two runs.
- The three panels are three runs: standard 1-bit, 1-bit sign quantization, and 1-bit at 3× the learning rate. Solid lines are training loss, dashed lines validation loss.
- In all three, STE stalls around 2.6 while g keeps learning, down to about 2.35.
- STE's activations blow up: the residual stream is about 340× larger than with g after 1000 steps (617 vs 1.8). LayerNorm hides this from the loss, but each layer's update is swamped, so learning stalls.
- The cause: at 1 bit STE puts the two levels at the row's x_min and x_max, which is too far out. g fits the two levels near the means of the two groups, so outputs stay the right size.
- At 3× the learning rate, STE's training loss turns upward after about step 75. Our reading: bigger steps trust a gradient that is blind to rounding error. g at the same rate trains best of all runs (2.29 validation).

## Slide 10 — Advantages and disadvantages

- 1-bit training works. A 1-bit multiply is an XNOR plus a bit count. The 256× figure is the paper's own energy proxy (activation bits × weight bits: 16 × 16 vs 1 × 1), not a measurement.
- Better reconstruction: g is the least-squares fit, so it is usually closer to x. In our check its error was about 5× smaller at 1 bit.
- Sparsity fits the same frame: threshold small values to zero, then quantize and reconstruct with g.
- The method is simple: one function changes, about five lines of code, and λ = 0.01 is used everywhere.
- Disadvantages: round still uses the straight-through surrogate; the 4-bit gain is only +0.002 in the most common setting; the energy savings are estimates with no hardware measurements.

## Slide 11 — Our takeaways

- What convinced us: the comparison is clean, since only the reconstruction changes. The gains are consistent and grow at fewer bits. Our own 1-bit run shows the same thing.
- What we question: the paper claims "no gradient estimation" (abstract, Table 8). But δ is detached exactly as in STE, so round's slope is still the identity surrogate. The new information about δ arrives through that surrogate path.
- Takeaway: a good approximation can still break optimization. Rounding error looks ignorable, but training needs its information. Fit parameters to the data wherever possible.

## Backup answers

- **"Isn't g just a better-scaled STE?"** Partly. The through-q path is still a surrogate, scaled by s_g/s_f. But g adds an exact through-x path, and its slope changes with δ. STE's slope never does.
- **The full Jacobian split (our derivation).** Write u = q − q̄1, r = x − g(q), ρ = s_g/s_f. The Jacobian of g is P + ρ(I − P) + c u rᵀ, with P = 11ᵀ/N + uuᵀ/(N(Var_q + λ)) and c = 1/(s_f N(Var_q + λ)). Only P (the through-x path) is exact; the other two terms pass through round's surrogate. On a random block of 64, the c u rᵀ term carries 0.5%, 4.7% and 0.3% of the gradient at 1, 2 and 4 bits. `implementation/jacobian_check.py` checks this against autograd.
- **What does the paper's own gradient analysis cover?** Appendix A.4 derives only ∂g/∂q, for the linear case. It leaves out the exact through-x path.
- **Are s_f and b_f really fixed?** They are recomputed from the block's max and min every step, and autograd flows through them. That affects only the two extreme entries of each block. Our equations hold s_f and b_f fixed, as the paper's text does.
- **Why λ = 0.01?** The paper uses it for every experiment. Table 13 shows a wide working range: λ = 1 gives 5.83 BLEU, 0.01 gives 21.42, 0.0001 gives 20.08, and 0 gives NaN.
- **Theorem 1 (if asked about efficiency).** Affine-quantized matrix multiplication reduces to one integer matmul plus two rank-1 corrections: Y = (s_X s_Wᵀ) ⊙ (Q_X Q_W − n q̄_X q̄_Wᵀ) + n x̄ w̄ᵀ. The proof uses mean-centering: the centred rows and columns cancel the cross terms.
- **Our smaller toy.** 4 blocks, 0.82M parameters, 3000 steps. Validation loss STE vs g: 2.123 vs 2.071 at 1 bit, 1.812 vs 1.790 at 2 bits, 1.592 vs 1.597 at 4 bits.
- **Animations.** `implementation/onebit/plots/` holds the full four-panel animations (loss, residual-stream size, input range of the 1-bit layers, one weight row) and `loss_three.gif` for slide 9.
