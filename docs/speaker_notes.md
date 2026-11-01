# Speaker notes (slides_v2)

Points taken off the slides to keep them clean. Say them out loud instead.

## Slide 6 — The new approach: a denoising reconstruction g

- **Forward pass, 2-bit example.** For x = (0.9, −0.3, 0.1, 0.6), the rounded integers are q = (3, 0, 1, 2). The standard rebuild x̂ = (0.9, −0.3, 0.1, 0.5) has squared error 0.010. The ridge rebuild g(q) = (0.935, −0.285, 0.122, 0.528) has squared error 0.007.
- **Backward pass.** s_g and b_g are computed from q = f(x) + δ, so ∂g/∂q changes with δ. STE's Jacobian, s_f·I, is the same for every δ.
- **λ is the denoising knob** (the paper's term, Sec. 3.3.1). As λ grows, s_g shrinks toward 0 and g falls back to the mean x̄. The noisy q is then ignored. λ also keeps the division safe when all q_i are equal (Var_q = 0); with λ = 0 training gives NaN (Table 13).
- **Cov_xq is not called the denoising factor in the paper.** It is the fit term: how well q lines up with x. If q tracks x poorly, Cov_xq is small, and that also shrinks s_g.

## Slide 7 — Why it works: the Jacobian trick

- **Exact part.** No guess is needed here. Inside one rounding step q stays fixed, but g still depends on x through x̄ and Cov_xq.
- **Error part.** Earlier layers learn from it: the term carries the error vector r = x − g(q) into their gradient.
- **Guess part at 4 bits.** At 4 bits ρ ≈ 1, so g behaves just like STE. This matches the paper's tiny 4-bit gain (+0.002) and our 4-bit tie.

## Slide 10 — Advantages and disadvantages

- **Simple.** Only one function changes, about five lines of code. One setting, λ = 0.01, is used for every experiment.
- **The 256× figure is an estimate.** It is the paper's own energy proxy (App. A.1.6): cost per multiply ∝ activation bits × weight bits, so 16 × 16 = 256 for BF16 against 1 × 1 for 1 bit. It is not a measurement.

## Slide 11 — Our takeaways

- **"No gradient estimation".** Since δ is detached, round's gradient is still guessed. The paper calls δ "additive noise", but the derivative used for round is the identity, the same as STE.
