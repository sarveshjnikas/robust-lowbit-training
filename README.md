# robust-lowbit-training

Study and reimplementation of **Robust Training of Neural Networks at Arbitrary Precision and Sparsity** (Ye et al., Google DeepMind, ICLR 2026) — [OpenReview](https://openreview.net/forum?id=e6nZrzSccj).

The paper replaces the Straight-Through Estimator's quantization-oblivious backward pass with a ridge-regression dequantization transform, so the rounding error enters the gradient. This makes 1-bit and sub-1-bit training stable.

## Contents

- `notes.tex` / `notes.pdf` — mathematical notes, built from the basics (in progress)

## Planned

- Minimal toy experiment: STE vs ridge dequantization at 1–4 bits
- Full implementation with tests and ablations

This is an independent study, not the authors' code.
