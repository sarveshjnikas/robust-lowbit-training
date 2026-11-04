# Why training rebuilds before multiplying, and inference multiplies first

## Setup

One linear layer, one output entry, over $N$ inputs. Activations and weights are quantized to integer codes with a scale and a shift:

$$\hat x_i = s_x q_{x,i} + b_x, \qquad \hat w_i = s_w q_{w,i} + b_w$$

**Order A, rebuild then multiply (training):**

$$y = \sum_{i=1}^{N} \hat x_i \hat w_i$$

**Order B, multiply then rebuild (inference).** Expand the product:

$$y = s_x s_w \sum_i q_{x,i} q_{w,i} + s_x b_w \sum_i q_{x,i} + b_x s_w \sum_i q_{w,i} + N b_x b_w$$

## Claim 1: same output

The expansion is distributivity, so A and B are the same function of $(q_x, q_w, s_x, s_w, b_x, b_w)$. They differ only by floating-point rounding.

## Claim 2: same gradients

The chain rule depends on the function, not on the order of evaluation, so every partial derivative is the same. For example:

$$\frac{\partial y}{\partial q_{w,j}} = s_w \hat x_j = s_x s_w q_{x,j} + b_x s_w$$

The left form comes from order A and the right form from order B. Rounding uses the same STE surrogate in both, $\partial q / \partial x = I / s_f$. For $g$, the derivatives through $s_g$ and $b_g$ are also unchanged. Neither order is needed for correctness.

## Claim 3: order B gives no benefit in training

1. **Gradients are real-valued.** Per layer, a training step does three matmuls: the forward pass $Y = \hat X \hat W^\top$, the input gradient $\nabla_{\hat X} L = \nabla_Y L \, \hat W$, and the weight gradient $\nabla_{\hat W} L = (\nabla_Y L)^\top \hat X$. Since $\nabla_Y L$ is real-valued, both backward matmuls are float whichever order the forward pass uses. An integer forward pass could speed up at most about a third of the work.
2. **Nothing can be precomputed.** The float weights change every step, so $q_W$, the scale and the shift are recomputed every step. For $g$, $s_g$ depends on $x$. Order B's advantage, stored integers and precomputed correction terms, does not exist during training.
3. **Tooling.** Autograd only differentiates float tensors, and integer matmul kernels do not support backprop. Order B would need custom kernels and a hand-written backward pass to compute the same numbers.

## Claim 4: order B is right at inference

The weights are fixed. $q_W$, $s_w$, $b_w$ and $\sum_i q_{w,i}$ are computed once and stored, and the float copy is dropped. There is no backward pass. Per output, the only $O(N)$ work is the integer dot product $\sum_i q_{x,i} q_{w,i}$, which at 1 bit is XNOR plus popcount. The correction terms cost $O(1)$ per output, given $\sum_i q_{x,i}$ once per token.

## Summary

Orders A and B compute the same function, so the outputs and gradients are equal. Training uses A because B saves nothing there and complicates the backward pass. Inference uses B because the integer matmul is almost all of the work.
