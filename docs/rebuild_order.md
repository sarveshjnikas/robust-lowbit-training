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

The left form comes from order A and the right form from order B. Rounding uses the same STE surrogate in both, $\partial q / \partial x = I / s_f$. For $g$, the derivatives through $s_g$ and $b_g$ are also unchanged. Neither order is needed for correctness. (Checked numerically: on a 64-value block at 2 bits, A and B agree to about $10^{-5}$ in output and gradients.)

## Claim 3: why this paper (and we) use order A in training

Order B can speed up training in principle: GPUs have fast integer units (an A100 runs int8 matmuls about twice as fast as fp16), and there is published work that runs the training forward pass in int8, e.g. SwitchBack (Wortsman et al., 2023). Order A is used here for practical reasons:

1. **QAT here is a simulation.** The goal is a model that works at 1 to 4 bits and a measurement of its accuracy, not faster training. Order A ("fake quantization") is a drop-in float layer that autograd handles directly. Autograd only differentiates float tensors (PyTorch: "Only Tensors of floating point and complex dtype can require gradients"), so order B needs custom kernels and a hand-written backward pass.
2. **No kernels for the low-bit cases.** There are no standard 1-, 2- or 4-bit training kernels, and none for $g$, whose scale $s_g$ depends on $x$. Order A works at any bit width without custom code.
3. **The backward pass is still real-valued.** Per layer, a training step does three matmuls: the forward pass $Y = \hat X \hat W^\top$, the input gradient $\nabla_{\hat X} L = \nabla_Y L \, \hat W$, and the weight gradient $\nabla_{\hat W} L = (\nabla_Y L)^\top \hat X$. Since $\nabla_Y L$ is real-valued, an integer forward pass speeds up at most about a third of the work. The weights also change every step, so they are re-quantized every step and nothing is precomputed.

## Claim 4: order B is right at inference

The weights are fixed. $q_W$, $s_w$, $b_w$ and $\sum_i q_{w,i}$ are computed once and stored, and the float copy is dropped. There is no backward pass. Per output, the only $O(N)$ work is the integer dot product $\sum_i q_{x,i} q_{w,i}$, which at 1 bit is XNOR plus popcount. The correction terms cost $O(1)$ per output, given $\sum_i q_{x,i}$ once per token.

## Summary

Orders A and B compute the same function, so the outputs and gradients are equal. Training uses A because it simulates quantization simply with standard float tools; integer training is possible but needs special kernels. Inference uses B because the integer matmul is almost all of the work.
