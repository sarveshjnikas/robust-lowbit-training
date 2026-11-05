# Talk script (slides_v4)

The key lines to say for each slide. The arrow (→) line at the end of each slide leads into the next one.

## 1. Title

"We present 'Robust Training of Neural Networks at Arbitrary Precision and Sparsity', by Ye et al. from Google DeepMind, at ICLR 2026."

"The main idea of the paper is a new technique to stabilize quantization-aware training of neural networks."

→ "First, why we want so few bits at all."

## 2. Why quantization at all

"Modern models have billions of weights, each stored in 16 or 32 bits. That is too large and too slow for a phone. Quantization stores each weight in 4, 2 or even 1 bit."

"At inference the model runs quantized. To make it adapt to that quantization step, we quantize during training too. This is called quantization-aware training."

"During training we keep the float weights x. The forward pass uses the quantized version, so the network learns to live with rounding."

"The pipeline: f rescales x onto an integer grid, round gives the integers q, and we rebuild x-hat. So x-hat equals x plus s_f times delta. Hence, the forward pass carries the rounding error."

→ "So where in the network does this happen?"

## 3. What gets quantized, and where

"Both the activations and the weights are quantized, block by block."

"During training, we rebuild both in float and do a normal matrix multiply. That way gradients can flow, and we update the float weights."

"At inference, we drop the float copy. We multiply the integers directly and apply the scale and shift afterwards. At 1 bit, that multiply is just XNOR and a bit count, which is where the savings come from."

→ "So far, so good. The trouble starts in the backward pass."

## 4. The problem in backprop: the true gradient is zero

"To train, we need the gradient of the loss with respect to x. By the chain rule, the scalings cancel, and we are left with the slope of round."

"Round is a staircase. Its slope is zero almost everywhere, and undefined at the jumps."

"So the true gradient is zero. The weights never move, so the model cannot be trained by gradient descent."

→ "The standard way out is the straight-through estimator."

## 5. The standard fix: STE

"STE replaces the derivative of round with 1. It treats round as the identity in the backward pass."

"Then the Jacobian is the identity, and the gradient passes straight through, as if there were no quantization."

"This is STE's blind spot. The forward pass is shifted by the rounding error, but the gradient never sees it."

"At 8 bits the error s_f times delta is tiny, so this does not matter. At 1 or 2 bits the error is as large as the values themselves. Hence, training with STE becomes unstable."

→ "Here are results from the paper that show this in practice."

## 6. STE at 1 bit: results from the paper

"On the left, a small GPT on Shakespeare with 1-bit weights and activations. STE's loss, in red, climbs. The paper's method, in blue, converges."

"On the right, affine quantization adds a shift, so it should help. Under STE it gives nothing. With the new method it gives a clear gain."

"Under STE, the levels sit at the extreme values of each block. With the new method, they are fitted to the data."

→ "So what is the new method?"

## 7. The new method: a denoising reconstruction g

"The idea is simple: keep f, keep round, and change only the rebuild."

"Instead of using the grid's own scale and shift, we fit a new affine map to x by ridge regression, with its own scale and shift. That gives us g of q. Lambda is the regularization parameter."

"In the backward pass, round is still straight through, exactly as in STE. The difference is that g's slope now depends on delta."

→ "Why does that change the gradient?"

## 8. Why it works: the backward pass sees delta

"Both methods treat round's slope as 1. Only the step after round differs."

"With STE, the rebuild has the same slope whatever the rounding error is, so the local gradient is just the identity."

"With g, the gradient reaches x by two paths: one through q, and one through x directly, via the fit of g."

"The path through x depends only on q. It never compares x with q, so it carries no rounding information."

"The path through q is different. Its scale s_g carries the covariance of x and q. Its two new terms carry the residual, x minus g of q, which is the rounding error left after the fit. Hence, the path through q carries the rounding error into backpropagation."

→ "Does it pay off in practice?"

## 9. Results from the paper

"Gemma 3 1B, trained from scratch. The score is the share of next tokens predicted correctly."

"The pattern: the fewer the bits, the bigger the gain. Almost nothing at 4 bits, plus 0.035 at 1 bit."

"The authors also test other domains, image classification and translation. With the new method, 4 bits match full precision."

"They also extend the approach to sparsity, treating thresholding as one more rounding step. A 4B model with 1-bit weights and half its weights pruned scores higher than a full-precision 1B model."

→ "We wanted to check this ourselves."

## 10. Our implementation 1: the paper's 1-bit GPT

"We reran the paper's exact Shakespeare setting at 1 bit, with both linear and affine quantization."

"g beats STE in both runs. STE stalls at similar losses in both cases, 2.48 linear and 2.41 affine, so the shift barely helps it. With g, affine reaches 1.65 against 2.06 for linear. So g unlocks the gain from the shift, as in the paper."

→ "Next, a real task outside language."

## 11. Our implementation 2: a real-world task

"We classify satellite images into 10 land types, with a small vision transformer of 2.7 million parameters. Each number is the mean of 3 seeds."

"At 4 bits, both methods match full precision. At 2 bits, g is about one point ahead."

"At 1 bit, STE drops to 65%, while g holds 91%. That is a gain of 26 points."

"The same pattern as the paper: the fewer the bits, the bigger the gain."

→ "So, what do we make of the paper?"

## 12. Advantages and disadvantages

"On the plus side, it makes 1-bit training work, its reconstruction is closer to x, and sparsity fits into the same framework."

"On the minus side, round is still straight through, so the claim of training 'without gradient estimation' is too strong."

"The gain at 4 bits, the most common setting, is very small. And the energy savings are estimates, with no hardware measurements."

→ "To wrap up."

## 13. Our takeaways

"What convinced us: a clean comparison where only the rebuild changes, consistent gains, and our own runs agree."

"Our open question: delta is detached exactly as in STE. The new information about delta arrives only after round, through g."

"The takeaway: a good approximation can still hurt optimization. Rounding error looks ignorable, but training needs it. Fitting parameters to the data is a good bet."

"Thank you. We are happy to take questions."
