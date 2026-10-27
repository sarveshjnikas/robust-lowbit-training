# Toy implementation

A small character-level GPT on Tiny Shakespeare, trained with quantized weights and activations. This mirrors the paper's nanoGPT experiment (Fig. 1a) at laptop scale.

## Question

At 1–2 bits, does STE training become unstable, and does the paper's ridge dequantization $g$ fix it?

## Runs

| method | bits |
|---|---|
| float | – (baseline) |
| STE | 1, 2, 4 |
| ridge $g$ ($\lambda = 0.01$) | 1, 2, 4 |

All runs use the same seed, data, model and optimizer. Only the dequantization step differs.

## Outputs

1. Training and validation loss against step at 1 bit, STE vs. ridge (our version of Fig. 1a).
2. Final validation loss against bits for float, STE and ridge.
3. A gradient check: perturbing the rounding error $\delta$ leaves STE's local gradient unchanged but changes ridge's.

## Files

- `config.py`: all hyperparameters and the quantization settings (`METHOD`, `BITS`, `LAMBDA`)
- `data.py`: dataset, character tokenizer, 90/10 split, `get_batch`
- `data/tinyshakespeare.txt`: the dataset (about 1.1M characters)

Still to come: `quant.py` (the quantizer and both dequantizers), `model.py` (GPT with quantized linear layers), `train.py`, and plots.

## Run

```bash
cd implementation
python data.py   # sanity check of the data pipeline
```
