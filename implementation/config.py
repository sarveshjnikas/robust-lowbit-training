# ALL KNOBS FOR THE TOY EXPERIMENT IN ONE PLACE
import torch

# DATA
DATA_PATH = "data/tinyshakespeare.txt"
TRAIN_FRAC = 0.9  # FIRST 90% TRAIN, LAST 10% VAL

# MODEL (SMALL CHAR-LEVEL GPT, SAME FAMILY AS THE PAPER'S NANOGPT SHAKESPEARE RUN)
block_size = 64   # CONTEXT LENGTH
n_embd = 128      # EMBEDDING DIMENSION
n_head = 4        # ATTENTION HEADS
n_layer = 4       # TRANSFORMER BLOCKS
dropout = 0.0     # OFF SO RUNS DIFFER ONLY IN THE QUANTIZER

# TRAINING
batch_size = 64
max_iters = 3000
eval_interval = 250
eval_iters = 100  # BATCHES PER LOSS ESTIMATE
learning_rate = 1e-3
seed = 1337

# QUANTIZATION
METHOD = "ste"    # "float" | "ste" | "ridge"
BITS = 1          # k: 1, 2 OR 4 (WEIGHTS AND ACTIVATIONS, A{k}W{k})
LAMBDA = 1e-2     # RIDGE PENALTY, PAPER USES 0.01 EVERYWHERE

device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
