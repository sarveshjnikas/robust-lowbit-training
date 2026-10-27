# TINY SHAKESPEARE, CHARACTER LEVEL
import torch
from config import DATA_PATH, TRAIN_FRAC, block_size, batch_size, device

with open(DATA_PATH, "r") as f:
    text = f.read()

# VOCAB: EVERY DISTINCT CHARACTER IS ONE TOKEN
chars = sorted(set(text))
vocab_size = len(chars)
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for i, c in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: "".join(itos[i] for i in l)

# SPLITS: CONTIGUOUS, NO SHUFFLING (NEXT-CHARACTER PREDICTION)
data = torch.tensor(encode(text), dtype=torch.long)
n = int(TRAIN_FRAC * len(data))
train_data, val_data = data[:n], data[n:]

def get_batch(split):
    d = train_data if split == "train" else val_data
    ix = torch.randint(len(d) - block_size, (batch_size,))  # batch_size RANDOM START OFFSETS
    x = torch.stack([d[i:i + block_size] for i in ix])
    y = torch.stack([d[i + 1:i + block_size + 1] for i in ix])  # TARGET = INPUT SHIFTED BY ONE
    return x.to(device), y.to(device)

if __name__ == "__main__":
    # SANITY CHECK
    print(f"chars: {len(text):,}  vocab: {vocab_size}  train: {len(train_data):,}  val: {len(val_data):,}")
    x, y = get_batch("train")
    print("x:", x.shape, "y:", y.shape, "device:", x.device)
    print(repr(decode(x[0].tolist())))
    print(repr(decode(y[0].tolist())))
    assert decode(encode(text[:1000])) == text[:1000]
