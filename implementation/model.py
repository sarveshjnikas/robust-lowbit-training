# CHAR-LEVEL GPT, ADAPTED FROM MY gpt.py (deep-learning/3_gpt)
# ONLY CHANGE: THE LINEAR LAYERS INSIDE THE BLOCKS ARE BUILT BY Lin(), SO THEY CAN BE SWAPPED FOR QUANTIZED ONES.
# EMBEDDINGS AND THE OUTPUT HEAD STAY FULL PRECISION (OUR CHOICE, COMMON IN QAT).
import torch
import torch.nn as nn
from torch.nn import functional as F
from config import block_size, n_embd, n_head, n_layer, dropout, METHOD, BITS, LAMBDA
from quant import QLinear

def Lin(fan_in, fan_out, bias=True):
    # FLOAT RUN: PLAIN LINEAR. OTHERWISE: QUANTIZED LINEAR WITH THE CONFIGURED METHOD AND BIT-WIDTH
    if METHOD == "float":
        return nn.Linear(fan_in, fan_out, bias=bias)
    return QLinear(fan_in, fan_out, bias=bias, method=METHOD, k=BITS, lam=LAMBDA)

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = Lin(n_embd, head_size, bias=False)
        self.query = Lin(n_embd, head_size, bias=False)
        self.value = Lin(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))  # LOWER TRIANGULAR: NO LOOKING AHEAD
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k, q, v = self.key(x), self.query(x), self.value(x)  # (B,T,HS)
        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5  # (B,T,T) SCALED SCORES
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = self.dropout(F.softmax(wei, dim=-1))
        return wei @ v  # (B,T,HS)

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = Lin(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(Lin(n_embd, 4 * n_embd), nn.ReLU(), Lin(4 * n_embd, n_embd), nn.Dropout(dropout))

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.sa = MultiHeadAttention(n_head, n_embd // n_head)
        self.ffwd = FeedForward(n_embd)
        self.ln1, self.ln2 = nn.LayerNorm(n_embd), nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class GPT(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)  # FULL PRECISION
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.token_embedding_table(idx) + self.position_embedding_table(torch.arange(T, device=idx.device))
        logits = self.lm_head(self.ln_f(self.blocks(x)))  # (B,T,vocab_size)
        loss = None if targets is None else F.cross_entropy(logits.view(B * T, -1), targets.view(B * T))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, _ = self(idx[:, -block_size:])
            probs = F.softmax(logits[:, -1, :], dim=-1)
            idx = torch.cat((idx, torch.multinomial(probs, num_samples=1)), dim=1)
        return idx

if __name__ == "__main__":
    # SANITY CHECK: FORWARD PASS, INITIAL LOSS ~ ln(vocab_size)
    import math
    from data import vocab_size, get_batch
    from config import device
    m = GPT(vocab_size).to(device)
    x, y = get_batch("train")
    _, loss = m(x, y)
    print(f"params: {sum(p.numel() for p in m.parameters())/1e6:.2f}M  loss: {loss.item():.3f}  ln(V): {math.log(vocab_size):.3f}")
