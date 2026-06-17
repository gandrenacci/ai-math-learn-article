"""
Scaled dot-product attention — the equation that changed everything.

    Attention(Q, K, V) = softmax( Q K^T / sqrt(d_k) ) V

This is the heart of the Transformer. In three steps:
    1. Each query compares with every key via a dot product Q K^T -> relevance scores.
    2. Scale by 1/sqrt(d_k) (keeps scores from growing with dimension) and apply
       softmax over the keys -> attention weights that sum to 1.
    3. Use those weights to take a weighted combination of the values V -> output.

The result lets a model dynamically focus on the most relevant parts of the
input: any token can directly attend to any other token, capturing long-range
dependencies. The same mechanism powers self-attention, cross-attention and
retrieval-style attention. Its cost is O(n^2) in sequence length n.
"""

import math

import torch
import torch.nn.functional as F


def scaled_dot_product_attention(Q, K, V):
    """
    Q: (batch, seq_q, d_k)
    K: (batch, seq_k, d_k)
    V: (batch, seq_k, d_v)
    returns: (output, attention_weights)
    """
    d_k = Q.size(-1)
    scores = torch.bmm(Q, K.transpose(1, 2)) / math.sqrt(d_k)  # (batch, seq_q, seq_k)
    weights = F.softmax(scores, dim=-1)  # distribution over keys
    output = torch.bmm(weights, V)  # (batch, seq_q, d_v)
    return output, weights


torch.manual_seed(0)

# 2 sequences of 5 tokens, 16-dim query/key/value vectors.
batch, seq_len, d_k = 2, 5, 16
Q = torch.randn(batch, seq_len, d_k)
K = torch.randn(batch, seq_len, d_k)
V = torch.randn(batch, seq_len, d_k)

output, attn = scaled_dot_product_attention(Q, K, V)
print("Scaled dot-product attention:")
print("  output shape        :", tuple(output.shape))  # (2, 5, 16)
print("  attention weights   :", tuple(attn.shape))  # (2, 5, 5)
print("  each query's weights sum to 1:", attn[0].sum(dim=-1).detach().numpy().round(4))

# ---------------------------------------------------------------------------
# A tiny interpretable example: one query attends to 3 values.
# The query is aligned with key 1, so most weight should land on value 1.
# ---------------------------------------------------------------------------
q = torch.tensor([[[1.0, 0.0]]])  # (1, 1, 2)
k = torch.tensor([[[0.1, 0.0], [1.0, 0.0], [0.0, 1.0]]])  # 3 keys
v = torch.tensor([[[10.0], [20.0], [30.0]]])  # 3 values
out, w = scaled_dot_product_attention(q, k, v)
print("\nInterpretable example (query aligned with key 1):")
print("  attention weights:", w.squeeze().detach().numpy().round(3))
print(f"  output (weighted mix of values): {out.item():.3f}  -> pulled toward value 20")
