"""
Deep network as function composition.

A neural network is just functions stacked on top of each other — the output of
one layer is the input of the next:

    h0 = x
    h_l = sigma_l( W_l h_{l-1} + b_l ),   l = 1 .. L
    f(x) = f_L( ... f_2( f_1(x) ) )

Each layer applies an affine transform followed by a nonlinearity. Composing
several such layers is what makes a network "deep" and able to represent complex
patterns. This script builds a small 3-layer network from scratch and checks it
against the equivalent PyTorch model.
"""

import torch
import torch.nn as nn

torch.manual_seed(0)

x = torch.randn(1, 4)  # one input sample, 4 features


# ---------------------------------------------------------------------------
# 1. A network written explicitly as a composition of layer functions.
# ---------------------------------------------------------------------------
def affine(h, W, b):
    return h @ W.T + b


# Layer parameters (normally learned during training).
W1, b1 = torch.randn(8, 4), torch.zeros(8)  # 4 -> 8
W2, b2 = torch.randn(8, 8), torch.zeros(8)  # 8 -> 8
W3, b3 = torch.randn(3, 8), torch.zeros(3)  # 8 -> 3

# f(x) = W3 . relu( W2 . relu( W1 x ) )  -- nonlinearity between each layer.
h1 = torch.relu(affine(x, W1, b1))
h2 = torch.relu(affine(h1, W2, b2))
out = affine(h2, W3, b3)  # output layer (logits, no activation)

print("Hand-built composition:")
print("  x  ->", tuple(x.shape), "-> h1", tuple(h1.shape), "-> h2", tuple(h2.shape), "-> out", tuple(out.shape))
print("  output:", out.detach().numpy().round(3))

# ---------------------------------------------------------------------------
# 2. The same idea expressed idiomatically with nn.Sequential.
#    Sequential literally composes the modules left to right.
# ---------------------------------------------------------------------------
model = nn.Sequential(
    nn.Linear(4, 8),
    nn.ReLU(),
    nn.Linear(8, 8),
    nn.ReLU(),
    nn.Linear(8, 3),
)
print("\nnn.Sequential is the same composition f3(relu(f2(relu(f1(x))))):")
print("  output shape:", tuple(model(x).shape))
print("  total parameters:", sum(p.numel() for p in model.parameters()))
