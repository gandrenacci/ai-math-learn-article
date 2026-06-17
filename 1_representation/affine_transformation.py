"""
Affine transformation — the core operation of every neural-network layer.

    z = W x + b

Where:
    x in R^d        input feature vector (d features)
    W in R^(m x d)  weight matrix (learned parameters)
    b in R^m        bias vector (offset / intercept)
    z in R^m        output vector (before activation)

Linear regression, logistic regression and a PyTorch nn.Linear layer are all
this same z = Wx + b, with different things wrapped around it. The matrix mixes
the d inputs into m outputs; the bias lets the layer shift its output away from
zero, exactly like the intercept in y = wx + b.
"""

import torch

# ---------------------------------------------------------------------------
# 1. Do the math by hand so the formula is explicit.
# ---------------------------------------------------------------------------
x = torch.tensor([1.0, 2.0, 3.0])  # one sample, d = 3 features

W = torch.tensor(
    [
        [0.5, -0.2, 0.1],  # row -> output neuron 1
        [0.3, 0.8, -0.5],  # row -> output neuron 2
    ]
)  # shape (m=2, d=3)

b = torch.tensor([0.1, -0.2])  # one bias per output neuron

z = W @ x + b  # matrix-vector product, then add the bias
print("Manual affine transform z = Wx + b:")
print("  z =", z.tolist())

# Spell out what each output neuron computes:
z1 = 0.5 * 1 + (-0.2) * 2 + 0.1 * 3 + 0.1
z2 = 0.3 * 1 + 0.8 * 2 + (-0.5) * 3 - 0.2
print("  checked by hand:", [round(z1, 4), round(z2, 4)])

# ---------------------------------------------------------------------------
# 2. The exact same operation, but as a real PyTorch layer.
#    nn.Linear stores W and b internally and learns them during training.
# ---------------------------------------------------------------------------
layer = torch.nn.Linear(in_features=4, out_features=3)
sample = torch.randn(1, 4)  # one sample with 4 features
out = layer(sample)  # computes sample @ W.T + b

print("\nnn.Linear(4, 3):")
print("  input shape :", tuple(sample.shape))  # (1, 4)
print("  output shape:", tuple(out.shape))  # (1, 3)
print("  W shape     :", tuple(layer.weight.shape))  # (3, 4) -> 12 weights
print("  b shape     :", tuple(layer.bias.shape))  # (3,)   -> 3 biases
print("  parameters  :", layer.weight.numel() + layer.bias.numel())
