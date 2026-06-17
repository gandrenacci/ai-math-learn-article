"""
Why nonlinearities matter — affine composition collapses.

Two affine layers in a row, with NO activation between them:

    z1 = W1 x + b1
    z2 = W2 z1 + b2 = W2(W1 x + b1) + b2 = (W2 W1) x + (W2 b1 + b2)
       = W' x + b'

So two stacked affine layers are mathematically identical to a *single* affine
layer with W' = W2 W1 and b' = W2 b1 + b2. No matter how many you stack, without
a nonlinearity the whole thing is still just one linear map.

Insert a nonlinearity (here ReLU) and that collapse no longer happens — the
network can represent functions a single affine layer never could.

    affine layers + nonlinearities  =>  expressive deep networks
"""

import torch

torch.manual_seed(0)

x = torch.randn(5, 3)  # batch of 5 samples, 3 features

W1, b1 = torch.randn(4, 3), torch.randn(4)
W2, b2 = torch.randn(2, 4), torch.randn(2)

# ---------------------------------------------------------------------------
# 1. No activation: two layers vs the single collapsed layer must match.
# ---------------------------------------------------------------------------
two_layers = (x @ W1.T + b1) @ W2.T + b2

W_prime = W2 @ W1  # (2, 3)
b_prime = W2 @ b1 + b2  # (2,)
one_layer = x @ W_prime.T + b_prime

print("Without activation, stacked affine == single affine layer:")
print("  max abs difference:", (two_layers - one_layer).abs().max().item())
print("  -> effectively zero, the two layers collapsed into one")

# ---------------------------------------------------------------------------
# 2. With a ReLU between them, no single affine layer can reproduce the output.
# ---------------------------------------------------------------------------
nonlinear = torch.relu(x @ W1.T + b1) @ W2.T + b2
print("\nWith ReLU between the layers:")
print("  difference vs collapsed affine:", (nonlinear - one_layer).abs().max().item())
print("  -> large: the nonlinearity broke the collapse, adding real expressive power")
