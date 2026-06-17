"""
Matrix multiplication — the workhorse of deep learning at scale.

Single sample:   z = W x + b
Batch of n:      Z = X W^T + b        (bias broadcast across all rows)

    X in R^(n x d)   n samples, each with d features
    W in R^(m x d)   weight matrix
    b in R^m         bias vector
    Z in R^(n x m)   one output row per input sample

Element-wise:    z_ij = sum_k X_ik * W_jk + b_j

Neural networks almost never process a single vector: they push a whole *batch*
of samples through the same weights at once. That turns the layer into one big
matrix multiplication, which is exactly what GPUs are built to accelerate.
"""

import torch

# A batch of 2 samples, each with d = 3 features.
X = torch.tensor(
    [
        [1.0, 2.0, 3.0],  # sample 1
        [4.0, 5.0, 6.0],  # sample 2
    ]
)  # shape (n=2, d=3)

W = torch.tensor(
    [
        [0.5, -0.2, 0.1],  # output neuron 1
        [0.3, 0.8, -0.5],  # output neuron 2
    ]
)  # shape (m=2, d=3)

b = torch.tensor([0.1, -0.2])  # shape (m=2,)

# Z = X W^T + b. W.T turns (m, d) into (d, m) so the inner dimensions match;
# the bias is broadcast (added to every row).
Z = X @ W.T + b
print("Batch output Z = X W^T + b:")
print(Z)
print("  shape:", tuple(Z.shape), "-> one row of m=2 outputs per sample")

# Verify a single element with the element-wise formula z_ij = sum_k X_ik W_jk + b_j
z_00 = sum(X[0, k] * W[0, k] for k in range(3)) + b[0]
print("  z[0,0] by hand:", round(z_00.item(), 4), "== ", round(Z[0, 0].item(), 4))

# This is precisely what nn.Linear does internally for a batch.
layer = torch.nn.Linear(3, 2)
print("\nnn.Linear(3, 2) on the same batch produces shape:", tuple(layer(X).shape))
