"""
L2 regularization (weight decay) — keeping models honest.

    L_total = L_original + lambda * sum_i theta_i^2

Adding a penalty proportional to the squared magnitude of the weights pushes
parameters toward zero unless the data strongly justifies a large value. This
discourages overfitting and encodes a prior that simpler solutions generalize
better (equivalently, a zero-centered Gaussian prior on the weights).

lambda controls the strength: 0 = no regularization, larger = simpler model.
This script shows the penalty term explicitly and via the optimizer's
weight_decay argument.
"""

import torch
import torch.nn as nn

torch.manual_seed(0)

model = nn.Linear(10, 1)
x = torch.randn(32, 10)
y = torch.randn(32, 1)
mse = nn.MSELoss()

# ---------------------------------------------------------------------------
# 1. The penalty written out explicitly and added to the task loss.
# ---------------------------------------------------------------------------
pred = model(x)
data_loss = mse(pred, y)
l2_penalty = sum(p.pow(2).sum() for p in model.parameters())  # sum of theta_i^2

lam = 0.01
total_loss = data_loss + lam * l2_penalty
print("Explicit L2 penalty:")
print(f"  data loss (MSE) = {data_loss.item():.4f}")
print(f"  L2 penalty sum(theta^2) = {l2_penalty.item():.4f}")
print(f"  total = data + {lam} * penalty = {total_loss.item():.4f}")

# ---------------------------------------------------------------------------
# 2. The same idea baked into the optimizer via weight_decay (= lambda).
#    AdamW applies the decay directly in the update step.
# ---------------------------------------------------------------------------
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=lam)
print(f"\nAdamW(weight_decay={lam}) applies the same shrinkage during each step.")

# Larger lambda shrinks the weights more — watch the weight norm fall.
print("\nWeight norm after 50 steps for different lambda (weight_decay):")
for wd in (0.0, 0.01, 0.1):
    m = nn.Linear(10, 1)
    opt = torch.optim.AdamW(m.parameters(), lr=0.05, weight_decay=wd)
    for _ in range(50):
        opt.zero_grad()
        loss = mse(m(x), y)
        loss.backward()
        opt.step()
    norm = sum(p.pow(2).sum() for p in m.parameters()).sqrt().item()
    print(f"  lambda = {wd:<5} -> ||theta|| = {norm:.4f}")
