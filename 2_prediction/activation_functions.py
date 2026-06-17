"""
Activation functions — adding nonlinearity.

    ReLU(x)    = max(0, x)
    sigmoid(z) = 1 / (1 + e^-z)        squashes to (0, 1)
    tanh(x)    = (e^x - e^-x)/(e^x + e^-x)   squashes to (-1, 1)

A stack of affine layers with nothing between them collapses to a single affine
layer (see affine_collapse.py). Inserting a nonlinear activation between layers
is what lets a deep network bend space and learn curved decision boundaries.

This script evaluates the three classic activations and saves a comparison plot.
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

x = torch.linspace(-5, 5, 200)

relu = torch.relu(x)
sigmoid = torch.sigmoid(x)
tanh = torch.tanh(x)

# Print a few sample values so the behaviour is visible in the console.
for name, fn in [("ReLU", torch.relu), ("sigmoid", torch.sigmoid), ("tanh", torch.tanh)]:
    pts = torch.tensor([-2.0, 0.0, 2.0])
    print(f"{name:8s} at [-2, 0, 2] -> {fn(pts).numpy().round(3)}")

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(x, relu, label="ReLU  max(0, x)", linewidth=2)
ax.plot(x, sigmoid, label="sigmoid  (0, 1)", linewidth=2)
ax.plot(x, tanh, label="tanh  (-1, 1)", linewidth=2)
ax.axhline(0, color="gray", linewidth=0.5)
ax.axvline(0, color="gray", linewidth=0.5)
ax.set_title("Common activation functions")
ax.set_xlabel("x")
ax.set_ylabel("activation(x)")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()

out_path = os.path.join(os.path.dirname(__file__), "..", "img", "activation_functions.png")
plt.savefig(out_path, dpi=150)
print("\nSaved plot ->", os.path.normpath(out_path))
