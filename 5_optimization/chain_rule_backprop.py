"""
The chain rule — why backpropagation works.

A network is a composition of functions f(x) = f_n(...f_1(x)). The chain rule
from calculus says the derivative of a composition is the product of the
derivatives along the chain:

    d/dx f(g(x)) = f'(g(x)) * g'(x)

Backpropagation is exactly this rule applied layer by layer, propagating
gradients from the output back to every parameter. Automatic-differentiation
engines (PyTorch autograd) build a computational graph and apply it for us.

This script verifies autograd against the hand-derived analytic gradient.
"""

import torch

# ---------------------------------------------------------------------------
# 1. Scalar composition: y = sin(x^2). Chain rule: dy/dx = cos(x^2) * 2x.
# ---------------------------------------------------------------------------
x = torch.tensor(2.0, requires_grad=True)
y = torch.sin(x**2)  # f(g(x)) with g(x)=x^2, f(z)=sin(z)
y.backward()  # autograd applies the chain rule backward

analytic = torch.cos(x**2) * 2 * x
print("y = sin(x^2) at x = 2:")
print(f"  autograd gradient: {x.grad.item():.6f}")
print(f"  analytic cos(x^2)*2x: {analytic.item():.6f}  (matches)")

# ---------------------------------------------------------------------------
# 2. A 2-layer chain so the "propagate backward" structure is visible.
#    L = (w2 * relu(w1 * x))^2  -- gradients flow L -> w2 -> w1.
# ---------------------------------------------------------------------------
xv = torch.tensor(1.5)
w1 = torch.tensor(0.8, requires_grad=True)
w2 = torch.tensor(-0.5, requires_grad=True)

h = torch.relu(w1 * xv)  # hidden activation
out = (w2 * h) ** 2  # scalar loss
out.backward()  # backprop computes dL/dw1 and dL/dw2

print("\nTwo-layer chain L = (w2 * relu(w1 * x))^2:")
print(f"  dL/dw1 (autograd) = {w1.grad.item():.6f}")
print(f"  dL/dw2 (autograd) = {w2.grad.item():.6f}")
# Analytic check for dL/dw2 = 2 * (w2 * h) * h (h is constant w.r.t. w2 here).
print(f"  dL/dw2 (analytic) = {(2 * (w2 * h) * h).item():.6f}")
