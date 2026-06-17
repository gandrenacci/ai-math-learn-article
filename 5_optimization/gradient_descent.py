"""
Gradient descent — the navigation strategy of training.

    theta <- theta - eta * grad_theta L

At each step: compute the gradient of the loss (the direction of steepest
*ascent*), then move the parameters a small step eta (the learning rate) in the
*opposite* direction, lowering the loss. Repeat until you reach a valley.

This script minimizes a 1D quadratic loss L(w) = (w - 3)^2, whose gradient is
2(w - 3), starting far from the minimum, and saves a plot of the descent.
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def loss(w):
    return (w - 3.0) ** 2


def grad(w):
    return 2.0 * (w - 3.0)  # analytic derivative of (w-3)^2


w = -2.0  # start far from the minimum at w = 3
eta = 0.15  # learning rate (step size)
history = [(w, loss(w))]

for _ in range(30):
    w = w - eta * grad(w)  # the update rule
    history.append((w, loss(w)))

ws, ls = zip(*history)
print(f"Start w = {ws[0]:.2f}, loss = {ls[0]:.2f}")
print(f"Final w = {ws[-1]:.4f}, loss = {ls[-1]:.4f}  (target w = 3.0)")

# The learning rate matters: too large diverges, too small crawls.
print("\nEffect of the learning rate (final w after 30 steps from w=-2):")
for test_eta in (0.01, 0.15, 0.9, 1.05):
    wt = -2.0
    for _ in range(30):
        wt = wt - test_eta * grad(wt)
    note = "diverges!" if abs(wt) > 1e3 else ""
    print(f"  eta = {test_eta:<5} -> w = {wt:10.4f}  {note}")

# Plot the loss curve with the descent steps.
w_range = np.linspace(-3, 6, 200)
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(w_range, loss(w_range), "b-", linewidth=2, label="L(w) = (w-3)^2")
ax.scatter(ws, ls, c="orange", zorder=5, s=30, label="gradient-descent steps")
ax.annotate("start", (ws[0], ls[0]), textcoords="offset points", xytext=(10, 10))
ax.annotate("min", (ws[-1], ls[-1]), textcoords="offset points", xytext=(10, 10))
ax.set_xlabel("w")
ax.set_ylabel("loss")
ax.set_title("Gradient descent converging to the minimum")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()

out_path = os.path.join(os.path.dirname(__file__), "..", "img", "gradient_descent.png")
plt.savefig(out_path, dpi=150)
print("\nSaved plot ->", os.path.normpath(out_path))
