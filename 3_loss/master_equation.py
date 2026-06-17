"""
The master equation of learning — empirical risk minimization.

    theta* = argmin_theta  (1/N) sum_i  L( f_theta(x_i), y_i )

Read it as: find the parameters theta that make the model f_theta least wrong on
our data, where "wrong" is measured by a loss function L. This single template
underlies essentially all of supervised learning; the choices that vary are the
model f_theta, the loss L, and the optimizer that performs the argmin.

Here f_theta is linear regression and L is mean squared error (MSE). We don't
train yet — we just show that better parameters yield a lower loss, which is
exactly the quantity training drives down.
"""

import numpy as np


# f_theta: a line. theta = (w, b).
def f_theta(x, w, b):
    return w * x + b


# L: mean squared error between predictions and targets.
def mse_loss(y_pred, y_true):
    return np.mean((y_pred - y_true) ** 2)


# Synthetic data generated from a known line y = 3x + 1.5 plus noise.
rng = np.random.default_rng(42)
x = rng.uniform(0, 10, 50)
y_true = 3.0 * x + 1.5 + rng.normal(0, 2, 50)

# Empirical risk for a few candidate parameter settings.
candidates = [(2.0, 0.0), (3.0, 1.5), (3.0, 1.0), (0.0, 0.0)]
print("Empirical risk  (1/N) sum L(f_theta(x), y)  for several theta = (w, b):")
for w, b in candidates:
    loss = mse_loss(f_theta(x, w, b), y_true)
    print(f"  w={w:<4} b={b:<4} -> loss = {loss:7.3f}")

# The true generating parameters give the lowest loss — the valley training seeks.
best = min(candidates, key=lambda wb: mse_loss(f_theta(x, *wb), y_true))
print(f"\nLowest-loss candidate: w={best[0]}, b={best[1]} (closest to the true 3.0, 1.5)")
print("Training = searching parameter space for this minimum (see ../5_optimization/).")
