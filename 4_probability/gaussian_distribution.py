"""
The Gaussian (normal) distribution — noise and priors.

    f(x) = 1 / (sigma sqrt(2 pi)) * exp( -(x - mu)^2 / (2 sigma^2) )

It is defined by a mean mu (the center) and a variance sigma^2 (the spread).
Because many random effects add up to an approximately bell-shaped result, the
Gaussian is the default model for measurement noise. It also appears as a prior
on model parameters (a zero-centered Gaussian prior favors small weights, which
is the probabilistic view of L2 regularization).

This script evaluates the density, samples from it, checks the 68-95-99.7 rule,
and saves a plot of three Gaussians.
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def gaussian_pdf(x, mu, sigma):
    return 1.0 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-((x - mu) ** 2) / (2 * sigma**2))


# Sample data and verify the empirical mean/std match the parameters.
rng = np.random.default_rng(0)
mu, sigma = 2.0, 1.5
samples = rng.normal(mu, sigma, size=100_000)
print(f"Gaussian(mu={mu}, sigma={sigma}):")
print(f"  sample mean = {samples.mean():.3f}, sample std = {samples.std():.3f}")

# The 68-95-99.7 rule: fraction of samples within 1, 2, 3 standard deviations.
for k in (1, 2, 3):
    frac = np.mean(np.abs(samples - mu) < k * sigma)
    print(f"  within {k} sigma: {frac:.1%}")

# Plot three Gaussians with different means/variances.
x = np.linspace(-6, 10, 400)
fig, ax = plt.subplots(figsize=(8, 4))
for m, s in [(0.0, 1.0), (2.0, 1.5), (-1.0, 0.5)]:
    ax.plot(x, gaussian_pdf(x, m, s), linewidth=2, label=f"mu={m}, sigma={s}")
ax.set_title("Gaussian probability density functions")
ax.set_xlabel("x")
ax.set_ylabel("density f(x)")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()

out_path = os.path.join(os.path.dirname(__file__), "..", "img", "gaussian_distribution.png")
plt.savefig(out_path, dpi=150)
print("\nSaved plot ->", os.path.normpath(out_path))
