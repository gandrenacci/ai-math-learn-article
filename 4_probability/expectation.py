"""
Expectation — the average operator behind risk.

    Discrete:   E[X] = sum_x  x * P(X = x)
    Continuous: E[X] = integral  x * f(x) dx

The expectation is the long-run average value of a random variable, weighting
each outcome by its probability. In machine learning, a loss is typically the
*expected* loss over the data distribution:

    R(theta) = E_{(x,y)~D} [ L(f_theta(x), y) ]

so training is minimizing an expectation. Expectation is the operator behind
risk, optimization and decision-making under uncertainty.
"""

import numpy as np

# ---------------------------------------------------------------------------
# 1. Discrete example: a fair vs. a loaded six-sided die.
# ---------------------------------------------------------------------------
outcomes = np.array([1, 2, 3, 4, 5, 6])

fair = np.full(6, 1 / 6)
exp_fair = np.sum(outcomes * fair)
print(f"Fair die   E[X] = {exp_fair:.3f}  (sum of x * P(x))")

loaded = np.array([0.05, 0.05, 0.05, 0.05, 0.30, 0.50])  # biased toward high rolls
exp_loaded = np.sum(outcomes * loaded)
print(f"Loaded die E[X] = {exp_loaded:.3f}")

# The Law of Large Numbers: the sample mean converges to the expectation.
rng = np.random.default_rng(0)
rolls = rng.choice(outcomes, size=1_000_000, p=loaded)
print(f"  empirical mean of 1,000,000 loaded rolls = {rolls.mean():.3f}")

# ---------------------------------------------------------------------------
# 2. Continuous example: E[X] of a Gaussian estimated by Monte Carlo.
# ---------------------------------------------------------------------------
mu = 3.0
samples = rng.normal(mu, 2.0, size=1_000_000)
print(f"\nGaussian(mu={mu}): Monte Carlo estimate of E[X] = {samples.mean():.3f}")

# ---------------------------------------------------------------------------
# 3. Expected loss (risk) — the quantity training minimizes.
# ---------------------------------------------------------------------------
errors = rng.normal(0, 1, size=100_000)  # prediction errors over the data
risk = np.mean(errors**2)  # expected squared error = E[L]
print(f"\nExpected squared-error loss (risk) over the dataset = {risk:.3f}")
