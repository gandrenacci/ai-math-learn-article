# Theme 4 — Probability

*Modeling uncertainty, noise, and beliefs.*

Probability gives AI a language for uncertainty. Bayes' theorem updates beliefs
from evidence; the Gaussian models noise and priors; expectation is the averaging
operator behind risk; entropy and KL divergence quantify uncertainty and the
mismatch between distributions.

| Script | Formula | What it shows |
|--------|---------|---------------|
| [`bayes_theorem.py`](./bayes_theorem.py) | `P(H\|E) = P(E\|H)P(H)/P(E)` | Updating beliefs (rare-disease test) |
| [`gaussian_distribution.py`](./gaussian_distribution.py) | `f(x) = e^{-(x-μ)²/2σ²}/(σ√2π)` | Noise/priors + 68-95-99.7 (saves a plot) |
| [`expectation.py`](./expectation.py) | `E[X] = Σ x P(x)` | The average operator behind risk |
| [`shannon_entropy.py`](./shannon_entropy.py) | `H(X) = -Σ p log p` | Uncertainty + `H(P,Q)=H(P)+D_KL` |
| [`kl_divergence.py`](./kl_divergence.py) | `D_KL(P‖Q) = Σ P log(P/Q)` | Distributional mismatch (asymmetric) |

Run any script from this folder, e.g.:

```bash
python bayes_theorem.py
```
