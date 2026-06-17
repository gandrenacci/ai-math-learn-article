# Theme 4 — Probability

*Modeling uncertainty, noise, and beliefs.*

Probability gives AI a language for uncertainty. Bayes' theorem updates beliefs
from evidence; the Gaussian models noise and priors; expectation is the averaging
operator behind risk; entropy and KL divergence quantify uncertainty and the
mismatch between distributions.

| Script | Formula | What it shows |
|--------|---------|---------------|
| [`bayes_theorem.py`](./bayes_theorem.py) | $P(H \mid E) = \dfrac{P(E \mid H)\, P(H)}{P(E)}$ | Updating beliefs (rare-disease test) |
| [`gaussian_distribution.py`](./gaussian_distribution.py) | $f(x) = \dfrac{1}{\sigma \sqrt{2\pi}}\, e^{-\frac{(x - \mu)^2}{2\sigma^2}}$ | Noise/priors + 68-95-99.7 (saves a plot) |
| [`expectation.py`](./expectation.py) | $\mathbb{E}[X] = \sum x\, P(x)$ | The average operator behind risk |
| [`shannon_entropy.py`](./shannon_entropy.py) | $H(X) = -\sum p \log p$ | Uncertainty + $H(P,Q) = H(P) + D_{\mathrm{KL}}$ |
| [`kl_divergence.py`](./kl_divergence.py) | $D_{\mathrm{KL}}(P \parallel Q) = \sum P \log \dfrac{P}{Q}$ | Distributional mismatch (asymmetric) |

Run any script from this folder, e.g.:

```bash
python bayes_theorem.py
```
