# The Math That Powers Modern AI — Code Examples

Runnable, commented Python examples for every formula in the article
**[The Math That Powers Modern AI](https://medium.com/data-bistrot/the-math-that-powers-modern-ai-9ba654bfde1c)**
on  [AI Bistrot](https://medium.com/data-bistrot).

Each script demonstrates one formula with a small, realistic example. The
docstring states the formula in plain text, inline comments tie the code to the
math, and the script prints its results. A few visual formulas also save a plot
to [`img/`](./img).

The examples are grouped into the article's **six core ideas**:

| # | Theme | Idea | Folder |
|---|-------|------|--------|
| 1 | Representation | Data as vectors and matrices | [`1_representation/`](./1_representation) |
| 2 | Prediction | Mapping representations to outputs | [`2_prediction/`](./2_prediction) |
| 3 | Loss | Measuring how wrong a prediction is | [`3_loss/`](./3_loss) |
| 4 | Probability | Modeling uncertainty and noise | [`4_probability/`](./4_probability) |
| 5 | Optimization | Adjusting parameters to reduce error | [`5_optimization/`](./5_optimization) |
| 6 | Attention | Focusing on what matters | [`6_attention/`](./6_attention) |

## Formula index

| Formula | Script |
|---------|--------|
| $z = Wx + b$ | [`1_representation/affine_transformation.py`](./1_representation/affine_transformation.py) |
| $Z = X W^{\top} + b$ | [`1_representation/matrix_multiplication.py`](./1_representation/matrix_multiplication.py) |
| $\cos\theta = \dfrac{u \cdot v}{\lVert u \rVert\, \lVert v \rVert}$ | [`1_representation/cosine_similarity.py`](./1_representation/cosine_similarity.py) |
| $A v = \lambda v$ (PCA) | [`1_representation/eigenvalues_pca.py`](./1_representation/eigenvalues_pca.py) |
| $f(x) = f_L(\dots f_1(x))$ | [`2_prediction/deep_network_composition.py`](./2_prediction/deep_network_composition.py) |
| affine collapse | [`2_prediction/affine_collapse.py`](./2_prediction/affine_collapse.py) |
| $\mathrm{ReLU},\ \sigma,\ \tanh$ | [`2_prediction/activation_functions.py`](./2_prediction/activation_functions.py) |
| $\sigma(z) = \dfrac{1}{1 + e^{-z}}$ | [`2_prediction/sigmoid.py`](./2_prediction/sigmoid.py) |
| $p_i = \dfrac{e^{z_i}}{\sum_j e^{z_j}}$ (+ temperature) | [`2_prediction/softmax.py`](./2_prediction/softmax.py) |
| $\theta^{*} = \arg\min \dfrac{1}{N} \sum L\big(f_\theta(x), y\big)$ | [`3_loss/master_equation.py`](./3_loss/master_equation.py) |
| $H(y, p) = -\sum_i y_i \log p_i$ | [`3_loss/cross_entropy.py`](./3_loss/cross_entropy.py) |
| $P(H \mid E) = \dfrac{P(E \mid H)\, P(H)}{P(E)}$ | [`4_probability/bayes_theorem.py`](./4_probability/bayes_theorem.py) |
| $f(x) = \dfrac{1}{\sigma \sqrt{2\pi}}\, e^{-\frac{(x - \mu)^2}{2\sigma^2}}$ | [`4_probability/gaussian_distribution.py`](./4_probability/gaussian_distribution.py) |
| $\mathbb{E}[X] = \sum x\, P(x)$ | [`4_probability/expectation.py`](./4_probability/expectation.py) |
| $H(X) = -\sum p \log p$ | [`4_probability/shannon_entropy.py`](./4_probability/shannon_entropy.py) |
| $D_{\mathrm{KL}}(P \parallel Q) = \sum P \log \dfrac{P}{Q}$ | [`4_probability/kl_divergence.py`](./4_probability/kl_divergence.py) |
| $\theta \leftarrow \theta - \eta\, \nabla L$ | [`5_optimization/gradient_descent.py`](./5_optimization/gradient_descent.py) |
| chain rule / backprop | [`5_optimization/chain_rule_backprop.py`](./5_optimization/chain_rule_backprop.py) |
| Adam update | [`5_optimization/adam_optimizer.py`](./5_optimization/adam_optimizer.py) |
| $L_{\text{total}} = L + \lambda \sum \theta^2$ | [`5_optimization/l2_regularization.py`](./5_optimization/l2_regularization.py) |
| $\mathrm{softmax}\!\left(\dfrac{Q K^{\top}}{\sqrt{d_k}}\right) V$ | [`6_attention/scaled_dot_product_attention.py`](./6_attention/scaled_dot_product_attention.py) |

## Prerequisites

- Python 3.10+

## Setup (one shared virtual environment)

A single `.venv` runs every example. From this folder:

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Running the examples

Each script is self-contained. Run any one directly:

```bash
python 1_representation/affine_transformation.py
python 6_attention/scaled_dot_product_attention.py
```

Run every example at once:

```bash
find . -name '*.py' | sort | xargs -n1 python
```

Scripts that produce figures save them under [`img/`](./img):
`eigenvalues_pca.png`, `activation_functions.png`, `gaussian_distribution.png`,
`gradient_descent.png`.

## License

MIT
