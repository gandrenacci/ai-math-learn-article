# The Math That Powers Modern AI — Code Examples

Runnable, well-commented Python examples for every formula in the article
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
| `z = Wx + b` | [`1_representation/affine_transformation.py`](./1_representation/affine_transformation.py) |
| `Z = X Wᵀ + b` | [`1_representation/matrix_multiplication.py`](./1_representation/matrix_multiplication.py) |
| `cos(θ) = (u·v)/(‖u‖‖v‖)` | [`1_representation/cosine_similarity.py`](./1_representation/cosine_similarity.py) |
| `A v = λ v` (PCA) | [`1_representation/eigenvalues_pca.py`](./1_representation/eigenvalues_pca.py) |
| `f(x) = f_L(…f_1(x))` | [`2_prediction/deep_network_composition.py`](./2_prediction/deep_network_composition.py) |
| affine collapse | [`2_prediction/affine_collapse.py`](./2_prediction/affine_collapse.py) |
| `ReLU, σ, tanh` | [`2_prediction/activation_functions.py`](./2_prediction/activation_functions.py) |
| `σ(z) = 1/(1+e^{-z})` | [`2_prediction/sigmoid.py`](./2_prediction/sigmoid.py) |
| `p_i = e^{z_i}/Σ e^{z_j}` (+ temperature) | [`2_prediction/softmax.py`](./2_prediction/softmax.py) |
| `θ* = argmin (1/N) Σ L(f_θ(x),y)` | [`3_loss/master_equation.py`](./3_loss/master_equation.py) |
| `H(y,p) = -Σ y_i log p_i` | [`3_loss/cross_entropy.py`](./3_loss/cross_entropy.py) |
| `P(H\|E) = P(E\|H)P(H)/P(E)` | [`4_probability/bayes_theorem.py`](./4_probability/bayes_theorem.py) |
| `f(x) = e^{-(x-μ)²/2σ²}/(σ√2π)` | [`4_probability/gaussian_distribution.py`](./4_probability/gaussian_distribution.py) |
| `E[X] = Σ x P(x)` | [`4_probability/expectation.py`](./4_probability/expectation.py) |
| `H(X) = -Σ p log p` | [`4_probability/shannon_entropy.py`](./4_probability/shannon_entropy.py) |
| `D_KL(P‖Q) = Σ P log(P/Q)` | [`4_probability/kl_divergence.py`](./4_probability/kl_divergence.py) |
| `θ ← θ - η ∇L` | [`5_optimization/gradient_descent.py`](./5_optimization/gradient_descent.py) |
| chain rule / backprop | [`5_optimization/chain_rule_backprop.py`](./5_optimization/chain_rule_backprop.py) |
| Adam update | [`5_optimization/adam_optimizer.py`](./5_optimization/adam_optimizer.py) |
| `L_total = L + λΣθ²` | [`5_optimization/l2_regularization.py`](./5_optimization/l2_regularization.py) |
| `softmax(QKᵀ/√d_k)V` | [`6_attention/scaled_dot_product_attention.py`](./6_attention/scaled_dot_product_attention.py) |

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
