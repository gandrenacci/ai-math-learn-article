# Theme 2 — Prediction

*Mapping representations to outputs, and why depth needs nonlinearity.*

A model turns inputs into predictions by composing layers. Affine layers alone
collapse into a single linear map, so nonlinear activations are what give deep
networks their power. At the output, sigmoid and softmax convert raw scores
(logits) into probabilities.

| Script | Formula | What it shows |
|--------|---------|---------------|
| [`deep_network_composition.py`](./deep_network_composition.py) | $f(x) = f_L(\dots f_1(x))$ | A network as composed layer functions |
| [`affine_collapse.py`](./affine_collapse.py) | $W_2(W_1 x + b_1) + b_2 = W' x + b'$ | Why a nonlinearity is essential |
| [`activation_functions.py`](./activation_functions.py) | $\mathrm{ReLU},\ \sigma,\ \tanh$ | The three classic activations (saves a plot) |
| [`sigmoid.py`](./sigmoid.py) | $\sigma(z) = \dfrac{1}{1 + e^{-z}}$ | Score → probability for binary classification |
| [`softmax.py`](./softmax.py) | $p_i = \dfrac{e^{z_i}}{\sum_j e^{z_j}}$ | Multi-class probabilities + temperature scaling |

Run any script from this folder, e.g.:

```bash
python softmax.py
```
