# Theme 5 — Optimization

*The algorithms that adjust parameters to reduce error.*

Once a loss is defined, optimization navigates the parameter landscape toward a
minimum. Gradient descent is the core step; the chain rule (backpropagation) is
how gradients are computed; Adam adapts the step size; L2 regularization keeps
solutions simple.

| Script | Formula | What it shows |
|--------|---------|---------------|
| [`gradient_descent.py`](./gradient_descent.py) | $\theta \leftarrow \theta - \eta\, \nabla L$ | Descent on $(w-3)^2$ + learning-rate effect (saves a plot) |
| [`chain_rule_backprop.py`](./chain_rule_backprop.py) | $\dfrac{d}{dx} f(g(x)) = f'(g(x))\, g'(x)$ | Autograd vs analytic gradients |
| [`adam_optimizer.py`](./adam_optimizer.py) | $\theta \leftarrow \theta - \eta\, \dfrac{\hat{m}}{\sqrt{\hat{v}} + \epsilon}$ | Adam by hand vs `torch.optim.Adam` |
| [`l2_regularization.py`](./l2_regularization.py) | $L_{\text{total}} = L + \lambda \sum \theta^2$ | Weight decay shrinks the weights |

Run any script from this folder, e.g.:

```bash
python gradient_descent.py
```
