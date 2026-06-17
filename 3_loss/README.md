# Theme 3 — Loss

*Measuring how wrong a prediction is.*

The loss function defines the terrain that training navigates. The master
equation frames all of supervised learning as minimizing an average loss;
cross-entropy is the specific loss behind classification and language modeling.

| Script | Formula | What it shows |
|--------|---------|---------------|
| [`master_equation.py`](./master_equation.py) | $\theta^{*} = \arg\min \dfrac{1}{N} \sum L\big(f_\theta(x), y\big)$ | Empirical risk; lower loss ↔ better params |
| [`cross_entropy.py`](./cross_entropy.py) | $H(y, p) = -\sum_i y_i \log p_i$ | The classification / next-token loss |

Run any script from this folder, e.g.:

```bash
python cross_entropy.py
```
