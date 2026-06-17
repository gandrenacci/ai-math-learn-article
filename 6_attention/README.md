# Theme 6 — Attention

*Letting a model focus on the most relevant parts of the input.*

Scaled dot-product attention is the defining equation of modern AI and the engine
of the Transformer. Each token compares itself to every other token, turns those
comparisons into a probability distribution with softmax, and uses it to mix the
values — capturing long-range relationships in language, vision and beyond.

| Script | Formula | What it shows |
|--------|---------|---------------|
| [`scaled_dot_product_attention.py`](./scaled_dot_product_attention.py) | `softmax(QKᵀ/√d_k)V` | Attention from scratch + an interpretable example |

Run it from this folder:

```bash
python scaled_dot_product_attention.py
```
