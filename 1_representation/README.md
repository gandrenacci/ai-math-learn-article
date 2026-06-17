# Theme 1 — Representation

*Turning raw data into vectors and matrices, and measuring its geometry.*

Modern AI starts by representing data as numbers and combining those numbers
with learned weights. Every layer is built on the affine transform, scaled up to
batches via matrix multiplication. Two more tools describe the *geometry* of
these representations: cosine similarity (angles between embeddings) and
eigenvalues (directions of greatest variance, used in PCA).

| Script | Formula | What it shows |
|--------|---------|---------------|
| [`affine_transformation.py`](./affine_transformation.py) | $z = Wx + b$ | The core layer operation, by hand and as `nn.Linear` |
| [`matrix_multiplication.py`](./matrix_multiplication.py) | $Z = X W^{\top} + b$ | The same op over a batch — what GPUs accelerate |
| [`cosine_similarity.py`](./cosine_similarity.py) | $\cos\theta = \dfrac{u \cdot v}{\lVert u \rVert\, \lVert v \rVert}$ | Directional similarity of embeddings |
| [`eigenvalues_pca.py`](./eigenvalues_pca.py) | $A v = \lambda v$ | Eigenvectors + PCA (saves a plot to `../img/`) |

Run any script from this folder, e.g.:

```bash
python affine_transformation.py
```
