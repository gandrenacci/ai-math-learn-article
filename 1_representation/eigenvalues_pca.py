"""
Eigenvalues and eigenvectors — the geometry of compression (PCA).

    A v = lambda v,   v != 0

A transforms the vector v into a scaled copy of itself: same direction, only
stretched (|lambda| > 1) or compressed (|lambda| < 1) by the eigenvalue lambda.
Such special directions are the eigenvectors.

In Principal Component Analysis (PCA) we take the eigenvectors of the data's
covariance matrix. The eigenvector with the largest eigenvalue points along the
direction of greatest variance. Keeping only the top directions lets us reduce
dimensionality while preserving most of the information — the "geometry of
compression."

This script generates correlated 2D data, finds its principal directions, and
saves a plot of the data with its eigenvectors overlaid.
"""

import os

import matplotlib

matplotlib.use("Agg")  # headless backend: render to file, no display needed
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# 1. A tiny, exact eigenvector check: A v = lambda v.
# ---------------------------------------------------------------------------
A = np.array([[2.0, 0.0], [0.0, 3.0]])
vals, vecs = np.linalg.eig(A)
print("Diagonal matrix A eigenvalues:", vals)  # [2, 3]
for i in range(2):
    Av = A @ vecs[:, i]
    lam_v = vals[i] * vecs[:, i]
    print(f"  A v = {Av.round(3)}   lambda v = {lam_v.round(3)}  (equal)")

# ---------------------------------------------------------------------------
# 2. PCA on synthetic correlated data.
# ---------------------------------------------------------------------------
rng = np.random.default_rng(0)
data = rng.multivariate_normal(mean=[0, 0], cov=[[3, 2], [2, 2]], size=200)

cov = np.cov(data.T)  # 2x2 covariance matrix of the features
eigenvalues, eigenvectors = np.linalg.eig(cov)

order = np.argsort(eigenvalues)[::-1]  # largest variance first
eigenvalues = eigenvalues[order]
eigenvectors = eigenvectors[:, order]

print("\nPCA on correlated 2D data:")
print("  eigenvalues (variance explained):", eigenvalues.round(2))
print("  1st principal component:", eigenvectors[:, 0].round(3))
explained = eigenvalues / eigenvalues.sum()
print("  variance ratio:", explained.round(3))

# ---------------------------------------------------------------------------
# 3. Plot the cloud with the principal directions scaled by sqrt(eigenvalue).
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(data[:, 0], data[:, 1], s=12, alpha=0.4, label="data")
center = data.mean(axis=0)
colors = ["crimson", "darkorange"]
for i in range(2):
    vec = eigenvectors[:, i] * np.sqrt(eigenvalues[i]) * 2
    ax.annotate(
        "",
        xy=center + vec,
        xytext=center,
        arrowprops=dict(color=colors[i], width=2, headwidth=10),
    )
    ax.text(*(center + vec), f"PC{i + 1}", color=colors[i], fontsize=12)

ax.set_aspect("equal")
ax.set_title("PCA: principal directions are eigenvectors of the covariance")
ax.set_xlabel("feature 1")
ax.set_ylabel("feature 2")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()

out_path = os.path.join(os.path.dirname(__file__), "..", "img", "eigenvalues_pca.png")
plt.savefig(out_path, dpi=150)
print("\nSaved plot ->", os.path.normpath(out_path))
