"""
Cosine similarity — the geometry of meaning.

    cosine_similarity(u, v) = cos(theta) = (u . v) / (||u|| ||v||)

It measures the *angle* between two vectors, ignoring their length:
    near  1  -> same direction  (semantically similar)
    near  0  -> orthogonal      (unrelated)
    near -1  -> opposite

When text, images or users are turned into embeddings (vectors in a high-
dimensional space), cosine similarity is the standard way to ask "how alike are
these two things?" — the basis of semantic search, recommendation and
clustering.
"""

import torch
import torch.nn.functional as F

# Toy 4-dimensional "embeddings" for two short sentences.
u = torch.tensor([1.0, 0.5, -0.3, 0.8])
v = torch.tensor([0.9, 0.6, -0.1, 0.7])  # points in nearly the same direction

# unsqueeze(0) adds a batch dimension, as F.cosine_similarity expects.
sim = F.cosine_similarity(u.unsqueeze(0), v.unsqueeze(0)).item()
print(f"cosine_similarity(u, v) = {sim:.4f}  (close to 1 -> very similar)")

# Verify the formula by hand: dot product over the product of norms.
manual = torch.dot(u, v) / (u.norm() * v.norm())
print(f"manual (u.v)/(|u||v|) = {manual.item():.4f}")

# Contrast: a vector pointing roughly the opposite way.
w = torch.tensor([-1.0, -0.4, 0.2, -0.9])
print(f"cosine_similarity(u, w) = {F.cosine_similarity(u.unsqueeze(0), w.unsqueeze(0)).item():.4f}  (negative -> opposite)")

# Magnitude does not matter, only direction: scaling v by 10 leaves cosine unchanged.
print(f"cosine_similarity(u, 10*v) = {F.cosine_similarity(u.unsqueeze(0), (10 * v).unsqueeze(0)).item():.4f}")
