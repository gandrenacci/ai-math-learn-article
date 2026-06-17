"""
Shannon entropy — measuring uncertainty.

    H(X) = - sum_x  p(x) log p(x)

Entropy quantifies how unpredictable a distribution is:
    maximum entropy  -> all outcomes equally likely (most uncertain)
    zero entropy     -> one outcome is certain

It connects to cross-entropy and KL divergence through the identity:

    H(P, Q) = H(P) + D_KL(P || Q)

i.e. cross-entropy = the data's intrinsic uncertainty PLUS the extra cost of
using the wrong distribution Q. With P fixed, minimizing cross-entropy is the
same as minimizing KL divergence — which is exactly what training does.
"""

import numpy as np


def entropy(p, base=2):
    p = np.asarray(p, dtype=float)
    p = p[p > 0]  # 0 log 0 := 0
    return -np.sum(p * (np.log(p) / np.log(base)))


# Entropy is maximal for a uniform distribution, zero for a certain one.
print("Shannon entropy H(X) in bits:")
print(f"  fair coin   [0.5, 0.5]      -> {entropy([0.5, 0.5]):.3f}  (max for 2 outcomes)")
print(f"  biased coin [0.9, 0.1]      -> {entropy([0.9, 0.1]):.3f}")
print(f"  certain     [1.0, 0.0]      -> {entropy([1.0, 0.0]):.3f}  (no uncertainty)")
print(f"  fair die    uniform over 6  -> {entropy([1 / 6] * 6):.3f}")

# Verify H(P, Q) = H(P) + D_KL(P || Q).
P = np.array([0.5, 0.3, 0.2])
Q = np.array([0.4, 0.4, 0.2])


def cross_entropy(p, q, base=2):
    return -np.sum(p * (np.log(q) / np.log(base)))


def kl(p, q, base=2):
    return np.sum(p * (np.log(p / q) / np.log(base)))


H_P = entropy(P)
H_PQ = cross_entropy(P, Q)
D = kl(P, Q)
print("\nIdentity  H(P, Q) = H(P) + D_KL(P || Q):")
print(f"  H(P)         = {H_P:.4f}")
print(f"  D_KL(P || Q) = {D:.4f}")
print(f"  H(P) + D_KL  = {H_P + D:.4f}")
print(f"  H(P, Q)      = {H_PQ:.4f}  (matches)")
