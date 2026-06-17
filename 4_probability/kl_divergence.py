"""
Kullback-Leibler (KL) divergence — distance between distributions.

    D_KL(P || Q) = sum_i  P(i) log( P(i) / Q(i) )

KL divergence measures the extra information lost when distribution Q is used to
approximate the true distribution P. Key properties:
    - always >= 0
    - equals 0 exactly when P == Q
    - NOT symmetric:  D_KL(P || Q) != D_KL(Q || P)   (hence "divergence", not distance)

It is the mismatch measure behind variational autoencoders, knowledge
distillation and RLHF — anywhere a model is trained to match a target
distribution.
"""

import torch
import torch.nn.functional as F

P = torch.tensor([0.5, 0.3, 0.2])  # true distribution
Q = torch.tensor([0.4, 0.4, 0.2])  # approximation

# F.kl_div expects log-probabilities for the FIRST argument (the approximation Q).
kl_pq = F.kl_div(Q.log(), P, reduction="sum")
print(f"D_KL(P || Q) = {kl_pq.item():.4f}")

# Verify against the explicit sum P(i) log(P(i)/Q(i)).
manual = torch.sum(P * torch.log(P / Q))
print(f"manual sum P log(P/Q) = {manual.item():.4f}")

# Asymmetry: swapping P and Q gives a different value.
kl_qp = F.kl_div(P.log(), Q, reduction="sum")
print(f"\nD_KL(Q || P) = {kl_qp.item():.4f}  -> not equal to D_KL(P || Q): asymmetric")

# Identical distributions -> zero divergence.
print(f"D_KL(P || P) = {F.kl_div(P.log(), P, reduction='sum').item():.4f}  (zero)")
