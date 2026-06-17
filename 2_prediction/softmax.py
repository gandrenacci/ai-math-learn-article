"""
Softmax — turning a vector of scores into a probability distribution.

    p_i = e^{z_i} / sum_j e^{z_j}

Softmax generalizes the sigmoid to k classes. Every output is non-negative and
they all sum to 1, so the result is a proper probability distribution over the
classes. It is the output layer of multi-class classifiers and a core piece of
Transformer attention.

Temperature scaling controls how "sharp" the distribution is:

    p_i(T) = e^{z_i / T} / sum_j e^{z_j / T},   T > 0

    T < 1  -> sharper / more deterministic (the top class dominates)
    T > 1  -> flatter / more random
    T = 1  -> standard softmax

This is the knob behind "creativity" when sampling from a language model.
"""

import torch
import torch.nn.functional as F

logits = torch.tensor([2.0, 1.0, 0.1])  # raw scores for 3 classes

probs = F.softmax(logits, dim=0)
print("logits      :", logits.tolist())
print("softmax     :", probs.numpy().round(3), " (sums to", round(probs.sum().item(), 3), ")")
# Expected ~ [0.659, 0.242, 0.099] -- "winner takes most".

# Binary case is just sigmoid on a single score.
print("sigmoid(1.5):", round(torch.sigmoid(torch.tensor(1.5)).item(), 4))

# Temperature scaling: same logits, three different sharpness levels.
print("\nTemperature scaling on the same logits:")
for T in (0.5, 1.0, 2.0):
    p_T = F.softmax(logits / T, dim=0)
    label = "sharper" if T < 1 else ("flatter" if T > 1 else "standard")
    print(f"  T = {T:<4} -> {p_T.numpy().round(3)}  ({label})")
