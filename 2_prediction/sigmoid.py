"""
Sigmoid — turning a score into a probability for binary classification.

    sigmoid(z) = 1 / (1 + e^-z)

The sigmoid (logistic) function squashes any real number into the open interval
(0, 1), so its output can be read as the model's confidence that the positive
class is correct:

    z -> +inf  =>  sigmoid(z) -> 1   (very confident: positive)
    z = 0      =>  sigmoid(z) = 0.5  (undecided)
    z -> -inf  =>  sigmoid(z) -> 0   (very confident: negative)

It is the output activation of a binary classifier.
"""

import torch

# A range of raw scores (logits) from very negative to very positive.
z = torch.tensor([-4.0, -1.0, 0.0, 1.0, 4.0])

probs = torch.sigmoid(z)
print("logit z      ->", z.tolist())
print("sigmoid(z)   ->", probs.numpy().round(4))

# Verify the closed form 1 / (1 + e^-z) matches torch.sigmoid.
manual = 1.0 / (1.0 + torch.exp(-z))
print("manual 1/(1+e^-z):", manual.numpy().round(4))

# A tiny worked example: a logistic-regression-style decision.
# Score = w . x + b, then sigmoid turns it into P(positive class).
w = torch.tensor([1.5, -2.0])
x = torch.tensor([1.0, 0.3])
b = torch.tensor(0.2)
score = torch.dot(w, x) + b
p_positive = torch.sigmoid(score)
print(f"\nlogistic regression: score = {score.item():.3f} -> P(positive) = {p_positive.item():.4f}")
print("decision:", "positive" if p_positive > 0.5 else "negative")
