"""
Cross-entropy — the standard loss for classification and language models.

    H(y, p) = - sum_i  y_i log p_i

For a one-hot target whose correct class is c it simplifies to:

    H(y, p) = - log p_c

It measures how far the model's predicted distribution p is from the true label
distribution y. The pipeline is always:
    logits  --softmax-->  probabilities p  --cross-entropy-->  loss

If the correct class gets high probability the loss is small; if it gets low
probability the loss is large. In language models the same loss scores how well
the model predicts the next token.
"""

import torch
import torch.nn.functional as F

# Logits for 2 samples over 4 classes.
logits = torch.tensor(
    [
        [1.2, 0.3, -0.5, 2.1],  # sample 1 -> favors class 3
        [0.1, 2.3, 0.5, -0.2],  # sample 2 -> favors class 1
    ]
)
targets = torch.tensor([3, 1])  # the correct class index for each sample

# F.cross_entropy applies softmax + negative-log-likelihood in one numerically
# stable step (do NOT softmax the logits yourself before calling it).
loss = F.cross_entropy(logits, targets)
print(f"Cross-entropy loss (mean over batch): {loss.item():.4f}")

# Show it really is -log(prob of the correct class), averaged over the batch.
probs = F.softmax(logits, dim=1)
p_correct = probs[range(len(targets)), targets]
print("probability of correct class per sample:", p_correct.detach().numpy().round(4))
manual = (-torch.log(p_correct)).mean()
print(f"manual mean(-log p_correct):           {manual.item():.4f}")

# Confident-and-right is cheap; confident-and-wrong is expensive.
print("\nLoss vs confidence on the true class:")
for p in (0.99, 0.5, 0.1, 0.01):
    print(f"  p_correct = {p:<5} -> -log p = {-torch.log(torch.tensor(p)).item():.3f}")
