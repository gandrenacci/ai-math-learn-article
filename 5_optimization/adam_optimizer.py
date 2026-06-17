"""
Adam — the default optimizer.

Adam (Adaptive Moment Estimation) improves plain gradient descent by tracking two
running averages of the gradient g_t for each parameter:

    m_t = beta1 * m_{t-1} + (1 - beta1) * g_t        first moment  (mean)
    v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2      second moment (variance)

    m_hat = m_t / (1 - beta1^t)     bias correction
    v_hat = v_t / (1 - beta2^t)

    theta <- theta - eta * m_hat / (sqrt(v_hat) + eps)

The mean term acts like momentum; the variance term adapts the step size per
parameter. This makes training faster, more stable and less sensitive to the
learning rate, which is why Adam is the default for most deep models.

Here we minimize the same L(w) = (w - 3)^2 with a hand-written Adam and confirm
it matches torch.optim.Adam.
"""

import torch

target = 3.0


def grad(w):
    return 2.0 * (w - target)


# ---------------------------------------------------------------------------
# 1. Adam implemented by hand, following the formulas above.
# ---------------------------------------------------------------------------
w = -2.0
m = v = 0.0
beta1, beta2, eta, eps = 0.9, 0.999, 0.1, 1e-8

for t in range(1, 201):
    g = grad(w)
    m = beta1 * m + (1 - beta1) * g
    v = beta2 * v + (1 - beta2) * g * g
    m_hat = m / (1 - beta1**t)
    v_hat = v / (1 - beta2**t)
    w = w - eta * m_hat / (v_hat**0.5 + eps)

print(f"Hand-written Adam:   w = {w:.4f}  (target {target})")

# ---------------------------------------------------------------------------
# 2. The same optimization with PyTorch's built-in Adam.
# ---------------------------------------------------------------------------
wt = torch.tensor(-2.0, requires_grad=True)
opt = torch.optim.Adam([wt], lr=0.1)
for _ in range(200):
    opt.zero_grad()
    loss = (wt - target) ** 2
    loss.backward()
    opt.step()

print(f"torch.optim.Adam:    w = {wt.item():.4f}")
print("Both converge to the minimum, adapting the step size as they go.")
