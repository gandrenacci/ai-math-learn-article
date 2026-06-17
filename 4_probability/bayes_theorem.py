"""
Bayes' theorem — updating beliefs from evidence.

    P(H | E) = P(E | H) P(H) / P(E)

    P(H)      prior     — belief in hypothesis H before seeing evidence
    P(E | H)  likelihood — how probable the evidence is if H is true
    P(H | E)  posterior — updated belief after seeing evidence E
    P(E)      marginal  — total probability of the evidence

It is the formal rule for learning from data: combine what you believed with how
well the hypothesis explains the new evidence. This is the conceptual backbone of
spam filters, medical diagnosis and Bayesian inference.

Worked example: a medical test for a rare disease.
"""


def bayes(prior, likelihood, false_positive):
    """P(disease | positive test) from prior, true-positive and false-positive rates."""
    # P(E) = P(E|H)P(H) + P(E|~H)P(~H)   -- the marginal probability of a positive test.
    p_evidence = likelihood * prior + false_positive * (1 - prior)
    return likelihood * prior / p_evidence


prior = 0.01  # 1% of people have the disease
sensitivity = 0.99  # P(positive | disease) — test catches 99% of real cases
false_positive = 0.05  # P(positive | healthy) — 5% of healthy people test positive

posterior = bayes(prior, sensitivity, false_positive)

print("Rare-disease test, Bayes' theorem:")
print(f"  prior P(disease)          = {prior:.2%}")
print(f"  sensitivity P(+ | disease) = {sensitivity:.2%}")
print(f"  false positive P(+ | well) = {false_positive:.2%}")
print(f"  -> posterior P(disease | +) = {posterior:.2%}")
print("\nEven after a positive test the probability is far below 100%,")
print("because the disease is rare — the prior dominates. That is Bayes at work.")

# A second positive test uses the first posterior as the new prior.
posterior_2 = bayes(posterior, sensitivity, false_positive)
print(f"\nAfter a SECOND independent positive test: P(disease) = {posterior_2:.2%}")
