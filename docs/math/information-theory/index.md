# Part 6: Information Theory

Information theory, founded by Claude Shannon in 1948, answers a fundamental question: how do you quantify information? For machine learning, information theory provides the loss functions we optimize (cross-entropy), the measures of model quality (KL divergence), and the theoretical limits of learning (mutual information). The connection between information theory and probability is deep — entropy is expected surprise, cross-entropy is the loss you pay for using the wrong model, and KL divergence measures how "wrong" your model is.

---

## Chapters

| Chapter | Topic | Key Results |
|---------|-------|-------------|
| 6.1 | [Entropy](entropy.md) | Shannon entropy, differential entropy, maximum entropy principle |
| 6.2 | [Divergence Measures](divergence-measures.md) | KL divergence, mutual information, f-divergences |
| 6.3 | [Cross-Entropy & Loss](cross-entropy-and-loss.md) | Cross-entropy loss, connection to MLE, softmax derivation |
| 6.4 | [Information Geometry](information-geometry.md) | Fisher information metric, natural gradient, statistical manifolds |

## Prerequisites

- [Part 4: Probability](../probability/index.md) — distributions, expectations
- [Part 3: Calculus](../calculus/index.md) — integration, logarithms
