# Part 5: Optimization

Training a machine learning model is an optimization problem: find the parameters $\theta^*$ that minimize a loss function $\mathcal{L}(\theta)$. The loss landscape of a neural network has billions of dimensions, saddle points, local minima, and flat regions. Understanding optimization theory — convexity, gradient methods, convergence rates, and constraints — is understanding *how* and *why* training works (or fails).

---

## Chapters

| Chapter | Topic | Key Results |
|---------|-------|-------------|
| 5.1 | [Convexity](convexity.md) | Convex sets, convex functions, Jensen's inequality |
| 5.2 | [Unconstrained Optimization](unconstrained.md) | Gradient descent, Newton's method, convergence analysis |
| 5.3 | [Constrained Optimization](constrained.md) | Lagrange multipliers, KKT conditions, duality |
| 5.4 | [Stochastic Optimization](stochastic-optimization.md) | SGD, momentum, Adam, convergence in expectation |
| 5.5 | [Variational Methods](variational-methods.md) | ELBO, variational inference, VAE derivation |

## Prerequisites

- [Part 2: Linear Algebra](../linear-algebra/index.md) — matrices, eigenvalues
- [Part 3: Calculus](../calculus/index.md) — gradients, Hessians
- [Part 4: Probability](../probability/index.md) — for stochastic optimization and variational methods
