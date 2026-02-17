# Part 3: Calculus

Calculus is the mathematics of change — and machine learning is fundamentally about how changes in parameters affect predictions. Every time you train a neural network, calculus is computing how a tiny change in each weight would change the loss. The gradient, the Jacobian, and the Hessian are the workhorses of optimization. Without calculus, there is no backpropagation, no gradient descent, and no modern deep learning.

---

## Chapters

| Chapter | Topic | Key Results |
|---------|-------|-------------|
| 3.1 | [Limits & Continuity](limits-and-continuity.md) | $\varepsilon$-$\delta$ definition, sequences, series, convergence |
| 3.2 | [Differentiation](differentiation.md) | Derivative rules, chain rule, mean value theorem |
| 3.3 | [Integration](integration.md) | Riemann integral, techniques, fundamental theorem of calculus |
| 3.4 | [Multivariable Calculus](multivariable-calculus.md) | Partial derivatives, gradient, Jacobian, Hessian |
| 3.5 | [Vector Calculus](vector-calculus.md) | Divergence, curl, Green's/Stokes' theorems |
| 3.6 | [Taylor Series](taylor-series.md) | Taylor/Maclaurin expansion, approximation, error bounds |
| 3.7 | [Calculus of Variations](calculus-of-variations.md) | Euler-Lagrange equation, variational inference |
| 3.8 | [Differential Equations](differential-equations.md) | ODEs, PDEs, Neural ODEs, SDEs, diffusion models |

## Prerequisites

- [Part 1: Foundations](../foundations/index.md)
- [Part 2: Linear Algebra](../linear-algebra/index.md) — for multivariable calculus and beyond

## The ML Connection

| Calculus Concept | Where It Appears in ML |
|-----------------|----------------------|
| Derivative | Gradient of loss w.r.t. single parameter |
| Chain rule | Backpropagation algorithm |
| Gradient ($\nabla f$) | Direction of steepest ascent for gradient descent |
| Jacobian | Normalizing flows, multi-output derivatives |
| Hessian | Second-order optimization, loss landscape analysis |
| Taylor expansion | Local approximation of loss, Newton's method |
| Integration | Marginalizing distributions, computing expectations |
| Calculus of variations | Variational inference (VAEs), optimal control |
| Differential equations | Neural ODEs, diffusion models, dynamical systems |
