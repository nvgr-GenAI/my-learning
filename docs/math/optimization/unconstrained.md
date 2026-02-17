# 5.2 Unconstrained Optimization

## Prerequisites

**Required Background:**
- Multivariate calculus (Chapter 4.1): gradients, Hessians, Taylor series
- Linear algebra (Chapter 3): positive definite matrices, eigenvalues, condition number
- Convex analysis (Chapter 5.1): convex functions, strong convexity, Lipschitz continuity

**Key Concepts to Review:**
- Gradient: $\nabla f(x) = [\frac{\partial f}{\partial x_1}, \ldots, \frac{\partial f}{\partial x_n}]^T$
- Hessian: $H(x) = \nabla^2 f(x)$ with entries $H_{ij} = \frac{\partial^2 f}{\partial x_i \partial x_j}$
- Taylor expansion: $f(x + p) \approx f(x) + \nabla f(x)^T p + \frac{1}{2} p^T H(x) p$

---

## 5.2.1 Problem Formulation

**Definition 5.2.1 (Unconstrained Optimization Problem)**
Find $x^* \in \mathbb{R}^n$ that minimizes an objective function:

$$
\min_{x \in \mathbb{R}^n} f(x)
$$

where $f: \mathbb{R}^n \to \mathbb{R}$ is continuously differentiable. The point $x^*$ is called a **global minimizer** if $f(x^*) \leq f(x)$ for all $x \in \mathbb{R}^n$.

**Notation:**
- $f(x)$: objective function (loss function in ML)
- $\nabla f(x)$: gradient at point $x$
- $H(x) = \nabla^2 f(x)$: Hessian matrix at point $x$
- $x^*$: optimal solution (minimizer)

**ML Connection:** Training a neural network amounts to solving:
$$
\min_{\theta \in \mathbb{R}^p} \mathcal{L}(\theta) = \frac{1}{n} \sum_{i=1}^n \ell(f_\theta(x_i), y_i)
$$
where $\theta$ are parameters, $f_\theta$ is the model, and $\ell$ is the loss function. This is unconstrained optimization in high-dimensional space (often $p > 10^6$ for deep networks).

---

## 5.2.2 Optimality Conditions

### First-Order Necessary Condition

**Theorem 5.2.1 (First-Order Necessary Condition)**
If $x^*$ is a local minimizer of $f$ and $f$ is continuously differentiable in a neighborhood of $x^*$, then:

$$
\nabla f(x^*) = 0
$$

**Proof sketch:** If $\nabla f(x^*) \neq 0$, we can move in direction $p = -\nabla f(x^*)$ to decrease $f$. By Taylor expansion: $f(x^* + \alpha p) \approx f(x^*) + \alpha \nabla f(x^*)^T p = f(x^*) - \alpha \|\nabla f(x^*)\|^2 < f(x^*)$ for small $\alpha > 0$. Contradiction.

**Definition 5.2.2 (Stationary Point)**
A point $x$ satisfying $\nabla f(x) = 0$ is called a **stationary point** or **critical point**.

**Example 5.2.1 (Finding Critical Points):**
Consider $f(x, y) = x^2 + y^2 - 2x - 4y$. To find critical points, solve $\nabla f = 0$:

$$
\nabla f = \begin{bmatrix} 2x - 2 \\ 2y - 4 \end{bmatrix} = \begin{bmatrix} 0 \\ 0 \end{bmatrix} \implies x = 1, \; y = 2
$$

The unique critical point is $(1, 2)$ with $f(1, 2) = 1 + 4 - 2 - 8 = -5$.

**Important:** $\nabla f(x^*) = 0$ is necessary but not sufficient. Stationary points can be:
- Local minimizers
- Local maximizers
- Saddle points (neither min nor max)

```
Function landscape:
                  saddle point
                       ↓
    local max     ___/‾\___      local min
        ↓        /         \         ↓
       /\       /           \       /\
      /  \     /             \     /  \
     /    \___/               \___/    \
                                ↑
                          global min
```

### Second-Order Conditions

**Theorem 5.2.2 (Second-Order Necessary Condition)**
If $x^*$ is a local minimizer and $f$ is twice continuously differentiable, then:
1. $\nabla f(x^*) = 0$ (first-order condition)
2. $H(x^*)$ is positive semidefinite ($H(x^*) \succeq 0$)

**Theorem 5.2.3 (Second-Order Sufficient Condition)**
If $\nabla f(x^*) = 0$ and $H(x^*)$ is positive definite ($H(x^*) \succ 0$), then $x^*$ is a **strict local minimizer**.

**Characterization of Stationary Points:**

| Condition | $\nabla f(x^*)$ | $H(x^*)$ | Type |
|-----------|-----------------|----------|------|
| First-order necessary | $= 0$ | — | Stationary point |
| Second-order necessary | $= 0$ | $\succeq 0$ | Local min candidate |
| Second-order sufficient | $= 0$ | $\succ 0$ | Strict local min |
| — | $= 0$ | $\prec 0$ | Local max |
| — | $= 0$ | indefinite | Saddle point |

**Example 5.2.2 (Saddle Point Verification):**
For $f(x, y) = x^2 - y^2$, the gradient is $\nabla f = [2x, -2y]^T = 0$ at $(0, 0)$.
The Hessian is constant:

$$
H = \begin{bmatrix} 2 & 0 \\ 0 & -2 \end{bmatrix}, \quad \text{eigenvalues: } \lambda_1 = 2, \; \lambda_2 = -2
$$

Since $H$ is indefinite (one positive, one negative eigenvalue), $(0, 0)$ is a **saddle point**.
Verify: $f(0.1, 0) = 0.01 > 0 = f(0,0)$ but $f(0, 0.1) = -0.01 < 0 = f(0,0)$.
The point is a minimum along $x$ but a maximum along $y$.

**Example 5.2.3 (Classifying a Minimum via Hessian):**
Continuing Example 5.2.1, for $f(x, y) = x^2 + y^2 - 2x - 4y$ at critical point $(1, 2)$:

$$
H = \begin{bmatrix} 2 & 0 \\ 0 & 2 \end{bmatrix}, \quad \text{eigenvalues: } \lambda_1 = \lambda_2 = 2 > 0
$$

Since $H \succ 0$ (positive definite), $(1, 2)$ is a **strict local minimum** by Theorem 5.2.3.
Moreover, $f$ is convex (Hessian is PD everywhere), so $(1, 2)$ is the **global minimum**.

**ML Connection:** In deep learning, most critical points are saddle points, not local minima. High-dimensional loss surfaces have exponentially many saddle points but relatively few local minima that aren't near the global minimum (empirically observed).

---

## 5.2.3 Gradient Descent

**Definition 5.2.3 (Gradient Descent Algorithm)**
Starting from $x_0 \in \mathbb{R}^n$, iterate:

$$
x_{k+1} = x_k - \alpha_k \nabla f(x_k)
$$

where $\alpha_k > 0$ is the **learning rate** (or step size) at iteration $k$.

**Algorithm 5.2.1 (Gradient Descent)**
```
Input: f, ∇f, x₀, {αₖ}, tolerance ε, max iterations K
Output: x* ≈ minimizer of f

1. Initialize k ← 0
2. while ‖∇f(xₖ)‖ > ε and k < K:
3.     xₖ₊₁ ← xₖ - αₖ ∇f(xₖ)
4.     k ← k + 1
5. return xₖ
```

**Geometric Intuition:**
```
Contour plot of f(x):

           ∇f(x₃)
             ↑
         x₃  |
          *--+
         /
    x₂ *        Level sets: f(x) = c
      /           ___
  x₁ *           /   \___
    /           |    x*  |
x₀ *            |___   __|
                    \_/

Gradient points toward steepest ascent
Negative gradient → steepest descent
```

**ML Connection:** Gradient descent is THE fundamental algorithm for training neural networks. Given training data $\{(x_i, y_i)\}_{i=1}^n$, minimize:
$$
\mathcal{L}(\theta) = \frac{1}{n} \sum_{i=1}^n \ell(f_\theta(x_i), y_i)
$$
via $\theta_{k+1} = \theta_k - \alpha_k \nabla_\theta \mathcal{L}(\theta_k)$ where $\nabla_\theta \mathcal{L}$ is computed by backpropagation.

**Example 5.2.4 (Gradient Descent Step-by-Step):**
Minimize $f(x) = x^2$ using $\alpha = 0.1$, starting at $x_0 = 5$. Here $f'(x) = 2x$.

$$
\begin{aligned}
k=0: \quad x_1 &= x_0 - 0.1 \cdot f'(x_0) = 5 - 0.1 \cdot 10 = 4.0, \quad f(x_1) = 16.0 \\
k=1: \quad x_2 &= 4.0 - 0.1 \cdot 8.0 = 3.2, \quad f(x_2) = 10.24 \\
k=2: \quad x_3 &= 3.2 - 0.1 \cdot 6.4 = 2.56, \quad f(x_3) = 6.5536
\end{aligned}
$$

Each iterate satisfies $x_k = 5 \cdot (1 - 2 \cdot 0.1)^k = 5 \cdot 0.8^k$. The error shrinks by factor $0.8$ per step (linear convergence), reaching $x_{10} \approx 0.537$ and $x_{20} \approx 0.058$.

### Convergence Analysis

**Assumption 5.2.1 (Lipschitz Continuous Gradient)**
The gradient $\nabla f$ is **L-Lipschitz continuous** if there exists $L > 0$ such that:
$$
\|\nabla f(x) - \nabla f(y)\| \leq L \|x - y\| \quad \forall x, y
$$

**Interpretation:** Gradient doesn't change too rapidly. Equivalent to $H(x) \preceq LI$ (Hessian eigenvalues $\leq L$).

**Theorem 5.2.4 (Convergence Rate - Convex Case)**
If $f$ is convex and $\nabla f$ is L-Lipschitz continuous, then gradient descent with constant step size $\alpha = \frac{1}{L}$ satisfies:

$$
f(x_k) - f(x^*) \leq \frac{2L \|x_0 - x^*\|^2}{k}
$$

**Convergence rate:** $O(1/k)$ (sublinear)

**Theorem 5.2.5 (Convergence Rate - Strongly Convex Case)**
If $f$ is $\mu$-strongly convex and $\nabla f$ is L-Lipschitz continuous, then gradient descent with $\alpha = \frac{2}{\mu + L}$ satisfies:

$$
\|x_k - x^*\|^2 \leq \left(1 - \frac{\mu}{L}\right)^k \|x_0 - x^*\|^2
$$

**Convergence rate:** $O(\rho^k)$ where $\rho = 1 - \frac{\mu}{L} < 1$ (linear/exponential convergence)

**Definition 5.2.4 (Condition Number)**
The **condition number** is $\kappa = \frac{L}{\mu}$. Larger $\kappa$ means slower convergence.

```
Convergence comparison:

f(xₖ) - f(x*)
    |
    |  Convex: O(1/k)
    |\___
    |    \___
    |        \___
    |            \___
    |
    |  Strongly convex: O((1-μ/L)ᵏ)
    |\
    | \__
    |    \__
    |       \__
    |          \_______
    +-------------------------> k
    0        10       20
```

**Example (Quadratic Function):**
For $f(x) = \frac{1}{2} x^T Q x$ where $Q \succ 0$:
- $\nabla f(x) = Qx$
- Eigenvalues: $\lambda_{\min}(Q) = \mu$, $\lambda_{\max}(Q) = L$
- Condition number: $\kappa(Q) = \frac{\lambda_{\max}}{\lambda_{\min}}$
- Well-conditioned ($\kappa \approx 1$): fast convergence
- Ill-conditioned ($\kappa \gg 1$): slow convergence

---

## 5.2.4 Learning Rate Selection

The learning rate $\alpha$ is the most critical hyperparameter in gradient descent.

### Fixed Learning Rate

**Too Large:** Algorithm diverges
```
Divergence with α too large:

  f(x)
    |     * x₃
    |    /
    |   * x₂
    |  /
    | * x₁
    |/
x*  *---------> x₀
    |
```

**Example 5.2.5 (Divergence with Large Learning Rate):**
Same function $f(x) = x^2$, $f'(x) = 2x$, starting at $x_0 = 5$, but now $\alpha = 1.1$ (exceeds $1/L = 1/2$):

$$
\begin{aligned}
k=0: \quad x_1 &= 5 - 1.1 \cdot 10 = -6.0, \quad f(x_1) = 36 \\
k=1: \quad x_2 &= -6 - 1.1 \cdot (-12) = 7.2, \quad f(x_2) = 51.84 \\
k=2: \quad x_3 &= 7.2 - 1.1 \cdot 14.4 = -8.64, \quad f(x_3) = 74.65
\end{aligned}
$$

The iterates oscillate with growing magnitude: $|x_k| = 5 \cdot 1.2^k \to \infty$. For $f(x) = x^2$ (where $L = 2$), any $\alpha > 1 = 1/L \cdot 2$ causes divergence.

**Too Small:** Slow convergence
```
Slow convergence with α too small:

    f(x)
      |
      |    x* (optimum)
      |     ↓
      | x₃ x₄ x₅ x₆ ...
      | x₂ •••••
      | x₁ ••
      | x₀ •
      +--------------->
```

**Optimal (Goldilocks):** Balanced progress
```
Optimal learning rate:

    f(x)
      |
      |    x*
      |    ↓
      | x₃ •
      | x₂  \
      | x₁   \
      | x₀    \
      +--------------->
```

**Rule of Thumb:** For Lipschitz constant $L$, use $\alpha \in (\frac{1}{L}, \frac{2}{L})$.

### Line Search

**Definition 5.2.5 (Exact Line Search)**
Choose $\alpha_k$ to minimize $f$ along the search direction:
$$
\alpha_k = \arg\min_{\alpha > 0} f(x_k - \alpha \nabla f(x_k))
$$

**Backtracking Line Search (Armijo Rule):**
1. Start with $\alpha = 1$
2. While $f(x_k - \alpha \nabla f(x_k)) > f(x_k) - c \alpha \|\nabla f(x_k)\|^2$:
   - $\alpha \leftarrow \beta \alpha$ (typical: $c = 10^{-4}$, $\beta = 0.5$)
3. Return $\alpha$

Ensures **sufficient decrease** in objective value.

**Example 5.2.6 (Exact Line Search):**
For $f(x) = x^2 + 4x + 5$, starting at $x_0 = 3$. Here $f'(x) = 2x + 4$, so $f'(3) = 10$.
The search direction is $p = -f'(3) = -10$. Minimize along this direction:

$$
g(\alpha) = f(3 - 10\alpha) = (3 - 10\alpha)^2 + 4(3 - 10\alpha) + 5
$$

Setting $g'(\alpha) = 0$: $\; 2(3 - 10\alpha)(-10) + 4(-10) = 0 \implies \alpha^* = 0.4$.
Then $x_1 = 3 - 10 \cdot 0.4 = -1$, and $f(-1) = 1 - 4 + 5 = 2$. Since $f'(-1) = -2 + 4 = 2 \neq 0$, we continue iterating.

### Learning Rate Schedules (ML)

**Step Decay:**
$$
\alpha_k = \alpha_0 \cdot \gamma^{\lfloor k/s \rfloor}
$$
Reduce learning rate by factor $\gamma$ every $s$ iterations.

**Exponential Decay:**
$$
\alpha_k = \alpha_0 \cdot e^{-\lambda k}
$$

**Cosine Annealing:**
$$
\alpha_k = \alpha_{\min} + \frac{1}{2}(\alpha_{\max} - \alpha_{\min})(1 + \cos(\frac{k\pi}{K}))
$$
Smoothly decreases from $\alpha_{\max}$ to $\alpha_{\min}$ over $K$ iterations.

**Warmup:**
Start with small learning rate, gradually increase to target value over first few epochs. Prevents instability in early training.

**ML Connection:** Modern deep learning uses **adaptive learning rates** (Adam, RMSprop) that adjust $\alpha$ per parameter based on gradient history. Standard practice: warmup for 1-5% of training, then cosine decay.

---

## 5.2.5 Momentum Methods

### Polyak's Heavy Ball Method

**Definition 5.2.6 (Momentum)**
Maintain velocity $v_k$ that accumulates gradients:

$$
\begin{aligned}
v_{k+1} &= \beta v_k - \alpha \nabla f(x_k) \\
x_{k+1} &= x_k + v_{k+1}
\end{aligned}
$$

where $\beta \in [0, 1)$ is the **momentum coefficient** (typical: $\beta = 0.9$).

**Physics Analogy:**
Imagine a ball rolling down a hill:
- Gradient: force pushing the ball
- Momentum: ball's velocity (resists sudden changes)
- Friction: $1 - \beta$ coefficient

```
Without momentum:        With momentum:

    \                        \
     \  zigzag               \___
      \/\/\                      \___
         \/\                         \___
           \                             ↓ x*
            ↓ x*                     (smooth path)
    (oscillates in ravine)
```

**Benefits:**
1. **Smooths noisy gradients:** Averages over past gradients
2. **Accelerates in consistent directions:** Builds up speed
3. **Dampens oscillations:** Reduces zigzagging in narrow valleys

**Algorithm 5.2.2 (Gradient Descent with Momentum)**
```
Input: f, ∇f, x₀, α, β, tolerance ε
Output: x* ≈ minimizer

1. Initialize v₀ ← 0, k ← 0
2. while ‖∇f(xₖ)‖ > ε:
3.     vₖ₊₁ ← β vₖ - α ∇f(xₖ)
4.     xₖ₊₁ ← xₖ + vₖ₊₁
5.     k ← k + 1
6. return xₖ
```

**Example 5.2.7 (Momentum Update Step-by-Step):**
Minimize $f(x) = x^2$ ($f'(x) = 2x$) with $\alpha = 0.1$, $\beta = 0.9$, starting at $x_0 = 5$, $v_0 = 0$.

$$
\begin{aligned}
k=0: \quad v_1 &= 0.9 \cdot 0 - 0.1 \cdot 2(5) = -1.0, \quad x_1 = 5 + (-1.0) = 4.0 \\
k=1: \quad v_2 &= 0.9 \cdot (-1.0) - 0.1 \cdot 2(4.0) = -1.7, \quad x_2 = 4.0 + (-1.7) = 2.3 \\
k=2: \quad v_3 &= 0.9 \cdot (-1.7) - 0.1 \cdot 2(2.3) = -1.99, \quad x_3 = 2.3 + (-1.99) = 0.31
\end{aligned}
$$

Compare: without momentum, $x_3 = 5 \cdot 0.8^3 = 2.56$. With momentum, $x_3 = 0.31$ -- the velocity accumulates and accelerates convergence significantly.

**Convergence:** For quadratic functions, optimal $\beta$ achieves convergence rate $O((\sqrt{\kappa}-1)/(\sqrt{\kappa}+1))^k$ vs $O((1-1/\kappa)^k)$ for standard GD. Significant speedup when $\kappa \gg 1$.

### Nesterov Accelerated Gradient

**Definition 5.2.7 (Nesterov Momentum)**
"Look-ahead" variant: evaluate gradient at predicted future position:

$$
\begin{aligned}
v_{k+1} &= \beta v_k - \alpha \nabla f(x_k + \beta v_k) \\
x_{k+1} &= x_k + v_{k+1}
\end{aligned}
$$

**Intuition:**
Standard momentum: compute gradient at current position, add momentum
Nesterov: jump ahead using momentum, then compute gradient

```
Nesterov vs standard momentum:

         x_k + β v_k (lookahead)
              ↓
    x_k ------•
     •        |
     |        ↓ ∇f evaluated here
     |
     ↓ ∇f evaluated here (standard)
```

**Theorem 5.2.6 (Nesterov Convergence)**
For convex functions with Lipschitz gradient, Nesterov accelerated gradient achieves:
$$
f(x_k) - f(x^*) = O(1/k^2)
$$
vs $O(1/k)$ for standard GD. This is **optimal** for first-order methods.

**ML Connection:** Momentum variants (especially Nesterov) are standard in deep learning. PyTorch/TensorFlow SGD optimizers include momentum. Helps escape saddle points and traverse flat regions faster.

---

## 5.2.6 Newton's Method

**Definition 5.2.8 (Newton's Method)**
Use second-order Taylor approximation to find step:

$$
x_{k+1} = x_k - H(x_k)^{-1} \nabla f(x_k)
$$

where $H(x_k) = \nabla^2 f(x_k)$ is the Hessian.

**Derivation:** Minimize quadratic model of $f$ around $x_k$:
$$
m_k(p) = f(x_k) + \nabla f(x_k)^T p + \frac{1}{2} p^T H(x_k) p
$$
Setting $\nabla_p m_k(p) = 0$ gives $p^* = -H(x_k)^{-1} \nabla f(x_k)$.

**Algorithm 5.2.3 (Newton's Method)**
```
Input: f, ∇f, ∇²f, x₀, tolerance ε
Output: x* ≈ minimizer

1. Initialize k ← 0
2. while ‖∇f(xₖ)‖ > ε:
3.     Solve H(xₖ) pₖ = -∇f(xₖ)  [Newton direction]
4.     xₖ₊₁ ← xₖ + pₖ
5.     k ← k + 1
6. return xₖ
```

**Comparison to Gradient Descent:**

| Aspect | Gradient Descent | Newton's Method |
|--------|------------------|-----------------|
| Direction | $-\nabla f(x_k)$ | $-H(x_k)^{-1} \nabla f(x_k)$ |
| Convergence rate | Linear: $O(\rho^k)$ | Quadratic: $O(\rho^{2^k})$ |
| Iteration cost | $O(n)$ | $O(n^3)$ (Hessian inversion) |
| Memory | $O(n)$ | $O(n^2)$ |
| Tuning | Learning rate $\alpha$ | None (scale-invariant) |
| Use case | Large-scale | Small-scale |

**Theorem 5.2.7 (Quadratic Convergence)**
If $f$ is twice continuously differentiable, $x^*$ is a minimizer with $H(x^*) \succ 0$, and $x_0$ is sufficiently close to $x^*$, then:
$$
\|x_{k+1} - x^*\| \leq M \|x_k - x^*\|^2
$$
for some constant $M > 0$.

**Quadratic convergence:** Number of correct digits doubles each iteration.

**Example 5.2.8 (Newton's Method Step-by-Step):**
Minimize $f(x) = x^2 - 4x + 5$. Here $f'(x) = 2x - 4$ and $f''(x) = 2$.
Start at $x_0 = 10$:

$$
x_1 = x_0 - \frac{f'(x_0)}{f''(x_0)} = 10 - \frac{2(10) - 4}{2} = 10 - \frac{16}{2} = 2
$$

Check: $f'(2) = 0$, so Newton converges to $x^* = 2$ **in one iteration**.
This is expected: Newton's method solves quadratic functions exactly in a single step because the quadratic Taylor model is exact.

For a non-quadratic, e.g., $f(x) = x^4 - 4x^2 + x$ with $f'(x) = 4x^3 - 8x + 1$, $f''(x) = 12x^2 - 8$, starting at $x_0 = 2$:

$$
\begin{aligned}
k=0: \quad x_1 &= 2 - \frac{4(8) - 16 + 1}{12(4) - 8} = 2 - \frac{17}{40} = 1.575 \\
k=1: \quad x_2 &= 1.575 - \frac{4(1.575)^3 - 8(1.575) + 1}{12(1.575)^2 - 8} \approx 1.445
\end{aligned}
$$

Multiple iterations are needed, but convergence is very fast once near $x^*$.

```
Convergence comparison (log scale):

log(error)
    |
    |  Gradient descent
    |\___
    |    \___
    |        \___
    |
    |  Newton
    |\
    | \_______________
    +-------------------------> k
    0    2    4    6    8
```

**Limitations:**
1. **Computational cost:** $O(n^3)$ to invert $H$ (or $O(n^2)$ to solve linear system)
2. **Memory:** Storing $n \times n$ Hessian
3. **Hessian computation:** Often unavailable or expensive
4. **Non-convexity:** May converge to saddle points or maxima

**ML Connection:** Newton's method is impractical for deep learning ($n > 10^6$ parameters). Used for small-scale problems (logistic regression with $n < 10^4$) or second-order optimizers on subsets of parameters.

---

## 5.2.7 Quasi-Newton Methods

**Idea:** Approximate $H^{-1}$ using only gradient information, avoiding expensive Hessian computation.

**Definition 5.2.9 (Quasi-Newton Update)**
Maintain approximation $B_k \approx H(x_k)^{-1}$ and update:
$$
x_{k+1} = x_k - \alpha_k B_k \nabla f(x_k)
$$

### BFGS Algorithm

**Broyden-Fletcher-Goldfarb-Shanno (BFGS)** is the most popular quasi-Newton method.

**Secant Condition:** Require $B_{k+1}$ to satisfy:
$$
B_{k+1} s_k = y_k
$$
where:
- $s_k = x_{k+1} - x_k$ (step)
- $y_k = \nabla f(x_{k+1}) - \nabla f(x_k)$ (gradient difference)

This mimics $H^{-1}$ behavior: $H^{-1} (\nabla f(x_{k+1}) - \nabla f(x_k)) \approx x_{k+1} - x_k$.

**BFGS Update Formula:**
$$
B_{k+1} = B_k + \frac{y_k y_k^T}{y_k^T s_k} - \frac{B_k s_k s_k^T B_k}{s_k^T B_k s_k}
$$

**Properties:**
- If $B_k \succ 0$ and $y_k^T s_k > 0$, then $B_{k+1} \succ 0$
- Superlinear convergence: faster than linear, slower than quadratic
- Complexity: $O(n^2)$ per iteration (vs $O(n^3)$ for Newton)

**Algorithm 5.2.4 (BFGS)**
```
Input: f, ∇f, x₀, tolerance ε
Output: x* ≈ minimizer

1. Initialize B₀ ← I, k ← 0
2. while ‖∇f(xₖ)‖ > ε:
3.     pₖ ← -Bₖ ∇f(xₖ)
4.     Find αₖ via line search
5.     xₖ₊₁ ← xₖ + αₖ pₖ
6.     sₖ ← xₖ₊₁ - xₖ, yₖ ← ∇f(xₖ₊₁) - ∇f(xₖ)
7.     Update Bₖ₊₁ via BFGS formula
8.     k ← k + 1
9. return xₖ
```

### Limited-Memory BFGS (L-BFGS)

For high-dimensional problems, storing $n \times n$ matrix $B_k$ is prohibitive.

**Definition 5.2.10 (L-BFGS)**
Store only last $m$ pairs $(s_k, y_k)$ (typical: $m = 10-20$) and compute $B_k \nabla f(x_k)$ implicitly via two-loop recursion.

**Memory:** $O(mn)$ vs $O(n^2)$ for full BFGS

**Algorithm complexity:** $O(mn)$ per iteration

**ML Connection:** L-BFGS is the go-to method for medium-scale ML problems ($10^3 < n < 10^5$):
- Logistic regression
- Small neural networks
- Fine-tuning with frozen layers
- Problems with cheap full gradient computation

**Comparison:**

| Method | Cost/iter | Memory | Convergence | Best for |
|--------|-----------|--------|-------------|----------|
| GD | $O(n)$ | $O(n)$ | Linear | $n > 10^6$ |
| Newton | $O(n^3)$ | $O(n^2)$ | Quadratic | $n < 10^3$ |
| BFGS | $O(n^2)$ | $O(n^2)$ | Superlinear | $n < 10^4$ |
| L-BFGS | $O(mn)$ | $O(mn)$ | Superlinear | $10^3 < n < 10^5$ |

---

## 5.2.8 Coordinate Descent

**Definition 5.2.11 (Coordinate Descent)**
Minimize $f$ by optimizing one coordinate at a time, holding others fixed.

**Algorithm 5.2.5 (Cyclic Coordinate Descent)**
```
Input: f, x₀, tolerance ε
Output: x* ≈ minimizer

1. Initialize k ← 0
2. while ‖∇f(xₖ)‖ > ε:
3.     for i = 1 to n:
4.         xₖ₊₁[i] ← argmin_z f(xₖ[1], ..., xₖ[i-1], z, xₖ[i+1], ..., xₖ[n])
5.         xₖ[i] ← xₖ₊₁[i]
6.     k ← k + 1
7. return xₖ
```

**Variants:**
- **Cyclic:** Update coordinates in fixed order $1, 2, \ldots, n$
- **Random:** Select coordinate uniformly at random
- **Greedy:** Choose coordinate with largest gradient component

**Example (Separable Function):**
For $f(x, y) = x^2 + 2y^2$:
```
Coordinate descent path:

  y
  |
  |  x* = (0,0)
  |    ↓
  | x₂ •
  |    |
  | x₁ •---• x₀
  |__________|___ x
```

**Example 5.2.10 (Coordinate Descent Iterations):**
Minimize $f(x, y) = x^2 + 2y^2 + xy - 4x - 6y$ starting at $(0, 0)$.
$\frac{\partial f}{\partial x} = 2x + y - 4$, $\frac{\partial f}{\partial y} = 4y + x - 6$.

**Iteration 1:** Fix $y = 0$, minimize over $x$: $\; 2x + 0 - 4 = 0 \implies x = 2$.
Fix $x = 2$, minimize over $y$: $\; 4y + 2 - 6 = 0 \implies y = 1$.
After iteration 1: $(x_1, y_1) = (2, 1)$, $f(2, 1) = 4 + 2 + 2 - 8 - 6 = -6$.

**Iteration 2:** Fix $y = 1$, minimize over $x$: $\; 2x + 1 - 4 = 0 \implies x = 1.5$.
Fix $x = 1.5$, minimize over $y$: $\; 4y + 1.5 - 6 = 0 \implies y = 1.125$.
After iteration 2: $(1.5, 1.125)$, $f \approx -6.125$.

The iterates converge to $(x^*, y^*) = (10/7, 8/7) \approx (1.43, 1.14)$.

**Advantages:**
1. **Simple:** No line search needed if subproblems have closed-form solutions
2. **Low memory:** Only work with one coordinate at a time
3. **Parallelizable:** Update disjoint coordinate blocks simultaneously

**Theorem 5.2.8 (Coordinate Descent Convergence)**
If $f$ is convex and coordinate-wise differentiable, cyclic coordinate descent converges to the optimum.

**ML Connection:** Coordinate descent is effective for:
- Lasso regression: $\min_\beta \frac{1}{2}\|y - X\beta\|^2 + \lambda \|\beta\|_1$ (closed-form updates)
- SVM training (SMO algorithm)
- Matrix factorization (alternating least squares)

---

## 5.2.9 Convergence Theory Summary

### Smoothness and Strong Convexity

**Definition 5.2.12 (L-Smoothness)**
$f$ is **L-smooth** if $\nabla f$ is L-Lipschitz continuous:
$$
\|\nabla f(x) - \nabla f(y)\| \leq L \|x - y\|
$$

**Equivalent conditions:**
- $f(y) \leq f(x) + \nabla f(x)^T (y-x) + \frac{L}{2}\|y-x\|^2$ (upper bound by quadratic)
- Eigenvalues of $H(x)$ bounded by $L$

**Definition 5.2.13 ($\mu$-Strong Convexity)**
$f$ is **$\mu$-strongly convex** if:
$$
f(y) \geq f(x) + \nabla f(x)^T (y-x) + \frac{\mu}{2}\|y-x\|^2
$$

**Equivalent conditions:**
- $f(x) - \frac{\mu}{2}\|x\|^2$ is convex
- Eigenvalues of $H(x)$ bounded below by $\mu$

```
Geometric interpretation:

f(x)
  |     L-smooth: bounded curvature above
  |    /‾‾\     quadratic upper bound
  | __/    \__
  |           ___
  |         _/   \___    μ-strongly convex:
  |       _/   f     \__ bounded curvature below
  |_____________________  quadratic lower bound
```

### Convergence Rate Table

| Function Class | Method | Convergence Rate | Notes |
|----------------|--------|------------------|-------|
| L-smooth convex | GD | $O(1/k)$ | Linear in function value |
| $\mu$-strongly convex, L-smooth | GD | $O(\rho^k)$, $\rho = 1 - \mu/L$ | Linear/exponential |
| L-smooth convex | Nesterov | $O(1/k^2)$ | Optimal for 1st-order |
| $\mu$-strongly convex, L-smooth | Nesterov | $O(\rho^k)$, $\rho = 1 - \sqrt{\mu/L}$ | Better than GD |
| Twice differentiable | Newton | $O(\rho^{2^k})$ | Quadratic (local) |
| L-smooth convex | Coordinate descent | $O(1/k)$ | Per-coordinate smoothness |

**Condition Number Impact:**
For strongly convex functions, convergence depends on $\kappa = L/\mu$:
- Well-conditioned ($\kappa \approx 1$): Fast convergence
- Ill-conditioned ($\kappa \gg 1$): Slow convergence

**Example:** For quadratic $f(x) = \frac{1}{2}x^T Q x$:
- $\kappa(Q) = 10$: GD needs ~100 iterations for high accuracy
- $\kappa(Q) = 1000$: GD needs ~10,000 iterations
- Newton: Always 1 iteration (quadratics are exactly modeled)

**Example 5.2.9 (Condition Number and Convergence):**
Consider $f(x, y) = \frac{1}{2}(x^2 + 100y^2)$. Here $Q = \text{diag}(1, 100)$, so $L = 100$, $\mu = 1$, $\kappa = 100$.
Optimal step size: $\alpha = \frac{2}{\mu + L} = \frac{2}{101} \approx 0.0198$. Convergence rate: $\rho = \frac{\kappa - 1}{\kappa + 1} = \frac{99}{101} \approx 0.98$.
Starting at $(1, 1)$:

$$
\begin{aligned}
k=0: \quad \nabla f &= (1, 100), \quad (x_1, y_1) = (1, 1) - 0.0198 \cdot (1, 100) = (0.980, -0.980) \\
k=1: \quad \nabla f &= (0.980, -98.0), \quad (x_2, y_2) = (0.980, -0.980) - 0.0198 \cdot (0.980, -98.0) = (0.961, 0.961)
\end{aligned}
$$

The $y$-coordinate oscillates wildly while $x$ barely moves -- a hallmark of ill-conditioned problems. GD needs $\sim 100 \cdot \ln(1/\epsilon)$ iterations, while Newton needs just 1.

### Practical Considerations

**When to use each method:**

1. **Gradient Descent:** Default for large-scale problems ($n > 10^6$)
   - Easy to implement, low memory, parallelizable
   - Use adaptive methods (Adam, RMSprop) in practice

2. **Momentum:** When objective has ravines or ill-conditioned
   - Neural network training (SGD with momentum)
   - Noisy gradient estimates (mini-batch)

3. **Newton:** Small-scale problems with cheap Hessian ($n < 10^3$)
   - Trust region methods for non-convex (avoid saddle points)
   - Need globalization strategy (line search or trust region)

4. **L-BFGS:** Medium-scale with expensive gradient ($10^3 < n < 10^5$)
   - Full-batch training on moderate datasets
   - Fine-tuning pre-trained models

5. **Coordinate Descent:** Structured problems with cheap coordinate updates
   - Lasso, elastic net, SVM
   - Separable objectives

---

## 5.2.10 ML Connections and Practical Insights

### Gradient Descent Variants in Deep Learning

**Full-Batch Gradient Descent:**
$$
\theta_{k+1} = \theta_k - \alpha_k \frac{1}{n} \sum_{i=1}^n \nabla_\theta \ell(f_\theta(x_i), y_i)
$$
- Exact gradient but expensive ($O(n)$ per iteration)
- Deterministic convergence

**Stochastic Gradient Descent (SGD):**
$$
\theta_{k+1} = \theta_k - \alpha_k \nabla_\theta \ell(f_\theta(x_{i_k}), y_{i_k})
$$
where $i_k$ is randomly sampled.
- Noisy but cheap ($O(1)$ per iteration)
- Can escape poor local minima
- Requires decreasing learning rate for convergence

**Mini-Batch SGD:**
$$
\theta_{k+1} = \theta_k - \alpha_k \frac{1}{|B_k|} \sum_{i \in B_k} \nabla_\theta \ell(f_\theta(x_i), y_i)
$$
where $B_k$ is a random mini-batch (typical size: 32-256).
- Balance between accuracy and efficiency
- Enables GPU parallelization
- Standard practice in deep learning

### Why Newton Fails for Deep Learning

1. **Dimension:** $n > 10^6$ parameters → Hessian has $> 10^{12}$ entries
2. **Non-convexity:** Many saddle points; Newton can converge to wrong type of critical point
3. **Computational cost:** One Newton step costs more than thousands of GD steps
4. **Storage:** Cannot fit Hessian in memory

**Alternative:** Use **diagonal approximations** (Adam, AdaGrad) or **K-FAC** (Kronecker-factored approximation).

### The Optimization Landscape of Neural Networks

**Empirical observations:**
1. Most critical points are saddle points (not local minima)
2. Local minima found by GD are typically near-optimal
3. Wide minima (flat loss landscape) generalize better than sharp minima
4. Loss surface has many saddle points in early layers, fewer in late layers

**Implication:** Simple first-order methods (SGD with momentum) work well despite non-convexity. Second-order information less critical than in classical optimization.

---

## Exercises

### Computational (★)

**Exercise 5.2.1**
Implement gradient descent to minimize $f(x, y) = x^2 + 4y^2$ starting from $(1, 1)$.
(a) Use fixed step size $\alpha = 0.1$. Plot the trajectory.
(b) Compute the condition number. How many iterations to reach $\|x_k - x^*\| < 10^{-6}$?
(c) Repeat with $\alpha = 0.01$ and $\alpha = 0.4$. Compare convergence.

**Exercise 5.2.2**
For the Rosenbrock function $f(x, y) = (1-x)^2 + 100(y-x^2)^2$:
(a) Compute $\nabla f$ and verify $(1, 1)$ is the minimizer.
(b) Run gradient descent from $(-1, 1)$ with $\alpha = 0.001$. Observe slow convergence.
(c) Add momentum ($\beta = 0.9$). Compare convergence speed.

### Theoretical (★★)

**Exercise 5.2.3**
Prove that for $f(x) = \frac{1}{2}x^T Q x - b^T x$ where $Q \succ 0$:
(a) The unique minimizer is $x^* = Q^{-1} b$.
(b) Gradient descent with $\alpha = \frac{2}{\lambda_{\min}(Q) + \lambda_{\max}(Q)}$ converges.
(c) The convergence rate is $O\left(\left(\frac{\kappa-1}{\kappa+1}\right)^k\right)$ where $\kappa = \lambda_{\max}/\lambda_{\min}$.

**Exercise 5.2.4**
Show that Newton's method converges in one iteration for quadratic functions $f(x) = \frac{1}{2}x^T Q x - b^T x + c$.

**Exercise 5.2.5**
Prove that if $f$ is $\mu$-strongly convex and L-smooth, then:
$$
f(x^*) \geq f(x) - \frac{1}{2\mu}\|\nabla f(x)\|^2
$$
Use this to bound the suboptimality in terms of gradient norm.

### Advanced (★★★)

**Exercise 5.2.6**
Prove Theorem 5.2.4: For convex $f$ with L-Lipschitz gradient, GD with $\alpha = 1/L$ satisfies:
$$
f(x_k) - f(x^*) \leq \frac{2L\|x_0 - x^*\|^2}{k}
$$
Hint: Show $f(x_{k+1}) \leq f(x_k) - \frac{1}{2L}\|\nabla f(x_k)\|^2$ and use convexity.

**Exercise 5.2.7 (Nesterov Lower Bound)**
Prove that for the class of L-smooth convex functions, no first-order method (using only gradient information) can achieve better than $O(1/k^2)$ convergence rate.

**Exercise 5.2.8**
Analyze coordinate descent for the least squares problem:
$$
\min_x \frac{1}{2}\|Ax - b\|^2
$$
(a) Derive the closed-form update for coordinate $i$.
(b) Show that the cost per iteration is $O(mn)$ where $A \in \mathbb{R}^{m \times n}$.
(c) Compare to gradient descent ($O(mn)$ per iteration). When is coordinate descent preferred?

---

## Related Topics

**Within this book:**
- Section 5.1: Convex Optimization (convexity, strong convexity, smoothness)
- Section 5.3: Constrained Optimization (KKT conditions, projected gradient descent)
- Section 5.4: Stochastic Optimization (SGD, variance reduction, adaptive methods)
- Chapter 8: Neural Network Training (backpropagation, optimizers, learning rate schedules)

**Further reading:**
- Nocedal & Wright, "Numerical Optimization" (2006): Comprehensive reference
- Nesterov, "Lectures on Convex Optimization" (2018): Optimal methods and complexity bounds
- Ruder, "An overview of gradient descent optimization algorithms" (2016): ML perspective
- Bottou et al., "Optimization Methods for Large-Scale Machine Learning" (2018): Deep learning focus

**Implementations:**
- SciPy: `scipy.optimize.minimize` (BFGS, L-BFGS-B, Newton-CG)
- PyTorch: `torch.optim` (SGD, Adam, RMSprop, LBFGS)
- TensorFlow: `tf.optimizers` (gradient descent variants, adaptive methods)

---

**Chapter Summary:** Unconstrained optimization seeks to minimize $f(x)$ without constraints. Optimality requires $\nabla f(x^*) = 0$ (necessary) and $H(x^*) \succ 0$ (sufficient). Gradient descent iterates $x_{k+1} = x_k - \alpha_k \nabla f(x_k)$ with convergence rate $O(1/k)$ for convex and $O(\rho^k)$ for strongly convex functions. Learning rate selection is critical: too large diverges, too small is inefficient. Momentum methods accelerate by accumulating past gradients; Nesterov achieves optimal $O(1/k^2)$ rate. Newton's method uses Hessian for quadratic convergence but costs $O(n^3)$ per iteration. Quasi-Newton (L-BFGS) approximates $H^{-1}$ with $O(n)$ memory and superlinear convergence. Coordinate descent optimizes one variable at a time, effective for structured problems. Convergence depends on condition number $\kappa = L/\mu$; ill-conditioned problems converge slowly. In ML, gradient descent (with momentum and adaptive learning rates) is the workhorse for training neural networks due to scalability.
