# Chapter 5.1: Convexity

## Prerequisites

- Linear algebra: vector spaces, inner products, norms
- Calculus: gradients, Hessian matrices, Taylor expansion
- Real analysis: continuity, differentiability
- Probability: expectation, random variables (for Jensen's inequality)

## Chapter Overview

Convexity is the cornerstone of tractable optimization in machine learning. Convex optimization problems possess a remarkable property: every local minimum is a global minimum. This chapter develops the mathematical foundations of convex sets and functions, culminating in their applications to ML optimization.

---

## 5.1.1 Convex Sets

### Definition 5.1.1 (Convex Set)

A set $C \subseteq \mathbb{R}^n$ is **convex** if for all $x, y \in C$ and all $\lambda \in [0, 1]$:

$$\lambda x + (1 - \lambda) y \in C$$

Geometrically, a set is convex if the line segment connecting any two points in the set lies entirely within the set.

**Example 5.1.A:** Consider $C = [1, 5] \subset \mathbb{R}$ (the closed interval from 1 to 5). Pick $x = 2$, $y = 4$, $\lambda = 0.3$:

$$\lambda x + (1 - \lambda) y = 0.3 \cdot 2 + 0.7 \cdot 4 = 0.6 + 2.8 = 3.4$$

Since $3.4 \in [1, 5]$, the line segment property holds. In fact, for any $\lambda \in [0,1]$, the result $\lambda \cdot 2 + (1-\lambda) \cdot 4 = 4 - 2\lambda \in [2, 4] \subset [1, 5]$, confirming $C$ is convex.

Now consider the non-convex set $S = [0, 1] \cup [3, 4]$. Pick $x = 0.5 \in [0,1]$ and $y = 3.5 \in [3,4]$ with $\lambda = 0.5$:

$$0.5 \cdot 0.5 + 0.5 \cdot 3.5 = 2.0 \notin S$$

The midpoint falls in the gap, so $S$ is not convex.

```
Convex Set                  Non-Convex Set
    ╱────╲                      ╱──╲
   ╱  •───•─╲                  ╱    ╲ •────•
  ╱   │   │  ╲                │      │      ╲
 ╱    │   │   ╲               │      │       │
╱─────•───•────╲              ╲────╱ ╲─────╱
  Line segment                  Segment exits set
  stays inside
```

### Theorem 5.1.1 (Intersection Preserves Convexity)

If $C_1, C_2, \ldots, C_m$ are convex sets, then their intersection $\bigcap_{i=1}^m C_i$ is convex.

**Proof:** Let $x, y \in \bigcap_{i=1}^m C_i$ and $\lambda \in [0,1]$. Then $x, y \in C_i$ for all $i$. Since each $C_i$ is convex, $\lambda x + (1-\lambda)y \in C_i$ for all $i$, hence $\lambda x + (1-\lambda)y \in \bigcap_{i=1}^m C_i$. □

**Example 5.1.B:** Let $C_1 = \{(x,y) : x^2 + y^2 \leq 4\}$ (disk of radius 2) and $C_2 = \{(x,y) : x \geq 0\}$ (right halfplane). Both are convex. Their intersection is $C_1 \cap C_2 = \{(x,y) : x^2 + y^2 \leq 4, \; x \geq 0\}$ (the right half-disk), which is convex by Theorem 5.1.1.

**Note:** Union of convex sets is generally not convex.

**Example 5.1.C (Convex Combination):** A **convex combination** of points $x_1, \ldots, x_k$ is $\sum_{i=1}^k \lambda_i x_i$ where $\lambda_i \geq 0$ and $\sum_i \lambda_i = 1$.

Consider three points in $\mathbb{R}^2$: $p_1 = (0, 0)$, $p_2 = (4, 0)$, $p_3 = (0, 3)$ with weights $\lambda_1 = 0.5, \lambda_2 = 0.3, \lambda_3 = 0.2$:

$$0.5 \cdot (0,0) + 0.3 \cdot (4,0) + 0.2 \cdot (0,3) = (0 + 1.2 + 0,\; 0 + 0 + 0.6) = (1.2, 0.6)$$

This point lies inside the triangle formed by $p_1, p_2, p_3$. The set of all convex combinations of these three points is the triangle itself (their **convex hull**).

### Example 5.1.1 (Fundamental Convex Sets)

1. **Hyperplane**: $\{x \in \mathbb{R}^n : a^T x = b\}$ for $a \neq 0$
   - Geometrically: an $(n-1)$-dimensional flat subspace

2. **Halfspace**: $\{x \in \mathbb{R}^n : a^T x \leq b\}$
   - Region on one side of a hyperplane

3. **Polyhedron**: $\{x \in \mathbb{R}^n : Ax \leq b\}$
   - Intersection of finitely many halfspaces
   - Example: simplex $\{x : x \geq 0, \mathbf{1}^T x \leq 1\}$

4. **Norm Ball**: $\{x \in \mathbb{R}^n : \|x - x_0\| \leq r\}$
   - Euclidean ball: $\ell_2$-norm
   - Manhattan ball: $\ell_1$-norm

5. **Ellipsoid**: $\{x : (x-x_c)^T P^{-1} (x-x_c) \leq 1\}$ for $P \succ 0$

```
Visualization in ℝ²:

Hyperplane        Halfspace         Norm Ball (ℓ₂)
    ╱                 ╱▓▓▓▓▓             ╱────╲
   ╱                 ╱▓▓▓▓▓▓           ╱        ╲
  ╱                 ╱▓▓▓▓▓▓▓         │     •    │
 ╱                 ╱▓▓▓▓▓▓▓▓          ╲        ╱
╱                 ╱▓▓▓▓▓▓▓▓▓           ╲────╱
                 (shaded region)
```

> **ML Connection:** The feasible region in constrained optimization (e.g., SVM margin constraints, $\ell_1$ regularization with $\|w\| \leq \tau$) is often a convex set formed by intersecting halfspaces and norm balls.

---

## 5.1.2 Convex Functions

### Definition 5.1.2 (Convex Function)

A function $f: C \to \mathbb{R}$ where $C \subseteq \mathbb{R}^n$ is convex is **convex** if for all $x, y \in C$ and $\lambda \in [0,1]$:

$$f(\lambda x + (1-\lambda)y) \leq \lambda f(x) + (1-\lambda)f(y)$$

Geometrically, the line segment connecting $(x, f(x))$ and $(y, f(y))$ lies above the graph of $f$.

```
Convex Function f(x)          Non-Convex Function g(x)
       │                              │
    f(y)•─────────•λf(x)+(1-λ)f(y)   │    •────────•
       │  ╲     ╱                     │   ╱  ╲    ╱
       │   ╲   ╱                      │  ╱    ╲  ╱
       │    ╲ ╱                       │ ╱      ╲╱
       │     •f(λx+(1-λ)y)            │╱        •g(λx+(1-λ)y)
       │    ╱ ╲                       •          (chord below)
       └───────────> x                └──────────> x
        Chord above graph
```

A function is **concave** if $-f$ is convex.

**Example 5.1.D:** Verify convexity of $f(x) = x^2$ using the definition. Take $x = 1$, $y = 3$, $\lambda = 0.4$:

- **LHS:** $f(\lambda x + (1-\lambda)y) = f(0.4 \cdot 1 + 0.6 \cdot 3) = f(2.2) = 4.84$
- **RHS:** $\lambda f(x) + (1-\lambda)f(y) = 0.4 \cdot 1 + 0.6 \cdot 9 = 0.4 + 5.4 = 5.8$

Since $4.84 \leq 5.8$, the convexity condition holds. The gap $5.8 - 4.84 = 0.96 > 0$ (strict inequality since $f(x) = x^2$ is strictly convex and $x \neq y$).

### Theorem 5.1.2 (First-Order Condition)

Let $f: C \to \mathbb{R}$ be differentiable on open convex $C$. Then $f$ is convex if and only if:

$$f(y) \geq f(x) + \nabla f(x)^T (y - x) \quad \forall x, y \in C$$

**Interpretation:** The first-order Taylor approximation (tangent hyperplane) is a global underestimator.

```
       │
    f(y)•
       │╱│ Actual function
       │ │╱
    f(x)•─────────────
       │╱ Tangent (linear approx)
       │
       └────────────> x
        x            y
```

### Theorem 5.1.3 (Second-Order Condition)

Let $f: C \to \mathbb{R}$ be twice differentiable on open convex $C$. Then:

1. $f$ is convex $\Leftrightarrow$ $\nabla^2 f(x) \succeq 0$ for all $x \in C$ (positive semidefinite Hessian)
2. If $\nabla^2 f(x) \succ 0$ for all $x \in C$ (positive definite), then $f$ is strictly convex

**Proof sketch (necessity):** For convex $f$, the first-order condition gives $f(x + t v) \geq f(x) + t \nabla f(x)^T v$. Taking limits as $t \to 0^+$ from the second-order Taylor expansion yields $v^T \nabla^2 f(x) v \geq 0$. □

**Example 5.1.E (Checking Convexity via Hessian):** Consider $f(x, y) = x^2 + xy + y^2$.

**Step 1 -- Compute the Hessian:**

$$\nabla f = \begin{pmatrix} 2x + y \\ x + 2y \end{pmatrix}, \quad H = \nabla^2 f = \begin{pmatrix} 2 & 1 \\ 1 & 2 \end{pmatrix}$$

**Step 2 -- Check positive semidefiniteness** via eigenvalues of $H$:

$$\det(H - \lambda I) = (2 - \lambda)^2 - 1 = 0 \implies \lambda = 1 \text{ or } \lambda = 3$$

Both eigenvalues are positive ($\lambda_1 = 1 > 0$, $\lambda_2 = 3 > 0$), so $H \succ 0$ (positive definite). Therefore $f$ is **strictly convex** (in fact, strongly convex with $m = 1$).

### Example 5.1.2 (Verifying Convexity via Hessian)

**Quadratic form:** $f(x) = \frac{1}{2} x^T Q x + b^T x + c$

$$\nabla f(x) = Qx + b, \quad \nabla^2 f(x) = Q$$

$f$ is convex $\Leftrightarrow$ $Q \succeq 0$ (symmetric positive semidefinite).

**Exponential:** $f(x) = e^{ax}$ for $x \in \mathbb{R}$

$$f'(x) = ae^{ax}, \quad f''(x) = a^2 e^{ax} \geq 0$$

Convex for all $a \in \mathbb{R}$.

**Example 5.1.I (Epigraph):** The **epigraph** of $f: \mathbb{R}^n \to \mathbb{R}$ is $\text{epi}(f) = \{(x, t) : t \geq f(x)\} \subseteq \mathbb{R}^{n+1}$. A function is convex if and only if its epigraph is a convex set.

For $f(x) = x^2$, the epigraph is $\{(x, t) : t \geq x^2\}$ -- the region on and above the parabola:

```text
    t
    │ ▓▓▓▓▓▓▓▓▓▓▓     epi(f) = shaded region
  4 │ ▓▓▓•▓▓▓▓•▓▓     Points (−2, 4) and (2, 4) are in epi(f)
    │ ▓▓▓▓▓▓▓▓▓▓▓
  2 │ ▓▓▓▓▓▓▓▓▓▓▓     Midpoint (0, 4) is also in epi(f)
    │ ▓▓▓╱──╲▓▓▓▓     since 4 ≥ 0² = 0 ✓
  0 │──╱────╲──────
    │ ╱      ╲         Points below parabola (e.g., (1, 0))
    └──────────── x    are NOT in epi(f) since 0 < 1² = 1
```

Verify: take $(x_1, t_1) = (-2, 5)$ and $(x_2, t_2) = (2, 6)$ in $\text{epi}(f)$, $\lambda = 0.5$:
midpoint $= (0, 5.5)$. Check: $5.5 \geq 0^2 = 0$, so $(0, 5.5) \in \text{epi}(f)$.

**Example 5.1.J (Convex Conjugate):** The **convex conjugate** (Fenchel conjugate) of $f$ is $f^*(s) = \sup_{x} \{sx - f(x)\}$.

Compute for $f(x) = x^2$:

$$f^*(s) = \sup_x \{sx - x^2\}$$

Take derivative and set to zero: $s - 2x = 0 \implies x = s/2$. Substitute:

$$f^*(s) = s \cdot \frac{s}{2} - \left(\frac{s}{2}\right)^2 = \frac{s^2}{2} - \frac{s^2}{4} = \frac{s^2}{4}$$

For example, at $s = 4$: $f^*(4) = 16/4 = 4$, meaning the maximum gap between the line $4x$ and $x^2$ is $4$ (achieved at $x = 2$, where $4 \cdot 2 - 2^2 = 4$).

---

## 5.1.3 Strict and Strong Convexity

### Definition 5.1.3 (Strictly Convex)

$f$ is **strictly convex** if for all $x \neq y$ and $\lambda \in (0, 1)$:

$$f(\lambda x + (1-\lambda)y) < \lambda f(x) + (1-\lambda)f(y)$$

Strict inequality ensures uniqueness of the minimizer.

**Example 5.1.F (Strict vs. Non-strict Convexity):** Compare $f(x) = |x|$ (convex) with $g(x) = x^2$ (strictly convex).

For $f(x) = |x|$, take $x = -2$, $y = 2$, $\lambda = 0.5$:

- **LHS:** $f(0.5 \cdot (-2) + 0.5 \cdot 2) = f(0) = 0$
- **RHS:** $0.5 \cdot |-2| + 0.5 \cdot |2| = 0.5 \cdot 2 + 0.5 \cdot 2 = 2$
- Result: $0 < 2$ (strict inequality here, but...)

Now take $x = 1$, $y = 3$, $\lambda = 0.5$ (both on the same linear piece):

- **LHS:** $f(2) = 2$
- **RHS:** $0.5 \cdot 1 + 0.5 \cdot 3 = 2$
- Result: $2 = 2$ (**equality** holds!)

Since equality is achieved for $x \neq y$, $f(x) = |x|$ is convex but **not strictly convex**. For $g(x) = x^2$, equality never holds when $x \neq y$, so it is strictly convex.

### Definition 5.1.4 (Strongly Convex)

$f$ is **$m$-strongly convex** ($m > 0$) if $f(x) - \frac{m}{2}\|x\|^2$ is convex, or equivalently:

$$f(y) \geq f(x) + \nabla f(x)^T(y-x) + \frac{m}{2}\|y-x\|^2$$

Second-order characterization: $\nabla^2 f(x) \succeq mI$ for all $x$.

**Example 5.1.G (Verifying Strong Convexity):** Show $f(x) = 3x^2 + 2x + 1$ is strongly convex and find $m$.

$$f''(x) = 6 \geq mI \quad \text{for all } x$$

So $f$ is $m$-strongly convex with $m = 6$. Verify the defining inequality at $x = 0$, $y = 1$:

- **LHS:** $f(1) = 3 + 2 + 1 = 6$
- **RHS:** $f(0) + f'(0)(1 - 0) + \frac{6}{2}\|1 - 0\|^2 = 1 + 2 \cdot 1 + 3 = 6$
- Result: $6 \geq 6$ (holds with equality -- the bound is tight for quadratics!)

```
Convexity Hierarchy:

 Strongly Convex (m > 0)
         ⊂
    Strictly Convex
         ⊂
       Convex

Example:
• f(x) = x² - Strongly convex (m=2)
• f(x) = x⁴ - Strictly convex, NOT strongly convex
• f(x) = x² + x⁴ - Strongly convex (m=2 asymptotically)
```

### Theorem 5.1.4 (Strong Convexity Implies Unique Minimum)

If $f$ is strongly convex on $\mathbb{R}^n$, then $f$ has at most one global minimum.

**Proof:** Suppose $x^*, y^*$ both minimize $f$. By strong convexity:

$$f(y^*) \geq f(x^*) + \nabla f(x^*)^T(y^* - x^*) + \frac{m}{2}\|y^* - x^*\|^2$$

If $\nabla f(x^*) = 0$ (optimality), this gives $f(y^*) \geq f(x^*) + \frac{m}{2}\|y^* - x^*\|^2$. By symmetry, $f(x^*) \geq f(y^*) + \frac{m}{2}\|y^* - x^*\|^2$. Both hold only if $x^* = y^*$. □

### Theorem 5.1.5 (Convergence Rate)

For $m$-strongly convex $f$ with $L$-Lipschitz gradient, gradient descent with step size $\alpha = 2/(m+L)$ achieves:

$$\|x_t - x^*\| \leq \left(1 - \frac{m}{L}\right)^t \|x_0 - x^*\|$$

Convergence rate depends on **condition number** $\kappa = L/m$.

> **ML Connection:** Adding $\ell_2$ regularization $\lambda \|w\|^2$ to any loss makes it $\lambda$-strongly convex, guaranteeing unique solution and faster convergence. Common in ridge regression, regularized logistic regression.

---

## 5.1.4 Jensen's Inequality

### Theorem 5.1.6 (Jensen's Inequality)

Let $f: \mathbb{R}^n \to \mathbb{R}$ be convex and $X$ a random variable with $\mathbb{E}[X]$ finite. Then:

$$f(\mathbb{E}[X]) \leq \mathbb{E}[f(X)]$$

Equality holds if $f$ is linear or $X$ is deterministic.

**Proof:** By definition of convexity with weights $\lambda_i = P(X = x_i)$ (discrete case):

$$f\left(\sum_i \lambda_i x_i\right) \leq \sum_i \lambda_i f(x_i)$$

Substitute $\lambda_i x_i = \mathbb{E}[X]$ and $\sum_i \lambda_i f(x_i) = \mathbb{E}[f(X)]$. □

**Example 5.1.H (Jensen's Applied to Variance):** Let $f(x) = x^2$ (convex) and $X$ be a discrete random variable: $P(X=1) = 0.5$, $P(X=3) = 0.5$.

Jensen's inequality says $f(\mathbb{E}[X]) \leq \mathbb{E}[f(X)]$, i.e., $(\mathbb{E}[X])^2 \leq \mathbb{E}[X^2]$:

- $\mathbb{E}[X] = 0.5 \cdot 1 + 0.5 \cdot 3 = 2$
- $(\mathbb{E}[X])^2 = 4$
- $\mathbb{E}[X^2] = 0.5 \cdot 1 + 0.5 \cdot 9 = 5$
- Check: $4 \leq 5$ (confirmed!)

The gap is $\mathbb{E}[X^2] - (\mathbb{E}[X])^2 = 5 - 4 = 1 = \text{Var}(X)$. Jensen's inequality thus proves $\text{Var}(X) \geq 0$.

```
Geometric Intuition:

       │
       │    •f(x₃)
       │   ╱ ╲
       │  ╱   ╲
       │ ╱     ╲    •f(x₂)
       │╱       ╲  ╱
    •──•─────────•╱  f(x₁)
    f(𝔼[X])      𝔼[f(X)]
       │
       └────────────> x
         x₁  𝔼[X]  x₂  x₃

The function value at the mean is below
the mean of function values.
```

### Example 5.1.3 (Applications of Jensen's Inequality)

1. **Arithmetic-Geometric Mean:** $f(x) = -\log x$ is convex on $x > 0$:
   $$\log(\mathbb{E}[X]) \geq \mathbb{E}[\log X] \implies \text{AM} \geq \text{GM}$$

2. **Log-Sum Inequality:** For $f(x) = x \log x$ (convex):
   $$\left(\sum_i p_i\right) \log\left(\sum_i p_i\right) \leq \sum_i p_i \log p_i$$

3. **Variance:** $f(x) = x^2$ is convex:
   $$(\mathbb{E}[X])^2 \leq \mathbb{E}[X^2] \implies \text{Var}(X) = \mathbb{E}[X^2] - (\mathbb{E}[X])^2 \geq 0$$

> **ML Connection:** Jensen's inequality is fundamental in variational inference (ELBO derivation), proving convergence of EM algorithm, and analyzing information-theoretic loss functions.

---

## 5.1.5 Convex Optimization Problems

### Definition 5.1.5 (Convex Optimization Problem)

A problem in **standard form**:

$$\begin{align}
\min_{x \in \mathbb{R}^n} \quad & f_0(x) \\
\text{subject to} \quad & f_i(x) \leq 0, \quad i = 1, \ldots, m \\
& a_j^T x = b_j, \quad j = 1, \ldots, p
\end{align}$$

is convex if:
- Objective $f_0$ is convex
- Inequality constraint functions $f_i$ are convex
- Equality constraints are affine ($a_j^T x = b_j$)

The **feasible set** $\mathcal{F} = \{x : f_i(x) \leq 0, a_j^T x = b_j\}$ is convex.

### Theorem 5.1.7 (Global Optimality)

For a convex optimization problem, any local minimum is a global minimum.

**Proof:** Suppose $x^*$ is a local minimum but not global. Let $y$ be a global minimum with $f_0(y) < f_0(x^*)$. By convexity of $\mathcal{F}$ and $f_0$:

$$f_0(\lambda y + (1-\lambda)x^*) \leq \lambda f_0(y) + (1-\lambda)f_0(x^*) < f_0(x^*)$$

for small $\lambda > 0$. This contradicts local optimality of $x^*$. □

```
Convex vs. Non-Convex Optimization:

Convex Landscape            Non-Convex Landscape
       │                           │
       │╲                          │  ╱╲    ╱╲
       │ ╲                         │ ╱  ╲  ╱  ╲╱╲
       │  ╲                        │╱    ╲╱
       │   ╲____                   •──────────────
       │       Global Min        Local   Global
       └──────────────            Minima  Minimum
```

### Theorem 5.1.8 (First-Order Optimality Condition)

For differentiable convex $f$ over convex $C$, $x^*$ is a global minimum if and only if:

$$\nabla f(x^*)^T (y - x^*) \geq 0 \quad \forall y \in C$$

For unconstrained problems ($C = \mathbb{R}^n$), this reduces to $\nabla f(x^*) = 0$.

**KKT Conditions:** For problems with inequality constraints, the Karush-Kuhn-Tucker conditions characterize optimality:

1. **Stationarity:** $\nabla f_0(x^*) + \sum_{i=1}^m \lambda_i \nabla f_i(x^*) + \sum_{j=1}^p \nu_j a_j = 0$
2. **Primal feasibility:** $f_i(x^*) \leq 0$, $a_j^T x^* = b_j$
3. **Dual feasibility:** $\lambda_i \geq 0$
4. **Complementary slackness:** $\lambda_i f_i(x^*) = 0$

For convex problems, KKT conditions are necessary and sufficient for global optimality.

---

## 5.1.6 Common Convex Functions in Machine Learning

### 1. Mean Squared Error (MSE)

$$\mathcal{L}(w) = \frac{1}{2n} \sum_{i=1}^n (y_i - w^T x_i)^2 = \frac{1}{2n}\|y - Xw\|^2$$

**Convexity:** Composition of affine map and squared norm.

$$\nabla^2 \mathcal{L}(w) = \frac{1}{n} X^T X \succeq 0$$

Strongly convex if $X$ has full column rank ($X^T X \succ 0$).

### 2. Cross-Entropy Loss (Logistic Regression)

For binary classification with $\hat{y}_i = \sigma(w^T x_i)$:

$$\mathcal{L}(w) = -\frac{1}{n}\sum_{i=1}^n \left[y_i \log \hat{y}_i + (1-y_i)\log(1-\hat{y}_i)\right]$$

**Convexity w.r.t. $w$:** The logistic function $\sigma(z) = 1/(1+e^{-z})$ is log-concave, making the negative log-likelihood convex.

$$\nabla^2 \mathcal{L}(w) = \frac{1}{n}\sum_{i=1}^n \sigma(w^T x_i)(1-\sigma(w^T x_i)) x_i x_i^T \succeq 0$$

**Important:** Cross-entropy is convex w.r.t. predictions only when viewed as Bregman divergence. In neural networks with multiple layers, loss is non-convex w.r.t. all parameters.

### 3. Norms

- **$\ell_p$-norms:** $\|x\|_p = \left(\sum_i |x_i|^p\right)^{1/p}$ convex for $p \geq 1$
  - $\ell_1$-norm: $\|x\|_1 = \sum_i |x_i|$ (sparsity-inducing)
  - $\ell_2$-norm: $\|x\|_2 = \sqrt{\sum_i x_i^2}$ (Euclidean)
- **Matrix norms:** Frobenius $\|A\|_F$, nuclear norm $\|A\|_* = \sum_i \sigma_i(A)$ (trace norm)

### 4. Log-Sum-Exp

$$f(x) = \log\left(\sum_{i=1}^n e^{x_i}\right)$$

**Convexity:** Hessian analysis shows $\nabla^2 f(x) \succeq 0$. Used as smooth approximation to $\max\{x_1, \ldots, x_n\}$.

**ML Application:** Softmax denominator in multi-class classification.

### 5. Negative Entropy

$$f(p) = \sum_{i=1}^n p_i \log p_i$$

Convex over probability simplex $\{p : p \geq 0, \sum_i p_i = 1\}$. Used in maximum entropy methods, KL divergence.

### Example 5.1.4 (Regularized Linear Regression)

**Ridge regression:**

$$\min_w \frac{1}{2n}\|y - Xw\|^2 + \frac{\lambda}{2}\|w\|^2$$

Sum of two convex functions, strongly convex with parameter $\lambda$.

**Lasso:**

$$\min_w \frac{1}{2n}\|y - Xw\|^2 + \lambda\|w\|_1$$

Convex but non-smooth at $w_i = 0$. Induces sparsity.

> **ML Connection:** Convex losses enable global optimization. Ridge, lasso, logistic regression, SVMs all solve convex problems. Gradient descent guaranteed to find global minimum.

---

## 5.1.7 Non-Convexity in Deep Learning

### Why Neural Networks Are Non-Convex

Consider a simple 2-layer network: $f(x; W_1, W_2) = W_2 \sigma(W_1 x)$

**Loss:** $\mathcal{L}(W_1, W_2) = \frac{1}{n}\sum_i \ell(f(x_i; W_1, W_2), y_i)$

Even if $\ell$ is convex in predictions, $\mathcal{L}$ is non-convex in $(W_1, W_2)$ due to:

1. **Composition with nonlinear activations:** $\sigma(\cdot)$ breaks convexity
2. **Weight symmetry:** Permuting neurons gives same function but different parameters
3. **Multiplication of parameters:** $W_2 W_1$ creates saddle points

```
Neural Network Loss Landscape (Schematic):

Convex Loss (Linear Model)     Non-Convex Loss (Neural Net)
         │                              │
         │╲                             │    ╱╲
         │ ╲                            │   ╱  ╲  ╱╲
         │  ╲                           │  ╱    ╲╱  ╲
         │   ╲____                      │ ╱          ╲╱╲____
         │       *                      │•──────•─────────•──────
         └──────────                    └──────────────────────
           Single                         Multiple local minima,
           Global Min                     saddle points
```

### Saddle Points

**Definition:** A point $x^*$ where $\nabla f(x^*) = 0$ but $\nabla^2 f(x^*)$ has both positive and negative eigenvalues.

```
Saddle Point Geometry (2D):

    z
    │     Positive curvature
    │        direction
    │          ╱│╲
    │         ╱ │ ╲
    │        ╱  •──╲──────> y (negative curvature)
    │       ╱  ╱│╲  ╲
    │      ╱  ╱ │ ╲  ╲
    │─────────────────────> x
            Saddle point: ∇f = 0
            but not a minimum
```

**Prevalence:** In high dimensions, saddle points exponentially outnumber local minima. Most critical points in neural networks are saddle points, not local minima.

### Loss Landscape Properties

1. **No bad local minima (empirically):** Local minima tend to have similar quality to global minimum for over-parameterized networks
2. **Flat regions:** Plateaus where $\|\nabla f\| \approx 0$ but not at optimum
3. **Symmetries:** Weight permutation creates manifolds of equivalent solutions

**Mode connectivity:** Different minima often connected by paths of low loss.

```
Mode Connectivity in Loss Landscape:

    Loss
     │
     │   *────────*  Two different minima
     │  ╱          ╲ connected by low-loss path
     │ ╱            ╲
     │╱              ╲
     •────────────────•────────> Parameter space
```

### Optimization in Non-Convex Settings

**Gradient Descent Behavior:**
- Can get stuck at saddle points (rare with noise)
- Converges to critical points, not necessarily minima
- Stochastic gradient descent (SGD) helps escape saddle points via noise

**Theoretical Results:**
- SGD finds $\epsilon$-stationary point in $O(1/\epsilon^2)$ iterations
- Perturbed GD escapes saddle points in poly-time
- Over-parameterization creates benign non-convex landscape

> **ML Connection:** Despite non-convexity, deep learning optimizers (SGD, Adam) work well in practice due to over-parameterization and implicit regularization. The optimization problem is non-convex, but the landscape has favorable structure.

---

## Key Takeaways

1. **Convex sets:** Closed under intersection; include hyperplanes, halfspaces, polyhedra, norm balls
2. **Convex functions:** Characterized by chord condition, first-order (tangent underestimates), second-order (PSD Hessian)
3. **Strong convexity:** Ensures unique minimum and fast convergence (linear rate)
4. **Jensen's inequality:** Fundamental tool relating function values and expectations
5. **Convex optimization:** Local minimum = global minimum; KKT conditions for optimality
6. **ML convex losses:** MSE, logistic loss, norms, regularizers
7. **Non-convexity:** Neural networks non-convex due to composition, but landscape has favorable properties

---

## Exercises

### ★ Basic Exercises

**Exercise 5.1.1:** Prove that the intersection of two halfspaces $\{x : a_1^T x \leq b_1\}$ and $\{x : a_2^T x \leq b_2\}$ is convex.

**Exercise 5.1.2:** Show that $f(x) = |x|$ is convex on $\mathbb{R}$ using the definition.

**Exercise 5.1.3:** Verify that the MSE loss $\mathcal{L}(w) = \|y - Xw\|^2$ is convex by computing its Hessian.

**Exercise 5.1.4:** Use Jensen's inequality to show that $e^{\mathbb{E}[X]} \leq \mathbb{E}[e^X]$ for any random variable $X$.

### ★★ Intermediate Exercises

**Exercise 5.1.5:** Prove that if $f$ and $g$ are convex, then $h(x) = \max\{f(x), g(x)\}$ is convex.

**Exercise 5.1.6:** Show that the log-sum-exp function $f(x) = \log(\sum_{i=1}^n e^{x_i})$ is convex by computing its Hessian and verifying $\nabla^2 f(x) \succeq 0$.

**Exercise 5.1.7:** Consider ridge regression $\min_w \|y - Xw\|^2 + \lambda \|w\|^2$.
- (a) Compute $\nabla^2 \mathcal{L}(w)$
- (b) Show that $\mathcal{L}$ is $\lambda$-strongly convex
- (c) What is the closed-form solution?

**Exercise 5.1.8:** For logistic regression loss $\mathcal{L}(w) = -\sum_i [y_i \log \sigma(w^T x_i) + (1-y_i)\log(1-\sigma(w^T x_i))]$:
- (a) Compute $\nabla \mathcal{L}(w)$
- (b) Show the Hessian is PSD

**Exercise 5.1.9:** Let $f: \mathbb{R}^n \to \mathbb{R}$ be $m$-strongly convex and $x^* = \arg\min f(x)$. Prove:
$$f(x) - f(x^*) \geq \frac{m}{2}\|x - x^*\|^2$$

### ★★★ Advanced Exercises

**Exercise 5.1.10:** Prove the equivalence of strong convexity definitions:
- (a) $f(x) - \frac{m}{2}\|x\|^2$ is convex
- (b) $f(y) \geq f(x) + \nabla f(x)^T(y-x) + \frac{m}{2}\|y-x\|^2$
- (c) $\nabla^2 f(x) \succeq mI$

**Exercise 5.1.11:** (KKT Conditions) Consider the constrained problem:
$$\min_w \frac{1}{2}\|y - Xw\|^2 \quad \text{subject to} \quad \|w\|_1 \leq t$$

Write down the KKT conditions and interpret the complementary slackness condition geometrically.

**Exercise 5.1.12:** (Neural Network Non-Convexity) For $f(x) = w_2 \sigma(w_1 x)$ with $\sigma(z) = \max\{0, z\}$ (ReLU):
- (a) Show that $f$ is linear in $w_2$ for fixed $w_1$
- (b) Compute $\frac{\partial^2 f}{\partial w_1 \partial w_2}$ and explain why loss is non-convex
- (c) Describe the weight symmetry: how can you permute weights to get the same function?

**Exercise 5.1.13:** (Jensen's Inequality Application) Use Jensen's inequality to prove the information inequality:
$$D_{KL}(P \| Q) = \sum_i p_i \log \frac{p_i}{q_i} \geq 0$$

with equality iff $P = Q$. (Hint: Consider $f(x) = -\log x$.)

**Exercise 5.1.14:** (Landscape Analysis) Consider $f(x,y) = x^2 - y^2$.
- (a) Find all critical points
- (b) Classify each as local min, local max, or saddle point using eigenvalues of Hessian
- (c) Sketch level sets and gradient flow

---

## Related Topics

- **Chapter 5.2:** Gradient Descent and Variants (SGD, momentum, adaptive methods)
- **Chapter 5.3:** Constrained Optimization and Lagrangian Duality
- **Chapter 5.4:** Proximal Methods and Non-Smooth Optimization (for $\ell_1$ regularization)
- **Chapter 6.1:** Optimization Algorithms for Deep Learning
- **Chapter 7.3:** Support Vector Machines as Convex Optimization

---

## References and Further Reading

1. Boyd & Vandenberghe, *Convex Optimization* (2004) - Comprehensive reference, free online
2. Nocedal & Wright, *Numerical Optimization* (2006) - Optimization algorithms
3. Shalev-Shwartz & Ben-David, *Understanding Machine Learning* (2014) - Convexity in ML context
4. Goodfellow et al., *Deep Learning* (2016), Ch. 4.3 - Non-convex optimization in neural networks
5. Bubeck, *Convex Optimization: Algorithms and Complexity* (2015) - Modern perspective

---

*This chapter provides the mathematical foundation for understanding when optimization problems in ML are tractable. While modern deep learning uses non-convex optimization, the principles of convexity remain essential for analyzing simpler models, designing regularizers, and understanding convergence guarantees.*
