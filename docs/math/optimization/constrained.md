# 5.3 Constrained Optimization

## Prerequisites

- **Required:** Unconstrained optimization (gradient descent, optimality conditions)
- **Required:** Linear algebra (vector spaces, projections, inner products)
- **Required:** Multivariate calculus (gradients, Jacobians, chain rule)
- **Recommended:** Convex analysis (convex sets, convex functions)
- **Recommended:** Duality theory basics

## Overview

Constrained optimization extends optimization to problems where the solution must satisfy additional requirements. These constraints arise naturally in machine learning:

- Support Vector Machines require margin constraints
- Neural network training may enforce fairness or robustness constraints
- Resource-limited learning has budget constraints
- Physics-informed models must satisfy physical laws

This chapter develops the mathematical foundations for solving constrained optimization problems, emphasizing connections to machine learning applications.

## 5.3.1 The Constrained Optimization Problem

**Definition 5.3.1** (Constrained Optimization Problem)
A constrained optimization problem has the form:

$$
\begin{align}
\min_{x \in \mathbb{R}^n} \quad & f(x) \\
\text{subject to} \quad & g_i(x) \leq 0, \quad i = 1, \ldots, m \\
& h_j(x) = 0, \quad j = 1, \ldots, p
\end{align}
$$

where:
- $f: \mathbb{R}^n \to \mathbb{R}$ is the **objective function**
- $g_i: \mathbb{R}^n \to \mathbb{R}$ are **inequality constraint functions**
- $h_j: \mathbb{R}^n \to \mathbb{R}$ are **equality constraint functions**

**Definition 5.3.2** (Feasible Set)
The **feasible set** or **constraint set** is:

$$
\mathcal{C} = \{x \in \mathbb{R}^n : g_i(x) \leq 0 \text{ for all } i, \, h_j(x) = 0 \text{ for all } j\}
$$

A point $x \in \mathcal{C}$ is called **feasible**. The problem is **feasible** if $\mathcal{C} \neq \emptyset$.

**Definition 5.3.3** (Active Constraints)
At a point $x \in \mathcal{C}$, an inequality constraint $g_i$ is **active** if $g_i(x) = 0$ and **inactive** if $g_i(x) < 0$. The **active set** at $x$ is:

$$
\mathcal{A}(x) = \{i : g_i(x) = 0\}
$$

**Example 5.3.1a:** (Active Set Identification)
Consider $\min f(x_1,x_2)$ subject to $g_1: x_1 \leq 2$, $g_2: x_2 \leq 3$, $g_3: x_1 + x_2 \leq 4$.
Rewrite as $g_1(x) = x_1 - 2 \leq 0$, $g_2(x) = x_2 - 3 \leq 0$, $g_3(x) = x_1 + x_2 - 4 \leq 0$.

At $x = (1, 2)$: $g_1 = -1 < 0$, $g_2 = -1 < 0$, $g_3 = -1 < 0$. Active set $\mathcal{A} = \emptyset$ (interior point).

At $x = (2, 2)$: $g_1 = 0$, $g_2 = -1 < 0$, $g_3 = 0$. Active set $\mathcal{A} = \{1, 3\}$ (on boundary of two constraints).

### Visualizing Constraints

```
Inequality constraint g(x) ≤ 0:
                                    g(x) = 0 (boundary)
    Feasible region                     |
    g(x) < 0                            |    Infeasible region
    ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░|    g(x) > 0
    ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░|
    ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░|
                                       ∇g(x) →
                          (points toward infeasible region)

Equality constraint h(x) = 0:

                    h(x) > 0
                        |
    ════════════════════════════════  h(x) = 0 (constraint surface)
                        |
                    h(x) < 0

              ∇h(x) ⊥ to surface
```

**Example 5.3.1** (Projection onto Ball)
Find the point in $\{x : \|x\|_2 \leq 1\}$ closest to $y \in \mathbb{R}^n$:

$$
\min_x \frac{1}{2}\|x - y\|^2 \quad \text{s.t.} \quad \|x\|^2 - 1 \leq 0
$$

Solution: If $\|y\| \leq 1$, then $x^* = y$ (constraint inactive). If $\|y\| > 1$, then $x^* = y/\|y\|$ (constraint active).

## 5.3.2 Lagrangian and Optimality Conditions

**Definition 5.3.4** (Lagrangian Function)
The **Lagrangian** $L: \mathbb{R}^n \times \mathbb{R}^m \times \mathbb{R}^p \to \mathbb{R}$ is:

$$
L(x, \lambda, \nu) = f(x) + \sum_{i=1}^m \lambda_i g_i(x) + \sum_{j=1}^p \nu_j h_j(x)
$$

where:
- $\lambda = (\lambda_1, \ldots, \lambda_m) \in \mathbb{R}^m$ are **dual variables** for inequality constraints
- $\nu = (\nu_1, \ldots, \nu_p) \in \mathbb{R}^p$ are **dual variables** for equality constraints
- $\lambda, \nu$ are also called **Lagrange multipliers**

The Lagrangian converts constrained optimization into unconstrained optimization by penalizing constraint violations.

**Example 5.3.2a:** (Lagrange Multiplier — Equality Constraint)
Minimize $f(x,y) = x^2 + y^2$ subject to $h(x,y) = x + y - 1 = 0$.

**Step 1:** Form the Lagrangian: $L(x,y,\nu) = x^2 + y^2 + \nu(x + y - 1)$.

**Step 2:** Stationarity conditions:

$$\frac{\partial L}{\partial x} = 2x + \nu = 0 \implies x = -\nu/2$$

$$\frac{\partial L}{\partial y} = 2y + \nu = 0 \implies y = -\nu/2$$

**Step 3:** Substitute into constraint: $x + y = 1 \implies -\nu/2 - \nu/2 = 1 \implies \nu = -1$.

**Step 4:** Solution: $x^* = 1/2$, $y^* = 1/2$, $\nu^* = -1$, with $f^* = 1/2$.

Geometric interpretation: the level curve $x^2 + y^2 = 1/2$ is tangent to the line $x + y = 1$ at $(1/2, 1/2)$.

### First-Order Optimality Conditions

For smooth functions, we cannot simply set $\nabla f(x^*) = 0$ because the optimum may lie on the boundary of the feasible set.

**Theorem 5.3.1** (Karush-Kuhn-Tucker Conditions)
Consider the problem in Definition 5.3.1 with $f, g_i, h_j$ continuously differentiable. If $x^*$ is a local minimum and a **constraint qualification** holds (e.g., gradients of active constraints are linearly independent), then there exist $\lambda^* \in \mathbb{R}^m$ and $\nu^* \in \mathbb{R}^p$ such that:

1. **Stationarity**: $\nabla_x L(x^*, \lambda^*, \nu^*) = 0$
   $$\nabla f(x^*) + \sum_{i=1}^m \lambda_i^* \nabla g_i(x^*) + \sum_{j=1}^p \nu_j^* \nabla h_j(x^*) = 0$$

2. **Primal Feasibility**: $g_i(x^*) \leq 0$ for all $i$, $h_j(x^*) = 0$ for all $j$

3. **Dual Feasibility**: $\lambda_i^* \geq 0$ for all $i$

4. **Complementary Slackness**: $\lambda_i^* g_i(x^*) = 0$ for all $i$

These are the **KKT conditions**. A point $(x^*, \lambda^*, \nu^*)$ satisfying them is called a **KKT point**.

**Example 5.3.2b:** (KKT Conditions — All Four Conditions)
Minimize $f(x,y) = (x-2)^2 + (y-1)^2$ subject to $g(x,y) = x + y - 1 \leq 0$.

Lagrangian: $L = (x-2)^2 + (y-1)^2 + \lambda(x + y - 1)$.

Write all four KKT conditions:

1. **Stationarity:** $2(x-2) + \lambda = 0$ and $2(y-1) + \lambda = 0$
2. **Primal feasibility:** $x + y - 1 \leq 0$
3. **Dual feasibility:** $\lambda \geq 0$
4. **Complementary slackness:** $\lambda(x + y - 1) = 0$

**Case 1:** $\lambda = 0$ (constraint inactive). Then $x = 2, y = 1$. Check: $g(2,1) = 2 > 0$ — **infeasible**. Reject.

**Case 2:** $\lambda > 0$, so $x + y = 1$ (constraint active). From stationarity: $x - 2 = y - 1$, so $x = y + 1$. Substituting: $(y+1) + y = 1 \implies y = 0, x = 1$. Then $\lambda = 2(2-1) = 2 > 0$. All KKT conditions satisfied.

**Solution:** $x^* = 1, y^* = 0, \lambda^* = 2$, with $f^* = 2$.

**Interpretation of KKT Conditions:**

```
Stationarity: ∇f(x*) lies in cone spanned by constraint gradients

    ∇g₁(x*)
         ↑  ∖
         |    ∖  -∇f(x*)
         |      ↙
    ─────●─────────  x*
         |      ∇g₂(x*)
         |    ↗

    ∇f(x*) = -λ₁*∇g₁(x*) - λ₂*∇g₂(x*)  (λᵢ ≥ 0)

Complementary slackness:
    - If gᵢ(x*) < 0 (inactive), then λᵢ* = 0 (multiplier is zero)
    - If λᵢ* > 0, then gᵢ(x*) = 0 (constraint is active)
```

**Theorem 5.3.2** (Sufficient Conditions for Convex Problems)
If $f$ and $g_i$ are convex, $h_j$ are affine, and $(x^*, \lambda^*, \nu^*)$ satisfy the KKT conditions, then $x^*$ is a **global minimum**.

This is why convexity is so powerful: KKT conditions become both necessary and sufficient.

**Example 5.3.2c:** (Complementary Slackness with Two Inequality Constraints)
Minimize $f(x,y) = -x - y$ subject to $g_1: x + 2y - 4 \leq 0$ and $g_2: x - 1 \leq 0$.

Lagrangian: $L = -x - y + \lambda_1(x + 2y - 4) + \lambda_2(x - 1)$.

Stationarity: $-1 + \lambda_1 + \lambda_2 = 0$ and $-1 + 2\lambda_1 = 0$.

From the second equation: $\lambda_1 = 1/2$. From the first: $\lambda_2 = 1/2$.

Complementary slackness: $\lambda_1(x + 2y - 4) = 0$ and $\lambda_2(x - 1) = 0$.

Since $\lambda_1 = 1/2 > 0$: $x + 2y = 4$. Since $\lambda_2 = 1/2 > 0$: $x = 1$.

Solving: $x^* = 1, y^* = 3/2$, with $f^* = -5/2$. Both constraints are active, and both multipliers are positive, consistent with complementary slackness.

**Example 5.3.2** (Equality-Constrained Quadratic)
Minimize $\frac{1}{2}x^T Q x + c^T x$ subject to $Ax = b$, where $Q \succ 0$.

Lagrangian: $L(x, \nu) = \frac{1}{2}x^T Q x + c^T x + \nu^T(Ax - b)$

KKT conditions:
- Stationarity: $Qx^* + c + A^T\nu^* = 0$
- Primal feasibility: $Ax^* = b$

Solution: Solve the linear system
$$\begin{bmatrix} Q & A^T \\ A & 0 \end{bmatrix} \begin{bmatrix} x^* \\ \nu^* \end{bmatrix} = \begin{bmatrix} -c \\ b \end{bmatrix}$$

This system appears in many ML applications, including constrained least squares.

## 5.3.3 Lagrange Duality

**Definition 5.3.5** (Lagrange Dual Function)
The **Lagrange dual function** $g: \mathbb{R}^m \times \mathbb{R}^p \to \mathbb{R} \cup \{-\infty\}$ is:

$$
g(\lambda, \nu) = \inf_{x \in \mathbb{R}^n} L(x, \lambda, \nu) = \inf_x \left[ f(x) + \sum_{i=1}^m \lambda_i g_i(x) + \sum_{j=1}^p \nu_j h_j(x) \right]
$$

For each fixed $(\lambda, \nu)$, $g(\lambda, \nu)$ is the minimum value of the Lagrangian over $x$.

**Theorem 5.3.3** (Weak Duality)
If $x$ is primal feasible and $\lambda \geq 0$, then:

$$
g(\lambda, \nu) \leq f(x)
$$

In particular, if $p^* = \inf_x f(x)$ subject to constraints, then $g(\lambda, \nu) \leq p^*$ for all $\lambda \geq 0, \nu$.

*Proof sketch:* For feasible $x$: $g(\lambda, \nu) \leq L(x, \lambda, \nu) = f(x) + \sum \lambda_i g_i(x) + \sum \nu_j h_j(x) \leq f(x)$ since $\lambda_i \geq 0, g_i(x) \leq 0$, and $h_j(x) = 0$. □

**Example 5.3.3a:** (Weak Duality — Numerical Verification)
Consider $\min x^2$ subject to $x \geq 2$ (i.e., $g(x) = 2 - x \leq 0$).

Primal optimal: $x^* = 2$, $p^* = 4$.

Lagrangian: $L(x, \lambda) = x^2 + \lambda(2-x)$. Dual function: $g(\lambda) = \inf_x [x^2 + \lambda(2-x)]$.

Setting $\frac{\partial L}{\partial x} = 2x - \lambda = 0$ gives $x = \lambda/2$. Substituting:

$$g(\lambda) = (\lambda/2)^2 + \lambda(2 - \lambda/2) = -\lambda^2/4 + 2\lambda$$

For any $\lambda \geq 0$: e.g., $\lambda = 1$ gives $g(1) = -1/4 + 2 = 7/4 \leq 4 = p^*$. Weak duality holds.

**Definition 5.3.6** (Dual Problem)
The **Lagrange dual problem** is:

$$
\begin{align}
\max_{\lambda, \nu} \quad & g(\lambda, \nu) \\
\text{subject to} \quad & \lambda \geq 0
\end{align}
$$

The optimal value $d^* = \sup_{\lambda \geq 0, \nu} g(\lambda, \nu)$ is the **dual optimal value**.

**Example 5.3.3b:** (Forming the Lagrange Dual Problem)
Continuing from Example 5.3.3a: $\min x^2$ subject to $2 - x \leq 0$.

Dual function: $g(\lambda) = -\lambda^2/4 + 2\lambda$ (derived above).

The **dual problem** is: $\max_{\lambda \geq 0} \; -\lambda^2/4 + 2\lambda$.

Solving: $g'(\lambda) = -\lambda/2 + 2 = 0 \implies \lambda^* = 4$.

Dual optimal value: $d^* = g(4) = -16/4 + 8 = 4$.

Since $p^* = 4 = d^*$, the duality gap is zero. This is expected: the problem is convex and Slater's condition holds (e.g., $x = 3$ is strictly feasible).

**Definition 5.3.7** (Duality Gap)
The difference $p^* - d^*$ is the **duality gap**. By weak duality, $p^* \geq d^*$ always holds.

**Definition 5.3.8** (Strong Duality)
**Strong duality** holds if $p^* = d^*$ (zero duality gap).

**Theorem 5.3.4** (Slater's Condition)
If the primal problem is convex ($f, g_i$ convex, $h_j$ affine) and there exists a **strictly feasible point** $x_0$ (i.e., $g_i(x_0) < 0$ for all $i$ and $h_j(x_0) = 0$ for all $j$), then strong duality holds.

**Example 5.3.3c:** (Strong Duality Verification for a Convex Problem)
Minimize $f(x) = x^2$ subject to $g(x) = 1 - x \leq 0$ (i.e., $x \geq 1$).

**Check Slater's condition:** $f$ is convex, $g$ is affine. Take $x_0 = 5$: $g(5) = -4 < 0$ (strictly feasible). Slater's condition holds.

**Primal:** The minimum of $x^2$ for $x \geq 1$ is $p^* = 1$ at $x^* = 1$.

**Dual:** Lagrangian $L(x,\lambda) = x^2 + \lambda(1-x)$. Minimizing over $x$: $2x - \lambda = 0 \implies x = \lambda/2$.

$$g(\lambda) = (\lambda/2)^2 + \lambda(1 - \lambda/2) = -\lambda^2/4 + \lambda$$

Maximizing: $g'(\lambda) = -\lambda/2 + 1 = 0 \implies \lambda^* = 2$. $d^* = g(2) = -1 + 2 = 1$.

**Result:** $p^* = 1 = d^*$. Zero duality gap confirms strong duality, as predicted by Slater's condition.

### Duality Structure

```
Primal Problem                  Dual Problem
─────────────                   ─────────────
min f(x)                        max g(λ,ν)
s.t. gᵢ(x) ≤ 0                  s.t. λ ≥ 0
     hⱼ(x) = 0

Optimal value: p*               Optimal value: d*

        Weak duality: d* ≤ p*

        Strong duality: d* = p*  (under Slater's condition for convex problems)

        ┌─────────────────────────┐
        │  Complementary Slackness│
        │  λᵢ* gᵢ(x*) = 0         │
        │  Links primal/dual      │
        └─────────────────────────┘
```

**Example 5.3.3** (Dual of Linear Program)
Primal: $\min c^T x$ s.t. $Ax \geq b$, $x \geq 0$.

Lagrangian: $L(x, \lambda, \mu) = c^T x + \lambda^T(b - Ax) + \mu^T(-x)$

For $g(\lambda, \mu)$ to be finite, need $c - A^T\lambda - \mu = 0$, so $\mu = c - A^T\lambda$.

Dual: $\max b^T \lambda$ s.t. $A^T\lambda \leq c$, $\lambda \geq 0$.

Strong duality holds for LPs (no Slater condition needed).

**Example 5.3.3d:** (Linear Programming — Graphical Solution)
Minimize $f(x_1,x_2) = -3x_1 - 2x_2$ subject to $x_1 + x_2 \leq 4$, $x_1 \leq 3$, $x_2 \leq 3$, $x_1, x_2 \geq 0$.

**Step 1:** Identify corner points of the feasible region: $(0,0)$, $(3,0)$, $(3,1)$, $(1,3)$, $(0,3)$.

**Step 2:** Evaluate objective at each vertex:

| Vertex  | $f$ value |
|---------|-----------|
| $(0,0)$ | $0$       |
| $(3,0)$ | $-9$      |
| $(3,1)$ | $-11$     |
| $(1,3)$ | $-9$      |
| $(0,3)$ | $-6$      |

**Step 3:** Minimum is $f^* = -11$ at $(x_1^*, x_2^*) = (3, 1)$.

The constraints $x_1 \leq 3$ and $x_1 + x_2 \leq 4$ are active at the optimum. The gradient $\nabla f = (-3, -2)$ is a non-negative combination of the active constraint normals $(1,0)$ and $(1,1)$, confirming KKT conditions.

## 5.3.4 Support Vector Machines

Support Vector Machines exemplify constrained optimization in ML. We derive the hard-margin SVM as a constrained quadratic program.

### Problem Setup

Given training data $(x_1, y_1), \ldots, (x_N, y_N)$ where $x_i \in \mathbb{R}^d$ and $y_i \in \{-1, +1\}$, find a hyperplane that separates the classes with maximum margin.

**Definition 5.3.9** (Separating Hyperplane)
A hyperplane $\{x : w^T x + b = 0\}$ **separates** the data if:

$$
y_i(w^T x_i + b) > 0 \quad \text{for all } i = 1, \ldots, N
$$

**Definition 5.3.10** (Margin)
The **margin** is the minimum distance from any point to the hyperplane:

$$
\gamma = \min_{i=1,\ldots,N} \frac{|w^T x_i + b|}{\|w\|} = \frac{1}{\|w\|} \min_i y_i(w^T x_i + b)
$$

### Hard-Margin SVM (Primal)

**Problem 5.3.1** (Hard-Margin SVM Primal)

$$
\begin{align}
\min_{w, b} \quad & \frac{1}{2}\|w\|^2 \\
\text{subject to} \quad & y_i(w^T x_i + b) \geq 1, \quad i = 1, \ldots, N
\end{align}
$$

This maximizes the margin $1/\|w\|$ by minimizing $\|w\|^2$ subject to correct classification with margin at least $1/\|w\|$.

### Geometric Intuition

```
                                w
        Class +1                ↑
          +        +         Separating
              +          +   hyperplane
    ─────────────────────────────────
            margin = 2/||w||
    ─────────────────────────────────
        -        -
            -         -       Class -1
              -

Support vectors: points on margin boundary
    y(w·x + b) = 1  (constraint is active)

Other points: y(w·x + b) > 1  (constraint inactive)
```

### Dual Formulation

Lagrangian: $L(w, b, \alpha) = \frac{1}{2}\|w\|^2 - \sum_{i=1}^N \alpha_i [y_i(w^T x_i + b) - 1]$

where $\alpha_i \geq 0$ are dual variables.

Stationarity conditions:
$$\nabla_w L = w - \sum_{i=1}^N \alpha_i y_i x_i = 0 \quad \Rightarrow \quad w = \sum_{i=1}^N \alpha_i y_i x_i$$

$$\frac{\partial L}{\partial b} = -\sum_{i=1}^N \alpha_i y_i = 0$$

Substituting back into $L$:

$$g(\alpha) = \sum_{i=1}^N \alpha_i - \frac{1}{2} \sum_{i,j=1}^N \alpha_i \alpha_j y_i y_j x_i^T x_j$$

**Problem 5.3.2** (Hard-Margin SVM Dual)

$$
\begin{align}
\max_{\alpha} \quad & \sum_{i=1}^N \alpha_i - \frac{1}{2} \sum_{i,j=1}^N \alpha_i \alpha_j y_i y_j x_i^T x_j \\
\text{subject to} \quad & \sum_{i=1}^N \alpha_i y_i = 0 \\
& \alpha_i \geq 0, \quad i = 1, \ldots, N
\end{align}
$$

This is a convex quadratic program in $\alpha \in \mathbb{R}^N$.

### Support Vectors and Sparsity

By complementary slackness: $\alpha_i [y_i(w^T x_i + b) - 1] = 0$.

- If $\alpha_i > 0$, then $y_i(w^T x_i + b) = 1$ (point is on margin boundary)
- If $y_i(w^T x_i + b) > 1$, then $\alpha_i = 0$ (point does not contribute to $w$)

Points with $\alpha_i > 0$ are **support vectors** — they "support" the hyperplane. Typically, only a small fraction are support vectors, giving sparsity.

### Kernel Trick Connection

The dual depends on data only through inner products $x_i^T x_j$. We can replace these with a kernel $k(x_i, x_j)$ to obtain nonlinear classifiers:

$$g(\alpha) = \sum_{i=1}^N \alpha_i - \frac{1}{2} \sum_{i,j=1}^N \alpha_i \alpha_j y_i y_j k(x_i, x_j)$$

Decision function: $f(x) = \sum_{i=1}^N \alpha_i y_i k(x_i, x) + b$

This is computed using only the support vectors where $\alpha_i > 0$.

## 5.3.5 Projected Gradient Descent

For constraints that are simple (e.g., box constraints, norm constraints), we can use **projected gradient descent**.

**Definition 5.3.11** (Projection onto Convex Set)
The **projection** of $y \in \mathbb{R}^n$ onto a closed convex set $\mathcal{C}$ is:

$$
\Pi_{\mathcal{C}}(y) = \arg\min_{x \in \mathcal{C}} \|x - y\|^2
$$

The projection is unique and satisfies $(y - \Pi_{\mathcal{C}}(y))^T(x - \Pi_{\mathcal{C}}(y)) \leq 0$ for all $x \in \mathcal{C}$.

**Algorithm 5.3.1** (Projected Gradient Descent)
```
Input: initial x₀ ∈ C, step sizes {αₖ}
For k = 0, 1, 2, ...
    zₖ₊₁ = xₖ - αₖ ∇f(xₖ)      (gradient step)
    xₖ₊₁ = Π_C(zₖ₊₁)            (project back to feasible set)
```

**Example 5.3.4a:** (Projected Gradient Descent — One Step with Box Constraint)
Minimize $f(x,y) = (x-3)^2 + (y-3)^2$ subject to $0 \leq x \leq 2$, $0 \leq y \leq 2$ (box constraint).

Starting at $x_0 = (1, 1)$, step size $\alpha = 0.5$.

**Gradient step:** $\nabla f(1,1) = (2(1-3), 2(1-3)) = (-4, -4)$.

$$z_1 = x_0 - \alpha \nabla f(x_0) = (1, 1) - 0.5(-4, -4) = (3, 3)$$

**Projection onto box $[0,2]^2$:** Clip each coordinate:

$$x_1 = \Pi_{[0,2]^2}(3, 3) = (\min(2, \max(0, 3)), \min(2, \max(0, 3))) = (2, 2)$$

The gradient pointed toward the unconstrained optimum $(3,3)$, but the projection pulled the iterate back to the nearest feasible point $(2,2)$, which is the constrained optimum.

**Theorem 5.3.5** (Convergence of Projected Gradient Descent)
If $f$ is convex and $L$-smooth, $\mathcal{C}$ is closed and convex, and $\alpha_k = 1/L$, then:

$$f(x_k) - f(x^*) \leq \frac{L\|x_0 - x^*\|^2}{2k}$$

### Common Projections

| Constraint Set | Projection Formula |
|----------------|-------------------|
| Box: $x \in [a, b]^n$ | $[\Pi_{\mathcal{C}}(y)]_i = \max(a_i, \min(y_i, b_i))$ |
| $\ell_2$ ball: $\|x\| \leq r$ | $\Pi_{\mathcal{C}}(y) = \begin{cases} y & \text{if } \|y\| \leq r \\ ry/\|y\| & \text{otherwise} \end{cases}$ |
| Simplex: $\sum x_i = 1, x \geq 0$ | (closed form via sorting, $O(n \log n)$) |
| $\ell_1$ ball: $\|x\|_1 \leq r$ | (soft thresholding) |

**Example 5.3.4** (Constrained Least Squares)
Minimize $\|Ax - b\|^2$ subject to $\|x\| \leq r$.

Projected gradient descent:
$$x_{k+1} = \Pi_{\|x\| \leq r}(x_k - \alpha A^T(Ax_k - b))$$

where the projection is $\Pi(y) = \min(1, r/\|y\|) \cdot y$.

### ML Application: Adversarial Robustness

Training robust models via:
$$\min_\theta \max_{\|\delta\| \leq \epsilon} \ell(f_\theta(x + \delta), y)$$

Inner maximization uses projected gradient ascent on $\delta$ with projection onto $\ell_\infty$ ball.

## 5.3.6 Penalty and Barrier Methods

When projections are expensive, penalty and barrier methods convert constrained problems to sequences of unconstrained problems.

### Penalty Methods

**Idea:** Add a penalty term to the objective that penalizes constraint violations.

**Quadratic Penalty Method:**
Solve a sequence of unconstrained problems:
$$\min_x f(x) + \frac{\rho}{2} \sum_{i=1}^m \max(0, g_i(x))^2 + \frac{\rho}{2} \sum_{j=1}^p h_j(x)^2$$

as $\rho \to \infty$. The penalties force feasibility in the limit.

**Example 5.3.5a:** (Quadratic Penalty Method — One Iteration)
Minimize $f(x) = x$ subject to $g(x) = 1 - x \leq 0$ (i.e., $x \geq 1$). True solution: $x^* = 1$.

Penalized problem: $\min_x \; x + \frac{\rho}{2}\max(0, 1-x)^2$.

For $x < 1$: $\phi(x) = x + \frac{\rho}{2}(1-x)^2$. Setting $\phi'(x) = 1 - \rho(1-x) = 0$ gives $x = 1 - 1/\rho$.

| $\rho$       | $x_\rho$ | $g(x_\rho)$             |
|--------------|----------|-------------------------|
| $1$          | $0$      | $1$ (infeasible)        |
| $10$         | $0.9$    | $0.1$ (nearly feasible) |
| $100$        | $0.99$   | $0.01$ (close)          |
| $\to\infty$  | $\to 1$  | $\to 0$ (feasible)      |

As $\rho$ increases, the solution approaches feasibility and $x_\rho \to x^* = 1$.

**Definition 5.3.12** (Augmented Lagrangian)
The **augmented Lagrangian** combines Lagrangian and penalty:

$$L_\rho(x, \lambda, \nu) = f(x) + \sum_{i=1}^m \lambda_i g_i(x) + \sum_{j=1}^p \nu_j h_j(x) + \frac{\rho}{2}\sum_{i=1}^m \max(0, g_i(x))^2 + \frac{\rho}{2}\sum_{j=1}^p h_j(x)^2$$

**Algorithm 5.3.2** (Augmented Lagrangian Method)
```
Input: initial x₀, λ⁰, ν⁰, ρ₀
For k = 0, 1, 2, ...
    xₖ = arg minₓ Lₚₖ(x, λᵏ, νᵏ)
    λᵢᵏ⁺¹ = max(0, λᵢᵏ + ρₖ gᵢ(xₖ))       (dual update)
    νⱼᵏ⁺¹ = νⱼᵏ + ρₖ hⱼ(xₖ)
    ρₖ₊₁ = γρₖ                          (increase penalty)
```

The augmented Lagrangian is better conditioned than pure penalty for moderate $\rho$.

### Barrier Methods

**Idea:** Add a barrier that approaches $\infty$ as $x$ approaches the boundary of the feasible set, keeping iterates strictly feasible.

**Logarithmic Barrier:**
For $g_i(x) < 0$, replace constraint with barrier:
$$\min_x f(x) - \mu \sum_{i=1}^m \log(-g_i(x))$$

where $\mu > 0$ is the **barrier parameter**.

As $\mu \to 0^+$, the solution approaches the constrained optimum.

**Algorithm 5.3.3** (Barrier Method / Interior Point Method)
```
Input: initial x₀ (strictly feasible), μ₀ > 0, γ ∈ (0,1)
For k = 0, 1, 2, ...
    xₖ = arg minₓ {f(x) - μₖ Σᵢ log(-gᵢ(x))}
    μₖ₊₁ = γμₖ                              (decrease barrier)
```

**Example 5.3.5b:** (Barrier Method — One Iteration)
Minimize $f(x) = -x$ subject to $x \leq 2$ (i.e., $g(x) = x - 2 \leq 0$). True solution: $x^* = 2$.

Barrier subproblem: $\min_x \; -x - \mu \log(2 - x)$ for $x < 2$.

Setting derivative to zero: $-1 + \frac{\mu}{2 - x} = 0 \implies x(\mu) = 2 - \mu$.

| $\mu$   | $x(\mu)$ | $f(x(\mu))$ |
|---------|----------|--------------|
| $1$     | $1$      | $-1$         |
| $0.1$   | $1.9$    | $-1.9$       |
| $0.01$  | $1.99$   | $-1.99$      |
| $\to 0$ | $\to 2$  | $\to -2$     |

As $\mu \to 0$, the barrier solution converges to the constrained optimum $x^* = 2$, $f^* = -2$.

**Barrier Effect:**
```
Objective: f(x)

Barrier: -μ log(-g(x))
         │
         │      ╱
         │    ╱
         │  ╱
    ─────│╱──────────────
         │              g(x) = 0 (boundary)
       ∞ │

As x → boundary, barrier → ∞, preventing infeasibility
```

Interior point methods are the foundation of modern convex optimization solvers (e.g., for linear and quadratic programming).

### ML Application: Log-Barrier for Non-Negative Constraints

Non-negative matrix factorization:
$$\min_{W \geq 0, H \geq 0} \|X - WH\|_F^2$$

Can apply barrier: $\min_{W,H} \|X - WH\|_F^2 - \mu \sum_{ij} (\log W_{ij} + \log H_{ij})$

## 5.3.7 ML Applications

### Constrained Neural Network Training

**Weight Norm Constraints:**
Control model capacity by constraining $\|W\|_F \leq r$ for weight matrices. Use projected gradient descent after each update.

**Fairness Constraints:**
Enforce demographic parity:
$$|P(\hat{Y}=1 | A=0) - P(\hat{Y}=1 | A=1)| \leq \epsilon$$

where $A$ is a protected attribute. Formulated as constrained optimization with fairness as inequality constraint.

**Physics-Informed Neural Networks:**
Enforce PDE constraints using Lagrange multipliers:
$$\mathcal{L} = \mathcal{L}_{\text{data}} + \lambda \mathcal{L}_{\text{PDE}}$$

where $\mathcal{L}_{\text{PDE}}$ measures PDE residual.

### Constrained Generation

**Conditional VAE:**
Latent space constraints using KKT conditions in the variational distribution.

**Constrained GAN:**
Generate samples satisfying constraints (e.g., physical feasibility) via augmented Lagrangian in generator loss.

### Safe Reinforcement Learning

Policy optimization with safety constraints:
$$\max_\pi \mathbb{E}[R] \quad \text{s.t.} \quad \mathbb{E}[C] \leq d$$

where $C$ is a cost function (e.g., collision). Solved using constrained policy optimization (CPO) with trust region methods.

### Robust Optimization

Adversarial training formulated as constrained optimization:
$$\min_\theta \mathbb{E}_{(x,y)} \max_{\|\delta\|_p \leq \epsilon} \ell(f_\theta(x + \delta), y)$$

Inner max is constrained optimization; often solved via projected gradient ascent.

## Summary

Constrained optimization extends ML's reach to problems requiring satisfaction of explicit requirements:

- **KKT Conditions** characterize solutions via Lagrange multipliers
- **Duality Theory** provides bounds and alternative formulations (crucial for SVMs)
- **SVMs** exemplify ML's deep connection to constrained quadratic programming
- **Projected Gradient Descent** handles simple constraints efficiently
- **Penalty/Barrier Methods** convert constrained to unconstrained problems

These tools enable principled handling of fairness, safety, physical laws, and resource limits in modern machine learning.

## Related Topics

- **5.1 Unconstrained Optimization**: Gradient descent, Newton's method (prerequisites)
- **5.2 Stochastic Optimization**: Extending SGD to constrained settings
- **6.3 Convex Optimization**: Efficient algorithms for convex constraints
- **8.2 Kernel Methods**: Kernel trick in SVMs and dual formulations
- **9.4 Adversarial Robustness**: Constrained optimization in adversarial training

## Exercises

### ★ Basic Exercises

**Exercise 5.3.1**
Consider $\min_{x \in \mathbb{R}^2} x_1^2 + x_2^2$ subject to $x_1 + x_2 = 1$.

(a) Write the Lagrangian.
(b) Find the KKT conditions.
(c) Solve for $x^*$.

**Exercise 5.3.2**
Project $y = (3, -1, 2)$ onto:
(a) The $\ell_2$ ball $\{x : \|x\| \leq 1\}$
(b) The box $\{x : 0 \leq x_i \leq 1\}$
(c) The simplex $\{x : \sum x_i = 1, x_i \geq 0\}$ (numerically)

**Exercise 5.3.3**
For $\min x_1 + x_2$ subject to $x_1^2 + x_2^2 \leq 1$, verify that strong duality holds by computing both primal and dual optimal values.

### ★★ Intermediate Exercises

**Exercise 5.3.4** (Soft-Margin SVM)
Extend the hard-margin SVM to allow misclassification:
$$\min_{w,b,\xi} \frac{1}{2}\|w\|^2 + C\sum_{i=1}^N \xi_i \quad \text{s.t.} \quad y_i(w^T x_i + b) \geq 1 - \xi_i, \; \xi_i \geq 0$$

(a) Write the Lagrangian with dual variables $\alpha_i, \mu_i$.
(b) Derive the dual problem.
(c) What are the support vectors in terms of $\alpha_i^*$?

**Exercise 5.3.5** (Projected Gradient for $\ell_1$ Ball)
Derive the projection formula for the $\ell_1$ ball $\{x : \|x\|_1 \leq r\}$. (Hint: Use soft thresholding with a threshold $\theta$ chosen so $\|\text{sign}(y) \max(|y| - \theta, 0)\|_1 = r$.)

**Exercise 5.3.6**
For the equality-constrained problem $\min \frac{1}{2}\|x\|^2$ subject to $Ax = b$ where $A$ has full row rank:

(a) Solve using KKT conditions.
(b) Show the solution is $x^* = A^T(AA^T)^{-1}b$.
(c) Interpret geometrically as a projection.

### ★★★ Advanced Exercises

**Exercise 5.3.7** (Dual of $\ell_1$-Regularized Problem)
Consider $\min_x \|Ax - b\|^2 + \lambda\|x\|_1$.

(a) Reformulate as a constrained problem with $\|x\|_1 \leq t$.
(b) Derive the Lagrange dual.
(c) Relate the dual to the Lasso solution path.

**Exercise 5.3.8** (ADMM for Constrained Learning)
The Alternating Direction Method of Multipliers (ADMM) solves:
$$\min_{x,z} f(x) + g(z) \quad \text{s.t.} \quad Ax + Bz = c$$

(a) Write the augmented Lagrangian.
(b) Derive the ADMM update steps.
(c) Apply to $\ell_1$-regularized logistic regression: $\min_w \sum_i \log(1 + e^{-y_i w^T x_i}) + \lambda\|w\|_1$.

**Exercise 5.3.9** (Slater's Condition)
Show that Slater's condition fails for the problem:
$$\min x_1 \quad \text{s.t.} \quad x_2^2 \leq 0, \; x_1^2 + x_2^2 \leq 1$$

Does strong duality hold? Verify by computing $p^*$ and $d^*$.

**Exercise 5.3.10** (Barrier Method Convergence)
For the logarithmic barrier method applied to LP:
$$\min c^T x \quad \text{s.t.} \quad Ax \leq b$$

(a) Write the barrier problem for parameter $\mu$.
(b) Show the barrier solution satisfies $\nabla f(x(\mu)) + \sum_i \frac{\mu}{b_i - a_i^T x} a_i = 0$.
(c) Prove that as $\mu \to 0$, the barrier path approaches the optimal solution.

---

*Next: Chapter 5.4 - Stochastic Gradient Methods*
