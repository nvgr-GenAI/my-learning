# Multivariable Calculus

Multivariable calculus is the engine of modern machine learning. Every neural network has millions of parameters, and training means computing how the loss changes with respect to *each one simultaneously*. The gradient tells you which direction to step, the Hessian tells you about the curvature of the loss landscape, and the chain rule — applied through a computational graph — is literally the backpropagation algorithm. If single-variable calculus asks "how does $y$ change when I nudge $x$?", multivariable calculus asks "how does $L$ change when I nudge *any* of a million weights at once?" This chapter is the mathematical heart of deep learning.

---

## Prerequisites

- [Differentiation](differentiation.md) — derivatives, chain rule, product rule
- [Linear Algebra — Matrices](../linear-algebra/matrices.md) — matrix operations, transpose, positive definiteness

---

## 1. Functions of Several Variables

**Definition 3.4.1 (Multivariable Function).** A *scalar-valued function of $n$ variables* is a map $f: \mathbb{R}^n \to \mathbb{R}$. We write $f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n)$.

A *vector-valued function* is a map $\mathbf{f}: \mathbb{R}^n \to \mathbb{R}^m$, where $\mathbf{f}(\mathbf{x}) = (f_1(\mathbf{x}), \ldots, f_m(\mathbf{x}))^T$.

*ML connection:* A loss function $L(\mathbf{w})$ maps the weight vector $\mathbf{w} \in \mathbb{R}^d$ (where $d$ can be billions) to a single scalar loss. A neural network layer maps $\mathbb{R}^n \to \mathbb{R}^m$ — it is a vector-valued function of its input.

---

## 2. Partial Derivatives

**Definition 3.4.2 (Partial Derivative).** The *partial derivative* of $f: \mathbb{R}^n \to \mathbb{R}$ with respect to $x_i$ is:

$$\frac{\partial f}{\partial x_i}(\mathbf{x}) = \lim_{h \to 0} \frac{f(x_1, \ldots, x_i + h, \ldots, x_n) - f(x_1, \ldots, x_i, \ldots, x_n)}{h}$$

Geometrically, $\frac{\partial f}{\partial x_i}$ measures the rate of change of $f$ when moving along the $x_i$-axis while holding all other variables fixed.

**Notation variants:** $\frac{\partial f}{\partial x_i}$, $f_{x_i}$, $\partial_i f$, $D_i f$.

**Example.** For $f(x, y) = x^2 y + \sin(xy)$:

$$\frac{\partial f}{\partial x} = 2xy + y\cos(xy) \qquad \frac{\partial f}{\partial y} = x^2 + x\cos(xy)$$

**Example 3.4.1:** Let $f(x, y) = x^2 y + 3xy^2$. Compute $\frac{\partial f}{\partial x}$ and $\frac{\partial f}{\partial y}$ at the point $(2, 1)$.

$$\frac{\partial f}{\partial x} = 2xy + 3y^2 \quad \Longrightarrow \quad \frac{\partial f}{\partial x}(2, 1) = 2(2)(1) + 3(1)^2 = 7$$

$$\frac{\partial f}{\partial y} = x^2 + 6xy \quad \Longrightarrow \quad \frac{\partial f}{\partial y}(2, 1) = (2)^2 + 6(2)(1) = 16$$

Interpretation: at $(2, 1)$, increasing $x$ by a small amount $\Delta x$ changes $f$ by approximately $7 \Delta x$, while increasing $y$ by $\Delta y$ changes $f$ by approximately $16 \Delta y$.

```
PARTIAL DERIVATIVE: SLICING A SURFACE

        z = f(x, y)
        │     ╱╲
        │   ╱    ╲
        │  ╱  ....╲........   ← slice at y = y₀
        │ ╱ .      ╲     .      (a curve in the xz-plane)
        │╱.         ╲   .
        ┼─────────────╲──────── y
       ╱.              .
      ╱  .   surface  .
     x    ...........

  ∂f/∂x at (x₀, y₀) = slope of this slice at x = x₀
```

*ML connection:* When training a model with weights $w_1, w_2, \ldots, w_d$, the partial derivative $\frac{\partial L}{\partial w_i}$ tells you how the loss changes if you adjust *only* weight $w_i$. Computing all $d$ partial derivatives gives the gradient.

**Theorem 3.4.1 (Clairaut / Schwarz).** If $f$ has continuous second partial derivatives, then the order of differentiation does not matter:

$$\frac{\partial^2 f}{\partial x_i \partial x_j} = \frac{\partial^2 f}{\partial x_j \partial x_i}$$

*Proof.* Consider $\phi(h, k) = f(a+h, b+k) - f(a+h, b) - f(a, b+k) + f(a, b)$. Applying the mean value theorem twice (first in $x$, then in $y$) gives $\phi(h,k) = hk \cdot f_{xy}(a + \theta_1 h, b + \theta_2 k)$ for some $\theta_1, \theta_2 \in (0,1)$. Applying MVT in the other order gives $\phi(h,k) = hk \cdot f_{yx}(a + \theta_3 h, b + \theta_4 k)$. Taking $h, k \to 0$ and using continuity of both mixed partials yields $f_{xy}(a,b) = f_{yx}(a,b)$. $\square$

**Example 3.4.2:** Verify Clairaut's theorem for $f(x, y) = x^2 y + 3xy^2$.

$$f_x = 2xy + 3y^2 \quad \Longrightarrow \quad f_{xy} = \frac{\partial}{\partial y}(2xy + 3y^2) = 2x + 6y$$

$$f_y = x^2 + 6xy \quad \Longrightarrow \quad f_{yx} = \frac{\partial}{\partial x}(x^2 + 6xy) = 2x + 6y$$

Indeed $f_{xy} = f_{yx} = 2x + 6y$, confirming the theorem. At $(2, 1)$: $f_{xy} = f_{yx} = 2(2) + 6(1) = 10$.

---

## 3. The Gradient

**Definition 3.4.3 (Gradient).** The *gradient* of $f: \mathbb{R}^n \to \mathbb{R}$ at $\mathbf{x}$ is the vector of all partial derivatives:

$$\nabla f(\mathbf{x}) = \begin{pmatrix} \frac{\partial f}{\partial x_1} \\ \frac{\partial f}{\partial x_2} \\ \vdots \\ \frac{\partial f}{\partial x_n} \end{pmatrix} \in \mathbb{R}^n$$

**Example 3.4.3:** Let $f(x, y) = x^2 + y^2$. Compute $\nabla f$ at the point $(3, 4)$.

$$\nabla f = \begin{pmatrix} 2x \\ 2y \end{pmatrix} \quad \Longrightarrow \quad \nabla f(3, 4) = \begin{pmatrix} 6 \\ 8 \end{pmatrix}$$

The magnitude is $\|\nabla f(3, 4)\| = \sqrt{6^2 + 8^2} = \sqrt{100} = 10$. The gradient points radially outward from the origin, which makes sense: $f = x^2 + y^2$ measures squared distance from the origin, and moving radially outward increases it fastest.

**Theorem 3.4.2 (Gradient Points in the Direction of Steepest Ascent).** Among all unit vectors $\mathbf{u} \in \mathbb{R}^n$, the directional derivative $D_{\mathbf{u}} f(\mathbf{x})$ is maximized when $\mathbf{u} = \frac{\nabla f}{\|\nabla f\|}$, and the maximum value is $\|\nabla f(\mathbf{x})\|$.

*Proof.* By Definition 3.4.4 below, $D_{\mathbf{u}} f = \nabla f \cdot \mathbf{u} = \|\nabla f\| \|\mathbf{u}\| \cos\theta = \|\nabla f\| \cos\theta$ where $\theta$ is the angle between $\nabla f$ and $\mathbf{u}$. This is maximized when $\theta = 0$, i.e., $\mathbf{u}$ points in the direction of $\nabla f$. $\square$

```
GRADIENT ON A CONTOUR PLOT

    x₂
    │
    │    ╭───╮
    │   ╭┤   ├╮
    │  ╭┤│ ● │├╮      ● = current point
    │  │││   │││      → = -∇f (steepest descent)
    │  ╰┤│ →→→→→→→
    │   ╰┤   ├╯      contour lines: f = c₁, c₂, c₃
    │    ╰───╯        (c₁ > c₂ > c₃, moving outward)
    │
    └──────────────── x₁

  Key insight: ∇f is PERPENDICULAR to contour lines
  -∇f points toward lower values (steepest descent)
```

*ML connection:* **Gradient descent** updates parameters by moving opposite to the gradient:

$$\mathbf{w}_{t+1} = \mathbf{w}_t - \eta \nabla L(\mathbf{w}_t)$$

where $\eta > 0$ is the learning rate. Since $-\nabla L$ points in the direction of steepest decrease of the loss, each step reduces $L$ (for sufficiently small $\eta$). This single equation is the foundation of training every neural network.

**Theorem 3.4.3 (Gradient is Orthogonal to Level Sets).** If $f: \mathbb{R}^n \to \mathbb{R}$ is differentiable and $\mathbf{x}_0$ lies on the level set $\{f = c\}$, then $\nabla f(\mathbf{x}_0)$ is orthogonal to the level set at $\mathbf{x}_0$.

*Proof.* Let $\mathbf{r}(t)$ be any smooth curve on the level set through $\mathbf{x}_0 = \mathbf{r}(0)$. Then $f(\mathbf{r}(t)) = c$ for all $t$. Differentiating: $\nabla f(\mathbf{r}(0)) \cdot \mathbf{r}'(0) = 0$ by the chain rule. Since $\mathbf{r}'(0)$ is tangent to the level set, $\nabla f$ is orthogonal to it. $\square$

---

## 4. Directional Derivatives

**Definition 3.4.4 (Directional Derivative).** The *directional derivative* of $f$ at $\mathbf{x}$ in the direction of a unit vector $\mathbf{u}$ is:

$$D_{\mathbf{u}} f(\mathbf{x}) = \lim_{t \to 0} \frac{f(\mathbf{x} + t\mathbf{u}) - f(\mathbf{x})}{t} = \nabla f(\mathbf{x}) \cdot \mathbf{u}$$

The second equality holds when $f$ is differentiable (a stronger condition than merely having partial derivatives).

Note that partial derivatives are special cases: $\frac{\partial f}{\partial x_i} = D_{\mathbf{e}_i} f$ where $\mathbf{e}_i$ is the $i$-th standard basis vector.

**Example 3.4.4:** Let $f(x, y) = x^2 + y^2$ with $\nabla f(3, 4) = (6, 8)$ (from Example 3.4.3). Compute the directional derivative at $(3, 4)$ in the direction of $\mathbf{v} = (1, 1)$.

First, normalize: $\mathbf{u} = \frac{\mathbf{v}}{\|\mathbf{v}\|} = \frac{1}{\sqrt{2}}(1, 1)$.

$$D_{\mathbf{u}} f(3, 4) = \nabla f(3, 4) \cdot \mathbf{u} = (6, 8) \cdot \frac{1}{\sqrt{2}}(1, 1) = \frac{6 + 8}{\sqrt{2}} = \frac{14}{\sqrt{2}} = 7\sqrt{2} \approx 9.90$$

Compare with the maximum rate $\|\nabla f(3,4)\| = 10$. Moving at $45°$ gives $99\%$ of the steepest ascent rate here because the gradient direction $(6, 8)$ is close to $(1, 1)$.

| Direction $\mathbf{u}$ | $D_{\mathbf{u}} f$ | Interpretation |
|------------------------|---------------------|----------------|
| $\mathbf{u} = \frac{\nabla f}{\|\nabla f\|}$ | $\|\nabla f\|$ (maximum) | Steepest ascent |
| $\mathbf{u} = -\frac{\nabla f}{\|\nabla f\|}$ | $-\|\nabla f\|$ (minimum) | Steepest descent |
| $\mathbf{u} \perp \nabla f$ | $0$ | Along contour line (no change) |

---

## 5. The Jacobian Matrix

**Definition 3.4.5 (Jacobian Matrix).** For a vector-valued function $\mathbf{f}: \mathbb{R}^n \to \mathbb{R}^m$, the *Jacobian* is the $m \times n$ matrix of all first-order partial derivatives:

$$J_{\mathbf{f}}(\mathbf{x}) = \begin{pmatrix} \frac{\partial f_1}{\partial x_1} & \frac{\partial f_1}{\partial x_2} & \cdots & \frac{\partial f_1}{\partial x_n} \\[4pt] \frac{\partial f_2}{\partial x_1} & \frac{\partial f_2}{\partial x_2} & \cdots & \frac{\partial f_2}{\partial x_n} \\[4pt] \vdots & \vdots & \ddots & \vdots \\[4pt] \frac{\partial f_m}{\partial x_1} & \frac{\partial f_m}{\partial x_2} & \cdots & \frac{\partial f_m}{\partial x_n} \end{pmatrix} \in \mathbb{R}^{m \times n}$$

Each row of $J$ is the gradient of one output component: row $i$ = $(\nabla f_i)^T$. For scalar-valued functions ($m = 1$), $J = (\nabla f)^T$ is a row vector.

**Example 3.4.5:** Let $\mathbf{F}(x, y) = (x^2 + y,\; xy)$. Compute the Jacobian at $(1, 2)$.

$$J_{\mathbf{F}} = \begin{pmatrix} \frac{\partial f_1}{\partial x} & \frac{\partial f_1}{\partial y} \\[4pt] \frac{\partial f_2}{\partial x} & \frac{\partial f_2}{\partial y} \end{pmatrix} = \begin{pmatrix} 2x & 1 \\ y & x \end{pmatrix} \quad \Longrightarrow \quad J_{\mathbf{F}}(1, 2) = \begin{pmatrix} 2 & 1 \\ 2 & 1 \end{pmatrix}$$

The Jacobian acts as a linear approximation: $\mathbf{F}(1 + \Delta x, 2 + \Delta y) \approx \mathbf{F}(1, 2) + J \begin{pmatrix} \Delta x \\ \Delta y \end{pmatrix} = \begin{pmatrix} 3 \\ 2 \end{pmatrix} + \begin{pmatrix} 2\Delta x + \Delta y \\ 2\Delta x + \Delta y \end{pmatrix}$.

Note: $\det J = 2(1) - 1(2) = 0$ here, so the linear map is singular at this point (both rows are identical, meaning both outputs change in the same way locally).

**Theorem 3.4.4 (Jacobian as Best Linear Approximation).** If $\mathbf{f}$ is differentiable at $\mathbf{x}$, then:

$$\mathbf{f}(\mathbf{x} + \mathbf{h}) = \mathbf{f}(\mathbf{x}) + J_{\mathbf{f}}(\mathbf{x})\,\mathbf{h} + o(\|\mathbf{h}\|)$$

The Jacobian is the linear map that best approximates $\mathbf{f}$ near $\mathbf{x}$.

*ML connection:* In **normalizing flows**, a simple distribution $\mathbf{z} \sim p_{\mathbf{z}}(\mathbf{z})$ is transformed through an invertible function $\mathbf{x} = g(\mathbf{z})$. The density of $\mathbf{x}$ uses the change-of-variables formula:

$$p_{\mathbf{x}}(\mathbf{x}) = p_{\mathbf{z}}(g^{-1}(\mathbf{x})) \cdot |\det J_{g^{-1}}(\mathbf{x})|$$

The Jacobian determinant measures how the transformation stretches or compresses volume. Normalizing flow architectures (RealNVP, Glow) are designed so that $\det J$ is cheap to compute.

---

## 6. The Hessian Matrix

**Definition 3.4.6 (Hessian Matrix).** The *Hessian* of $f: \mathbb{R}^n \to \mathbb{R}$ is the $n \times n$ matrix of second partial derivatives:

$$H_f(\mathbf{x}) = \nabla^2 f(\mathbf{x}) = \begin{pmatrix} \frac{\partial^2 f}{\partial x_1^2} & \frac{\partial^2 f}{\partial x_1 \partial x_2} & \cdots & \frac{\partial^2 f}{\partial x_1 \partial x_n} \\[4pt] \frac{\partial^2 f}{\partial x_2 \partial x_1} & \frac{\partial^2 f}{\partial x_2^2} & \cdots & \frac{\partial^2 f}{\partial x_2 \partial x_n} \\[4pt] \vdots & \vdots & \ddots & \vdots \\[4pt] \frac{\partial^2 f}{\partial x_n \partial x_1} & \frac{\partial^2 f}{\partial x_n \partial x_2} & \cdots & \frac{\partial^2 f}{\partial x_n^2} \end{pmatrix}$$

By Clairaut's theorem (Theorem 3.4.1), $H$ is **symmetric** when $f$ has continuous second derivatives.

**Example 3.4.6:** Compute the Hessian of $f(x, y) = x^3 + xy^2$ at the point $(1, 2)$.

First partial derivatives: $f_x = 3x^2 + y^2$, $f_y = 2xy$.

Second partial derivatives: $f_{xx} = 6x$, $f_{xy} = 2y$, $f_{yy} = 2x$.

$$H_f = \begin{pmatrix} 6x & 2y \\ 2y & 2x \end{pmatrix} \quad \Longrightarrow \quad H_f(1, 2) = \begin{pmatrix} 6 & 4 \\ 4 & 2 \end{pmatrix}$$

Eigenvalues: $\lambda = \frac{8 \pm \sqrt{64 - 4(12-16)}}{2} = \frac{8 \pm \sqrt{80}}{2} = 4 \pm 2\sqrt{5}$. Since $4 - 2\sqrt{5} \approx -0.47 < 0$ and $4 + 2\sqrt{5} \approx 8.47 > 0$, the Hessian is **indefinite** at $(1, 2)$, indicating saddle-like curvature.

**Theorem 3.4.5 (Second Derivative Test for Multivariable Functions).** Let $\nabla f(\mathbf{x}_0) = \mathbf{0}$ (critical point). Then:

| Hessian $H_f(\mathbf{x}_0)$ | Conclusion |
|------------------------------|------------|
| Positive definite (all eigenvalues $> 0$) | Local minimum |
| Negative definite (all eigenvalues $< 0$) | Local maximum |
| Indefinite (eigenvalues of both signs) | Saddle point |
| Semi-definite (some eigenvalue $= 0$) | Inconclusive |

*Proof (sketch for local minimum case).* By the multivariable Taylor expansion (Section 8), near the critical point: $f(\mathbf{x}_0 + \mathbf{h}) = f(\mathbf{x}_0) + \underbrace{\nabla f(\mathbf{x}_0)^T \mathbf{h}}_{= 0} + \frac{1}{2}\mathbf{h}^T H_f(\mathbf{x}_0)\mathbf{h} + o(\|\mathbf{h}\|^2)$. If $H$ is positive definite, then $\mathbf{h}^T H \mathbf{h} > 0$ for all $\mathbf{h} \neq \mathbf{0}$, so $f(\mathbf{x}_0 + \mathbf{h}) > f(\mathbf{x}_0)$ for sufficiently small $\|\mathbf{h}\|$. $\square$

**Example 3.4.7:** Find and classify the critical points of $f(x, y) = x^2 + y^2 - 2x - 4y + 5$.

**Step 1 -- Find critical points:** Set $\nabla f = \mathbf{0}$:

$$f_x = 2x - 2 = 0 \implies x = 1, \qquad f_y = 2y - 4 = 0 \implies y = 2$$

The only critical point is $(1, 2)$.

**Step 2 -- Compute the Hessian:**

$$H_f = \begin{pmatrix} f_{xx} & f_{xy} \\ f_{xy} & f_{yy} \end{pmatrix} = \begin{pmatrix} 2 & 0 \\ 0 & 2 \end{pmatrix}$$

**Step 3 -- Classify:** Eigenvalues are both $2 > 0$, so $H$ is positive definite. Therefore $(1, 2)$ is a **local minimum** with $f(1, 2) = 1 + 4 - 2 - 8 + 5 = 0$.

```
SADDLE POINT: HESSIAN IS INDEFINITE

  f(x,y) = x² - y²      Hessian = ┌  2   0 ┐
                                    └  0  -2 ┘
         z                eigenvalues: +2, -2  (indefinite)
         │   ╲  ╱
         │    ╲╱         minimum along x-axis
         │────●────      maximum along y-axis
         │    ╱╲         saddle at origin
         │   ╱  ╲
         └──────────
        (looks like a horse saddle or a Pringles chip)

  In high-dimensional loss landscapes, most critical
  points are saddle points, not local minima.
```

*ML connection:* The **Hessian eigenvalue spectrum** reveals the geometry of the loss landscape:

- **Large positive eigenvalues** = steep, narrow valleys (high curvature directions)
- **Near-zero eigenvalues** = flat directions (the loss barely changes)
- **Negative eigenvalues** = saddle point directions (can escape by moving along these)

Research shows that in large neural networks, critical points are almost always saddle points rather than local minima. The ratio of negative eigenvalues is related to the loss value at the critical point.

---

## 7. The Multivariable Chain Rule

**Theorem 3.4.6 (Chain Rule for Multivariable Functions).** If $\mathbf{g}: \mathbb{R}^n \to \mathbb{R}^m$ is differentiable at $\mathbf{x}$ and $\mathbf{f}: \mathbb{R}^m \to \mathbb{R}^k$ is differentiable at $\mathbf{g}(\mathbf{x})$, then the composition $\mathbf{f} \circ \mathbf{g}: \mathbb{R}^n \to \mathbb{R}^k$ is differentiable at $\mathbf{x}$ with:

$$J_{\mathbf{f} \circ \mathbf{g}}(\mathbf{x}) = J_{\mathbf{f}}(\mathbf{g}(\mathbf{x})) \cdot J_{\mathbf{g}}(\mathbf{x})$$

In other words: **the Jacobian of a composition is the product of the Jacobians**.

**Example 3.4.8:** Let $f(x, y) = xy$, where $x(t) = t^2$ and $y(t) = 3t$. Compute $\frac{df}{dt}$ at $t = 2$.

$$\frac{df}{dt} = \frac{\partial f}{\partial x}\frac{dx}{dt} + \frac{\partial f}{\partial y}\frac{dy}{dt} = y \cdot 2t + x \cdot 3$$

Substituting $x(2) = 4$, $y(2) = 6$:

$$\frac{df}{dt}\bigg|_{t=2} = 6 \cdot 4 + 4 \cdot 3 = 24 + 12 = 36$$

Verification: $f(t) = t^2 \cdot 3t = 3t^3$, so $\frac{df}{dt} = 9t^2 \implies 9(4) = 36$. Confirmed.

**Example 3.4.11:** (Jacobian product) Let $\mathbf{g}(x, y) = (x + y,\; xy)$ and $f(u, v) = u^2 + v$. Compute $\nabla_{(x,y)}(f \circ \mathbf{g})$ at $(1, 2)$ using the chain rule $J_{f \circ g} = J_f \cdot J_g$.

$$J_{\mathbf{g}} = \begin{pmatrix} 1 & 1 \\ y & x \end{pmatrix}\bigg|_{(1,2)} = \begin{pmatrix} 1 & 1 \\ 2 & 1 \end{pmatrix}$$

At $(1, 2)$: $\mathbf{g}(1, 2) = (3, 2)$, so $J_f = (\nabla f)^T = (2u,\; 1)\big|_{(3,2)} = (6, 1)$.

$$J_{f \circ g} = J_f \cdot J_g = (6, 1) \begin{pmatrix} 1 & 1 \\ 2 & 1 \end{pmatrix} = (6 + 2,\; 6 + 1) = (8, 7)$$

So $\nabla_{(x,y)}(f \circ \mathbf{g})(1,2) = (8, 7)^T$. This is how backpropagation works: multiply the upstream gradient $(6, 1)$ by the local Jacobian $J_g$.

For the special case of a scalar loss $L$ composed with layers:

$$\frac{\partial L}{\partial x_i} = \sum_{j=1}^m \frac{\partial L}{\partial g_j} \cdot \frac{\partial g_j}{\partial x_i}$$

*Proof.* Since $\mathbf{g}$ is differentiable at $\mathbf{x}$: $\mathbf{g}(\mathbf{x}+\mathbf{h}) = \mathbf{g}(\mathbf{x}) + J_{\mathbf{g}}(\mathbf{x})\mathbf{h} + o(\|\mathbf{h}\|)$. Let $\mathbf{k} = \mathbf{g}(\mathbf{x}+\mathbf{h}) - \mathbf{g}(\mathbf{x})$. Since $\mathbf{f}$ is differentiable at $\mathbf{g}(\mathbf{x})$: $\mathbf{f}(\mathbf{g}(\mathbf{x}) + \mathbf{k}) = \mathbf{f}(\mathbf{g}(\mathbf{x})) + J_{\mathbf{f}}(\mathbf{g}(\mathbf{x}))\mathbf{k} + o(\|\mathbf{k}\|)$. Substituting $\mathbf{k} = J_{\mathbf{g}}\mathbf{h} + o(\|\mathbf{h}\|)$ and using $\|\mathbf{k}\| = O(\|\mathbf{h}\|)$ gives $(\mathbf{f} \circ \mathbf{g})(\mathbf{x}+\mathbf{h}) = (\mathbf{f} \circ \mathbf{g})(\mathbf{x}) + J_{\mathbf{f}} J_{\mathbf{g}} \mathbf{h} + o(\|\mathbf{h}\|)$. $\square$

**This theorem IS backpropagation.** Consider a neural network as a composition:

$$L = L \circ f_L \circ f_{L-1} \circ \cdots \circ f_1$$

```
COMPUTATIONAL GRAPH AND THE CHAIN RULE (BACKPROPAGATION)

FORWARD PASS (left to right):

  x ──→ [ f₁ ] ──→ h₁ ──→ [ f₂ ] ──→ h₂ ──→ [ f₃ ] ──→ L
         W₁,b₁           W₂,b₂           loss fn

BACKWARD PASS (right to left):

  ∂L    ∂L   ∂h₂    ∂L   ∂h₂  ∂h₁    ∂L   ∂h₂  ∂h₁
  ── = ──── ────    ──── ──── ────   ──── ──── ────
  ∂h₂  ∂h₂          ∂h₂  ∂h₁         ∂h₂  ∂h₁  ∂x

  ∂L/∂L ←── ∂L/∂h₂ ←── ∂L/∂h₁ ←── ∂L/∂x
    1    J₃ᵀ          J₂ᵀ          J₁ᵀ

  At each node: multiply incoming gradient by LOCAL Jacobian
  This is just the chain rule applied systematically!
```

*ML connection:* Backpropagation computes the chain rule efficiently by storing intermediate values (forward pass) and propagating gradients backward. The key insight: you do NOT compute the full Jacobian at each layer. Instead, you compute **Jacobian-vector products** $J^T \mathbf{v}$ (where $\mathbf{v}$ is the incoming gradient), which costs $O(mn)$ rather than $O(mn \cdot n)$ for the full Jacobian. This is called **reverse-mode automatic differentiation**.

**Forward mode vs. reverse mode:**

| Property          | Forward Mode (JVP)                 | Reverse Mode (VJP)                 |
|-------------------|------------------------------------|------------------------------------|
| Computes          | $J \mathbf{v}$ (column of $J$)     | $J^T \mathbf{v}$ (row of $J$)      |
| One pass gives    | Derivative w.r.t. one input        | Derivative w.r.t. all inputs       |
| Cost for full $J$ | $O(n)$ passes ($n$ = num inputs)   | $O(m)$ passes ($m$ = num outputs)  |
| Best when         | Few inputs, many outputs           | Many inputs, few outputs (ML!)     |

Since neural networks have millions of inputs (weights) but a single scalar output (loss), reverse mode (backpropagation) computes the full gradient in **one backward pass** — an enormous computational advantage.

---

## 8. Taylor Expansion in Multiple Variables

**Theorem 3.4.7 (Multivariable Taylor Expansion).** If $f: \mathbb{R}^n \to \mathbb{R}$ is sufficiently smooth, then near $\mathbf{x}_0$:

$$f(\mathbf{x}_0 + \mathbf{h}) = f(\mathbf{x}_0) + \nabla f(\mathbf{x}_0)^T \mathbf{h} + \frac{1}{2}\mathbf{h}^T H_f(\mathbf{x}_0)\mathbf{h} + O(\|\mathbf{h}\|^3)$$

| Term | Order | Object | Information |
|------|-------|--------|-------------|
| $f(\mathbf{x}_0)$ | 0th | Scalar | Function value |
| $\nabla f^T \mathbf{h}$ | 1st | Linear in $\mathbf{h}$ | Slope (gradient) |
| $\frac{1}{2}\mathbf{h}^T H \mathbf{h}$ | 2nd | Quadratic in $\mathbf{h}$ | Curvature (Hessian) |

**Example.** For $f(x, y) = e^{x+y}$ near $(0, 0)$ where $f(0,0) = 1$, $\nabla f = (1, 1)^T$, and $H = \begin{pmatrix} 1 & 1 \\ 1 & 1 \end{pmatrix}$:

$$f(h_1, h_2) \approx 1 + h_1 + h_2 + \frac{1}{2}(h_1^2 + 2h_1 h_2 + h_2^2) = 1 + (h_1 + h_2) + \frac{1}{2}(h_1 + h_2)^2$$

**Example 3.4.9:** Use the Taylor expansion to approximate $f(x, y) = x^2 + xy$ at $(1.1, 2.05)$, expanding around $\mathbf{x}_0 = (1, 2)$.

At $\mathbf{x}_0 = (1, 2)$: $f(1, 2) = 1 + 2 = 3$. Displacement: $\mathbf{h} = (0.1, 0.05)$.

$$\nabla f = (2x + y,\; x) \implies \nabla f(1, 2) = (4, 1)$$

$$H_f = \begin{pmatrix} 2 & 1 \\ 1 & 0 \end{pmatrix}$$

$$f(1.1, 2.05) \approx 3 + (4, 1) \cdot (0.1, 0.05) + \frac{1}{2}(0.1, 0.05)\begin{pmatrix} 2 & 1 \\ 1 & 0 \end{pmatrix}\begin{pmatrix} 0.1 \\ 0.05 \end{pmatrix}$$

$$= 3 + 0.45 + \frac{1}{2}(0.1, 0.05) \cdot (0.25, 0.1) = 3 + 0.45 + \frac{1}{2}(0.03) = 3.465$$

Exact value: $(1.1)^2 + (1.1)(2.05) = 1.21 + 2.255 = 3.465$. The 2nd-order approximation is exact here because $f$ is a polynomial of degree $\leq 2$.

*ML connection:* The Taylor expansion explains why different optimizers exist:

- **Gradient descent** uses the 1st-order approximation: step in the direction $-\nabla f$
- **Newton's method** uses the 2nd-order approximation: $\mathbf{h}^* = -H^{-1} \nabla f$, solving for the minimum of the quadratic approximation
- Newton's method converges faster (quadratically vs. linearly) but requires computing and inverting the $n \times n$ Hessian — infeasible when $n$ is in the millions
- **Quasi-Newton methods** (L-BFGS, Adam) approximate the Hessian cheaply

| Optimizer         | Taylor Order    | Per-step Cost   | Convergence Rate |
|-------------------|-----------------|-----------------|------------------|
| Gradient descent  | 1st             | $O(n)$          | Linear           |
| Newton's method   | 2nd             | $O(n^3)$        | Quadratic        |
| L-BFGS            | ~2nd (approx)   | $O(n)$          | Superlinear      |
| Adam              | Diagonal 2nd    | $O(n)$          | Adaptive         |

---

## 9. Implicit Function Theorem

**Theorem 3.4.8 (Implicit Function Theorem).** Let $F: \mathbb{R}^{n+m} \to \mathbb{R}^m$ be continuously differentiable and suppose $F(\mathbf{x}_0, \mathbf{y}_0) = \mathbf{0}$. If the $m \times m$ matrix $\frac{\partial F}{\partial \mathbf{y}}(\mathbf{x}_0, \mathbf{y}_0)$ is invertible, then there exists a neighborhood of $\mathbf{x}_0$ and a unique continuously differentiable function $\mathbf{g}$ such that:

$$F(\mathbf{x}, \mathbf{g}(\mathbf{x})) = \mathbf{0} \qquad \text{with} \qquad \frac{\partial \mathbf{g}}{\partial \mathbf{x}} = -\left(\frac{\partial F}{\partial \mathbf{y}}\right)^{-1} \frac{\partial F}{\partial \mathbf{x}}$$

The key point: even when $\mathbf{y}$ is defined *implicitly* by an equation $F(\mathbf{x}, \mathbf{y}) = \mathbf{0}$, we can still differentiate $\mathbf{y}$ with respect to $\mathbf{x}$.

*ML connection:* **Implicit differentiation** appears in:

- **Implicit layers** (Deep Equilibrium Models): the output $\mathbf{z}^*$ satisfies $\mathbf{z}^* = f_\theta(\mathbf{z}^*, \mathbf{x})$ implicitly, and the IFT provides the gradient without unrolling
- **Bilevel optimization** (hyperparameter optimization, meta-learning): the inner optimization defines parameters implicitly as a function of hyperparameters
- **Constrained optimization** via Lagrange multipliers: the optimal point is implicitly defined by the KKT conditions

---

## 10. Applications in ML

### 10.1 Gradient Descent and Variants

The gradient descent update is:

$$\mathbf{w}_{t+1} = \mathbf{w}_t - \eta \nabla L(\mathbf{w}_t)$$

| Variant | Key Idea | Uses |
|---------|----------|------|
| SGD | Approximate $\nabla L$ with mini-batch gradient | Noise helps escape saddle points |
| Momentum | Accumulate past gradients: $\mathbf{v}_t = \beta \mathbf{v}_{t-1} + \nabla L$ | Accelerates through narrow valleys |
| Adam | Adaptive learning rate per parameter using 1st and 2nd moment estimates | Default choice for most deep learning |
| Newton's | Step $\mathbf{h} = -H^{-1}\nabla L$ (exact quadratic step) | Infeasible for large $n$, but motivates Adam |

**Example 3.4.10:** Perform one gradient descent step on $f(x, y) = x^2 + 4y^2$ starting from $\mathbf{w}_0 = (2, 1)$ with learning rate $\eta = 0.1$.

$$\nabla f = (2x,\; 8y) \quad \Longrightarrow \quad \nabla f(2, 1) = (4, 8)$$

$$\mathbf{w}_1 = \mathbf{w}_0 - \eta\, \nabla f(\mathbf{w}_0) = (2, 1) - 0.1 \cdot (4, 8) = (2 - 0.4,\; 1 - 0.8) = (1.6, 0.2)$$

Loss decreased: $f(2, 1) = 4 + 4 = 8 \;\to\; f(1.6, 0.2) = 2.56 + 0.16 = 2.72$. The step reduced $f$ by $66\%$ in one iteration.

### 10.2 Backpropagation as Chain Rule

For a two-layer network $L = \ell(W_2 \sigma(W_1 \mathbf{x} + \mathbf{b}_1) + \mathbf{b}_2, \mathbf{y})$:

$$\frac{\partial L}{\partial W_1} = \frac{\partial L}{\partial \mathbf{h}_2} \cdot \frac{\partial \mathbf{h}_2}{\partial \mathbf{h}_1} \cdot \frac{\partial \mathbf{h}_1}{\partial W_1}$$

where $\mathbf{h}_1 = W_1 \mathbf{x} + \mathbf{b}_1$ and $\mathbf{h}_2 = W_2 \sigma(\mathbf{h}_1) + \mathbf{b}_2$. Each factor is a local Jacobian, and the chain rule composes them.

### 10.3 Hessian for Second-Order Optimization

**Newton's method** at each step solves:

$$\mathbf{w}_{t+1} = \mathbf{w}_t - [H_L(\mathbf{w}_t)]^{-1} \nabla L(\mathbf{w}_t)$$

This converges quadratically near a minimum, but requires $O(n^2)$ storage and $O(n^3)$ computation for the Hessian. Practical alternatives:

- **Hessian-free optimization:** compute $H\mathbf{v}$ (Hessian-vector products) without forming $H$ explicitly
- **L-BFGS:** approximate $H^{-1}$ from past gradient differences
- **Natural gradient:** use the Fisher information matrix (expected Hessian under the model) instead of the true Hessian

### 10.4 Loss Landscape Analysis via Hessian Eigenvalues

The eigenvalue spectrum of $H_L$ at a critical point reveals:

| Eigenvalue Pattern | Loss Landscape Shape | Implication |
|-------------------|---------------------|-------------|
| All large positive | Sharp minimum | Poor generalization (sensitive to perturbation) |
| Mix of small positive | Flat minimum | Better generalization |
| Some negative | Saddle point | Optimizer can escape |
| Many near-zero | Flat directions | Loss is insensitive to many parameter combinations |

### 10.5 Gradient of Softmax and Cross-Entropy

The softmax function $\sigma: \mathbb{R}^n \to \mathbb{R}^n$ is defined by $\sigma_i(\mathbf{z}) = \frac{e^{z_i}}{\sum_j e^{z_j}}$.

The Jacobian of softmax is:

$$\frac{\partial \sigma_i}{\partial z_j} = \sigma_i(\delta_{ij} - \sigma_j) = \begin{cases} \sigma_i(1 - \sigma_i) & \text{if } i = j \\ -\sigma_i \sigma_j & \text{if } i \neq j \end{cases}$$

In matrix form: $J_\sigma = \text{diag}(\boldsymbol{\sigma}) - \boldsymbol{\sigma}\boldsymbol{\sigma}^T$.

When combined with cross-entropy loss $L = -\sum_i y_i \log \sigma_i(\mathbf{z})$ (where $\mathbf{y}$ is one-hot), the gradient simplifies remarkably:

$$\frac{\partial L}{\partial \mathbf{z}} = \boldsymbol{\sigma}(\mathbf{z}) - \mathbf{y}$$

This elegant result — the gradient is just "prediction minus target" — is why softmax + cross-entropy is the standard output layer for classification.

### 10.6 Jacobian in Normalizing Flows

A normalizing flow transforms a simple distribution $p_z(\mathbf{z})$ through invertible maps $\mathbf{x} = g_K \circ \cdots \circ g_1(\mathbf{z})$:

$$\log p_x(\mathbf{x}) = \log p_z(\mathbf{z}) - \sum_{k=1}^K \log |\det J_{g_k}|$$

The $\log|\det J|$ term accounts for volume change at each transformation step. Architectures like RealNVP use coupling layers where the Jacobian is triangular, making the determinant a simple product of diagonal entries.

---

## Exercises

**★ Basic**

1. Compute the gradient of $f(x, y, z) = x^2 y + yz^2 + \sin(xz)$ at the point $(1, 2, 0)$.

2. For $f(x, y) = x^2 + 4y^2$, sketch the contour lines and verify that $\nabla f$ is perpendicular to them at $(1, 1)$.

3. Compute the Jacobian of the function $\mathbf{f}(x, y) = (x^2 + y, xy, e^x)$.

4. Find the Hessian of $f(x, y) = x^3 - 3xy + y^3$ and classify the critical points.

**★★ Intermediate**

5. Prove that the gradient of a quadratic form $f(\mathbf{x}) = \frac{1}{2}\mathbf{x}^T A \mathbf{x} + \mathbf{b}^T \mathbf{x} + c$ (where $A$ is symmetric) is $\nabla f = A\mathbf{x} + \mathbf{b}$, and the Hessian is $H = A$.

6. For MSE loss $L(\mathbf{w}) = \frac{1}{n}\|X\mathbf{w} - \mathbf{y}\|^2$, compute $\nabla L$ and $H_L$. Show that setting $\nabla L = \mathbf{0}$ gives the normal equations $X^T X \mathbf{w} = X^T \mathbf{y}$.

7. Verify the softmax Jacobian formula: compute $\frac{\partial \sigma_i}{\partial z_j}$ and show that $J_\sigma = \text{diag}(\boldsymbol{\sigma}) - \boldsymbol{\sigma}\boldsymbol{\sigma}^T$.

8. A function $f: \mathbb{R}^2 \to \mathbb{R}$ has $\nabla f(1, 2) = (3, -4)$. In what direction should you move from $(1, 2)$ to increase $f$ the fastest? What is the rate of change in the direction $(1, 1)/\sqrt{2}$?

**★★★ Challenging**

9. For a two-layer neural network $f(\mathbf{x}) = W_2 \text{ReLU}(W_1 \mathbf{x})$ with scalar output, derive $\frac{\partial f}{\partial W_1}$ using the chain rule. Why does the gradient depend on the input $\mathbf{x}$?

10. Prove that for the change-of-variables formula in normalizing flows, if $g = g_2 \circ g_1$ then $\det J_g = \det J_{g_2} \cdot \det J_{g_1}$. How does this justify composing simple transformations?

11. Consider $f(x, y) = x^4 + y^4 - 4xy + 1$. Find all critical points, compute the Hessian at each, and classify them. Identify which critical point is a saddle point.

---

## Related Topics

- [Differentiation](differentiation.md) -- single-variable derivatives and the chain rule
- [Linear Algebra — Matrices](../linear-algebra/matrices.md) -- matrix operations, positive definiteness
- [Linear Algebra — Eigenvalues](../linear-algebra/eigenvalues.md) -- Hessian eigenvalue analysis
- [Optimization — Gradient Methods](../optimization/index.md) -- gradient descent and its variants
- [Taylor Series](taylor-series.md) -- detailed treatment of series expansion
- [Vector Calculus](vector-calculus.md) -- divergence, curl, and integral theorems
