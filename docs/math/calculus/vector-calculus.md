# Vector Calculus

Many of the most powerful ideas in modern generative modelling rest on the classical machinery of vector calculus. Normalizing flows compute determinants of Jacobians to track how probability density changes under smooth transformations. Score-based diffusion models learn the *divergence* of a score field to reverse a noise process. Physics-informed neural networks (PINNs) embed curl-free and divergence-free constraints directly into their loss functions. And the graph Laplacian --- the engine behind spectral graph neural networks --- is a discrete analogue of the Laplacian operator $\nabla^2$. This chapter develops the core differential operators on vector fields, states the three great integral theorems of vector calculus, and connects each concept to its role in machine learning and physics.

---

## Prerequisites

- [Multivariable Calculus](multivariable-calculus.md) --- partial derivatives, gradient, Jacobian, Hessian
- [Linear Algebra](../linear-algebra/index.md) --- vectors, matrices, determinants, cross product

---

## 1. Scalar Fields and Vector Fields

**Definition 3.5.1 (Scalar Field).** A *scalar field* is a function $f: \mathbb{R}^n \to \mathbb{R}$ that assigns a real number to each point in space.

**Definition 3.5.2 (Vector Field).** A *vector field* is a function $\mathbf{F}: \mathbb{R}^n \to \mathbb{R}^n$ that assigns a vector to each point in space. In $\mathbb{R}^3$:

$$\mathbf{F}(x, y, z) = P(x,y,z)\,\mathbf{i} + Q(x,y,z)\,\mathbf{j} + R(x,y,z)\,\mathbf{k}$$

```
SCALAR FIELD vs VECTOR FIELD  (2D cross-section)

  Scalar field f(x,y)              Vector field F(x,y)
  (temperature on a plate)         (wind velocity on a map)

    y                                y
    ^  30  35  40  42               ^  ->  ->  -->  -->
    |  25  30  35  38               |  ->  ->  ->   -->
    |  20  25  28  30               |  ^   ->  ->   ->
    |  15  18  20  22               |  ^   ^   ->   ->
    +-----------> x                 +---------------> x

  Each point has ONE number        Each point has a VECTOR
  (scalar value)                   (magnitude + direction)
```

*ML connection:* In ML, a loss landscape $L(\boldsymbol{\theta})$ is a scalar field over parameter space. The gradient $\nabla L$ converts it into a vector field --- the field that gradient descent follows. Understanding vector fields helps visualize optimization dynamics: convergence basins, saddle points, and limit cycles.

---

## 2. Gradient as a Vector Field

**Definition 3.5.3 (Gradient Field).** Given a scalar field $f: \mathbb{R}^n \to \mathbb{R}$, the *gradient* $\nabla f: \mathbb{R}^n \to \mathbb{R}^n$ is the vector field:

$$\nabla f = \left(\frac{\partial f}{\partial x_1}, \frac{\partial f}{\partial x_2}, \dots, \frac{\partial f}{\partial x_n}\right)$$

The gradient points in the direction of steepest ascent, and $\|\nabla f\|$ gives the rate of steepest increase.

**Example 3.5.1:** Let $f(x, y) = x^2 y + \sin(y)$. Compute the gradient vector field $\nabla f$ and evaluate it at $(1, 0)$.

$$\nabla f = \left(\frac{\partial f}{\partial x},\; \frac{\partial f}{\partial y}\right) = \left(2xy,\; x^2 + \cos y\right)$$

At $(1, 0)$: $\nabla f = (0,\; 2)$. The steepest ascent direction is purely in the $y$-direction, with rate $\|\nabla f\| = 2$.

**Definition 3.5.4 (Conservative Field).** A vector field $\mathbf{F}$ is *conservative* (or a *gradient field*) if there exists a scalar field $f$ such that $\mathbf{F} = \nabla f$. The function $f$ is called the *potential function*.

**Theorem 3.5.1 (Characterization of Conservative Fields).** A smooth vector field $\mathbf{F}$ on a simply connected domain is conservative if and only if $\nabla \times \mathbf{F} = \mathbf{0}$ (curl-free).

**Example 3.5.2:** Is $\mathbf{F}(x, y, z) = (2xy + z,\; x^2,\; x)$ conservative? If so, find the potential $f$.

**Check curl:** $\frac{\partial R}{\partial y} - \frac{\partial Q}{\partial z} = 0 - 0 = 0$, $\;\frac{\partial P}{\partial z} - \frac{\partial R}{\partial x} = 1 - 1 = 0$, $\;\frac{\partial Q}{\partial x} - \frac{\partial P}{\partial y} = 2x - 2x = 0$.

Since $\nabla \times \mathbf{F} = \mathbf{0}$, the field is conservative. **Find** $f$:

$$f = \int P\, dx = \int (2xy + z)\, dx = x^2 y + xz + g(y,z)$$

$$\frac{\partial f}{\partial y} = x^2 + g_y = Q = x^2 \;\Rightarrow\; g_y = 0 \;\Rightarrow\; g = h(z)$$

$$\frac{\partial f}{\partial z} = x + h'(z) = R = x \;\Rightarrow\; h'(z) = 0 \;\Rightarrow\; f(x,y,z) = x^2 y + xz + C$$

*ML connection:* Not every vector field encountered in ML is conservative. The update dynamics of GANs, for example, involve a non-conservative field (the simultaneous gradient of two competing objectives), which is why GAN training can exhibit rotational behavior rather than converging to a fixed point.

---

## 3. Divergence

**Definition 3.5.5 (Divergence).** The *divergence* of a vector field $\mathbf{F} = (F_1, F_2, \dots, F_n)$ is the scalar field:

$$\nabla \cdot \mathbf{F} = \frac{\partial F_1}{\partial x_1} + \frac{\partial F_2}{\partial x_2} + \cdots + \frac{\partial F_n}{\partial x_n} = \sum_{i=1}^n \frac{\partial F_i}{\partial x_i}$$

Divergence measures the net *outflow* of a vector field at a point --- whether the field is acting as a source ($\nabla \cdot \mathbf{F} > 0$) or a sink ($\nabla \cdot \mathbf{F} < 0$).

**Example 3.5.3:** Let $\mathbf{F}(x, y, z) = (xy,\; y^2,\; xz)$. Compute $\nabla \cdot \mathbf{F}$.

$$\nabla \cdot \mathbf{F} = \frac{\partial}{\partial x}(xy) + \frac{\partial}{\partial y}(y^2) + \frac{\partial}{\partial z}(xz) = y + 2y + x = x + 3y$$

At the point $(1, 2, 3)$: $\nabla \cdot \mathbf{F} = 1 + 6 = 7 > 0$, so the field acts as a **source** there.

```
DIVERGENCE: SOURCE vs SINK vs INCOMPRESSIBLE

  Positive divergence       Negative divergence      Zero divergence
  (source / expanding)      (sink / contracting)     (incompressible)

       \  |  /                   \  |  /
        \ | /                     v v v
    <--- (.) --->             -->  (.)  <--              --> --> -->
        / | \                     ^ ^ ^                  --> --> -->
       /  |  \                   /  |  \                 --> --> -->

  div F > 0                 div F < 0                div F = 0
  "stuff created"           "stuff absorbed"         "stuff preserved"
  e.g. heat source          e.g. drain               e.g. steady flow
```

**Theorem 3.5.2 (Divergence of a Gradient: the Laplacian).** For a scalar field $f$:

$$\nabla \cdot (\nabla f) = \nabla^2 f = \sum_{i=1}^n \frac{\partial^2 f}{\partial x_i^2}$$

This is the *Laplacian* $\Delta f = \nabla^2 f$, which measures how $f$ at a point deviates from the average of $f$ in a neighborhood.

**Example 3.5.4:** Let $f(x, y, z) = x^2 y + y^3 z + z^2$. Compute $\nabla^2 f$.

$$\frac{\partial^2 f}{\partial x^2} = 2y, \quad \frac{\partial^2 f}{\partial y^2} = 6yz, \quad \frac{\partial^2 f}{\partial z^2} = 2$$

$$\nabla^2 f = 2y + 6yz + 2$$

At $(1, 1, 1)$: $\nabla^2 f = 2 + 6 + 2 = 10 > 0$, meaning $f(1,1,1)$ is *below* the local average (like a valley bottom for the heat equation: heat flows in).

*ML connection:* In **score-based diffusion models** (e.g., DDPM, score matching), the score function $\mathbf{s}_\theta(\mathbf{x}) \approx \nabla_\mathbf{x} \log p(\mathbf{x})$ is a vector field. Denoising score matching and sliced score matching use the divergence $\nabla \cdot \mathbf{s}_\theta$ as a key quantity (via the identity in Stein's lemma). Computing or estimating this divergence efficiently is a central algorithmic challenge --- Hutchinson's trace estimator approximates $\nabla \cdot \mathbf{F} = \operatorname{tr}(\mathbf{J}_\mathbf{F})$ stochastically.

---

## 4. Curl

**Definition 3.5.6 (Curl).** For a vector field $\mathbf{F} = (P, Q, R)$ in $\mathbb{R}^3$, the *curl* is:

$$\nabla \times \mathbf{F} = \begin{vmatrix} \mathbf{i} & \mathbf{j} & \mathbf{k} \\ \frac{\partial}{\partial x} & \frac{\partial}{\partial y} & \frac{\partial}{\partial z} \\ P & Q & R \end{vmatrix} = \left(\frac{\partial R}{\partial y} - \frac{\partial Q}{\partial z}\right)\mathbf{i} + \left(\frac{\partial P}{\partial z} - \frac{\partial R}{\partial x}\right)\mathbf{j} + \left(\frac{\partial Q}{\partial x} - \frac{\partial P}{\partial y}\right)\mathbf{k}$$

Curl measures the local *rotation* (circulation density) of the field.

**Example 3.5.5:** Let $\mathbf{F}(x, y, z) = (xy,\; y^2,\; xz)$. Compute $\nabla \times \mathbf{F}$.

$$\nabla \times \mathbf{F} = \left(\frac{\partial(xz)}{\partial y} - \frac{\partial(y^2)}{\partial z},\; \frac{\partial(xy)}{\partial z} - \frac{\partial(xz)}{\partial x},\; \frac{\partial(y^2)}{\partial x} - \frac{\partial(xy)}{\partial y}\right)$$

$$= (0 - 0,\; 0 - z,\; 0 - x) = (0,\; -z,\; -x)$$

At $(1, 2, 3)$: $\nabla \times \mathbf{F} = (0, -3, -1)$. The field has nonzero curl, so it is **not conservative**.

```
CURL: ROTATION OF A VECTOR FIELD  (viewed from above, 2D slice)

  Nonzero curl                  Zero curl
  (rotational field)            (irrotational / conservative)

       ^                            ^     ^     ^
      / \                           |     |     |
     |   |                          |     |     |
     |  (.)  <-- axis of            |     |     |
     |   |       rotation           |     |     |
      \ v                           |     |     |
       ¯                            v     v     v

  curl F != 0                   curl F = 0
  "local spinning"              "no spinning"
  e.g. whirlpool                e.g. gravity field
```

**Theorem 3.5.3 (Fundamental Identities).**

1. $\nabla \times (\nabla f) = \mathbf{0}$ --- the curl of any gradient is zero
2. $\nabla \cdot (\nabla \times \mathbf{F}) = 0$ --- the divergence of any curl is zero

*Proof of (1).* The $i$-th component of $\nabla \times (\nabla f)$ involves terms like $\frac{\partial^2 f}{\partial y \partial z} - \frac{\partial^2 f}{\partial z \partial y} = 0$ by equality of mixed partials (Clairaut's theorem). The same cancellation occurs in every component. $\square$

*Proof of (2).* $\nabla \cdot (\nabla \times \mathbf{F}) = \frac{\partial}{\partial x}\!\left(\frac{\partial R}{\partial y} - \frac{\partial Q}{\partial z}\right) + \frac{\partial}{\partial y}\!\left(\frac{\partial P}{\partial z} - \frac{\partial R}{\partial x}\right) + \frac{\partial}{\partial z}\!\left(\frac{\partial Q}{\partial x} - \frac{\partial P}{\partial y}\right) = 0$ by Clairaut's theorem (all six terms cancel pairwise). $\square$

These identities encode an important structural fact: gradient fields are always curl-free, and curl fields are always divergence-free.

**Example 3.5.6:** Verify $\nabla \cdot (\nabla \times \mathbf{F}) = 0$ for $\mathbf{F} = (xy,\; y^2,\; xz)$.

From Example 3.5.5, $\nabla \times \mathbf{F} = (0,\; -z,\; -x)$. Now compute the divergence:

$$\nabla \cdot (0,\; -z,\; -x) = \frac{\partial(0)}{\partial x} + \frac{\partial(-z)}{\partial y} + \frac{\partial(-x)}{\partial z} = 0 + 0 + 0 = 0 \;\checkmark$$

---

## 5. Line Integrals and Surface Integrals

### 5.1 Line Integrals

**Definition 3.5.7 (Line Integral of a Vector Field).** Given a vector field $\mathbf{F}$ and a smooth curve $C$ parametrized by $\mathbf{r}(t)$, $a \leq t \leq b$:

$$\int_C \mathbf{F} \cdot d\mathbf{r} = \int_a^b \mathbf{F}(\mathbf{r}(t)) \cdot \mathbf{r}'(t)\, dt$$

This measures the total *work* done by $\mathbf{F}$ along $C$ --- the accumulated component of $\mathbf{F}$ in the direction of motion.

**Theorem 3.5.4 (Path Independence for Conservative Fields).** If $\mathbf{F} = \nabla f$, then:

$$\int_C \mathbf{F} \cdot d\mathbf{r} = f(\mathbf{r}(b)) - f(\mathbf{r}(a))$$

The integral depends only on the endpoints, not the path.

**Example 3.5.7:** Evaluate $\int_C \mathbf{F} \cdot d\mathbf{r}$ where $\mathbf{F} = (2x,\; 3y)$ and $C$ is the line segment from $(0, 0)$ to $(1, 2)$.

**Parametrize:** $\mathbf{r}(t) = (t,\; 2t)$, $0 \leq t \leq 1$, so $\mathbf{r}'(t) = (1,\; 2)$.

$$\mathbf{F}(\mathbf{r}(t)) = (2t,\; 6t), \quad \mathbf{F} \cdot \mathbf{r}' = 2t \cdot 1 + 6t \cdot 2 = 14t$$

$$\int_C \mathbf{F} \cdot d\mathbf{r} = \int_0^1 14t\, dt = 7t^2 \Big|_0^1 = 7$$

**Check via potential:** $\mathbf{F} = \nabla f$ where $f(x,y) = x^2 + \frac{3}{2}y^2$. So $f(1,2) - f(0,0) = 1 + 6 = 7$ $\checkmark$

### 5.2 Surface Integrals

**Definition 3.5.8 (Surface Integral of a Vector Field).** Given a vector field $\mathbf{F}$ and a surface $S$ with outward unit normal $\hat{\mathbf{n}}$:

$$\iint_S \mathbf{F} \cdot d\mathbf{S} = \iint_S \mathbf{F} \cdot \hat{\mathbf{n}}\, dA$$

This measures the total *flux* of $\mathbf{F}$ through $S$ --- the net amount of $\mathbf{F}$ passing through the surface.

**Example 3.5.8:** Compute the flux of $\mathbf{F} = (0,\; 0,\; z)$ upward through the square $S: 0 \leq x \leq 1,\; 0 \leq y \leq 1,\; z = 3$.

On this flat surface, $\hat{\mathbf{n}} = (0, 0, 1)$ (upward) and $dA = dx\,dy$. At $z = 3$: $\mathbf{F} = (0, 0, 3)$.

$$\iint_S \mathbf{F} \cdot \hat{\mathbf{n}}\, dA = \int_0^1 \int_0^1 3\, dx\, dy = 3$$

The flux is $3$, representing the uniform upward flow of magnitude $3$ through a unit square.

*ML connection:* Path independence (Theorem 3.5.4) explains why gradient descent on a convex loss always reaches the same minimum regardless of initialization path. For non-convex losses the gradient field is still conservative (since $\nabla L$ is a gradient by construction), but path independence only guarantees the *integral* is path-independent --- the discrete iterates of gradient descent can still converge to different local minima.

---

## 6. Green's Theorem

**Theorem 3.5.5 (Green's Theorem).** Let $D$ be a simply connected region in $\mathbb{R}^2$ with positively oriented boundary curve $\partial D$, and let $P, Q$ have continuous partial derivatives on $D$. Then:

$$\oint_{\partial D} (P\, dx + Q\, dy) = \iint_D \left(\frac{\partial Q}{\partial x} - \frac{\partial P}{\partial y}\right) dA$$

Green's theorem converts a line integral around a closed curve into a double integral over the enclosed region. The integrand $\frac{\partial Q}{\partial x} - \frac{\partial P}{\partial y}$ is the 2D analogue of curl.

```
GREEN'S THEOREM: BOUNDARY <--> INTERIOR

    ┌─────────────────┐
    │                 │
    │    ∂Q   ∂P     │     ∮ (P dx + Q dy)   =   ∬ (∂Q/∂x - ∂P/∂y) dA
    │    ── - ──      │      ∂D                     D
    │    ∂x   ∂y      │
    │                 │     "circulation       =   "total curl
    │       D         │      around boundary"       inside region"
    │                 │
    └──→──→──→──→──→──┘
           ∂D
    (counterclockwise)
```

**Example 3.5.9:** Verify Green's theorem for $P = y$, $Q = x^2$ over the unit square $D = [0,1] \times [0,1]$.

**Right side (double integral):** $\frac{\partial Q}{\partial x} - \frac{\partial P}{\partial y} = 2x - 1$.

$$\iint_D (2x - 1)\, dA = \int_0^1 \int_0^1 (2x - 1)\, dx\, dy = \int_0^1 [x^2 - x]_0^1\, dy = \int_0^1 0\, dy = 0$$

**Left side (line integral):** Traverse the four edges of $[0,1]^2$ counterclockwise:

- **Bottom** $(y=0, x: 0 \to 1)$: $\int_0^1 0\, dx + x^2 \cdot 0 = 0$
- **Right** $(x=1, y: 0 \to 1)$: $\int_0^1 y \cdot 0 + 1\, dy = 1$
- **Top** $(y=1, x: 1 \to 0)$: $\int_1^0 1\, dx + x^2 \cdot 0 = -1$
- **Left** $(x=0, y: 1 \to 0)$: $\int_1^0 y \cdot 0 + 0\, dy = 0$

$$\oint_{\partial D} = 0 + 1 + (-1) + 0 = 0 \;\checkmark$$

---

## 7. Stokes' Theorem

**Theorem 3.5.6 (Stokes' Theorem).** Let $S$ be an oriented smooth surface in $\mathbb{R}^3$ bounded by a simple closed curve $\partial S$, and let $\mathbf{F}$ be a smooth vector field. Then:

$$\oint_{\partial S} \mathbf{F} \cdot d\mathbf{r} = \iint_S (\nabla \times \mathbf{F}) \cdot d\mathbf{S}$$

Stokes' theorem generalizes Green's theorem to surfaces in 3D: the circulation of $\mathbf{F}$ around the boundary equals the flux of the curl through the surface.

**Theorem 3.5.7 (Green's Theorem as Special Case).** Green's theorem is Stokes' theorem applied to a flat surface in the $xy$-plane with $\mathbf{F} = (P, Q, 0)$.

*Proof sketch.* With $S$ lying in the $xy$-plane, $d\mathbf{S} = \mathbf{k}\, dA$. Then $(\nabla \times \mathbf{F}) \cdot \mathbf{k} = \frac{\partial Q}{\partial x} - \frac{\partial P}{\partial y}$, and the surface integral reduces to $\iint_D \left(\frac{\partial Q}{\partial x} - \frac{\partial P}{\partial y}\right) dA$. $\square$

---

## 8. Divergence Theorem (Gauss's Theorem)

**Theorem 3.5.8 (Divergence Theorem).** Let $V$ be a compact region in $\mathbb{R}^3$ with outward-oriented boundary surface $\partial V$, and let $\mathbf{F}$ be a smooth vector field. Then:

$$\iint_{\partial V} \mathbf{F} \cdot d\mathbf{S} = \iiint_V (\nabla \cdot \mathbf{F})\, dV$$

The total flux of $\mathbf{F}$ out through the boundary equals the integral of the divergence inside.

**Example 3.5.10:** Verify the divergence theorem for $\mathbf{F} = (x,\; y,\; z)$ over the unit sphere $x^2 + y^2 + z^2 \leq 1$.

**Volume integral:** $\nabla \cdot \mathbf{F} = 1 + 1 + 1 = 3$.

$$\iiint_V 3\, dV = 3 \cdot \frac{4\pi}{3} = 4\pi$$

**Surface integral:** On the unit sphere, $\hat{\mathbf{n}} = (x, y, z)$ and $\mathbf{F} \cdot \hat{\mathbf{n}} = x^2 + y^2 + z^2 = 1$.

$$\iint_{\partial V} \mathbf{F} \cdot d\mathbf{S} = \iint_{\partial V} 1\, dA = 4\pi \;\checkmark$$

```
DIVERGENCE THEOREM: FLUX THROUGH BOUNDARY = TOTAL DIVERGENCE INSIDE

         ╭──────────────╮
        ╱  ∇·F  ∇·F     ╲         ∬  F · dS   =   ∭ (∇·F) dV
       │   ∇·F  ∇·F  ∇·F │        ∂V                V
       │   ∇·F  ∇·F  ∇·F │
        ╲  ∇·F  ∇·F     ╱      "net outward      "total source
         ╰──────────────╯        flux through  =    strength
         ←─── F·dS ───→         boundary"          inside volume"
     (outward flux on ∂V)
```

### 8.1 The Three Theorems Unified

All three theorems share the same structure: *the integral of a derivative over a region equals the integral of the function over the boundary*.

| Theorem | Region | Boundary | "Derivative" | Equation |
|---------|--------|----------|-------------|----------|
| Green's | 2D region $D$ | Curve $\partial D$ | 2D curl | $\oint P\,dx + Q\,dy = \iint \left(\frac{\partial Q}{\partial x} - \frac{\partial P}{\partial y}\right) dA$ |
| Stokes' | Surface $S$ | Curve $\partial S$ | Curl $\nabla \times$ | $\oint \mathbf{F} \cdot d\mathbf{r} = \iint (\nabla \times \mathbf{F}) \cdot d\mathbf{S}$ |
| Divergence | Volume $V$ | Surface $\partial V$ | Divergence $\nabla \cdot$ | $\iint \mathbf{F} \cdot d\mathbf{S} = \iiint (\nabla \cdot \mathbf{F})\, dV$ |

In the language of differential forms, all three are instances of the *generalized Stokes' theorem*: $\int_{\partial \Omega} \omega = \int_\Omega d\omega$.

---

## 9. Applications in Machine Learning

### 9.1 Normalizing Flows and the Change of Variables

A normalizing flow transforms a simple base distribution $p_\mathbf{z}(\mathbf{z})$ into a complex distribution $p_\mathbf{x}(\mathbf{x})$ via an invertible map $\mathbf{x} = f(\mathbf{z})$. The change of variables formula from multivariable calculus gives:

$$p_\mathbf{x}(\mathbf{x}) = p_\mathbf{z}(f^{-1}(\mathbf{x})) \left|\det \frac{\partial f^{-1}}{\partial \mathbf{x}}\right|$$

The Jacobian determinant $\det(\partial f / \partial \mathbf{z})$ measures how the flow locally expands or contracts volume --- this is precisely the divergence theorem in infinitesimal form. Architectures like RealNVP and GLOW are designed so that this determinant is cheap to compute (triangular Jacobians give $O(n)$ cost instead of $O(n^3)$).

### 9.2 Divergence in Score-Based Diffusion Models

Score matching trains a neural network $\mathbf{s}_\theta(\mathbf{x})$ to approximate $\nabla_\mathbf{x} \log p_{\text{data}}(\mathbf{x})$. The explicit score matching objective involves $\nabla \cdot \mathbf{s}_\theta$:

$$\mathcal{L}(\theta) = \mathbb{E}_{p_{\text{data}}}\!\left[\frac{1}{2}\|\mathbf{s}_\theta(\mathbf{x})\|^2 + \nabla \cdot \mathbf{s}_\theta(\mathbf{x})\right]$$

Computing $\nabla \cdot \mathbf{s}_\theta = \sum_i \frac{\partial s_{\theta,i}}{\partial x_i} = \operatorname{tr}(\mathbf{J}_{\mathbf{s}_\theta})$ costs $O(n)$ backward passes naively. Hutchinson's estimator reduces this: $\operatorname{tr}(\mathbf{J}) = \mathbb{E}_{\mathbf{v}}[\mathbf{v}^T \mathbf{J} \mathbf{v}]$ for random $\mathbf{v}$ with $\mathbb{E}[\mathbf{v}\mathbf{v}^T] = \mathbf{I}$.

### 9.3 Physics-Informed Neural Networks (PINNs)

PINNs embed physical laws (PDEs) as soft constraints. Many PDEs are stated directly in terms of div, curl, and Laplacian:

| PDE | Vector Calculus Form | Application |
|-----|---------------------|-------------|
| Heat equation | $\frac{\partial u}{\partial t} = \alpha \nabla^2 u$ | Thermal diffusion |
| Wave equation | $\frac{\partial^2 u}{\partial t^2} = c^2 \nabla^2 u$ | Acoustics, seismology |
| Navier-Stokes | $\frac{\partial \mathbf{v}}{\partial t} + (\mathbf{v} \cdot \nabla)\mathbf{v} = -\nabla p + \nu \nabla^2 \mathbf{v}$ | Fluid dynamics |
| Maxwell's equations | $\nabla \cdot \mathbf{E} = \rho/\varepsilon_0$, $\nabla \times \mathbf{B} = \mu_0 \mathbf{J} + \mu_0\varepsilon_0 \frac{\partial \mathbf{E}}{\partial t}$ | Electromagnetics |

A PINN loss function includes terms like $\|\nabla^2 u_\theta - f\|^2$, requiring automatic differentiation to compute Laplacians and curls of the network output.

### 9.4 Graph Laplacian in Graph Neural Networks

The *graph Laplacian* $L = D - A$ (where $D$ is the degree matrix and $A$ the adjacency matrix) is a discrete analogue of the continuous Laplacian $\nabla^2$:

$$(\nabla^2 f)(x) \approx \frac{1}{|N(i)|}\sum_{j \in N(i)} (f(j) - f(i))$$

The continuous Laplacian measures how a function deviates from its local average; the graph Laplacian does the same over graph neighborhoods. Spectral GNNs (e.g., ChebNet, GCN) define convolution via the eigendecomposition $L = U\Lambda U^T$, directly using the spectrum of this discrete divergence-of-gradient operator.

---

## Summary of Differential Operators

```
OPERATOR RELATIONSHIPS (del operator ∇)

  Scalar field f ──∇──> Vector field ∇f ──∇·──> Scalar field ∇²f
      (potential)       (gradient field)          (Laplacian)

  Scalar field f ──∇──> Vector field ∇f ──∇×──> Zero vector
      (any smooth f)    (conservative)          (always! Thm 3.5.3)

  Vector field F ──∇×──> Vector field ∇×F ──∇·──> Zero scalar
      (any smooth F)     (curl field)             (always! Thm 3.5.3)


  OPERATOR TABLE (in R³):

  ┌───────────┬──────────────────────┬─────────────────┬──────────┐
  │ Operator  │ Input → Output       │ Formula         │ Measures │
  ├───────────┼──────────────────────┼─────────────────┼──────────┤
  │ ∇f        │ scalar → vector      │ (∂f/∂xᵢ)       │ slope    │
  │ ∇·F       │ vector → scalar      │ Σ ∂Fᵢ/∂xᵢ      │ source   │
  │ ∇×F       │ vector → vector      │ det[i,j,k;∂;F] │ rotation │
  │ ∇²f       │ scalar → scalar      │ Σ ∂²f/∂xᵢ²     │ curvature│
  └───────────┴──────────────────────┴─────────────────┴──────────┘
```

---

## Exercises

### Foundations (★)

**Exercise 3.5.1.** Compute the divergence and curl of $\mathbf{F}(x, y, z) = (x^2, y^2, z^2)$. Interpret the divergence physically.

**Exercise 3.5.2.** Show that the vector field $\mathbf{F} = (y, -x, 0)$ has zero divergence but nonzero curl. Sketch the field and explain why it "rotates without expanding."

**Exercise 3.5.3.** Let $f(x, y) = x^2 + y^2$. Compute $\nabla f$, then $\nabla \cdot (\nabla f)$, and verify you obtain the Laplacian $\nabla^2 f = 4$.

### Core Theory (★★)

**Exercise 3.5.4.** Verify Theorem 3.5.3 (identity 1) explicitly: take $f(x,y,z) = x^2 y + yz^3$ and compute $\nabla \times (\nabla f)$ to confirm it equals $\mathbf{0}$.

**Exercise 3.5.5.** Use the divergence theorem to evaluate $\iint_S \mathbf{F} \cdot d\mathbf{S}$ where $\mathbf{F} = (x, y, z)$ and $S$ is the unit sphere. (Hint: $\nabla \cdot \mathbf{F} = 3$, and the volume of the unit sphere is $4\pi/3$.)

**Exercise 3.5.6.** Let $\mathbf{F} = \nabla f$ be a conservative field. Use Stokes' theorem to explain why $\oint_C \mathbf{F} \cdot d\mathbf{r} = 0$ for any closed curve $C$.

### ML Applications (★★★)

**Exercise 3.5.7.** **(Score matching)** Given a neural network $\mathbf{s}_\theta: \mathbb{R}^d \to \mathbb{R}^d$, write out the divergence $\nabla \cdot \mathbf{s}_\theta$ in terms of the Jacobian $\mathbf{J}_{\mathbf{s}_\theta}$. Explain why Hutchinson's estimator $\hat{\operatorname{tr}}(\mathbf{J}) = \mathbf{v}^T \mathbf{J} \mathbf{v}$ (for random $\mathbf{v}$ with $\mathbb{E}[\mathbf{v}\mathbf{v}^T] = \mathbf{I}$) gives an unbiased estimate.

**Exercise 3.5.8.** **(Normalizing flows)** Consider the flow $f: \mathbb{R}^2 \to \mathbb{R}^2$ defined by $f(z_1, z_2) = (z_1, z_2 + \alpha z_1^2)$ for constant $\alpha$. (a) Compute the Jacobian and its determinant. (b) If $\mathbf{z} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$, write the density $p_\mathbf{x}(\mathbf{x})$. (c) Explain why $|\det \mathbf{J}| = 1$ means this flow preserves volume.

**Exercise 3.5.9.** **(Graph Laplacian)** For the path graph $1 - 2 - 3$, write out the graph Laplacian $L = D - A$ and apply it to the signal $f = (1, 3, 2)^T$. Verify that $(Lf)_i = \sum_{j \sim i}(f(i) - f(j))$ and interpret the result as a discrete "second derivative."

---

## Related Topics

- **Previous:** [Multivariable Calculus](multivariable-calculus.md) --- gradient, Jacobian, Hessian
- **Next:** [Taylor Series](taylor-series.md) --- local approximation, error bounds
- **Differential Forms:** The generalized Stokes' theorem unifies Green's, Stokes', and Divergence theorems
- **PDEs:** Heat equation, wave equation, Navier-Stokes --- all use $\nabla^2$, $\nabla \cdot$, $\nabla \times$
- **Diffusion Models:** Score matching, denoising score matching --- divergence of score function
- **Graph Neural Networks:** Spectral methods via graph Laplacian eigendecomposition
