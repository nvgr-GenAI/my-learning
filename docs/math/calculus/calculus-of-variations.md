# Calculus of Variations

Classical calculus optimizes functions: find the $x$ that minimizes $f(x)$. Calculus of variations optimizes over *function spaces*: find the function $y(x)$ that minimizes a functional $J[y]$. This leap from finite-dimensional to infinite-dimensional optimization is exactly what modern ML demands. Variational inference asks: among all probability distributions $q$, which one best approximates the true posterior $p(\mathbf{z}|\mathbf{x})$? VAEs, optimal transport, and maximum entropy methods all reduce to variational problems --- optimizing not over parameter vectors, but over entire functions and distributions.

---

## Prerequisites

- [Multivariable Calculus](multivariable-calculus.md) --- partial derivatives, gradient, chain rule
- [Integration](integration.md) --- definite integrals, integration by parts

---

## 1. Functionals

**Definition 3.7.1 (Functional).** A *functional* is a map $J: \mathcal{F} \to \mathbb{R}$ that assigns a real number to each function $y$ in some function space $\mathcal{F}$. We write $J[y]$ to distinguish functionals (square brackets) from ordinary functions (parentheses).

**Canonical form.** Most functionals in applications take the *integral form*:

$$J[y] = \int_a^b L(x, y(x), y'(x)) \, dx$$

where $L$ is the *Lagrangian* (a function of three variables) and $y$ ranges over sufficiently smooth functions satisfying boundary conditions $y(a) = y_a$, $y(b) = y_b$.

```
FUNCTIONS vs FUNCTIONALS

  Function:  f: R^n → R            Functional:  J: {functions} → R
  Input: a vector x                Input: an entire function y(x)
  Output: a number f(x)            Output: a number J[y]

  Example:                         Example:
  f(x) = x² + 1                   J[y] = ∫₀¹ (y'(x))² dx
  f(3) = 10                        J[sin] = ∫₀¹ cos²(x) dx ≈ 0.73

  Minimize: find x*               Minimize: find y*(x)
  (finite-dimensional)             (infinite-dimensional)
```

**Example 3.7.1:** Consider the functional $J[y] = \int_0^1 (y'(x))^2 \, dx$ with $y(0) = 0$, $y(1) = 1$. Evaluate $J$ for two candidate functions.

*Candidate 1:* $y(x) = x$ (straight line). Then $y' = 1$, so

$$J[x] = \int_0^1 1^2 \, dx = 1$$

*Candidate 2:* $y(x) = x^2$ (parabola). Then $y' = 2x$, so

$$J[x^2] = \int_0^1 (2x)^2 \, dx = \int_0^1 4x^2 \, dx = \frac{4}{3} \approx 1.33$$

The straight line gives a smaller value. (We will prove shortly that it is the minimizer.)

*ML connection:* Every loss function in machine learning over function classes is a functional. The expected risk $R[f] = \mathbb{E}_{(x,y)}[\ell(f(x), y)]$ maps a predictor function $f$ to a real number. Empirical risk minimization restricts the search to a parametric family, but the variational viewpoint reveals the true problem structure.

---

## 2. Variations and the First Variation

**Definition 3.7.2 (Variation).** Let $y$ be a candidate function and $\eta$ a smooth function with $\eta(a) = \eta(b) = 0$ (it vanishes at the boundary). The *variation* of $y$ in direction $\eta$ is the perturbed function $y + \varepsilon\eta$ for small $\varepsilon$.

**Definition 3.7.3 (First Variation / Gateaux Derivative).** The *first variation* of $J$ at $y$ in direction $\eta$ is:

$$\delta J[y; \eta] = \lim_{\varepsilon \to 0} \frac{J[y + \varepsilon\eta] - J[y]}{\varepsilon} = \frac{d}{d\varepsilon}\bigg|_{\varepsilon=0} J[y + \varepsilon\eta]$$

This is the infinite-dimensional analogue of the directional derivative.

**Theorem 3.7.1 (First Variation of Integral Functionals).** For $J[y] = \int_a^b L(x, y, y') \, dx$, the first variation is:

$$\delta J[y; \eta] = \int_a^b \left( \frac{\partial L}{\partial y}\eta + \frac{\partial L}{\partial y'}\eta' \right) dx$$

*Proof.* Substitute $y + \varepsilon\eta$ and differentiate under the integral sign:

$$\frac{d}{d\varepsilon}\bigg|_{\varepsilon=0} \int_a^b L(x, y + \varepsilon\eta, y' + \varepsilon\eta') \, dx = \int_a^b \left( \frac{\partial L}{\partial y}\eta + \frac{\partial L}{\partial y'}\eta' \right) dx$$

by the chain rule applied to $L$ as a function of $\varepsilon$. $\square$

**Example 3.7.2:** Compute $\delta J[y; \eta]$ for $J[y] = \int_0^1 (y')^2 \, dx$ at $y(x) = x$ in direction $\eta(x) = \sin(\pi x)$.

Here $L(x, y, y') = (y')^2$, so $\frac{\partial L}{\partial y} = 0$ and $\frac{\partial L}{\partial y'} = 2y'$. At $y = x$ we have $y' = 1$, so:

$$\delta J = \int_0^1 \left(0 \cdot \sin(\pi x) + 2(1) \cdot \pi\cos(\pi x)\right) dx = 2\pi \int_0^1 \cos(\pi x) \, dx$$

$$= 2\pi \left[\frac{\sin(\pi x)}{\pi}\right]_0^1 = 2\pi \cdot \frac{0 - 0}{\pi} = 0$$

The first variation vanishes, consistent with $y = x$ being an extremum of $J$.

---

## 3. The Euler-Lagrange Equation

**Theorem 3.7.2 (Euler-Lagrange Equation).** If $y^*$ is an extremum of $J[y] = \int_a^b L(x, y, y') \, dx$ among functions with $y(a) = y_a$, $y(b) = y_b$, then $y^*$ satisfies:

$$\frac{\partial L}{\partial y} - \frac{d}{dx}\frac{\partial L}{\partial y'} = 0$$

This is a second-order ODE for $y^*(x)$ --- the variational analogue of $\nabla f = 0$.

*Proof.* At an extremum, $\delta J[y^*; \eta] = 0$ for all admissible $\eta$. From Theorem 3.7.1:

$$\int_a^b \left( \frac{\partial L}{\partial y}\eta + \frac{\partial L}{\partial y'}\eta' \right) dx = 0$$

Integrate the second term by parts:

$$\int_a^b \frac{\partial L}{\partial y'}\eta' \, dx = \left[\frac{\partial L}{\partial y'}\eta\right]_a^b - \int_a^b \frac{d}{dx}\frac{\partial L}{\partial y'}\eta \, dx$$

The boundary term vanishes since $\eta(a) = \eta(b) = 0$. Therefore:

$$\int_a^b \left( \frac{\partial L}{\partial y} - \frac{d}{dx}\frac{\partial L}{\partial y'} \right)\eta \, dx = 0 \quad \text{for all } \eta$$

By the *fundamental lemma of calculus of variations* (if $\int_a^b f(x)\eta(x)\,dx = 0$ for all smooth $\eta$ vanishing at the endpoints, then $f \equiv 0$), the integrand must be identically zero. $\square$

```
ANALOGY: FINITE vs INFINITE OPTIMIZATION

  Finite-dimensional              Calculus of Variations
  ─────────────────               ──────────────────────
  Variable: x ∈ R^n               Variable: y(x) ∈ function space
  Objective: f(x)                 Objective: J[y] = ∫ L dx
  Necessary condition:            Necessary condition:
    ∇f = 0                          ∂L/∂y − d/dx(∂L/∂y') = 0
  Gives: system of equations      Gives: differential equation
  Second order: Hessian           Second order: second variation
```

**Example 3.7.3 (Shortest Path).** Find the curve $y(x)$ of shortest arc length from $(0, 0)$ to $(1, 1)$.

The arc length functional is $J[y] = \int_0^1 \sqrt{1 + (y')^2} \, dx$, so $L = \sqrt{1 + (y')^2}$.

Compute the partials: $\frac{\partial L}{\partial y} = 0$ and $\frac{\partial L}{\partial y'} = \frac{y'}{\sqrt{1+(y')^2}}$.

The Euler-Lagrange equation gives $\frac{d}{dx}\frac{y'}{\sqrt{1+(y')^2}} = 0$, so $\frac{y'}{\sqrt{1+(y')^2}} = C$ (constant).

Solving: $y' = \text{const}$, meaning $y(x) = ax + b$. With $y(0)=0, y(1)=1$: $y(x) = x$.

The shortest path is the straight line, as expected.

**Example 3.7.4 (Geodesic on a Flat Plane).** A *geodesic* minimizes arc length. On the Euclidean plane with a parameterized curve $(x(t), y(t))$ for $t \in [0, 1]$, the length functional is:

$$J[x, y] = \int_0^1 \sqrt{(\dot{x})^2 + (\dot{y})^2} \, dt$$

Applying Euler-Lagrange to each coordinate with $L = \sqrt{\dot{x}^2 + \dot{y}^2}$:

$$\frac{d}{dt}\frac{\dot{x}}{\sqrt{\dot{x}^2+\dot{y}^2}} = 0, \quad \frac{d}{dt}\frac{\dot{y}}{\sqrt{\dot{x}^2+\dot{y}^2}} = 0$$

Both expressions are constant, so $\dot{x}/\dot{y} = \text{const}$, giving $y = mx + b$ --- a straight line. Geodesics on flat space are straight lines, confirming that the Euclidean metric produces no curvature effects.

*ML connection:* The Euler-Lagrange equation is the theoretical backbone of variational inference. When we seek the distribution $q^*$ that minimizes a divergence functional $D[q \| p]$, the Euler-Lagrange equation yields the optimal $q^*$ in closed form (when it exists). This is how we derive that the optimal variational distribution for mean-field VI is a product of exponential family members.

---

## 4. The Brachistochrone Problem

**Problem.** Find the curve $y(x)$ connecting $(0, 0)$ to $(x_1, y_1)$ along which a frictionless bead slides fastest under gravity.

The travel time is:

$$T[y] = \int_0^{x_1} \frac{\sqrt{1 + (y')^2}}{\sqrt{2gy}} \, dx$$

Here $L(x, y, y') = \frac{\sqrt{1 + (y')^2}}{\sqrt{2gy}}$.

**Applying Euler-Lagrange:** After computation (using the Beltrami identity since $L$ does not depend on $x$ explicitly), the solution is a *cycloid*:

$$x(\theta) = r(\theta - \sin\theta), \quad y(\theta) = r(1 - \cos\theta)$$

```
THE BRACHISTOCHRONE — FASTEST DESCENT

  Start (0,0)
    ●─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ● Straight line (not fastest!)
    \                     /
     \   Cycloid         /
      \  (fastest!)     /
       ╲              ╱
        ╲            ╱
         ╲          ╱  Circular arc
          ╲        ╱
           ●──────●
           End (x₁, y₁)

  The cycloid beats ALL other curves
  (even though it's longer than the straight line)
```

This 1696 problem, posed by Johann Bernoulli, launched the field.

**Example 3.7.5 (Brachistochrone Setup).** Set up the functional for a bead sliding from $(0, 0)$ to $(1, 1)$ (taking $g = 10 \, \text{m/s}^2$).

By conservation of energy, the speed at height $y$ (measured downward) is $v = \sqrt{2gy} = \sqrt{20y}$. An infinitesimal arc element has length $ds = \sqrt{1+(y')^2}\,dx$, so the travel time is:

$$T[y] = \int_0^1 \frac{\sqrt{1 + (y')^2}}{\sqrt{20y}} \, dx$$

Here $L(y, y') = \frac{\sqrt{1+(y')^2}}{\sqrt{20y}}$. Since $L$ has no explicit $x$-dependence, we can apply the Beltrami identity $L - y'\frac{\partial L}{\partial y'} = C$ to reduce to a first-order ODE. For the straight line $y = x$: $T_{\text{line}} = \int_0^1 \frac{\sqrt{2}}{\sqrt{20x}}\,dx = \frac{\sqrt{2}}{\sqrt{20}} \cdot 2\sqrt{x}\big|_0^1 = \frac{2\sqrt{2}}{\sqrt{20}} \approx 0.632\,\text{s}$. The cycloid solution achieves a shorter time.

---

## 5. Constrained Variational Problems

### 5.1 Isoperimetric Problems

**Definition 3.7.4 (Isoperimetric Problem).** Minimize (or maximize) $J[y] = \int_a^b L(x, y, y') \, dx$ subject to a constraint $K[y] = \int_a^b G(x, y, y') \, dx = C$.

**Theorem 3.7.3 (Isoperimetric Theorem).** The constrained extremum satisfies the Euler-Lagrange equation for the *augmented Lagrangian*:

$$\tilde{L} = L + \lambda G$$

where $\lambda \in \mathbb{R}$ is a Lagrange multiplier determined by the constraint $K[y] = C$.

*Proof.* Define $\Phi(\varepsilon_1, \varepsilon_2) = J[y + \varepsilon_1\eta_1 + \varepsilon_2\eta_2]$ and $\Psi(\varepsilon_1, \varepsilon_2) = K[y + \varepsilon_1\eta_1 + \varepsilon_2\eta_2]$. At the constrained extremum, $\nabla\Phi = \lambda\nabla\Psi$ at $(\varepsilon_1, \varepsilon_2) = (0,0)$ by finite-dimensional Lagrange multipliers. Choosing $\eta_1, \eta_2$ appropriately and applying the fundamental lemma gives the Euler-Lagrange equation for $L + \lambda G$. $\square$

**Example 3.7.6 (Isoperimetric Setup).** Among curves $y(x)$ from $(0,0)$ to $(1,0)$, find the one that maximizes the enclosed area $A[y] = \int_0^1 y \, dx$ subject to fixed arc length $\ell[y] = \int_0^1 \sqrt{1+(y')^2}\,dx = \pi/2$.

Set $J[y] = \int_0^1 y\,dx$ (maximize) and $K[y] = \int_0^1 \sqrt{1+(y')^2}\,dx = \pi/2$ (constraint). Form the augmented Lagrangian:

$$\tilde{L} = y + \lambda\sqrt{1+(y')^2}$$

The Euler-Lagrange equation for $\tilde{L}$ gives: $1 - \lambda\frac{d}{dx}\frac{y'}{\sqrt{1+(y')^2}} = 0$. This is the equation for a circular arc. With $\lambda = 1/2$ and the boundary conditions, the solution is the semicircle $y(x) = \sqrt{(1/4) - (x - 1/2)^2}$ of radius $r = 1/2$, enclosing area $A = \pi/8 \approx 0.393$.

**Classical example.** Among all closed curves of perimeter $P$, find the one enclosing maximum area. Solution: the circle (this is the original *isoperimetric inequality*: $A \leq P^2/(4\pi)$).

### 5.2 Multiple Constraints

For $m$ constraints $K_i[y] = C_i$, the augmented Lagrangian is $\tilde{L} = L + \sum_{i=1}^m \lambda_i G_i$, and we solve the Euler-Lagrange equation for $\tilde{L}$ with $m$ multipliers.

---

## 6. Applications in Machine Learning

### 6.1 Variational Inference and the ELBO

Given observed data $\mathbf{x}$ and latent variables $\mathbf{z}$, we want $p(\mathbf{z}|\mathbf{x})$ but it is intractable. Variational inference seeks the best approximation $q^*(\mathbf{z})$ by minimizing:

$$J[q] = \text{KL}(q \| p(\cdot|\mathbf{x})) = \int q(\mathbf{z}) \log \frac{q(\mathbf{z})}{p(\mathbf{z}|\mathbf{x})} \, d\mathbf{z}$$

This is a *functional* of $q$ (a function over latent space). Minimizing KL is equivalent to maximizing the *Evidence Lower Bound* (ELBO):

$$\text{ELBO}[q] = \mathbb{E}_q[\log p(\mathbf{x}, \mathbf{z})] - \mathbb{E}_q[\log q(\mathbf{z})] = \log p(\mathbf{x}) - \text{KL}(q \| p(\cdot|\mathbf{x}))$$

**Theorem 3.7.4 (Optimal Variational Distribution).** Among all distributions $q$ (subject to $\int q = 1$, $q \geq 0$), the minimizer of $\text{KL}(q \| p(\cdot|\mathbf{x}))$ is $q^* = p(\cdot|\mathbf{x})$.

*Proof.* This is a constrained variational problem with Lagrangian $\tilde{L} = q\log(q/p) + \lambda q$. The Euler-Lagrange condition (here just the functional derivative) gives $\log(q/p) + 1 + \lambda = 0$, so $q = p \cdot e^{-(1+\lambda)}$. The constraint $\int q = 1$ forces $q = p(\cdot|\mathbf{x})$. $\square$

**Example 3.7.7 (KL Divergence as a Functional).** Let the true posterior be $p(z|x) = \mathcal{N}(3, 1)$ and consider two candidate approximations: $q_1(z) = \mathcal{N}(3, 1)$ (exact match) and $q_2(z) = \mathcal{N}(0, 1)$ (poor approximation).

For Gaussians, $\text{KL}(\mathcal{N}(\mu_q, \sigma_q^2) \| \mathcal{N}(\mu_p, \sigma_p^2)) = \log\frac{\sigma_p}{\sigma_q} + \frac{\sigma_q^2 + (\mu_q - \mu_p)^2}{2\sigma_p^2} - \frac{1}{2}$.

$$J[q_1] = \text{KL}(q_1 \| p) = \log\frac{1}{1} + \frac{1 + 0}{2} - \frac{1}{2} = 0 \quad \text{(perfect match)}$$

$$J[q_2] = \text{KL}(q_2 \| p) = \log\frac{1}{1} + \frac{1 + 9}{2} - \frac{1}{2} = 4.5 \quad \text{(large divergence)}$$

The functional $J[q] = \text{KL}(q \| p)$ assigns a real number to each distribution $q$. Variational inference seeks the $q^*$ in a tractable family that minimizes this functional.

In practice, we restrict $q$ to a tractable family (e.g., factored Gaussians in mean-field VI), and the calculus of variations within that family determines the optimal parameters.

### 6.2 Variational Autoencoders (VAEs)

VAEs parameterize both $q_\phi(\mathbf{z}|\mathbf{x})$ (encoder) and $p_\theta(\mathbf{x}|\mathbf{z})$ (decoder) as neural networks and maximize:

$$\text{ELBO}(\theta, \phi) = \mathbb{E}_{q_\phi}[\log p_\theta(\mathbf{x}|\mathbf{z})] - \text{KL}(q_\phi(\mathbf{z}|\mathbf{x}) \| p(\mathbf{z}))$$

The first term encourages reconstruction; the second regularizes the latent space. The reparameterization trick $\mathbf{z} = \boldsymbol{\mu} + \boldsymbol{\sigma} \odot \boldsymbol{\epsilon}$ ($\boldsymbol{\epsilon} \sim \mathcal{N}(0, I)$) makes the variational objective differentiable w.r.t. $\phi$.

### 6.3 Optimal Transport and Wasserstein Distance

**Definition 3.7.5 (Kantorovich Problem).** Given distributions $\mu$ and $\nu$ on $\mathcal{X}$, the *Wasserstein-$p$ distance* is:

$$W_p(\mu, \nu) = \left( \inf_{\gamma \in \Pi(\mu,\nu)} \int_{\mathcal{X} \times \mathcal{X}} d(x,y)^p \, d\gamma(x,y) \right)^{1/p}$$

where $\Pi(\mu, \nu)$ is the set of all couplings (joint distributions with marginals $\mu$ and $\nu$). This is a variational problem: optimize over the function space of transport plans $\gamma$.

Wasserstein distances provide geometrically meaningful loss functions for generative models (Wasserstein GANs) because, unlike KL divergence, they metrize weak convergence and handle non-overlapping supports gracefully.

### 6.4 Maximum Entropy Distributions

**Theorem 3.7.5 (Maximum Entropy).** Among all distributions $q$ on $\mathbb{R}$ satisfying $\mathbb{E}_q[x] = \mu$ and $\mathbb{E}_q[x^2] = \mu^2 + \sigma^2$, the one maximizing the entropy $H[q] = -\int q\log q \, dx$ is the Gaussian $\mathcal{N}(\mu, \sigma^2)$.

*Proof sketch.* Maximize $J[q] = -\int q\log q$ subject to $\int q = 1$, $\int xq = \mu$, $\int x^2 q = \mu^2 + \sigma^2$. The augmented Lagrangian is $\tilde{L} = -q\log q + \lambda_0 q + \lambda_1 xq + \lambda_2 x^2 q$. Setting the functional derivative to zero: $-\log q - 1 + \lambda_0 + \lambda_1 x + \lambda_2 x^2 = 0$, so $q(x) = \exp(\lambda_0 - 1 + \lambda_1 x + \lambda_2 x^2)$. The constraints determine $\lambda_0, \lambda_1, \lambda_2$, yielding the Gaussian. $\square$

This result justifies the Gaussian assumption throughout ML: when you know only the mean and variance, the Gaussian is the *least biased* (most uncertain) choice.

### 6.5 Physics-Informed Neural Networks (PINNs)

PINNs solve PDEs by minimizing a variational loss:

$$\mathcal{L}[\hat{u}_\theta] = \underbrace{\int_\Omega \left| \mathcal{D}[\hat{u}_\theta] - f \right|^2 dx}_{\text{PDE residual}} + \underbrace{\lambda \int_{\partial\Omega} \left| \hat{u}_\theta - g \right|^2 ds}_{\text{boundary condition}}$$

where $\hat{u}_\theta$ is a neural network approximating the solution $u$, and $\mathcal{D}$ is the differential operator. This converts a variational problem (find $u$ satisfying the PDE) into a neural network training problem.

---

## 7. The Second Variation

**Definition 3.7.6 (Second Variation).** The second variation of $J$ at $y$ is:

$$\delta^2 J[y; \eta] = \frac{d^2}{d\varepsilon^2}\bigg|_{\varepsilon=0} J[y + \varepsilon\eta]$$

**Theorem 3.7.6 (Legendre Condition).** A necessary condition for $y^*$ to be a (weak) local minimum is:

$$\frac{\partial^2 L}{\partial (y')^2}\bigg|_{y^*} \geq 0 \quad \text{for all } x \in [a, b]$$

This is the infinite-dimensional analogue of the Hessian being positive semi-definite. The sufficient condition (Jacobi condition) additionally requires that the *conjugate point* lies outside $[a, b]$.

---

## Exercises

**★ Basic**

1. Compute the first variation of $J[y] = \int_0^1 (y')^2 \, dx$ at $y(x) = x$ in the direction $\eta(x) = \sin(\pi x)$.

2. Write down the Euler-Lagrange equation for $J[y] = \int_0^1 \left[(y')^2 + y^2\right] dx$. What familiar ODE do you obtain?

3. Verify that $y(x) = \cosh(x)$ satisfies the Euler-Lagrange equation for $J[y] = \int_0^1 y\sqrt{1 + (y')^2} \, dx$ (the minimal surface of revolution).

**★★ Intermediate**

4. Derive the Euler-Lagrange equation for $J[y] = \int_0^T \frac{1}{2}m(y')^2 - V(y) \, dt$ (Lagrangian mechanics). Identify the resulting equation as Newton's second law $F = ma$.

5. Among all curves $y(x)$ from $(0, 0)$ to $(1, 1)$ with $\int_0^1 y \, dx = 1/3$, find the one that minimizes $\int_0^1 (y')^2 \, dx$. (Hint: use the isoperimetric method.)

6. Show that the entropy functional $H[q] = -\int q \log q \, dx$ is concave (i.e., $\delta^2 H \leq 0$), confirming that the maximum entropy solution is a true maximum.

**★★★ Challenging**

7. The *Dirichlet principle* states that the function minimizing $J[u] = \int_\Omega |\nabla u|^2 \, dx$ subject to $u|_{\partial\Omega} = g$ solves Laplace's equation $\nabla^2 u = 0$. Derive this from the Euler-Lagrange equation in multiple dimensions.

8. In variational inference with a mean-field factorization $q(\mathbf{z}) = \prod_i q_i(z_i)$, use the calculus of variations to show that the optimal $q_j^*$ satisfies $\log q_j^*(z_j) = \mathbb{E}_{q_{-j}}[\log p(\mathbf{x}, \mathbf{z})] + \text{const}$, where $q_{-j}$ denotes all factors except $j$.

9. For the brachistochrone, use the Beltrami identity ($L - y'\frac{\partial L}{\partial y'} = C$ when $L$ does not depend on $x$ explicitly) to reduce the Euler-Lagrange equation to a first-order ODE, and verify the cycloid solution.

---

## Related Topics

- [Multivariable Calculus](multivariable-calculus.md) --- gradients and the finite-dimensional analogue of variational derivatives
- [Integration](integration.md) --- the integrals that define functionals
- [Taylor Series](taylor-series.md) --- local expansions underpinning the second variation
- [Optimization](../optimization/index.md) --- finite-dimensional optimization that variational methods generalize
- [Information Theory](../information-theory/index.md) --- KL divergence, entropy, and their variational characterizations
