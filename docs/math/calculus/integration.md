# Integration

Integration is the mathematics of accumulation — and machine learning is built on it. Every time you compute an expected loss, marginalize over latent variables, or normalize a probability distribution, you are integrating. The fundamental challenge of Bayesian deep learning is that the integrals defining the posterior are intractable in high dimensions, driving the development of Monte Carlo methods, variational inference, and the entire field of approximate inference. Understanding integration from Riemann sums to numerical quadrature provides the foundation for these critical techniques.

---

## Prerequisites

- [Limits & Continuity](limits-and-continuity.md) — $\varepsilon$-$\delta$ definitions, convergence, supremum and infimum
- [Differentiation](differentiation.md) — derivative rules, chain rule, mean value theorem

---

## 1. The Riemann Integral

### 1.1 Partitions and Sums

**Definition 3.3.1 (Partition).** A *partition* $P$ of the interval $[a, b]$ is a finite set of points:

$$P = \{x_0, x_1, \ldots, x_n\} \quad \text{with} \quad a = x_0 < x_1 < \cdots < x_n = b$$

The *mesh* (or *norm*) of $P$ is $\|P\| = \max_{1 \leq i \leq n} (x_i - x_{i-1})$.

**Definition 3.3.2 (Upper and Lower Sums).** For a bounded function $f: [a,b] \to \mathbb{R}$ and partition $P$:

$$L(f, P) = \sum_{i=1}^n m_i \cdot \Delta x_i \qquad U(f, P) = \sum_{i=1}^n M_i \cdot \Delta x_i$$

where $m_i = \inf_{x \in [x_{i-1}, x_i]} f(x)$, $M_i = \sup_{x \in [x_{i-1}, x_i]} f(x)$, and $\Delta x_i = x_i - x_{i-1}$.

```
RIEMANN SUMS FOR f(x) = x²  on [0, 1]

   Upper Sum (overestimate)        Lower Sum (underestimate)

   f(x)│                           f(x)│
    1  │        ┌──┐                1  │           ╱
       │     ┌──┤  │                   │        ╱──┘
       │  ┌──┤  │  │                   │     ╱──┘
       │──┤  │  │  │                   │  ╱──┘
       │  │  │  │  │                   │──┘
   ────┼──┴──┴──┴──┴──              ────┼──┴──┴──┴──┴──
       0              1                 0              1

   U(f,P) ≥ ∫₀¹ x² dx ≥ L(f,P)

   As ‖P‖ → 0:  U(f,P) → 1/3 ← L(f,P)
```

**Example 3.3.1:** Compute the left and right Riemann sums for $f(x) = x^2$ on $[0, 1]$ with $n = 4$ subintervals.

With $\Delta x = 1/4$ and partition points $\{0, 0.25, 0.5, 0.75, 1\}$:

$$L(f,P) = \sum_{i=1}^{4} f(x_{i-1})\,\Delta x = \tfrac{1}{4}\left[0^2 + 0.25^2 + 0.5^2 + 0.75^2\right] = \tfrac{1}{4}(0 + 0.0625 + 0.25 + 0.5625) = 0.21875$$

$$U(f,P) = \sum_{i=1}^{4} f(x_i)\,\Delta x = \tfrac{1}{4}\left[0.25^2 + 0.5^2 + 0.75^2 + 1^2\right] = \tfrac{1}{4}(0.0625 + 0.25 + 0.5625 + 1) = 0.46875$$

The exact value $\int_0^1 x^2\, dx = 1/3 \approx 0.333$ lies between these bounds: $0.21875 \leq 0.333 \leq 0.46875$.

**Definition 3.3.3 (Riemann Integral).** A bounded function $f: [a,b] \to \mathbb{R}$ is *Riemann integrable* if:

$$\sup_P L(f, P) = \inf_P U(f, P)$$

and the common value is called the *Riemann integral*:

$$\int_a^b f(x)\, dx = \sup_P L(f, P) = \inf_P U(f, P)$$

**Theorem 3.3.1 (Integrability Criterion).** A bounded function $f$ on $[a,b]$ is Riemann integrable if and only if for every $\varepsilon > 0$, there exists a partition $P$ such that:

$$U(f, P) - L(f, P) < \varepsilon$$

**Example 3.3.1b:** Evaluate $\displaystyle\int_0^{\pi} \sin(x)\, dx$.

Since $\sin(x)$ is continuous on $[0, \pi]$, it is Riemann integrable. An antiderivative is $-\cos(x)$:

$$\int_0^{\pi} \sin(x)\, dx = \left[-\cos(x)\right]_0^{\pi} = -\cos(\pi) - (-\cos(0)) = -(-1) + 1 = 2$$

Geometrically, this is the area of one "hump" of the sine curve.

**Theorem 3.3.2.** Every continuous function on $[a,b]$ is Riemann integrable.

*Proof.* A continuous function on a closed bounded interval is uniformly continuous: for every $\varepsilon > 0$, there exists $\delta > 0$ such that $|x - y| < \delta \implies |f(x) - f(y)| < \varepsilon/(b-a)$. Choose any partition $P$ with $\|P\| < \delta$. Then $M_i - m_i < \varepsilon/(b-a)$ on each subinterval, so $U(f,P) - L(f,P) = \sum (M_i - m_i)\Delta x_i < \frac{\varepsilon}{b-a} \sum \Delta x_i = \varepsilon$. $\square$

*ML connection:* The Riemann sum $\sum_{i} f(x_i) \Delta x_i$ is the discrete approximation that underlies numerical integration. When computing expected loss $\mathbb{E}[\mathcal{L}] = \int \mathcal{L}(x) p(x)\, dx$ in practice, we approximate with a finite sample mean $\frac{1}{N}\sum_{i=1}^N \mathcal{L}(x_i)$ — a Monte Carlo version of a Riemann sum.

---

## 2. Properties of the Integral

**Theorem 3.3.3 (Linearity).** If $f$ and $g$ are integrable on $[a,b]$ and $\alpha, \beta \in \mathbb{R}$:

$$\int_a^b [\alpha f(x) + \beta g(x)]\, dx = \alpha \int_a^b f(x)\, dx + \beta \int_a^b g(x)\, dx$$

**Example 3.3.2:** Compute $\displaystyle\int_0^1 (3x^2 + 4x)\, dx$ using linearity.

$$\int_0^1 (3x^2 + 4x)\, dx = 3\int_0^1 x^2\, dx + 4\int_0^1 x\, dx = 3 \cdot \frac{1}{3} + 4 \cdot \frac{1}{2} = 1 + 2 = 3$$

**Theorem 3.3.4 (Monotonicity).** If $f(x) \leq g(x)$ for all $x \in [a,b]$, then:

$$\int_a^b f(x)\, dx \leq \int_a^b g(x)\, dx$$

**Theorem 3.3.5 (Additivity over Intervals).** For $a < c < b$:

$$\int_a^b f(x)\, dx = \int_a^c f(x)\, dx + \int_c^b f(x)\, dx$$

**Theorem 3.3.6 (Triangle Inequality for Integrals).**

$$\left|\int_a^b f(x)\, dx\right| \leq \int_a^b |f(x)|\, dx$$

**Example 3.3.8 (Area Between Curves):** Find the area between $f(x) = x + 1$ and $g(x) = x^2$ on $[0, 1]$.

Since $f(x) \geq g(x)$ on $[0,1]$ (verify: $x + 1 \geq x^2 \iff x^2 - x - 1 \leq 0$, which holds on $[0,1]$):

$$A = \int_0^1 \left[(x+1) - x^2\right]\, dx = \int_0^1 (x + 1 - x^2)\, dx = \left[\frac{x^2}{2} + x - \frac{x^3}{3}\right]_0^1 = \frac{1}{2} + 1 - \frac{1}{3} = \frac{7}{6}$$

---

## 3. The Fundamental Theorem of Calculus

**Theorem 3.3.7 (FTC, Part I).** Let $f$ be continuous on $[a,b]$. Define $F(x) = \int_a^x f(t)\, dt$. Then $F$ is differentiable on $(a,b)$ and:

$$F'(x) = f(x)$$

*Proof.* For $h \neq 0$:

$$\frac{F(x+h) - F(x)}{h} = \frac{1}{h}\int_x^{x+h} f(t)\, dt$$

By the mean value theorem for integrals, there exists $c_h$ between $x$ and $x+h$ such that $\int_x^{x+h} f(t)\, dt = f(c_h) \cdot h$. Thus $\frac{F(x+h) - F(x)}{h} = f(c_h)$. As $h \to 0$, $c_h \to x$, and by continuity of $f$, $f(c_h) \to f(x)$. $\square$

**Theorem 3.3.8 (FTC, Part II).** Let $f$ be continuous on $[a,b]$ and let $F$ be any antiderivative of $f$ (i.e., $F' = f$). Then:

$$\int_a^b f(x)\, dx = F(b) - F(a)$$

*Proof.* Let $G(x) = \int_a^x f(t)\, dt$. By Part I, $G'(x) = f(x)$, so $G$ and $F$ are both antiderivatives of $f$. Since two antiderivatives differ by a constant, $F(x) = G(x) + C$ for some $C$. Then $F(b) - F(a) = G(b) - G(a) = \int_a^b f(t)\, dt - 0 = \int_a^b f(t)\, dt$. $\square$

```
FTC VISUALIZED: DIFFERENTIATION AND INTEGRATION ARE INVERSES

        differentiate
   F(x)  ──────────▶  f(x) = F'(x)
          ◀──────────
         integrate

   f(x) = 2x   ←──differentiate──   F(x) = x²
                ──── integrate ───▶   ∫₀ˣ 2t dt = x²

   Area under f(x) = 2x from 0 to 3:
   ∫₀³ 2x dx = [x²]₀³ = 9 - 0 = 9
```

**Example 3.3.3:** Evaluate $\displaystyle\int_0^2 x^2\, dx$ using FTC Part II.

An antiderivative of $f(x) = x^2$ is $F(x) = \frac{x^3}{3}$. By the FTC:

$$\int_0^2 x^2\, dx = F(2) - F(0) = \frac{2^3}{3} - \frac{0^3}{3} = \frac{8}{3} - 0 = \frac{8}{3}$$

*ML connection:* The FTC justifies computing cumulative distribution functions from density functions: $F(x) = \int_{-\infty}^x f(t)\, dt$, and conversely $f(x) = F'(x)$. This relationship is fundamental in probability theory and statistical learning.

---

## 4. Integration Techniques

### 4.1 Substitution (Change of Variables)

**Theorem 3.3.9 (Substitution Rule).** If $g: [a,b] \to \mathbb{R}$ is differentiable with continuous derivative and $f$ is continuous on the range of $g$, then:

$$\int_a^b f(g(x)) g'(x)\, dx = \int_{g(a)}^{g(b)} f(u)\, du$$

*Proof.* Let $F$ be an antiderivative of $f$. By the chain rule, $\frac{d}{dx}[F(g(x))] = f(g(x)) g'(x)$. Applying FTC Part II to both sides gives the result. $\square$

**Example 3.3.4:** Compute $\displaystyle\int 2x \cos(x^2)\, dx$ using substitution.

Let $u = x^2$, so $du = 2x\, dx$:

$$\int 2x \cos(x^2)\, dx = \int \cos(u)\, du = \sin(u) + C = \sin(x^2) + C$$

Verify: $\frac{d}{dx}[\sin(x^2)] = \cos(x^2) \cdot 2x = 2x\cos(x^2)$ $\checkmark$

*ML connection:* The substitution rule generalizes to the *change of variables formula* in probability: if $Y = g(X)$ and $g$ is invertible, then $p_Y(y) = p_X(g^{-1}(y)) \cdot |{(g^{-1})'(y)}|$. This is the 1D version of the Jacobian formula used in normalizing flows.

### 4.2 Integration by Parts

**Theorem 3.3.10 (Integration by Parts).** If $u$ and $v$ are differentiable with continuous derivatives on $[a,b]$:

$$\int_a^b u(x) v'(x)\, dx = [u(x)v(x)]_a^b - \int_a^b u'(x) v(x)\, dx$$

*Proof.* From the product rule, $(uv)' = u'v + uv'$. Integrating both sides over $[a,b]$ and rearranging by the FTC gives the result. $\square$

**Example.** $\int x e^x\, dx$: let $u = x$, $v' = e^x$, so $u' = 1$, $v = e^x$:

$$\int x e^x\, dx = x e^x - \int e^x\, dx = xe^x - e^x + C = (x-1)e^x + C$$

**Example 3.3.5:** Evaluate $\displaystyle\int_0^1 x e^x\, dx$ (definite version).

Using the antiderivative from above: $\int_0^1 x e^x\, dx = \left[(x-1)e^x\right]_0^1 = (1-1)e^1 - (0-1)e^0 = 0 - (-1) = 1$.

*ML connection:* Integration by parts is used to derive the ELBO (Evidence Lower Bound) in variational inference and appears in deriving properties of entropy: $H(X) = -\int p(x) \log p(x)\, dx$.

### 4.3 Partial Fractions

For rational functions $P(x)/Q(x)$ where $\deg P < \deg Q$, decompose into simpler fractions:

$$\frac{1}{(x-a)(x-b)} = \frac{1}{a-b}\left(\frac{1}{x-a} - \frac{1}{x-b}\right)$$

Each term integrates to a logarithm: $\int \frac{1}{x-a}\, dx = \ln|x-a| + C$.

**Example 3.3.6:** Decompose and integrate $\displaystyle\int \frac{1}{x^2 - 1}\, dx$.

Factor: $x^2 - 1 = (x-1)(x+1)$. Apply partial fractions:

$$\frac{1}{(x-1)(x+1)} = \frac{1}{-2}\left(\frac{1}{x+1} - \frac{1}{x-1}\right) = \frac{1}{2}\left(\frac{1}{x-1} - \frac{1}{x+1}\right)$$

$$\int \frac{1}{x^2 - 1}\, dx = \frac{1}{2}\ln|x-1| - \frac{1}{2}\ln|x+1| + C = \frac{1}{2}\ln\left|\frac{x-1}{x+1}\right| + C$$

---

## 5. Improper Integrals

**Definition 3.3.4 (Improper Integral — Infinite Limits).** For $f$ integrable on $[a, R]$ for every $R > a$:

$$\int_a^\infty f(x)\, dx = \lim_{R \to \infty} \int_a^R f(x)\, dx$$

if the limit exists (the integral *converges*); otherwise it *diverges*.

**Example 3.3.7:** Evaluate $\displaystyle\int_1^\infty \frac{1}{x^2}\, dx$.

$$\int_1^\infty \frac{1}{x^2}\, dx = \lim_{R \to \infty} \int_1^R x^{-2}\, dx = \lim_{R \to \infty} \left[-\frac{1}{x}\right]_1^R = \lim_{R \to \infty} \left(-\frac{1}{R} + 1\right) = 0 + 1 = 1$$

The integral converges because $p = 2 > 1$. Compare with $\int_1^\infty \frac{1}{x}\, dx = \lim_{R\to\infty} \ln R = \infty$ (diverges, $p = 1$).

**Definition 3.3.5 (Improper Integral — Unbounded Function).** If $f$ is unbounded near $x = a$ but integrable on $[a + \varepsilon, b]$ for every $\varepsilon > 0$:

$$\int_a^b f(x)\, dx = \lim_{\varepsilon \to 0^+} \int_{a+\varepsilon}^b f(x)\, dx$$

**Key examples:**

| Integral | Converges? | Value |
|----------|-----------|-------|
| $\int_1^\infty x^{-p}\, dx$ | $p > 1$ | $\frac{1}{p-1}$ |
| $\int_1^\infty x^{-p}\, dx$ | $p \leq 1$ | Diverges |
| $\int_0^\infty e^{-x}\, dx$ | Yes | $1$ |
| $\int_{-\infty}^{\infty} e^{-x^2/2}\, dx$ | Yes | $\sqrt{2\pi}$ |

**Theorem 3.3.11 (Gaussian Integral).**

$$\int_{-\infty}^{\infty} e^{-x^2}\, dx = \sqrt{\pi}$$

*Proof sketch.* Let $I = \int_{-\infty}^{\infty} e^{-x^2}\, dx$. Then $I^2 = \int\int e^{-(x^2+y^2)}\, dx\, dy$. Switching to polar coordinates $(r, \theta)$: $I^2 = \int_0^{2\pi}\int_0^\infty e^{-r^2} r\, dr\, d\theta = 2\pi \cdot \frac{1}{2} = \pi$. Thus $I = \sqrt{\pi}$. $\square$

*ML connection:* The Gaussian integral is why the normal distribution $\mathcal{N}(\mu, \sigma^2)$ has the normalizing constant $(2\pi\sigma^2)^{-1/2}$. Every time you write $p(x) = \frac{1}{\sqrt{2\pi}\sigma}\exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)$, the Gaussian integral guarantees that $\int p(x)\, dx = 1$.

---

## 6. Numerical Integration

When antiderivatives have no closed form — which is the common case in ML — we resort to numerical methods.

### 6.1 Trapezoidal Rule

**Definition 3.3.6 (Trapezoidal Rule).** Approximate $\int_a^b f(x)\, dx$ using $n$ trapezoids with uniform spacing $h = (b-a)/n$:

$$T_n = \frac{h}{2}\left[f(x_0) + 2f(x_1) + 2f(x_2) + \cdots + 2f(x_{n-1}) + f(x_n)\right]$$

**Example 3.3.9:** Approximate $\int_0^1 x^2\, dx$ using the trapezoidal rule with $n = 4$ subintervals.

With $h = (1-0)/4 = 0.25$ and points $x_0=0, x_1=0.25, x_2=0.5, x_3=0.75, x_4=1$:

$$T_4 = \frac{0.25}{2}\left[0^2 + 2(0.25)^2 + 2(0.5)^2 + 2(0.75)^2 + 1^2\right] = 0.125\left[0 + 0.125 + 0.5 + 1.125 + 1\right] = 0.125 \times 2.75 = 0.34375$$

The exact value is $1/3 \approx 0.33333$, so the error is $\approx 0.0104$.

**Theorem 3.3.12 (Trapezoidal Error).** If $f \in C^2[a,b]$:

$$\left|\int_a^b f(x)\, dx - T_n\right| \leq \frac{(b-a)^3}{12n^2} \max_{x \in [a,b]} |f''(x)|$$

The error is $O(h^2)$ — doubling the number of points reduces the error by a factor of 4.

### 6.2 Simpson's Rule

**Definition 3.3.7 (Simpson's Rule).** Using $n$ subintervals (with $n$ even) and uniform spacing $h = (b-a)/n$:

$$S_n = \frac{h}{3}\left[f(x_0) + 4f(x_1) + 2f(x_2) + 4f(x_3) + \cdots + 4f(x_{n-1}) + f(x_n)\right]$$

**Example 3.3.11:** Approximate $\int_0^1 x^2\, dx$ using Simpson's rule with $n = 4$ subintervals.

With $h = 0.25$ and the same points as before:

$$S_4 = \frac{0.25}{3}\left[0^2 + 4(0.25)^2 + 2(0.5)^2 + 4(0.75)^2 + 1^2\right] = \frac{0.25}{3}\left[0 + 0.25 + 0.5 + 2.25 + 1\right] = \frac{0.25 \times 4}{3} = \frac{1}{3}$$

Simpson's rule gives the exact answer here because it integrates polynomials of degree $\leq 3$ exactly.

**Theorem 3.3.13 (Simpson's Error).** If $f \in C^4[a,b]$:

$$\left|\int_a^b f(x)\, dx - S_n\right| \leq \frac{(b-a)^5}{180n^4} \max_{x \in [a,b]} |f^{(4)}(x)|$$

The error is $O(h^4)$ — significantly more accurate than the trapezoidal rule for smooth functions.

### 6.3 Monte Carlo Integration

**Definition 3.3.8 (Monte Carlo Estimator).** To estimate $I = \int_a^b f(x)\, dx$, draw $N$ samples $x_1, \ldots, x_N$ uniformly from $[a,b]$:

$$\hat{I}_N = \frac{b-a}{N} \sum_{i=1}^N f(x_i)$$

**Example 3.3.10:** Estimate $\int_0^1 x^2\, dx$ using Monte Carlo with samples $x_1 = 0.2, x_2 = 0.5, x_3 = 0.7, x_4 = 0.9$.

$$\hat{I}_4 = \frac{1-0}{4}\left[(0.2)^2 + (0.5)^2 + (0.7)^2 + (0.9)^2\right] = \frac{1}{4}(0.04 + 0.25 + 0.49 + 0.81) = \frac{1.59}{4} = 0.3975$$

The exact value is $\int_0^1 x^2\, dx = 1/3 \approx 0.3333$. The estimate improves as $N$ increases.

**Theorem 3.3.14 (Monte Carlo Convergence).** $\mathbb{E}[\hat{I}_N] = I$ (unbiased), and the standard error is:

$$\text{SE} = \frac{(b-a) \cdot \text{std}(f)}{{\sqrt{N}}}$$

The error is $O(N^{-1/2})$ regardless of dimension — this is the key advantage.

```
COMPARISON OF NUMERICAL METHODS

  Method          Error Rate    Curse of Dimensionality?
  ─────────────   ──────────    ────────────────────────
  Trapezoidal     O(n⁻²)       Yes — exponential in d
  Simpson's       O(n⁻⁴)       Yes — exponential in d
  Monte Carlo     O(n⁻¹/²)     No  — independent of d  ← key for ML

  In 100 dimensions:
  - Trapezoidal with 10 points/dim → 10¹⁰⁰ evaluations (impossible)
  - Monte Carlo with 10,000 samples → 10,000 evaluations (feasible)
```

*ML connection:* Monte Carlo integration is the backbone of modern ML. Stochastic gradient descent estimates $\nabla \mathbb{E}[\mathcal{L}]$ with mini-batch samples. MCMC methods (Metropolis-Hastings, Hamiltonian Monte Carlo) sample from posterior distributions to approximate intractable integrals. The $O(N^{-1/2})$ rate, while slow, is dimension-independent — making it the only viable option for high-dimensional integrals.

---

## 7. Multiple Integrals

**Definition 3.3.9 (Double Integral).** For $f: \mathbb{R}^2 \to \mathbb{R}$ over a region $R$:

$$\iint_R f(x, y)\, dA = \int_a^b \int_{g_1(x)}^{g_2(x)} f(x, y)\, dy\, dx$$

**Theorem 3.3.15 (Fubini's Theorem).** If $f$ is continuous on the rectangle $[a,b] \times [c,d]$, the order of integration can be exchanged:

$$\int_a^b \int_c^d f(x,y)\, dy\, dx = \int_c^d \int_a^b f(x,y)\, dx\, dy$$

**Example 3.3.12:** Evaluate $\displaystyle\iint_R (x + 2y)\, dA$ over the rectangle $R = [0, 1] \times [0, 2]$.

$$\int_0^1 \int_0^2 (x + 2y)\, dy\, dx = \int_0^1 \left[xy + y^2\right]_0^2\, dx = \int_0^1 (2x + 4)\, dx$$

$$= \left[x^2 + 4x\right]_0^1 = 1 + 4 = 5$$

Verifying with reversed order: $\int_0^2 \int_0^1 (x + 2y)\, dx\, dy = \int_0^2 \left[\tfrac{x^2}{2} + 2xy\right]_0^1 dy = \int_0^2 (\tfrac{1}{2} + 2y)\, dy = \left[\tfrac{y}{2} + y^2\right]_0^2 = 1 + 4 = 5$ $\checkmark$

**Theorem 3.3.16 (Change of Variables in Multiple Integrals).** If $T: \mathbb{R}^n \to \mathbb{R}^n$ is a differentiable bijection on region $R$:

$$\int_{T(R)} f(\mathbf{x})\, d\mathbf{x} = \int_R f(T(\mathbf{u})) \left|\det\left(\frac{\partial T}{\partial \mathbf{u}}\right)\right| d\mathbf{u}$$

The Jacobian determinant $|\det(J_T)|$ corrects for how $T$ stretches or compresses volume.

*ML connection:* Multiple integrals appear everywhere in probabilistic ML:

- **Marginalization:** $p(x) = \int p(x, y)\, dy$ — integrating out latent variables
- **Expectations:** $\mathbb{E}[g(X)] = \int g(x) p(x)\, dx$ over potentially high-dimensional spaces
- **Normalizing flows:** the change of variables formula with Jacobian determinant transforms simple distributions to complex ones (see [Determinants](../linear-algebra/determinants.md))

---

## 8. Integration in Machine Learning

### 8.1 Computing Expectations

The expected value of a function $g(X)$ under distribution $p$ is:

$$\mathbb{E}_p[g(X)] = \int g(x) p(x)\, dx$$

Every loss function in ML is optimized in expectation: $\min_\theta \mathbb{E}_{(x,y) \sim \mathcal{D}}[\mathcal{L}(f_\theta(x), y)]$.

### 8.2 Marginalizing Distributions

Given a joint distribution $p(x, y)$, the marginal is obtained by integrating out $y$:

$$p(x) = \int p(x, y)\, dy = \int p(x|y) p(y)\, dy$$

In latent variable models (VAEs, mixture models), the marginal likelihood involves integrating over latent variables $\mathbf{z}$:

$$p(\mathbf{x}) = \int p(\mathbf{x}|\mathbf{z}) p(\mathbf{z})\, d\mathbf{z}$$

### 8.3 Bayesian Evidence and Intractable Integrals

Bayes' theorem requires the *evidence* (marginal likelihood):

$$p(\theta | \mathcal{D}) = \frac{p(\mathcal{D}|\theta) p(\theta)}{p(\mathcal{D})} \qquad \text{where} \quad p(\mathcal{D}) = \int p(\mathcal{D}|\theta) p(\theta)\, d\theta$$

```
THE INTRACTABILITY PROBLEM IN BAYESIAN ML

   p(θ | D) = p(D|θ) p(θ) / p(D)
                                    ↑
                              p(D) = ∫ p(D|θ) p(θ) dθ
                                         ↑
                                    Integral over ALL
                                    parameter values
                                    (often millions of dims)

   Solutions:
   ┌─────────────────────┬──────────────────────────────────────┐
   │ Method              │ How It Avoids the Integral           │
   ├─────────────────────┼──────────────────────────────────────┤
   │ MCMC                │ Samples from p(θ|D) without          │
   │                     │ computing p(D) — uses ratios only    │
   │ Variational Inf.    │ Approximates p(θ|D) ≈ q(θ) by       │
   │                     │ optimizing a tractable lower bound   │
   │ Laplace Approx.     │ Gaussian approx. around the MAP      │
   │                     │ estimate using the Hessian           │
   │ MAP Estimation      │ Finds mode only — ignores the        │
   │                     │ integral entirely (point estimate)   │
   └─────────────────────┴──────────────────────────────────────┘
```

### 8.4 Normalizing Constants

For a probability distribution, $\int p(x)\, dx = 1$. Given an unnormalized density $\tilde{p}(x)$:

$$p(x) = \frac{\tilde{p}(x)}{Z} \qquad \text{where} \quad Z = \int \tilde{p}(x)\, dx$$

Computing $Z$ is often the bottleneck. In energy-based models, $\tilde{p}(x) = e^{-E(x)}$ and $Z = \int e^{-E(x)}\, dx$ is intractable, motivating contrastive learning and score matching.

### 8.5 Area Under the ROC Curve (AUC)

The AUC metric for binary classifiers is literally an integral:

$$\text{AUC} = \int_0^1 \text{TPR}(\text{FPR}^{-1}(t))\, dt$$

This equals the probability that a randomly chosen positive example is ranked higher than a randomly chosen negative example: $\text{AUC} = P(s(x^+) > s(x^-))$.

---

## Exercises

**★ Basic**

1. Compute $\int_0^2 (3x^2 + 2x)\, dx$ using the FTC.

2. Using the substitution $u = x^2 + 1$, evaluate $\int_0^1 2x(x^2+1)^3\, dx$.

3. Evaluate $\int_1^\infty \frac{1}{x^3}\, dx$. For what values of $p$ does $\int_1^\infty x^{-p}\, dx$ converge?

4. Approximate $\int_0^1 e^{-x^2}\, dx$ using the trapezoidal rule with $n = 4$ subintervals.

**★★ Intermediate**

5. Prove that integration by parts gives $\int_0^\infty x e^{-x}\, dx = 1$, and interpret this as $\mathbb{E}[X]$ for the exponential distribution with rate 1.

6. Show that the normalizing constant of the Gaussian distribution is correct: verify $\int_{-\infty}^{\infty} \frac{1}{\sqrt{2\pi}\sigma} e^{-(x-\mu)^2/(2\sigma^2)}\, dx = 1$ by substitution and the Gaussian integral.

7. **(Monte Carlo)** Generate $N = 1000$ uniform samples on $[0,1]$ and estimate $\int_0^1 e^{-x^2}\, dx$. Compute the standard error and compare with the trapezoidal rule using the same number of function evaluations.

8. Using Fubini's theorem, compute $\int_0^1 \int_0^x e^{x+y}\, dy\, dx$ by choosing the most convenient order of integration.

**★★★ Challenging**

9. **(Bayesian)** For a Beta-Binomial model with prior $p(\theta) = \text{Beta}(\alpha, \beta)$ and likelihood $p(k|n,\theta) = \binom{n}{k}\theta^k(1-\theta)^{n-k}$, compute the evidence $p(k|n) = \int_0^1 p(k|n,\theta) p(\theta)\, d\theta$ in closed form using the Beta function $B(\alpha,\beta) = \int_0^1 t^{\alpha-1}(1-t)^{\beta-1}\, dt$.

10. **(Variational Inference)** The ELBO is $\mathcal{L}(q) = \int q(\theta) \log \frac{p(\mathcal{D},\theta)}{q(\theta)}\, d\theta$. Show that $\log p(\mathcal{D}) \geq \mathcal{L}(q)$ for any distribution $q$, with equality iff $q(\theta) = p(\theta|\mathcal{D})$. (Hint: use Jensen's inequality and the concavity of $\log$.)

11. Prove that the Monte Carlo estimator $\hat{I}_N = \frac{1}{N}\sum_{i=1}^N f(x_i)$ with $x_i \sim p(x)$ is unbiased for $\mathbb{E}_p[f(X)]$ and has variance $\text{Var}(\hat{I}_N) = \text{Var}_p(f(X))/N$. Why does this make SGD a valid optimization algorithm?

---

## Related Topics

- [Differentiation](differentiation.md) — the inverse operation; FTC connects them
- [Multivariable Calculus](multivariable-calculus.md) — gradients, Jacobians, and change of variables in higher dimensions
- [Taylor Series](taylor-series.md) — approximation of integrands, Laplace's method
- [Probability Distributions](../probability/index.md) — integrals define expectations, CDFs, and normalizing constants
- [Optimization](../optimization/index.md) — SGD approximates integrals via sampling
- [Determinants](../linear-algebra/determinants.md) — Jacobian determinants in the change of variables formula
