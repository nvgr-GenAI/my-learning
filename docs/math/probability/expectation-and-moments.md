# Expectation and Moments

Expectation is the bridge between probability and optimization. Every loss function in machine learning is an expected value — we minimize $\mathbb{E}[\ell(\hat{y}, y)]$ over some distribution of data. Variance tells us how noisy our estimates are, covariance reveals which features move together (the foundation of PCA), and moment generating functions give us a complete fingerprint of a distribution in a single analytic function. Jensen's inequality, a seemingly abstract convexity result, is the engine behind the ELBO in variational autoencoders and the convergence of EM algorithms.

---

## Prerequisites

- [Random Variables](random-variables.md) — PMF, PDF, CDF, transformations
- [Distributions](distributions.md) — Bernoulli, Gaussian, Poisson, and their parameters

---

## 1. Expectation

**Definition 4.5.1 (Expected Value — Discrete).** For a discrete random variable $X$ with PMF $p_X(x)$:

$$\mathbb{E}[X] = \sum_{x \in \mathcal{X}} x \, p_X(x)$$

provided the sum converges absolutely ($\sum |x| \, p_X(x) < \infty$).

**Example 4.4.1 (Loaded die).** A loaded die has $P(X=6) = 1/2$ and $P(X=k) = 1/10$ for $k = 1,2,3,4,5$.

$$\mathbb{E}[X] = 1\!\cdot\!\tfrac{1}{10} + 2\!\cdot\!\tfrac{1}{10} + 3\!\cdot\!\tfrac{1}{10} + 4\!\cdot\!\tfrac{1}{10} + 5\!\cdot\!\tfrac{1}{10} + 6\!\cdot\!\tfrac{1}{2} = \frac{1+2+3+4+5}{10} + 3 = \frac{15}{10} + 3 = 4.5$$

Compare with a fair die where $\mathbb{E}[X] = 3.5$. The loaded die's mean shifts toward the heavy face.

**Definition 4.5.2 (Expected Value — Continuous).** For a continuous random variable $X$ with PDF $f_X(x)$:

$$\mathbb{E}[X] = \int_{-\infty}^{\infty} x \, f_X(x) \, dx$$

provided the integral converges absolutely.

**Example 4.4.2 (Continuous expected value).** Let $X$ have PDF $f(x) = 2x$ on $[0, 1]$ (zero elsewhere). This is a valid density since $\int_0^1 2x \, dx = x^2 \big|_0^1 = 1$.

$$\mathbb{E}[X] = \int_0^1 x \cdot 2x \, dx = 2\int_0^1 x^2 \, dx = 2 \cdot \frac{x^3}{3}\bigg|_0^1 = \frac{2}{3} \approx 0.667$$

The mean is above $0.5$ because the density puts more weight near $x = 1$ than near $x = 0$.

```
EXPECTATION AS CENTER OF MASS

  Discrete: die roll X ∈ {1,2,3,4,5,6}, each with p = 1/6

  P(X=x)
  1/6  ┊  █  █  █  █  █  █
       ┊──█──█──█──█──█──█──
       1  2  3  4  5  6
                ▲
                E[X] = 3.5   (center of mass)

  Continuous: X ~ Uniform(0,1)

  f(x)
   1  ┊──────────────
      ┊              │
      ┊              │
      ┊──────────────┊──
      0     0.5      1
             ▲
             E[X] = 0.5
```

**Theorem 4.5.1 (LOTUS — Law of the Unconscious Statistician).** For a function $g: \mathbb{R} \to \mathbb{R}$:

$$\mathbb{E}[g(X)] = \begin{cases} \sum_x g(x) \, p_X(x) & \text{discrete} \\[6pt] \int_{-\infty}^{\infty} g(x) \, f_X(x) \, dx & \text{continuous} \end{cases}$$

*Proof (continuous case).* Let $Y = g(X)$. By the change-of-variables formula for densities and the definition of expectation of $Y$, we can show the result without needing to derive $f_Y$. Formally, for monotone $g$ with inverse $h = g^{-1}$:

$$\mathbb{E}[Y] = \int y \, f_Y(y) \, dy = \int g(x) \, f_X(x) \, |h'(g(x))| \cdot |g'(x)| \, dx$$

Since $|h'(g(x))| \cdot |g'(x)| = 1$, this reduces to $\int g(x) f_X(x) \, dx$. The general case follows by partitioning the domain into monotone pieces. $\square$

*ML connection:* LOTUS is why we can compute $\mathbb{E}[\ell(h(X), Y)]$ directly — we integrate the loss over the data distribution without first deriving the distribution of the loss itself. Every empirical risk $\frac{1}{n}\sum_{i=1}^n \ell(h(x_i), y_i)$ is a Monte Carlo estimate of this expectation.

**Example 4.4.3 (LOTUS).** Let $X$ be a fair die roll. Compute $\mathbb{E}[X^2]$ without finding the distribution of $X^2$.

$$\mathbb{E}[X^2] = \sum_{x=1}^{6} x^2 \cdot \tfrac{1}{6} = \frac{1^2 + 2^2 + 3^2 + 4^2 + 5^2 + 6^2}{6} = \frac{1 + 4 + 9 + 16 + 25 + 36}{6} = \frac{91}{6} \approx 15.17$$

We applied $g(x) = x^2$ directly to each value of $X$ — no need to first derive the PMF of $Y = X^2$.

---

## 2. Properties of Expectation

**Theorem 4.5.2 (Linearity of Expectation).** For any random variables $X, Y$ (not necessarily independent) and constants $a, b \in \mathbb{R}$:

$$\mathbb{E}[aX + b] = a\mathbb{E}[X] + b$$

$$\mathbb{E}[X + Y] = \mathbb{E}[X] + \mathbb{E}[Y]$$

Linearity **always** holds — no independence required.

*Proof.* $\mathbb{E}[aX + b] = \int (ax + b) f_X(x) \, dx = a\int x f_X(x) \, dx + b\int f_X(x) \, dx = a\mathbb{E}[X] + b$. For the sum, integrate over the joint density and apply Fubini's theorem to separate the integrals. $\square$

**Example 4.4.4 (Linearity — numerical verification).** Let $X$ be a fair die roll, so $\mathbb{E}[X] = 3.5$. Define $Y = 3X + 2$. Verify linearity:

$$\text{Direct: } \mathbb{E}[Y] = \sum_{x=1}^{6} (3x+2) \cdot \tfrac{1}{6} = \frac{5 + 8 + 11 + 14 + 17 + 20}{6} = \frac{75}{6} = 12.5$$

$$\text{Via linearity: } \mathbb{E}[3X + 2] = 3 \cdot \mathbb{E}[X] + 2 = 3(3.5) + 2 = 12.5 \; \checkmark$$

**Theorem 4.5.3 (Monotonicity).** If $X \leq Y$ almost surely, then $\mathbb{E}[X] \leq \mathbb{E}[Y]$.

**Theorem 4.5.4 (Independence and Products).** If $X$ and $Y$ are independent:

$$\mathbb{E}[XY] = \mathbb{E}[X] \cdot \mathbb{E}[Y]$$

*Proof.* $\mathbb{E}[XY] = \iint xy \, f_{X,Y}(x,y) \, dx \, dy = \iint xy \, f_X(x) f_Y(y) \, dx \, dy = \left(\int x f_X(x) \, dx\right)\left(\int y f_Y(y) \, dy\right)$. $\square$

The converse is false: $\mathbb{E}[XY] = \mathbb{E}[X]\mathbb{E}[Y]$ does not imply independence.

---

## 3. Variance

**Definition 4.5.3 (Variance).** The variance of $X$ is:

$$\text{Var}(X) = \mathbb{E}\left[(X - \mathbb{E}[X])^2\right]$$

**Theorem 4.5.5 (Computational Formula).**

$$\text{Var}(X) = \mathbb{E}[X^2] - \left(\mathbb{E}[X]\right)^2$$

*Proof.* Let $\mu = \mathbb{E}[X]$. Then $\text{Var}(X) = \mathbb{E}[(X - \mu)^2] = \mathbb{E}[X^2 - 2\mu X + \mu^2] = \mathbb{E}[X^2] - 2\mu\mathbb{E}[X] + \mu^2 = \mathbb{E}[X^2] - \mu^2$. $\square$

**Example 4.4.5 (Variance — step by step).** Let $X$ be a fair die roll. We already know $\mathbb{E}[X] = 3.5$ and $\mathbb{E}[X^2] = 91/6$ (Example 4.4.3).

$$\text{Var}(X) = \mathbb{E}[X^2] - (\mathbb{E}[X])^2 = \frac{91}{6} - (3.5)^2 = \frac{91}{6} - \frac{49}{4} = \frac{182 - 147}{12} = \frac{35}{12} \approx 2.917$$

Cross-check via definition: $\text{Var}(X) = \frac{1}{6}\left[(1-3.5)^2 + (2-3.5)^2 + \cdots + (6-3.5)^2\right] = \frac{6.25 + 2.25 + 0.25 + 0.25 + 2.25 + 6.25}{6} = \frac{17.5}{6} = \frac{35}{12} \; \checkmark$

**Theorem 4.5.6 (Properties of Variance).**

| Property | Formula | Independence needed? |
|----------|---------|---------------------|
| Non-negativity | $\text{Var}(X) \geq 0$, with equality iff $X$ is constant a.s. | No |
| Scaling | $\text{Var}(aX) = a^2 \text{Var}(X)$ | No |
| Shift invariance | $\text{Var}(X + b) = \text{Var}(X)$ | No |
| Affine | $\text{Var}(aX + b) = a^2 \text{Var}(X)$ | No |
| Sum (independent) | $\text{Var}(X + Y) = \text{Var}(X) + \text{Var}(Y)$ | **Yes** |
| Sum (general) | $\text{Var}(X + Y) = \text{Var}(X) + \text{Var}(Y) + 2\text{Cov}(X,Y)$ | No |

---

## 4. Standard Deviation

**Definition 4.5.4 (Standard Deviation).**

$$\sigma_X = \sqrt{\text{Var}(X)}$$

Standard deviation has the same units as $X$ (unlike variance, which has squared units), making it more interpretable. For a Gaussian $X \sim \mathcal{N}(\mu, \sigma^2)$, approximately 68% of probability mass lies within $\mu \pm \sigma$, 95% within $\mu \pm 2\sigma$, and 99.7% within $\mu \pm 3\sigma$.

**Example 4.4.6 (Standard deviation — interpretation).** From the fair die (Example 4.4.5), $\text{Var}(X) = 35/12$.

$$\sigma_X = \sqrt{35/12} = \sqrt{2.917} \approx 1.708$$

Interpretation: a typical die roll deviates about $1.7$ from the mean of $3.5$. The interval $\mu \pm \sigma = [1.79,\; 5.21]$ contains the outcomes $\{2, 3, 4, 5\}$, which carry probability $4/6 \approx 67\%$ — consistent with the one-sigma rule.

*ML connection:* Standardization transforms features to zero mean and unit variance: $z = (x - \mu)/\sigma$. This is exactly what **batch normalization** does within a neural network — for each mini-batch, it computes $\hat{x}_i = (x_i - \mu_B)/\sigma_B$ before applying learnable scale and shift parameters.

---

## 5. Covariance

**Definition 4.5.5 (Covariance).**

$$\text{Cov}(X, Y) = \mathbb{E}\left[(X - \mathbb{E}[X])(Y - \mathbb{E}[Y])\right] = \mathbb{E}[XY] - \mathbb{E}[X]\mathbb{E}[Y]$$

```
COVARIANCE — GEOMETRIC INTUITION

  Cov > 0 (positive)         Cov ≈ 0 (none)          Cov < 0 (negative)

  Y ▲    · ·                 Y ▲  ·   ·               Y ▲ · ·
    │      · ·                 │ ·  ·  · ·               │   · ·
    │        · ·               │  · · ·  ·               │     · ·
    │          · ·             │ · ·  · ·                │       · ·
    └──────────▶ X             └──────────▶ X            └──────────▶ X
  X up ⟹ Y up               X tells nothing           X up ⟹ Y down
                              about Y
```

**Example 4.4.7 (Covariance — joint distribution).** Let $(X, Y)$ have the joint PMF:

|       | $Y=0$ | $Y=1$ |
| ----- | ----- | ----- |
| $X=0$ | $1/4$ | $1/4$ |
| $X=1$ | $1/4$ | $1/4$ |

Compute: $\mathbb{E}[X] = 0 \cdot \tfrac{1}{2} + 1 \cdot \tfrac{1}{2} = \tfrac{1}{2}$, $\;\mathbb{E}[Y] = \tfrac{1}{2}$ (by symmetry).

$$\mathbb{E}[XY] = 0\!\cdot\!0\!\cdot\!\tfrac{1}{4} + 0\!\cdot\!1\!\cdot\!\tfrac{1}{4} + 1\!\cdot\!0\!\cdot\!\tfrac{1}{4} + 1\!\cdot\!1\!\cdot\!\tfrac{1}{4} = \tfrac{1}{4}$$

$$\text{Cov}(X,Y) = \mathbb{E}[XY] - \mathbb{E}[X]\mathbb{E}[Y] = \tfrac{1}{4} - \tfrac{1}{2}\cdot\tfrac{1}{2} = 0$$

Zero covariance: $X$ and $Y$ are in fact independent here (the joint factors into marginals).

Now modify to: $P(0,0) = 3/8$, $P(0,1) = 1/8$, $P(1,0) = 1/8$, $P(1,1) = 3/8$. The marginals are still $\mathbb{E}[X] = \mathbb{E}[Y] = 1/2$, but:

$$\mathbb{E}[XY] = 1\!\cdot\!1\!\cdot\!\tfrac{3}{8} = \tfrac{3}{8}, \qquad \text{Cov}(X,Y) = \tfrac{3}{8} - \tfrac{1}{4} = \tfrac{1}{8} > 0$$

Positive covariance: $X$ and $Y$ tend to take the same value.

**Theorem 4.5.7 (Properties of Covariance).**

1. $\text{Cov}(X, X) = \text{Var}(X)$
2. $\text{Cov}(X, Y) = \text{Cov}(Y, X)$ (symmetric)
3. $\text{Cov}(aX + b, Y) = a \, \text{Cov}(X, Y)$ (bilinear)
4. $\text{Cov}(X + Y, Z) = \text{Cov}(X, Z) + \text{Cov}(Y, Z)$
5. If $X, Y$ independent, then $\text{Cov}(X, Y) = 0$ (converse is false)

**Definition 4.5.6 (Covariance Matrix).** For a random vector $\mathbf{X} = (X_1, \ldots, X_n)^T$:

$$\Sigma = \text{Cov}(\mathbf{X}) = \mathbb{E}\left[(\mathbf{X} - \boldsymbol{\mu})(\mathbf{X} - \boldsymbol{\mu})^T\right]$$

where $\Sigma_{ij} = \text{Cov}(X_i, X_j)$ and $\boldsymbol{\mu} = \mathbb{E}[\mathbf{X}]$.

**Theorem 4.5.8.** The covariance matrix $\Sigma$ is symmetric and positive semi-definite.

*Proof.* Symmetry: $\Sigma_{ij} = \text{Cov}(X_i, X_j) = \text{Cov}(X_j, X_i) = \Sigma_{ji}$. PSD: for any $\mathbf{a} \in \mathbb{R}^n$, $\mathbf{a}^T \Sigma \mathbf{a} = \text{Var}(\mathbf{a}^T \mathbf{X}) \geq 0$. $\square$

*ML connection:* PCA eigendecomposes $\Sigma = Q\Lambda Q^T$. The eigenvectors in $Q$ are the principal directions, and the eigenvalues in $\Lambda$ are the variances along those directions. Projecting onto the top $k$ eigenvectors gives the best $k$-dimensional representation minimizing reconstruction error.

---

## 6. Correlation

**Definition 4.5.7 (Pearson Correlation Coefficient).**

$$\rho(X, Y) = \frac{\text{Cov}(X, Y)}{\sigma_X \sigma_Y}$$

**Theorem 4.5.9 (Bounds on Correlation).** $-1 \leq \rho(X,Y) \leq 1$, with equality iff $Y = aX + b$ for some constants $a \neq 0$ (positive $a$ gives $\rho = 1$, negative $a$ gives $\rho = -1$).

*Proof.* Apply Cauchy-Schwarz to centered variables: $|\text{Cov}(X,Y)|^2 \leq \text{Var}(X)\text{Var}(Y)$, with equality iff $X - \mu_X$ and $Y - \mu_Y$ are linearly dependent. $\square$

**Example 4.4.8 (Correlation — compute and interpret).** Using the modified joint distribution from Example 4.4.7 ($\text{Cov}(X,Y) = 1/8$), compute $\rho$.

We need $\text{Var}(X)$ and $\text{Var}(Y)$. Since $X \sim \text{Bernoulli}(1/2)$: $\text{Var}(X) = \tfrac{1}{2}(1 - \tfrac{1}{2}) = \tfrac{1}{4}$. By symmetry, $\text{Var}(Y) = \tfrac{1}{4}$.

$$\rho(X,Y) = \frac{\text{Cov}(X,Y)}{\sigma_X \sigma_Y} = \frac{1/8}{\sqrt{1/4}\cdot\sqrt{1/4}} = \frac{1/8}{1/4} = \frac{1}{2} = 0.5$$

Interpretation: a moderate positive linear association. When $X = 1$, $Y$ is more likely to be $1$ (probability $3/4$ vs. $1/4$).

| $\rho$ value | Interpretation |
|-------------|---------------|
| $\rho = 1$ | Perfect positive linear relationship |
| $0.7 < \rho < 1$ | Strong positive |
| $0 < \rho < 0.3$ | Weak positive |
| $\rho = 0$ | No **linear** relationship (not necessarily independent!) |
| $\rho = -1$ | Perfect negative linear relationship |

**Correlation $\neq$ Causation.** Classic example: ice cream sales and drowning deaths are positively correlated, but neither causes the other — temperature is the confounding variable. In ML, spurious correlations in training data can lead to models that fail under distribution shift.

---

## 7. Higher Moments

**Definition 4.5.8 (Moments).** The $k$-th moment of $X$ is $\mathbb{E}[X^k]$. The $k$-th central moment is $\mathbb{E}[(X - \mu)^k]$.

| Moment | Formula | Measures |
|--------|---------|----------|
| 1st (mean) | $\mu = \mathbb{E}[X]$ | Location (center) |
| 2nd central (variance) | $\sigma^2 = \mathbb{E}[(X-\mu)^2]$ | Spread |
| 3rd standardized (skewness) | $\gamma_1 = \mathbb{E}\!\left[\left(\frac{X-\mu}{\sigma}\right)^3\right]$ | Asymmetry |
| 4th standardized (kurtosis) | $\gamma_2 = \mathbb{E}\!\left[\left(\frac{X-\mu}{\sigma}\right)^4\right]$ | Tail heaviness |

```
SKEWNESS AND KURTOSIS

  Left-skewed (γ₁ < 0)     Symmetric (γ₁ = 0)      Right-skewed (γ₁ > 0)

       ╱╲                       ╱╲                          ╱╲
      ╱  ╲                     ╱  ╲                        ╱  ╲
    ╱    ╲╲                   ╱    ╲                      ╱╱    ╲
  ╱╱       ╲╲               ╱╱      ╲╲                  ╱╱       ╲╲
  ──────────────            ──────────────              ──────────────
  tail ← mean              mean = median               mean → tail

  High kurtosis: heavy tails (more outliers than Gaussian)
  Gaussian has kurtosis = 3 (excess kurtosis = 0)
```

*ML connection:* Skewness and kurtosis help diagnose feature distributions before training. Heavy-tailed features (high kurtosis) may need robust scaling or log-transforms. The Gaussian has excess kurtosis 0 — distributions with higher kurtosis have heavier tails and produce more extreme values.

**Example 4.4.9 (Skewness and kurtosis — coin-flip distribution).** Let $X \sim \text{Bernoulli}(p)$ with $p = 0.9$. Then $\mu = 0.9$, $\sigma^2 = 0.9 \cdot 0.1 = 0.09$, $\sigma = 0.3$.

Skewness: The standardized variable $(X - \mu)/\sigma$ takes value $(0 - 0.9)/0.3 = -3$ with probability $0.1$ and $(1 - 0.9)/0.3 = 1/3$ with probability $0.9$.

$$\gamma_1 = \mathbb{E}\!\left[\left(\tfrac{X-\mu}{\sigma}\right)^3\right] = 0.1 \cdot (-3)^3 + 0.9 \cdot (1/3)^3 = 0.1(-27) + 0.9(1/27) = -2.7 + 1/30 \approx -2.667$$

Kurtosis: $\gamma_2 = 0.1 \cdot (-3)^4 + 0.9 \cdot (1/3)^4 = 0.1(81) + 0.9(1/81) = 8.1 + 1/90 \approx 8.111$

Excess kurtosis $= 8.111 - 3 = 5.111$. The strong negative skew ($\gamma_1 \approx -2.67$) reflects the long left tail (rare $0$'s far from the mean). The high excess kurtosis indicates extreme concentration with occasional outliers.

---

## 8. Moment Generating Functions

**Definition 4.5.9 (MGF).** The moment generating function of $X$ is:

$$M_X(t) = \mathbb{E}[e^{tX}]$$

defined for all $t$ in a neighborhood of $0$ where the expectation exists.

**Theorem 4.5.10 (Moments from the MGF).** If $M_X(t)$ exists in a neighborhood of $0$:

$$\mathbb{E}[X^k] = M_X^{(k)}(0) = \left.\frac{d^k}{dt^k} M_X(t)\right|_{t=0}$$

*Proof.* $M_X(t) = \mathbb{E}[e^{tX}] = \mathbb{E}\!\left[\sum_{k=0}^{\infty} \frac{(tX)^k}{k!}\right] = \sum_{k=0}^{\infty} \frac{t^k}{k!} \mathbb{E}[X^k]$, interchanging sum and expectation (justified by dominated convergence when the MGF exists in a neighborhood). Differentiating $k$ times and setting $t = 0$ extracts $\mathbb{E}[X^k]$. $\square$

**Example 4.4.10 (MGF — Exponential distribution).** Let $X \sim \text{Exp}(\lambda)$ with $\lambda = 2$, so $f(x) = 2e^{-2x}$ for $x \geq 0$.

$$M_X(t) = \int_0^\infty e^{tx} \cdot 2e^{-2x}\, dx = 2\int_0^\infty e^{-(2-t)x}\, dx = \frac{2}{2-t}, \quad t < 2$$

Extract moments by differentiating:

$$M_X'(t) = \frac{2}{(2-t)^2} \implies \mathbb{E}[X] = M_X'(0) = \frac{2}{4} = \frac{1}{2}$$

$$M_X''(t) = \frac{4}{(2-t)^3} \implies \mathbb{E}[X^2] = M_X''(0) = \frac{4}{8} = \frac{1}{2}$$

$$\text{Var}(X) = \mathbb{E}[X^2] - (\mathbb{E}[X])^2 = \frac{1}{2} - \frac{1}{4} = \frac{1}{4} = \frac{1}{\lambda^2} \;\checkmark$$

**Theorem 4.5.11 (Uniqueness).** If $M_X(t) = M_Y(t)$ for all $t$ in a neighborhood of $0$, then $X$ and $Y$ have the same distribution.

| Distribution | MGF $M_X(t)$ | Derived moments |
|-------------|---------------|-----------------|
| Bernoulli($p$) | $1 - p + pe^t$ | $\mathbb{E}[X] = p$, $\text{Var}(X) = p(1-p)$ |
| Poisson($\lambda$) | $e^{\lambda(e^t - 1)}$ | $\mathbb{E}[X] = \lambda$, $\text{Var}(X) = \lambda$ |
| $\mathcal{N}(\mu, \sigma^2)$ | $e^{\mu t + \sigma^2 t^2/2}$ | $\mathbb{E}[X] = \mu$, $\text{Var}(X) = \sigma^2$ |
| Exponential($\lambda$) | $\frac{\lambda}{\lambda - t}$, $t < \lambda$ | $\mathbb{E}[X] = 1/\lambda$, $\text{Var}(X) = 1/\lambda^2$ |

**Definition 4.5.10 (Characteristic Function).** When the MGF may not exist (e.g., Cauchy distribution), the characteristic function always does:

$$\varphi_X(t) = \mathbb{E}[e^{itX}]$$

This exists for every distribution and uniquely determines it. It is the Fourier transform of the density.

---

## 9. Jensen's Inequality

**Theorem 4.5.12 (Jensen's Inequality).** If $g$ is a convex function and $\mathbb{E}[X]$ exists:

$$\mathbb{E}[g(X)] \geq g(\mathbb{E}[X])$$

If $g$ is concave, the inequality reverses: $\mathbb{E}[g(X)] \leq g(\mathbb{E}[X])$.

*Proof.* Since $g$ is convex, it lies above every tangent line. At the point $\mu = \mathbb{E}[X]$, there exists a supporting line $g(x) \geq g(\mu) + g'(\mu)(x - \mu)$ for all $x$ (where $g'(\mu)$ is a subgradient if $g$ is not differentiable). Taking expectations of both sides:

$$\mathbb{E}[g(X)] \geq g(\mu) + g'(\mu)\underbrace{(\mathbb{E}[X] - \mu)}_{= 0} = g(\mu) = g(\mathbb{E}[X]) \quad \square$$

```
JENSEN'S INEQUALITY — VISUAL

  g(x) convex (curves upward)

  g(x) │           ·
       │         ·
       │       ·           E[g(X)]  ← always higher
       │    ·  ─ ─ ─ ─ ─ ─ ─ ● ─ ─
       │  ·        ·       ╱
       │·     ─ ─ ● ─ ─ ─    g(E[X])  ← always lower
       │       g(E[X])
       └──────────┼──────────── x
               E[X]

  The curve g lies above the chord connecting any two points.
  Averaging inputs then applying g ≤ applying g then averaging.
```

**Example 4.4.11 (Jensen's inequality — concave log).** Let $X$ take values $1$ and $3$ each with probability $1/2$. Since $g(x) = \ln(x)$ is concave, Jensen gives $\mathbb{E}[\ln X] \leq \ln(\mathbb{E}[X])$.

$$\mathbb{E}[X] = \tfrac{1}{2}(1) + \tfrac{1}{2}(3) = 2$$

$$\mathbb{E}[\ln X] = \tfrac{1}{2}\ln 1 + \tfrac{1}{2}\ln 3 = 0 + \tfrac{1}{2}(1.099) = 0.549$$

$$\ln(\mathbb{E}[X]) = \ln 2 = 0.693$$

Indeed $0.549 \leq 0.693$ $\checkmark$. The Jensen gap is $0.693 - 0.549 = 0.144$. This gap equals $D_{KL}$ in the ELBO derivation — a tighter bound means the gap (and the approximation error) is smaller.

*ML connection:* Jensen's inequality is the key step in deriving the **Evidence Lower Bound (ELBO)** for variational autoencoders. Since $\log$ is concave:

$$\log p(\mathbf{x}) = \log \mathbb{E}_{q(\mathbf{z}|\mathbf{x})}\!\left[\frac{p(\mathbf{x}, \mathbf{z})}{q(\mathbf{z}|\mathbf{x})}\right] \geq \mathbb{E}_{q}\!\left[\log \frac{p(\mathbf{x}, \mathbf{z})}{q(\mathbf{z}|\mathbf{x})}\right] = \text{ELBO}$$

Maximizing the ELBO is a tractable surrogate for maximizing the intractable log-evidence $\log p(\mathbf{x})$.

---

## 10. Applications in ML

### 10.1 Loss Functions as Expected Values

Every loss function is an expectation over the data distribution:

$$\text{Risk}(h) = \mathbb{E}_{(X,Y) \sim \mathcal{D}}\left[\ell(h(X), Y)\right]$$

Since the true distribution $\mathcal{D}$ is unknown, we minimize the **empirical risk**:

$$\hat{R}(h) = \frac{1}{n}\sum_{i=1}^n \ell(h(x_i), y_i) \approx \mathbb{E}[\ell(h(X), Y)]$$

By the law of large numbers, $\hat{R}(h) \to R(h)$ as $n \to \infty$.

### 10.2 Bias-Variance Tradeoff

For a model $\hat{f}$ trained on data, the expected prediction error decomposes as:

$$\mathbb{E}\left[(Y - \hat{f}(X))^2\right] = \underbrace{\text{Bias}(\hat{f})^2}_{\text{systematic error}} + \underbrace{\text{Var}(\hat{f})}_{\text{sensitivity to training data}} + \underbrace{\sigma^2}_{\text{irreducible noise}}$$

```
BIAS-VARIANCE TRADEOFF

  Error
    ▲
    │ ╲  Total Error
    │  ╲  ╱
    │   ╲╱     ╱
    │    ╲    ╱  Variance
    │   ╱ ╲  ╱
    │  ╱   ╲╱
    │ ╱    ╱╲
    │╱    ╱  ╲
    │    ╱    Bias²
    │   ╱
    └──────────────────▶ Model complexity
      Simple            Complex
      (high bias,       (low bias,
       low variance)     high variance)
```

### 10.3 Batch Normalization

Given a mini-batch $\{x_1, \ldots, x_m\}$ within a neural network layer:

$$\mu_B = \frac{1}{m}\sum_{i=1}^m x_i, \qquad \sigma_B^2 = \frac{1}{m}\sum_{i=1}^m (x_i - \mu_B)^2$$

$$\hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}, \qquad y_i = \gamma \hat{x}_i + \beta$$

This standardizes activations to zero mean and unit variance (using $\mathbb{E}$ and $\text{Var}$), then applies learnable parameters $\gamma, \beta$. It stabilizes training by reducing internal covariate shift.

### 10.4 Covariance Matrix in PCA

PCA finds the orthogonal directions of maximum variance by eigendecomposing the sample covariance matrix $\hat{\Sigma} = \frac{1}{n}\sum_{i=1}^n (\mathbf{x}_i - \bar{\mathbf{x}})(\mathbf{x}_i - \bar{\mathbf{x}})^T$. The $k$-th principal component is the eigenvector corresponding to the $k$-th largest eigenvalue, and the eigenvalue equals the variance along that direction.

### 10.5 Jensen's Inequality in the ELBO

In variational inference, we seek to approximate an intractable posterior $p(\mathbf{z}|\mathbf{x})$ with a tractable $q(\mathbf{z}|\mathbf{x})$. Jensen's inequality (applied to the concave $\log$) gives:

$$\log p(\mathbf{x}) \geq \underbrace{\mathbb{E}_q[\log p(\mathbf{x}|\mathbf{z})]}_{\text{reconstruction}} - \underbrace{D_{KL}(q(\mathbf{z}|\mathbf{x}) \| p(\mathbf{z}))}_{\text{regularization}} = \text{ELBO}$$

The gap between $\log p(\mathbf{x})$ and the ELBO is exactly $D_{KL}(q \| p(\mathbf{z}|\mathbf{x})) \geq 0$, so a tighter ELBO means a better approximation.

---

## Exercises

**★ Basic**

1. Compute $\mathbb{E}[X]$ and $\text{Var}(X)$ for a fair six-sided die ($X$ uniform on $\{1, 2, 3, 4, 5, 6\}$).

2. If $X \sim \mathcal{N}(3, 4)$, compute $\mathbb{E}[2X + 5]$ and $\text{Var}(2X + 5)$.

3. Show that $\text{Var}(X) = 0$ implies $X = \mathbb{E}[X]$ with probability 1.

4. For $X \sim \text{Bernoulli}(p)$, derive the MGF $M_X(t) = 1 - p + pe^t$ and use it to compute $\mathbb{E}[X]$ and $\mathbb{E}[X^2]$.

**★★ Intermediate**

5. Prove that for any random variable $X$: $\text{Var}(X) = \min_{c \in \mathbb{R}} \mathbb{E}[(X - c)^2]$. (Hint: the minimum is achieved at $c = \mathbb{E}[X]$.)

6. Let $X_1, \ldots, X_n$ be i.i.d. with mean $\mu$ and variance $\sigma^2$. Show that $\text{Var}(\bar{X}) = \sigma^2/n$ where $\bar{X} = \frac{1}{n}\sum X_i$. Why does this matter for stochastic gradient descent?

7. If $(X, Y)$ has covariance matrix $\Sigma = \begin{pmatrix} 4 & 3 \\ 3 & 9 \end{pmatrix}$, find $\text{Var}(X)$, $\text{Var}(Y)$, $\text{Cov}(X,Y)$, $\rho(X,Y)$, and $\text{Var}(2X - Y)$.

8. Use the MGF of $X \sim \mathcal{N}(\mu, \sigma^2)$ to prove that $aX + b \sim \mathcal{N}(a\mu + b, \, a^2\sigma^2)$.

**★★★ Challenging**

9. **(Jensen's gap)** Let $X \sim \text{Uniform}(1, 3)$. Compute $\mathbb{E}[X^2]$ and $(\mathbb{E}[X])^2$. Verify that Jensen's inequality holds (since $g(x) = x^2$ is convex) and compute the gap $\mathbb{E}[X^2] - (\mathbb{E}[X])^2$. Relate this to the variance.

10. **(Bias-variance decomposition)** Derive the bias-variance-noise decomposition for squared loss: starting from $\mathbb{E}_{D,\epsilon}[(y - \hat{f}(x))^2]$ where $y = f(x) + \epsilon$, expand and group terms to obtain $\text{Bias}^2 + \text{Var} + \sigma^2$.

11. **(ELBO derivation)** Starting from $\log p(\mathbf{x}) = \log \int p(\mathbf{x}, \mathbf{z}) \, d\mathbf{z}$, introduce an arbitrary distribution $q(\mathbf{z})$, apply Jensen's inequality to the concave $\log$, and derive the ELBO. Show that the gap is $D_{KL}(q(\mathbf{z}) \| p(\mathbf{z}|\mathbf{x}))$.

12. **(Stein's lemma)** Prove that if $X \sim \mathcal{N}(\mu, \sigma^2)$ and $g$ is differentiable with $\mathbb{E}[|g'(X)|] < \infty$, then $\text{Cov}(X, g(X)) = \sigma^2 \mathbb{E}[g'(X)]$. Explain why this is useful for variance reduction in Monte Carlo estimation.

---

## Related Topics

- [Random Variables](random-variables.md) -- PMF, PDF, CDF definitions used throughout
- [Distributions](distributions.md) -- computing moments for specific distributions
- [Joint Distributions](joint-distributions.md) -- covariance and correlation in multivariate settings
- [Limit Theorems](limit-theorems.md) -- LLN justifies empirical risk, CLT gives confidence intervals
- [Bayesian Inference](bayesian-inference.md) -- posterior expectations and the ELBO
- [Maximum Likelihood](maximum-likelihood.md) -- Fisher information involves $\text{Var}(\nabla \log p)$
- [Eigenvalues & Eigenvectors](../linear-algebra/eigenvalues.md) -- covariance matrix eigendecomposition in PCA
