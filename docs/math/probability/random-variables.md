# Random Variables

A random variable is a bridge between the abstract world of probability spaces and the concrete world of numbers. When we say "the model's loss is 0.34" or "the latent code is $z = (-0.7, 1.2)$," we are evaluating random variables — measurable functions that assign numerical values to random outcomes. Every quantity in machine learning that involves uncertainty — predictions, losses, latent representations, gradients under dropout — is formally a random variable. Making this precise unlocks the machinery of expectation, variance, and the distributions that power modern generative models.

---

## Prerequisites

- [Probability Foundations](probability-foundations.md) — sample spaces, $\sigma$-algebras, probability measures
- [Calculus - Integration](../calculus/integration.md) — Riemann and Lebesgue integration for continuous densities

---

## 1. Definition

**Definition 4.3.1 (Random Variable).** Let $(\Omega, \mathcal{F}, P)$ be a probability space. A *random variable* is a function $X : \Omega \to \mathbb{R}$ that is *measurable* with respect to $\mathcal{F}$, meaning:

$$X^{-1}((-\infty, x]) = \{\omega \in \Omega : X(\omega) \leq x\} \in \mathcal{F} \quad \text{for all } x \in \mathbb{R}$$

The measurability condition ensures that we can compute $P(X \leq x)$ for any real number $x$ — without it, the probability would be undefined.

```
THE RANDOM VARIABLE AS A FUNCTION

  Sample Space Ω              Real Line ℝ
  (abstract outcomes)         (numbers we compute with)

  ┌─────────────┐      X      ┌──────────────────────┐
  │  ω₁ ●───────│────────────▶│─── x₁               │
  │  ω₂ ●───────│────────────▶│─── x₂               │
  │  ω₃ ●───────│────────────▶│─── x₁  (same value) │
  │  ω₄ ●───────│────────────▶│─── x₃               │
  └─────────────┘             └──────────────────────┘

  X maps outcomes to numbers.
  Multiple outcomes can map to the same number.
  "X = x₁" really means {ω ∈ Ω : X(ω) = x₁} = {ω₁, ω₃}
```

**Example 4.3.1:** Roll a fair six-sided die. The sample space is $\Omega = \{\omega_1, \omega_2, \omega_3, \omega_4, \omega_5, \omega_6\}$ (the six faces). Define $X(\omega_i) = i$. Then $X$ is a random variable mapping abstract outcomes to numbers:

$$X(\omega_3) = 3, \qquad P(X \leq 4) = P(\{\omega_1, \omega_2, \omega_3, \omega_4\}) = \tfrac{4}{6} = \tfrac{2}{3}$$

The pre-image $X^{-1}((-\infty, 4]) = \{\omega_1, \omega_2, \omega_3, \omega_4\} \in \mathcal{F}$, so the measurability condition is satisfied.

*ML connection:* In a generative model, the sample space $\Omega$ might be the set of all possible images. A random variable $X$ could extract a single pixel intensity, or more commonly, we work with random vectors $\mathbf{X} : \Omega \to \mathbb{R}^d$ where $d$ is the dimension of the latent space. When we write $\mathbf{z} \sim \mathcal{N}(\mathbf{0}, I)$ for a VAE's latent code, $\mathbf{z}$ is a random vector — each component $z_i$ is a random variable.

---

## 2. Discrete Random Variables

**Definition 4.3.2 (Discrete Random Variable).** A random variable $X$ is *discrete* if it takes values in a countable set $\{x_1, x_2, x_3, \ldots\}$.

**Definition 4.3.3 (Probability Mass Function).** The *PMF* of a discrete random variable $X$ is:

$$p_X(x) = P(X = x) = P(\{\omega \in \Omega : X(\omega) = x\})$$

**Properties of the PMF:**

1. $p_X(x) \geq 0$ for all $x$
2. $\sum_{x \in \text{support}} p_X(x) = 1$
3. $P(X \in A) = \sum_{x \in A} p_X(x)$ for any set $A$

**Definition 4.3.4 (Support).** The *support* of $X$ is $\text{supp}(X) = \{x : p_X(x) > 0\}$.

```
PMF — PROBABILITY MASS FUNCTION

  p(x)
  0.30 │        ┌───┐
       │        │   │
  0.25 │  ┌───┐ │   │
       │  │   │ │   │
  0.20 │  │   │ │   │ ┌───┐
       │  │   │ │   │ │   │
  0.15 │  │   │ │   │ │   │
       │  │   │ │   │ │   │ ┌───┐
  0.10 │  │   │ │   │ │   │ │   │
       │  │   │ │   │ │   │ │   │
  0.05 │  │   │ │   │ │   │ │   │
       │  │   │ │   │ │   │ │   │
  0.00 └──┴───┴─┴───┴─┴───┴─┴───┴──
          x₁     x₂     x₃     x₄

  Each bar height IS the probability: p(x₂) = 0.30
  Bar heights sum to 1.
  Probability is concentrated at discrete points.
```

**Example 4.3.2:** Let $X$ be the outcome of rolling a fair six-sided die. The PMF is:

$$p_X(k) = \frac{1}{6} \quad \text{for } k \in \{1, 2, 3, 4, 5, 6\}$$

**Verification:** $\sum_{k=1}^{6} p_X(k) = 6 \times \frac{1}{6} = 1$ $\checkmark$

To compute $P(X \in \{2, 3, 5\})$: sum the PMF values $= \frac{1}{6} + \frac{1}{6} + \frac{1}{6} = \frac{1}{2}$.

**Example 4.3.3 (Indicator Random Variable):** Let $A$ = "die shows an even number" = $\{2, 4, 6\}$. The *indicator random variable* $I_A$ is defined as:

$$I_A(\omega) = \begin{cases} 1 & \text{if } \omega \in A \\ 0 & \text{if } \omega \notin A \end{cases}$$

Its PMF is $p_{I_A}(1) = P(A) = \frac{3}{6} = \frac{1}{2}$ and $p_{I_A}(0) = 1 - P(A) = \frac{1}{2}$. Thus:

$$\mathbb{E}[I_A] = 1 \cdot P(A) + 0 \cdot P(A^c) = P(A) = \frac{1}{2}$$

This illustrates the general result: $\mathbb{E}[I_A] = P(A)$ for any event $A$.

*ML connection:* The output of a classifier after softmax is a PMF over classes. If a model outputs $\text{softmax}(\mathbf{z}) = (0.7, 0.2, 0.1)$ for classes (cat, dog, bird), this defines a categorical random variable $Y$ with $p_Y(\text{cat}) = 0.7$, $p_Y(\text{dog}) = 0.2$, $p_Y(\text{bird}) = 0.1$. Cross-entropy loss $-\sum_k y_k \log \hat{p}_k$ measures how well this PMF matches the true label.

---

## 3. Continuous Random Variables

**Definition 4.3.5 (Continuous Random Variable).** A random variable $X$ is *continuous* if there exists a non-negative function $f_X : \mathbb{R} \to [0, \infty)$ such that for every interval $[a, b]$:

$$P(a \leq X \leq b) = \int_a^b f_X(x) \, dx$$

The function $f_X$ is called the *probability density function* (PDF).

**Theorem 4.3.1 (The density is NOT a probability).** For a continuous random variable $X$:

$$P(X = x) = \int_x^x f_X(t)\, dt = 0 \quad \text{for every specific } x$$

The density $f_X(x)$ can exceed $1$. It represents probability *per unit length*, not probability itself.

*Proof.* $P(X = x) = P(x \leq X \leq x) = \int_x^x f_X(t)\, dt = 0$ since the integral over an interval of zero width is zero. $\square$

**Properties of the PDF:**

1. $f_X(x) \geq 0$ for all $x$
2. $\int_{-\infty}^{\infty} f_X(x) \, dx = 1$
3. $f_X(x)$ can be greater than $1$ (e.g., $\text{Uniform}(0, 0.5)$ has $f(x) = 2$)

```
PDF — PROBABILITY DENSITY FUNCTION

  f(x)
       │
       │         ╱╲
       │        ╱  ╲          Density CAN exceed 1
       │       ╱    ╲         (it's not a probability!)
       │      ╱ ░░░░ ╲
       │     ╱ ░░░░░░ ╲       Shaded area = P(a ≤ X ≤ b)
       │    ╱ ░░░░░░░░ ╲      (THIS is the probability)
       │   ╱ ░░░░░░░░░░ ╲
       │──╱──░░░░░░░░░░──╲──────
       └──────a────────b──────── x

  P(a ≤ X ≤ b) = ∫ₐᵇ f(x) dx = area under the curve
  P(X = a) = 0  (single point has zero area)
  Total area under curve = 1
```

**Example 4.3.4:** Let $f(x) = 2x$ for $x \in [0, 1]$ and $f(x) = 0$ otherwise. Verify this is a valid PDF:

$$\int_{-\infty}^{\infty} f(x)\, dx = \int_0^1 2x\, dx = \left[x^2\right]_0^1 = 1 - 0 = 1 \quad \checkmark$$

Also, $f(x) = 2x \geq 0$ for all $x \in [0, 1]$, so both PDF properties hold. Note that $P(X = 0.5) = 0$ even though $f(0.5) = 1$; the density is not a probability. To find $P(0.5 \leq X \leq 0.75)$:

$$P(0.5 \leq X \leq 0.75) = \int_{0.5}^{0.75} 2x\, dx = \left[x^2\right]_{0.5}^{0.75} = 0.5625 - 0.25 = 0.3125$$

*ML connection:* When we say a latent variable $z$ is "drawn from $\mathcal{N}(0, 1)$," we mean $z$ has PDF $f(z) = \frac{1}{\sqrt{2\pi}} e^{-z^2/2}$. The probability of $z$ being in any specific interval $[a, b]$ is the area under this bell curve between $a$ and $b$. The density at the mode $f(0) = 1/\sqrt{2\pi} \approx 0.399$ — less than $1$ in this case, but for a $\mathcal{N}(0, 0.1)$ the peak density is about $3.99$.

---

## 4. Cumulative Distribution Function

**Definition 4.3.6 (CDF).** The *cumulative distribution function* of any random variable $X$ is:

$$F_X(x) = P(X \leq x)$$

**Theorem 4.3.2 (Properties of the CDF).** For any random variable $X$:

1. $F_X$ is non-decreasing: $x_1 < x_2 \implies F_X(x_1) \leq F_X(x_2)$
2. $\lim_{x \to -\infty} F_X(x) = 0$ and $\lim_{x \to +\infty} F_X(x) = 1$
3. $F_X$ is right-continuous: $\lim_{h \to 0^+} F_X(x + h) = F_X(x)$
4. $P(a < X \leq b) = F_X(b) - F_X(a)$

*Proof of (4).* $\{X \leq b\} = \{X \leq a\} \cup \{a < X \leq b\}$, and these events are disjoint. By additivity: $F_X(b) = F_X(a) + P(a < X \leq b)$. $\square$

**Theorem 4.3.3 (CDF-PMF relationship).** For a discrete random variable:

$$F_X(x) = \sum_{x_i \leq x} p_X(x_i)$$

The CDF is a step function with jumps of size $p_X(x_i)$ at each point in the support.

**Example 4.3.5:** For the fair die ($X \in \{1,2,3,4,5,6\}$, each with probability $\frac{1}{6}$), the CDF is:

$$F_X(x) = \begin{cases} 0 & x < 1 \\ \frac{1}{6} & 1 \leq x < 2 \\ \frac{2}{6} & 2 \leq x < 3 \\ \frac{3}{6} & 3 \leq x < 4 \\ \frac{4}{6} & 4 \leq x < 5 \\ \frac{5}{6} & 5 \leq x < 6 \\ 1 & x \geq 6 \end{cases}$$

Check: $F_X(3.5) = \frac{3}{6} = \frac{1}{2}$ (three values $\leq 3.5$). Also $P(2 < X \leq 5) = F_X(5) - F_X(2) = \frac{5}{6} - \frac{2}{6} = \frac{1}{2}$.

**Theorem 4.3.4 (CDF-PDF relationship).** For a continuous random variable (where $f_X$ is continuous):

$$F_X(x) = \int_{-\infty}^x f_X(t) \, dt \qquad \text{and} \qquad f_X(x) = F_X'(x)$$

*Proof.* The first equation follows from $F_X(x) = P(X \leq x) = \int_{-\infty}^x f_X(t)\, dt$. The second follows by the Fundamental Theorem of Calculus. $\square$

**Example 4.3.6:** For the PDF $f(x) = 2x$ on $[0, 1]$ (from Example 4.3.4), compute the CDF:

$$F_X(x) = \int_{-\infty}^x f(t)\, dt = \int_0^x 2t\, dt = \left[t^2\right]_0^x = x^2 \quad \text{for } x \in [0, 1]$$

So $F_X(x) = \begin{cases} 0 & x < 0 \\ x^2 & 0 \leq x \leq 1 \\ 1 & x > 1 \end{cases}$

**Verify:** $F_X'(x) = 2x = f(x)$ $\checkmark$, and $F_X(0) = 0$, $F_X(1) = 1$ $\checkmark$. Also: $P(X \leq 0.5) = F_X(0.5) = 0.25$.

```
CDF — DISCRETE (step function)          CDF — CONTINUOUS (smooth)

  F(x)                                    F(x)
  1.0 ┤                ●─────             1.0 ┤                    ────────
      │                                       │                 ╱
  0.8 ┤         ●──────○                  0.8 ┤               ╱
      │                                       │             ╱
  0.5 ┤    ●────○                         0.5 ┤           ╱
      │                                       │         ╱
  0.2 ┤●───○                              0.2 ┤       ╱
      │                                       │     ╱
  0.0 ┤────                               0.0 ┤────
      └──┬────┬──────┬──── x               └────────────────────── x
         x₁   x₂    x₃
                                           F'(x) = f(x) (the PDF)
  Jump at xᵢ = p(xᵢ) (the PMF)
  ● = included, ○ = excluded
```

*ML connection:* The CDF is central to computing $p$-values in hypothesis testing ("is this model improvement statistically significant?") and to probability integral transform in normalizing flows: if $X$ has CDF $F_X$, then $F_X(X) \sim \text{Uniform}(0, 1)$. This fact underlies the transformation between simple and complex distributions.

---

## 5. Functions of Random Variables

**Definition 4.3.7 (Transformation).** If $X$ is a random variable and $g : \mathbb{R} \to \mathbb{R}$ is a (measurable) function, then $Y = g(X)$ is also a random variable.

**Theorem 4.3.5 (CDF method).** For $Y = g(X)$:

$$F_Y(y) = P(g(X) \leq y)$$

Compute the right-hand side by solving $g(X) \leq y$ for $X$ and using the known distribution of $X$.

**Example 4.3.7 (Transformation via CDF method):** Let $X \sim \text{Uniform}(0, 1)$ and $Y = X^2$. Find the distribution of $Y$.

**Step 1 — CDF of $Y$:** For $0 \leq y \leq 1$:

$$F_Y(y) = P(Y \leq y) = P(X^2 \leq y) = P(X \leq \sqrt{y}) = \sqrt{y}$$

(since $X \geq 0$ and $F_X(x) = x$ for Uniform$(0,1)$)

**Step 2 — PDF of $Y$:** Differentiate:

$$f_Y(y) = F_Y'(y) = \frac{1}{2\sqrt{y}} \quad \text{for } y \in (0, 1]$$

**Verify:** $\int_0^1 \frac{1}{2\sqrt{y}}\, dy = \left[\sqrt{y}\right]_0^1 = 1$ $\checkmark$. Note that $f_Y(y) \to \infty$ as $y \to 0^+$, illustrating that densities can be unbounded.

**Theorem 4.3.6 (Change of variables — monotone case).** If $g$ is strictly monotone and differentiable with inverse $g^{-1}$, and $X$ has PDF $f_X$, then $Y = g(X)$ has PDF:

$$f_Y(y) = f_X(g^{-1}(y)) \cdot \left|\frac{d}{dy} g^{-1}(y)\right|$$

*Proof.* Assume $g$ is strictly increasing (the decreasing case is similar). Then $F_Y(y) = P(g(X) \leq y) = P(X \leq g^{-1}(y)) = F_X(g^{-1}(y))$. Differentiating by the chain rule: $f_Y(y) = f_X(g^{-1}(y)) \cdot (g^{-1})'(y)$. The absolute value handles both increasing and decreasing $g$. $\square$

**Example.** Let $X \sim \mathcal{N}(0, 1)$ and $Y = e^X$ (so $Y$ is log-normal). Here $g(x) = e^x$, $g^{-1}(y) = \ln y$, $(g^{-1})'(y) = 1/y$. Then:

$$f_Y(y) = \frac{1}{\sqrt{2\pi}} e^{-(\ln y)^2 / 2} \cdot \frac{1}{y} \quad \text{for } y > 0$$

*ML connection:* Normalizing flows build complex distributions by composing invertible transformations $\mathbf{z}_K = g_K \circ \cdots \circ g_1(\mathbf{z}_0)$, starting from a simple base distribution $\mathbf{z}_0 \sim \mathcal{N}(\mathbf{0}, I)$. The change-of-variables formula (generalized to multiple dimensions with the Jacobian determinant) gives the density of the output, enabling exact likelihood computation.

---

## 6. The Reparameterization Trick

**Definition 4.3.8 (Reparameterization).** Instead of sampling $z \sim q_\phi(z)$ directly (which blocks gradient flow through the sampling operation), express the random variable as a deterministic, differentiable function of a parameter-free noise source:

$$z = g(\phi, \epsilon) \quad \text{where } \epsilon \sim p(\epsilon) \text{ is independent of } \phi$$

**Theorem 4.3.7 (Gaussian reparameterization).** If $z \sim \mathcal{N}(\mu, \sigma^2)$, then:

$$z = \mu + \sigma \cdot \epsilon \qquad \text{where } \epsilon \sim \mathcal{N}(0, 1)$$

This is a valid reparameterization since $\mu + \sigma\epsilon$ has mean $\mu$ and variance $\sigma^2$.

**Example 4.3.9:** A VAE encoder outputs $\mu = 2.0$ and $\sigma = 0.5$. We draw $\epsilon = 1.3$ from $\mathcal{N}(0,1)$. Then:

$$z = \mu + \sigma \cdot \epsilon = 2.0 + 0.5 \times 1.3 = 2.65$$

The gradients needed for backpropagation are:

$$\frac{\partial z}{\partial \mu} = 1, \qquad \frac{\partial z}{\partial \sigma} = \epsilon = 1.3$$

These are well-defined constants (given the sampled $\epsilon$), so the loss gradient flows back through $z$ to update $\mu$ and $\sigma$. Without reparameterization, $z \sim \mathcal{N}(2.0, 0.25)$ would be a stochastic node with no gradient.

*Proof.* Let $\epsilon \sim \mathcal{N}(0, 1)$. Then $\mathbb{E}[\mu + \sigma\epsilon] = \mu + \sigma \cdot 0 = \mu$ and $\text{Var}(\mu + \sigma\epsilon) = \sigma^2 \text{Var}(\epsilon) = \sigma^2$. Since affine transformations of Gaussians are Gaussian, $\mu + \sigma\epsilon \sim \mathcal{N}(\mu, \sigma^2)$. $\square$

```
WHY REPARAMETERIZATION MATTERS

  WITHOUT reparameterization           WITH reparameterization
  (cannot backpropagate)               (gradients flow through)

  ┌────────┐     z ~ N(μ,σ²)          ┌────────┐    ε ~ N(0,1)
  │Encoder │──▶ SAMPLE ──▶ Decoder    │Encoder │──▶ z = μ + σε ──▶ Decoder
  │ (μ,σ)  │     ╳ no grad            │ (μ,σ)  │     ✓ grad flows!
  └────────┘                           └────────┘
                                            │ ∂z/∂μ = 1
  Sampling is stochastic —                  │ ∂z/∂σ = ε
  no gradient defined                       ▼
                                       Backprop works!
```

*ML connection:* The reparameterization trick is the key insight that makes Variational Autoencoders (VAEs) trainable via standard backpropagation. The encoder outputs $\mu$ and $\sigma$, we sample $\epsilon \sim \mathcal{N}(0, 1)$, and compute $z = \mu + \sigma \odot \epsilon$ (element-wise for vectors). Because $z$ is now a deterministic function of $\mu, \sigma, \epsilon$, gradients $\partial \mathcal{L}/\partial \mu$ and $\partial \mathcal{L}/\partial \sigma$ are well-defined. This same idea extends to other distributions and appears in stochastic variational inference, policy gradient methods (Gumbel-softmax for discrete variables), and diffusion models.

---

## 7. The Quantile Function

**Definition 4.3.9 (Quantile Function).** The *quantile function* (inverse CDF) of a random variable $X$ is:

$$Q_X(p) = F_X^{-1}(p) = \inf\{x \in \mathbb{R} : F_X(x) \geq p\} \qquad \text{for } p \in (0, 1)$$

For continuous, strictly increasing CDFs, this simplifies to the ordinary inverse: $Q_X(p) = F_X^{-1}(p)$.

**Example 4.3.8 (Finding the median from the CDF):** Using the CDF from Example 4.3.6, $F_X(x) = x^2$ for $x \in [0, 1]$ (where $f(x) = 2x$). Find the median.

The median is $Q_X(0.5)$: solve $F_X(x) = 0.5$:

$$x^2 = 0.5 \implies x = \sqrt{0.5} = \frac{\sqrt{2}}{2} \approx 0.707$$

So $P(X \leq 0.707) = 0.5$ and $P(X > 0.707) = 0.5$. Notice the median is above $0.5$ because the density $f(x) = 2x$ puts more probability mass near $x = 1$ than near $x = 0$.

Similarly, the 25th percentile: $x^2 = 0.25 \implies x = 0.5$, so $Q_X(0.25) = 0.5$.

**Theorem 4.3.8 (Inverse transform sampling).** If $U \sim \text{Uniform}(0, 1)$ and $F_X$ is a CDF, then:

$$X = Q_X(U) = F_X^{-1}(U) \sim F_X$$

*Proof.* $P(F_X^{-1}(U) \leq x) = P(U \leq F_X(x)) = F_X(x)$ since $U$ is uniform on $(0,1)$. $\square$

**Example 4.3.10 (Inverse transform sampling):** Generate a sample from $f(x) = 2x$ on $[0, 1]$ using a uniform random number. From Example 4.3.6, $F_X(x) = x^2$, so $Q_X(u) = F_X^{-1}(u) = \sqrt{u}$.

Suppose we draw $U = 0.36$ from $\text{Uniform}(0, 1)$. Then:

$$X = Q_X(0.36) = \sqrt{0.36} = 0.6$$

This produces a sample from $f(x) = 2x$. Repeating with many uniform draws yields values concentrated toward $x = 1$ (since the density increases linearly), matching the shape of $f$.

**Key quantiles:**

| Quantile | $p$ | Meaning |
|----------|-----|---------|
| Median | $0.5$ | Half the probability mass on each side |
| Lower quartile | $0.25$ | 25th percentile |
| 95th percentile | $0.95$ | Used in VaR (finance), confidence intervals |
| $z_{0.975} = 1.96$ | $0.975$ | Gives the 95% CI for $\mathcal{N}(0,1)$: $\mu \pm 1.96\sigma$ |

*ML connection:* Quantile regression predicts specific quantiles of the target distribution rather than just the mean. Instead of minimizing squared error, it minimizes the *pinball loss* $\rho_\tau(y - \hat{y}) = \max(\tau(y - \hat{y}), (\tau - 1)(y - \hat{y}))$ for quantile level $\tau$. This is valuable when you need prediction intervals (e.g., "the delivery will arrive between 2pm and 4pm with 90% confidence") rather than point predictions.

---

## 8. Applications in ML

### 8.1 Latent Variables in Generative Models

Generative models introduce latent (hidden) random variables $\mathbf{z}$ that capture the unobserved structure in data. The generative process is:

$$\mathbf{z} \sim p(\mathbf{z}) = \mathcal{N}(\mathbf{0}, I) \qquad \text{then} \qquad \mathbf{x} = G_\theta(\mathbf{z})$$

Here $\mathbf{z}$ is a continuous random vector (each component is a continuous random variable), and $G_\theta$ is a deterministic neural network that transforms the simple distribution $p(\mathbf{z})$ into a complex data distribution $p(\mathbf{x})$. This is precisely Section 5's change-of-variables in action — $G_\theta$ is the transformation $g$, and the Jacobian determines how density changes.

### 8.2 Model Outputs as Random Variables

A neural network classifier with softmax output defines a categorical random variable:

$$P(Y = k \mid \mathbf{x}) = \frac{e^{z_k}}{\sum_j e^{z_j}} \qquad \text{(PMF from Section 2)}$$

The logits $z_k$ are deterministic given $\mathbf{x}$, but $Y$ is a discrete random variable whose PMF is the softmax output. Sampling from this PMF (e.g., in language model generation) uses inverse transform sampling (Section 7) or the Gumbel-max trick.

### 8.3 CDFs in Statistical Testing

After training two models $A$ and $B$, testing whether $A$ is significantly better requires:

1. Compute test statistic $T$ (e.g., difference in accuracy)
2. Under the null hypothesis $H_0$: $T$ follows a known distribution with CDF $F_T$
3. $p$-value = $1 - F_T(t_{\text{observed}})$ = probability of seeing a result this extreme
4. Reject $H_0$ if $p$-value $< \alpha$ (typically $0.05$)

The CDF (Section 4) is the computational tool that converts a test statistic into a probability statement.

---

## Exercises

**★ Basic**

1. A fair die is rolled. Let $X$ be the number showing. Write the PMF, compute $F_X(3.5)$, and verify that $\sum p_X(x) = 1$.

2. Let $X$ have PDF $f(x) = 2x$ for $x \in [0, 1]$ and $0$ otherwise. Verify this is a valid PDF, compute $F_X(x)$, and find $P(0.25 \leq X \leq 0.75)$.

3. If $X \sim \text{Uniform}(0, 1)$, compute the CDF and PDF of $Y = -\ln(X)$. What named distribution is this?

**★★ Intermediate**

4. Let $X$ have CDF $F_X(x) = 1 - e^{-\lambda x}$ for $x \geq 0$. Derive the PDF. Using the quantile function, find the median of $X$ in terms of $\lambda$.

5. Prove that if $X$ is a continuous random variable with CDF $F_X$, then $U = F_X(X) \sim \text{Uniform}(0, 1)$. (This is the *probability integral transform*.)

6. In a VAE, the encoder outputs $\mu = 1.5$ and $\sigma = 0.8$. Given a noise sample $\epsilon = -0.3$, compute the latent value $z$ using the reparameterization trick. Write expressions for $\partial z / \partial \mu$ and $\partial z / \partial \sigma$.

**★★★ Challenging**

7. Let $X \sim \mathcal{N}(0, 1)$ and $Y = X^2$. Using the CDF method and change of variables, derive the PDF of $Y$. What named distribution is this? (Hint: it is fundamental to the chi-squared distribution.)

8. Prove the general change-of-variables formula (Theorem 4.3.6) for the strictly decreasing case, carefully handling the sign. Then show that both cases (increasing and decreasing) unify under the absolute value of the derivative.

9. Consider a normalizing flow with a single affine layer: $z = \sigma x + \mu$ where $x \sim \mathcal{N}(0, 1)$. Compute $f_Z(z)$ via the change-of-variables formula. Generalize to $\mathbf{z} = A\mathbf{x} + \boldsymbol{\mu}$ where $\mathbf{x} \sim \mathcal{N}(\mathbf{0}, I)$ and $A$ is an invertible matrix. Express $f_{\mathbf{Z}}(\mathbf{z})$ in terms of $|\det A|$.

---

## Related Topics

- [Probability Foundations](probability-foundations.md) — the measure-theoretic base this chapter builds on
- [Conditional Probability](conditional-probability.md) — conditioning random variables on events and other variables
- [Distributions](distributions.md) — the named distributions (Gaussian, Bernoulli, etc.) that random variables follow
- [Expectation & Moments](expectation-and-moments.md) — $\mathbb{E}[X]$, $\text{Var}(X)$, and higher moments
- [Joint Distributions](joint-distributions.md) — multiple random variables together, marginals, and independence
