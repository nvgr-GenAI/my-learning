# Chapter 6.2: Divergence Measures

## Prerequisites

- **Information Theory Fundamentals** (Chapter 6.1): Entropy $H(X)$, joint entropy $H(X,Y)$, conditional entropy $H(X|Y)$
- **Probability Theory**: Probability distributions, expectations, Jensen's inequality
- **Calculus**: Logarithms, convexity, optimization basics
- **Linear Algebra**: Inner products, norms (for MMD and Wasserstein distance)

## Overview

Divergence measures quantify the dissimilarity between probability distributions. Unlike traditional distance metrics, most divergences are **asymmetric** and do not satisfy the triangle inequality. They play a fundamental role in machine learning, particularly in:

- **Variational inference**: KL divergence in the ELBO
- **Generative modeling**: JSD in GANs, Wasserstein distance in WGANs
- **Representation learning**: Mutual information maximization
- **Model selection**: Information criteria (AIC, BIC)

```
Conceptual Hierarchy:

    f-Divergences (General Family)
           |
    +------+------+------+
    |      |      |      |
   KL   Rev-KL  χ²    Hellinger
    |
    +----> Jensen-Shannon (symmetrized KL)
    +----> Mutual Information (KL from independence)

    Alternative Frameworks:
    - Optimal Transport → Wasserstein Distance
    - Kernel Methods → Maximum Mean Discrepancy
```

---

## 6.2.1 Kullback-Leibler Divergence

### Definition 6.2.1 (KL Divergence)

For discrete probability distributions $p$ and $q$ with support $\mathcal{X}$, the **Kullback-Leibler divergence** (or relative entropy) from $q$ to $p$ is:

$$
D_{\text{KL}}(p \| q) = \sum_{x \in \mathcal{X}} p(x) \log \frac{p(x)}{q(x)}
$$

For continuous distributions with densities $p(x)$ and $q(x)$:

$$
D_{\text{KL}}(p \| q) = \int p(x) \log \frac{p(x)}{q(x)} \, dx
$$

**Conventions:**
- $0 \log \frac{0}{q} = 0$ (by continuity)
- $p \log \frac{p}{0} = \infty$ if $p > 0$
- Requires $q(x) > 0$ whenever $p(x) > 0$ (absolute continuity: $p \ll q$)

**Example 6.2.2: (Computing KL Divergence)**

Let $p = (0.7, 0.3)$ and $q = (0.5, 0.5)$ over $\mathcal{X} = \{a, b\}$. Using natural logarithms:

$$
\begin{align}
D_{\text{KL}}(p \| q) &= p(a) \ln \frac{p(a)}{q(a)} + p(b) \ln \frac{p(b)}{q(b)} \\
&= 0.7 \ln \frac{0.7}{0.5} + 0.3 \ln \frac{0.3}{0.5} \\
&= 0.7 \ln(1.4) + 0.3 \ln(0.6) \\
&= 0.7 \times 0.3365 + 0.3 \times (-0.5108) \\
&= 0.2356 - 0.1532 = 0.0824 \text{ nats}
\end{align}
$$

Note: $D_{\text{KL}}(p \| q) > 0$ since $p \neq q$, consistent with Gibbs' inequality below.

### Theorem 6.2.1 (Gibbs' Inequality)

For any probability distributions $p$ and $q$:

$$
D_{\text{KL}}(p \| q) \geq 0
$$

with equality if and only if $p = q$ almost everywhere.

**Proof:** By Jensen's inequality, since $-\log$ is strictly convex:

$$
-D_{\text{KL}}(p \| q) = \sum_x p(x) \log \frac{q(x)}{p(x)} \leq \log \left( \sum_x p(x) \frac{q(x)}{p(x)} \right) = \log(1) = 0
$$

Equality holds iff $\frac{q(x)}{p(x)}$ is constant wherever $p(x) > 0$, which implies $p = q$. $\square$

### Properties

1. **Asymmetry**: $D_{\text{KL}}(p \| q) \neq D_{\text{KL}}(q \| p)$ in general
2. **Not a metric**: Fails symmetry and triangle inequality
3. **Convexity**: $D_{\text{KL}}(p \| q)$ is jointly convex in $(p, q)$
4. **Relation to cross-entropy**: $D_{\text{KL}}(p \| q) = H(p, q) - H(p)$ where $H(p,q) = -\sum_x p(x) \log q(x)$

**Example 6.2.3: (Asymmetry of KL Divergence)**

Using the same $p = (0.7, 0.3)$ and $q = (0.5, 0.5)$, we already computed $D_{\text{KL}}(p \| q) = 0.0824$ nats. Now compute the reverse:

$$
\begin{align}
D_{\text{KL}}(q \| p) &= q(a) \ln \frac{q(a)}{p(a)} + q(b) \ln \frac{q(b)}{p(b)} \\
&= 0.5 \ln \frac{0.5}{0.7} + 0.5 \ln \frac{0.5}{0.3} \\
&= 0.5 \times (-0.3365) + 0.5 \times 0.5108 \\
&= -0.1682 + 0.2554 = 0.0871 \text{ nats}
\end{align}
$$

Thus $D_{\text{KL}}(p \| q) = 0.0824 \neq 0.0871 = D_{\text{KL}}(q \| p)$, confirming asymmetry. The reverse KL is larger here because $q$ is uniform and "wastes" probability on regions where $p$ is small.

### Example 6.2.1 (Bernoulli Distributions)

Let $p \sim \text{Ber}(\theta_1)$ and $q \sim \text{Ber}(\theta_2)$:

$$
D_{\text{KL}}(p \| q) = \theta_1 \log \frac{\theta_1}{\theta_2} + (1-\theta_1) \log \frac{1-\theta_1}{1-\theta_2}
$$

For $\theta_1 = 0.7$, $\theta_2 = 0.3$:
- $D_{\text{KL}}(p \| q) = 0.7 \log(7/3) + 0.3 \log(3/7) \approx 0.337$ nats
- $D_{\text{KL}}(q \| p) \approx 0.337$ nats (happens to be similar here, but not always)

---

## 6.2.2 Forward vs Reverse KL Divergence

The asymmetry of KL divergence has profound implications for optimization.

### Forward KL: $D_{\text{KL}}(p \| q)$ (Mode-Covering)

**Minimizing over $q$:** $\arg\min_q D_{\text{KL}}(p \| q)$ where $p$ is the target distribution.

$$
D_{\text{KL}}(p \| q) = \mathbb{E}_{x \sim p} \left[ \log \frac{p(x)}{q(x)} \right]
$$

- **Behavior**: Penalizes heavily when $q(x)$ is small where $p(x)$ is large
- **Result**: $q$ tries to **cover all modes** of $p$
- **Averaging effect**: $q$ spreads mass across all regions where $p$ has mass

### Reverse KL: $D_{\text{KL}}(q \| p)$ (Mode-Seeking)

**Minimizing over $q$:** $\arg\min_q D_{\text{KL}}(q \| p)$

$$
D_{\text{KL}}(q \| p) = \mathbb{E}_{x \sim q} \left[ \log \frac{q(x)}{p(x)} \right]
$$

- **Behavior**: Penalizes when $q(x)$ is large where $p(x)$ is small
- **Result**: $q$ **seeks a single mode** of $p$
- **Concentration effect**: $q$ focuses mass on one peak of $p$

### Visual Comparison

```
True distribution p(x):        Bimodal distribution

     *         *
    * *       * *
   *   *     *   *
  *     *   *     *
 *       * *       *
*         *         *
─────────────────────────

Forward KL: min D_KL(p||q)    Reverse KL: min D_KL(q||p)
q spreads to cover modes      q concentrates on one mode

       ******                        *
      *      *                      * *
     *        *                    *   *
    *          *                  *     *
   *            *                *       *
  *              *              *         *
──────────────────            ──────────────────

q tries to match p            q picks one mode
everywhere (mode-covering)    (mode-seeking)
```

**Example 6.2.4: (Forward vs Reverse KL — Mode-Covering vs Mode-Seeking)**

Let the target be a bimodal distribution $p = (0.5, 0.5, 0, 0)$ over $\{1,2,3,4\}$ (two modes at positions 1 and 2). Suppose we can only fit a unimodal $q$.

*Forward KL* — minimizing $D_{\text{KL}}(p \| q)$: Try $q_1 = (0.25, 0.50, 0.25, 0)$ (spreads mass to cover both modes):

$$
D_{\text{KL}}(p \| q_1) = 0.5 \ln \frac{0.5}{0.25} + 0.5 \ln \frac{0.5}{0.50} = 0.5 \ln 2 + 0 = 0.347 \text{ nats}
$$

*Reverse KL* — minimizing $D_{\text{KL}}(q \| p)$: Try $q_2 = (0.9, 0.1, 0, 0)$ (concentrates on one mode):

$$
D_{\text{KL}}(q_2 \| p) = 0.9 \ln \frac{0.9}{0.5} + 0.1 \ln \frac{0.1}{0.5} = 0.9 \times 0.588 + 0.1 \times (-1.609) = 0.368 \text{ nats}
$$

But if $q_2$ placed any mass where $p = 0$ (e.g., $q_2(3) > 0$), the reverse KL would be $\infty$. This forces reverse KL to avoid zero-probability regions of $p$, leading to mode-seeking behavior.

### ML Application: Variational Inference

In variational inference, we approximate a complex posterior $p(z|x)$ with a simpler distribution $q(z)$.

**ELBO Maximization:**
$$
\log p(x) = \mathcal{L}(q) + D_{\text{KL}}(q(z) \| p(z|x))
$$

where $\mathcal{L}(q) = \mathbb{E}_q[\log p(x,z)] - \mathbb{E}_q[\log q(z)]$ is the evidence lower bound.

Since $\log p(x)$ is constant w.r.t. $q$, maximizing $\mathcal{L}(q)$ is equivalent to:
$$
\min_q D_{\text{KL}}(q(z) \| p(z|x))
$$

This is **reverse KL**, leading to **mode-seeking** behavior:
- Mean-field approximations often underestimate uncertainty
- $q$ may miss modes of the true posterior
- **Forward KL** would be more conservative but requires sampling from $p(z|x)$ (intractable)

---

## 6.2.3 Mutual Information

### Definition 6.2.2 (Mutual Information)

For random variables $X$ and $Y$ with joint distribution $p(x,y)$ and marginals $p(x)$, $p(y)$:

$$
I(X; Y) = D_{\text{KL}}(p(x,y) \| p(x)p(y))
$$

Equivalently:
$$
I(X; Y) = \sum_{x,y} p(x,y) \log \frac{p(x,y)}{p(x)p(y)}
$$

### Theorem 6.2.2 (Equivalent Formulations)

$$
\begin{align}
I(X; Y) &= H(X) - H(X|Y) \\
        &= H(Y) - H(Y|X) \\
        &= H(X) + H(Y) - H(X,Y) \\
        &= \mathbb{E}_{p(x,y)} \left[ \log \frac{p(x,y)}{p(x)p(y)} \right]
\end{align}
$$

**Proof of $I(X;Y) = H(X) - H(X|Y)$:**

$$
\begin{align}
I(X;Y) &= \sum_{x,y} p(x,y) \log \frac{p(x,y)}{p(x)p(y)} \\
       &= \sum_{x,y} p(x,y) \log \frac{p(x|y)}{p(x)} \\
       &= \sum_{x,y} p(x,y) \log p(x|y) - \sum_{x,y} p(x,y) \log p(x) \\
       &= -H(X|Y) - \sum_x p(x) \log p(x) \\
       &= H(X) - H(X|Y) \quad \square
\end{align}
$$

### Venn Diagram Interpretation

```
          ┌─────────────────┐
          │                 │
          │  H(X|Y)         │  H(Y|X)
          │       ┌─────────┼─────────┐
          │       │         │         │
          │       │  I(X;Y) │         │
          │       │         │         │
          └───────┼─────────┘         │
                  │                   │
                  │                   │
                  └───────────────────┘

H(X,Y) = H(X|Y) + I(X;Y) + H(Y|X)
H(X) = H(X|Y) + I(X;Y)
H(Y) = H(Y|X) + I(X;Y)
```

### Properties

1. **Non-negativity**: $I(X;Y) \geq 0$ with equality iff $X \perp Y$
2. **Symmetry**: $I(X;Y) = I(Y;X)$
3. **Bounded**: $I(X;Y) \leq \min(H(X), H(Y))$
4. **Data processing inequality**: If $X \to Y \to Z$ forms a Markov chain, then $I(X;Z) \leq I(X;Y)$

### ML Application: Representation Learning

**Mutual Information Maximization:** Learn representations $Z = f(X)$ that preserve information about labels $Y$:

$$
\max_f I(Z; Y) = H(Y) - H(Y|Z)
$$

- **InfoNCE** (Contrastive Learning): Lower bound on $I(X;Y)$ using contrastive samples
- **InfoGAN**: Maximize $I(c; G(z,c))$ where $c$ are latent codes, $G$ is generator
- **Deep InfoMax**: Maximize $I(X; E(X))$ between input and encoded representation

---

## 6.2.4 f-Divergences

### Definition 6.2.3 (f-Divergence)

For a convex function $f: \mathbb{R}^+ \to \mathbb{R}$ with $f(1) = 0$:

$$
D_f(p \| q) = \sum_x q(x) f\left( \frac{p(x)}{q(x)} \right)
$$

For continuous distributions:
$$
D_f(p \| q) = \int q(x) f\left( \frac{p(x)}{q(x)} \right) dx
$$

The function $f$ is called the **generator** of the divergence.

### Theorem 6.2.3 (Non-negativity)

If $f$ is convex with $f(1) = 0$, then $D_f(p \| q) \geq 0$ with equality iff $p = q$.

**Proof:** By Jensen's inequality:

$$
D_f(p \| q) = \sum_x q(x) f\left( \frac{p(x)}{q(x)} \right) \geq f\left( \sum_x q(x) \frac{p(x)}{q(x)} \right) = f\left( \sum_x p(x) \right) = f(1) = 0
$$

Equality requires $\frac{p(x)}{q(x)}$ constant, implying $p = q$. $\square$

### Common f-Divergences

| Name | Generator $f(t)$ | Divergence Expression |
|------|-----------------|----------------------|
| **KL** | $t \log t$ | $\sum_x p(x) \log \frac{p(x)}{q(x)}$ |
| **Reverse KL** | $-\log t$ | $\sum_x q(x) \log \frac{q(x)}{p(x)}$ |
| **Chi-squared** | $(t-1)^2$ | $\sum_x \frac{(p(x) - q(x))^2}{q(x)}$ |
| **Squared Hellinger** | $(\sqrt{t} - 1)^2$ | $\sum_x (\sqrt{p(x)} - \sqrt{q(x)})^2$ |
| **Total Variation** | $\frac{1}{2} \|t-1\|$ | $\frac{1}{2} \sum_x \|p(x) - q(x)\|$ |
| **Jensen-Shannon** | $-(t+1)\log\frac{t+1}{2} + t\log t$ | (See Section 6.2.5) |

**Example 6.2.5: (KL Divergence as an f-Divergence)**

Verify that choosing $f(t) = t \ln t$ recovers the KL divergence. Using $p = (0.7, 0.3)$, $q = (0.5, 0.5)$:

$$
\begin{align}
D_f(p \| q) &= \sum_x q(x) \, f\!\left(\frac{p(x)}{q(x)}\right) = \sum_x q(x) \cdot \frac{p(x)}{q(x)} \ln \frac{p(x)}{q(x)} \\
&= \sum_x p(x) \ln \frac{p(x)}{q(x)} = D_{\text{KL}}(p \| q)
\end{align}
$$

Numerically: $f(1.4) = 1.4 \ln 1.4 = 0.4711$ and $f(0.6) = 0.6 \ln 0.6 = -0.3065$.

$$
D_f = 0.5 \times 0.4711 + 0.5 \times (-0.3065) = 0.2356 - 0.1532 = 0.0824 \text{ nats}
$$

This matches $D_{\text{KL}}(p \| q) = 0.0824$ from Example 6.2.2, confirming KL is indeed the f-divergence with generator $f(t) = t \ln t$.

**Example 6.2.6: (Total Variation as an f-Divergence)**

Using the same $p = (0.7, 0.3)$, $q = (0.5, 0.5)$ with $f(t) = \frac{1}{2}|t - 1|$:

$$
\begin{align}
\text{TV}(p, q) &= \frac{1}{2} \sum_x |p(x) - q(x)| \\
&= \frac{1}{2}(|0.7 - 0.5| + |0.3 - 0.5|) \\
&= \frac{1}{2}(0.2 + 0.2) = 0.2
\end{align}
$$

Verify via f-divergence: $D_f = 0.5 \cdot \frac{1}{2}|1.4 - 1| + 0.5 \cdot \frac{1}{2}|0.6 - 1| = 0.5 \times 0.2 + 0.5 \times 0.2 = 0.2$. Confirmed.

We can also check **Pinsker's inequality**: $\text{TV}(p,q) \leq \sqrt{\frac{1}{2} D_{\text{KL}}(p \| q)} = \sqrt{0.0412} = 0.203$. Indeed $0.2 \leq 0.203$. $\checkmark$

### ML Application: f-GAN

**f-GAN Framework:** Generalize GAN training to any f-divergence.

$$
\min_G \max_D \left( \mathbb{E}_{x \sim p_{\text{data}}}[T(x)] - \mathbb{E}_{x \sim p_G}[f^*(T(x))] \right)
$$

where $f^*$ is the Fenchel conjugate of $f$, and $T$ is a discriminator network.

**Examples:**
- Original GAN: Jensen-Shannon divergence ($f(t) = t\log t - (t+1)\log(t+1)$)
- Least-squares GAN: Pearson $\chi^2$ divergence ($f(t) = (t-1)^2$)
- Choosing different $f$ gives different generator behavior (mode-seeking vs mode-covering)

---

## 6.2.5 Jensen-Shannon Divergence

### Definition 6.2.4 (Jensen-Shannon Divergence)

For distributions $p$ and $q$, define the mixture $m = \frac{1}{2}(p + q)$. The **Jensen-Shannon divergence** is:

$$
\text{JSD}(p \| q) = \frac{1}{2} D_{\text{KL}}(p \| m) + \frac{1}{2} D_{\text{KL}}(q \| m)
$$

More generally, with weights $\pi_1, \pi_2 \geq 0$, $\pi_1 + \pi_2 = 1$:

$$
\text{JSD}_{\pi}(p \| q) = \pi_1 D_{\text{KL}}(p \| m) + \pi_2 D_{\text{KL}}(q \| m)
$$

where $m = \pi_1 p + \pi_2 q$.

### Properties

1. **Symmetry**: $\text{JSD}(p \| q) = \text{JSD}(q \| p)$
2. **Bounded**: $0 \leq \text{JSD}(p \| q) \leq \log 2$ (for base-2 logarithms: $\leq 1$ bit)
3. **Metric**: $\sqrt{\text{JSD}(p \| q)}$ is a true metric (satisfies triangle inequality)
4. **Relation to mutual information**:
   $$
   \text{JSD}(p \| q) = I(X; Z)
   $$
   where $Z \sim \text{Ber}(1/2)$ selects distribution $p$ or $q$, and $X$ is sampled from the selected distribution

**Example 6.2.7: (Computing Jensen-Shannon Divergence)**

Let $p = (0.7, 0.3)$ and $q = (0.5, 0.5)$. First compute the mixture $m = \frac{1}{2}(p + q)$:

$$
m = \left(\frac{0.7+0.5}{2}, \frac{0.3+0.5}{2}\right) = (0.6, 0.4)
$$

Now compute each KL term:

$$
\begin{align}
D_{\text{KL}}(p \| m) &= 0.7 \ln \frac{0.7}{0.6} + 0.3 \ln \frac{0.3}{0.4} \\
&= 0.7 \times 0.1542 + 0.3 \times (-0.2877) = 0.1080 - 0.0863 = 0.0216 \\[6pt]
D_{\text{KL}}(q \| m) &= 0.5 \ln \frac{0.5}{0.6} + 0.5 \ln \frac{0.5}{0.4} \\
&= 0.5 \times (-0.1823) + 0.5 \times 0.2231 = -0.0912 + 0.1116 = 0.0204
\end{align}
$$

Therefore:

$$
\text{JSD}(p \| q) = \frac{1}{2}(0.0216) + \frac{1}{2}(0.0204) = 0.0210 \text{ nats}
$$

Note that $\text{JSD}(p \| q) = 0.0210 \leq \ln 2 = 0.693$, satisfying the upper bound. Also, JSD is much smaller than either $D_{\text{KL}}(p \| q) = 0.0824$ or $D_{\text{KL}}(q \| p) = 0.0871$, since averaging with the mixture $m$ brings $p$ and $q$ closer.

### Theorem 6.2.4 (JSD Upper Bound)

For binary logarithms:
$$
\text{JSD}(p \| q) \leq 1 \text{ bit}
$$

**Proof:** Let $m = \frac{1}{2}(p+q)$. Then:

$$
\begin{align}
\text{JSD}(p \| q) &= H(m) - \frac{1}{2}H(p) - \frac{1}{2}H(q) \\
                   &\leq H\left(\frac{1}{2}\right) = 1 \text{ bit}
\end{align}
$$

since the mixture entropy $H(m)$ is maximized when $p$ and $q$ are disjoint (non-overlapping support). $\square$

### ML Application: Original GAN

**GAN Objective:** The original GAN discriminator minimizes:

$$
L_D = -\mathbb{E}_{x \sim p_{\text{data}}}[\log D(x)] - \mathbb{E}_{x \sim p_G}[\log(1 - D(x))]
$$

**Proposition 6.2.1:** For fixed generator $G$, the optimal discriminator is:

$$
D^*(x) = \frac{p_{\text{data}}(x)}{p_{\text{data}}(x) + p_G(x)}
$$

Substituting $D^*(x)$ into the generator objective:

$$
\min_G \max_D V(D, G) = -2\log 2 + 2 \cdot \text{JSD}(p_{\text{data}} \| p_G)
$$

Thus, training GANs implicitly minimizes the Jensen-Shannon divergence between data and generator distributions.

**Problem:** JSD saturates when distributions have disjoint supports (common in high dimensions), leading to vanishing gradients.

---

## 6.2.6 Wasserstein Distance

### Definition 6.2.5 (Wasserstein Distance)

For probability distributions $p$ and $q$ on a metric space $(\mathcal{X}, d)$, the **1-Wasserstein distance** (Earth Mover's Distance) is:

$$
W_1(p, q) = \inf_{\gamma \in \Pi(p,q)} \mathbb{E}_{(x,y) \sim \gamma}[d(x, y)]
$$

where $\Pi(p,q)$ is the set of all joint distributions (couplings) with marginals $p$ and $q$.

**Interpretation:** Minimum "work" to transform distribution $p$ into $q$, where work = mass × distance.

### Kantorovich-Rubinstein Duality

For $d(x,y) = \|x - y\|$:

$$
W_1(p, q) = \sup_{\|f\|_L \leq 1} \left( \mathbb{E}_{x \sim p}[f(x)] - \mathbb{E}_{x \sim q}[f(x)] \right)
$$

where $\|f\|_L \leq 1$ means $f$ is 1-Lipschitz: $|f(x) - f(y)| \leq \|x - y\|$.

### Comparison to KL Divergence

```
KL Divergence:                    Wasserstein Distance:
- Asymmetric                      - Symmetric (true metric)
- Not a metric                    - Satisfies triangle inequality
- Infinite if supports disjoint   - Finite even for disjoint supports
- Mode-seeking/covering           - Considers geometric structure

Example: Two Gaussians p = N(0,1), q = N(μ,1)

D_KL(p||q) = μ²/2               W_1(p,q) = |μ|

As μ → ∞:
D_KL → ∞                         W_1 → ∞ linearly

For disjoint support (e.g., μ >> 1):
D_KL = ∞ immediately             W_1 = distance between means
```

**Example 6.2.8: (Wasserstein Distance for 1D Discrete Distributions)**

Let $p = (0.5, 0.5, 0)$ and $q = (0, 0.5, 0.5)$ on points $\{1, 2, 3\}$ with $d(x,y) = |x - y|$.

For 1D distributions, $W_1$ has a closed form using CDFs $F_p$ and $F_q$:

$$
W_1(p, q) = \int |F_p(x) - F_q(x)| \, dx = \sum_{k} |F_p(k) - F_q(k)|
$$

Compute the CDFs at each point:

| $x$ | $F_p(x)$ | $F_q(x)$ | $|F_p - F_q|$ |
|-----|-----------|-----------|----------------|
| 1   | 0.5       | 0.0       | 0.5            |
| 2   | 1.0       | 0.5       | 0.5            |
| 3   | 1.0       | 1.0       | 0.0            |

$$
W_1(p, q) = 0.5 + 0.5 + 0.0 = 1.0
$$

Interpretation: we must move 0.5 units of mass from position 1 to position 3 (distance 2), giving total work $0.5 \times 2 = 1.0$. Note that $D_{\text{KL}}(p \| q) = \infty$ here since $p(1) = 0.5 > 0$ but $q(1) = 0$, illustrating why Wasserstein distance is preferred when supports do not fully overlap.

### ML Application: Wasserstein GAN (WGAN)

**WGAN Objective:** Minimize Wasserstein distance instead of JSD:

$$
\min_G W_1(p_{\text{data}}, p_G)
$$

Using Kantorovich-Rubinstein duality:

$$
W_1(p_{\text{data}}, p_G) = \max_{\|f\|_L \leq 1} \left( \mathbb{E}_{x \sim p_{\text{data}}}[f(x)] - \mathbb{E}_{x \sim p_G}[f(x)] \right)
$$

**Practical Implementation:**
- Discriminator $f_w$ (critic) is enforced to be 1-Lipschitz via:
  - **Weight clipping** (original WGAN): clip weights to $[-c, c]$
  - **Gradient penalty** (WGAN-GP): add term $\lambda \mathbb{E}[(\|\nabla_{\hat{x}} f_w(\hat{x})\|_2 - 1)^2]$

**Advantages over original GAN:**
- Meaningful loss metric (correlates with sample quality)
- No mode collapse
- More stable training (no vanishing gradients)

---

## 6.2.7 Maximum Mean Discrepancy

### Definition 6.2.6 (Maximum Mean Discrepancy)

For distributions $p$ and $q$ on $\mathcal{X}$, and a reproducing kernel Hilbert space (RKHS) $\mathcal{H}$ with kernel $k$:

$$
\text{MMD}^2(p, q) = \sup_{\|f\|_{\mathcal{H}} \leq 1} \left( \mathbb{E}_{x \sim p}[f(x)] - \mathbb{E}_{y \sim q}[f(y)] \right)^2
$$

**Biased Empirical Estimate:** Given samples $\{x_i\}_{i=1}^n \sim p$ and $\{y_j\}_{j=1}^m \sim q$:

$$
\widehat{\text{MMD}}^2 = \frac{1}{n^2} \sum_{i,i'} k(x_i, x_{i'}) + \frac{1}{m^2} \sum_{j,j'} k(y_j, y_{j'}) - \frac{2}{nm} \sum_{i,j} k(x_i, y_j)
$$

### Theorem 6.2.5 (Kernel Mean Embedding)

If $k$ is a **characteristic kernel** (e.g., Gaussian RBF):

$$
\text{MMD}(p, q) = 0 \iff p = q
$$

Common characteristic kernels:
- **Gaussian RBF**: $k(x,y) = \exp\left(-\frac{\|x-y\|^2}{2\sigma^2}\right)$
- **Laplacian**: $k(x,y) = \exp\left(-\frac{\|x-y\|}{\sigma}\right)$

### Properties

1. **Symmetric**: $\text{MMD}(p, q) = \text{MMD}(q, p)$
2. **Metric**: Satisfies triangle inequality
3. **Computationally efficient**: $O(n^2)$ for $n$ samples
4. **Differentiable**: Can backpropagate through kernel evaluations

### ML Application: Generative Models and Two-Sample Testing

**MMD-GAN:** Replace adversarial training with MMD minimization:

$$
\min_G \text{MMD}^2(p_{\text{data}}, p_G)
$$

- No discriminator network needed
- Direct optimization of generator
- More stable than adversarial training

**Two-Sample Testing:** Test if two datasets come from same distribution:

$$
H_0: p = q \quad \text{vs} \quad H_1: p \neq q
$$

**Test statistic:** $\widehat{\text{MMD}}^2$ follows asymptotic distribution under $H_0$, allowing p-value computation.

**Domain Adaptation:** Minimize MMD between source and target feature distributions to learn domain-invariant representations.

---

## 6.2.8 Summary and Connections

### Comparison of Divergence Measures

| Divergence | Symmetric | Metric | Bounded | Disjoint Supports | Computation |
|------------|-----------|--------|---------|-------------------|-------------|
| **KL** | ✗ | ✗ | ✗ | $\infty$ | Tractable (if $q$ known) |
| **Reverse KL** | ✗ | ✗ | ✗ | $\infty$ | Tractable (if $p$ known) |
| **JSD** | ✓ | $\sqrt{\text{JSD}}$ is | ✓ (≤ log 2) | Bounded | Tractable |
| **Wasserstein** | ✓ | ✓ | ✗ | Finite | Requires optimization |
| **MMD** | ✓ | ✓ | ✗ | Finite | $O(n^2)$ kernel evals |

### ML Use Cases

```
Problem Domain          Recommended Divergence    Reason
─────────────────────────────────────────────────────────────────
Variational Inference   Reverse KL                Tractable ELBO
                        (D_KL(q||p))

Expectation-            Forward KL                Moment matching
Propagation             (D_KL(p||q))

Generative Models:
  - Original GAN        JSD                       Discriminator MLE
  - Wasserstein GAN     Wasserstein (W_1)         Stable, meaningful loss
  - MMD-GAN             MMD                       No adversarial training

Two-Sample Testing      MMD, Wasserstein          Finite sample guarantees

Domain Adaptation       MMD                       Efficient, differentiable

Model Compression       Forward KL                Student covers teacher

Reinforcement Learning  KL (various)              Policy regularization
```

### Information-Theoretic Inequalities

**Pinsker's Inequality:** Relates KL to total variation:

$$
\text{TV}(p, q) \leq \sqrt{\frac{1}{2} D_{\text{KL}}(p \| q)}
$$

**Bretagnolle-Huber Inequality:** Relates KL to Hellinger distance:

$$
H^2(p, q) \leq D_{\text{KL}}(p \| q) \leq -\log(1 - H^2(p, q))
$$

where $H^2(p,q) = 1 - \sum_x \sqrt{p(x)q(x)}$ is squared Hellinger distance.

---

## Exercises

### Computational Exercises

**Exercise 6.2.1** (★)
Compute $D_{\text{KL}}(p \| q)$ and $D_{\text{KL}}(q \| p)$ for:
- $p = \text{Ber}(0.8)$, $q = \text{Ber}(0.2)$
- $p = \mathcal{N}(0, 1)$, $q = \mathcal{N}(2, 1)$ (use closed form: $D_{\text{KL}}(p \| q) = \frac{(\mu_p - \mu_q)^2}{2\sigma_q^2} + \frac{1}{2}\left(\frac{\sigma_p^2}{\sigma_q^2} - 1 - \log\frac{\sigma_p^2}{\sigma_q^2}\right)$)

**Exercise 6.2.2** (★)
For categorical distributions $p = [0.5, 0.3, 0.2]$ and $q = [0.33, 0.33, 0.34]$:
1. Compute $D_{\text{KL}}(p \| q)$, $D_{\text{KL}}(q \| p)$, and $\text{JSD}(p \| q)$
2. Verify that $\text{JSD}(p \| q) \leq \frac{1}{2}(D_{\text{KL}}(p \| q) + D_{\text{KL}}(q \| p))$

**Exercise 6.2.3** (★★)
Show that mutual information decomposes as:
$$
I(X; Y, Z) = I(X; Y) + I(X; Z | Y)
$$
(This is the **chain rule for mutual information**.)

### Theoretical Exercises

**Exercise 6.2.4** (★★)
Prove that KL divergence is jointly convex: for $\lambda \in [0,1]$,
$$
D_{\text{KL}}(\lambda p_1 + (1-\lambda) p_2 \| \lambda q_1 + (1-\lambda) q_2) \leq \lambda D_{\text{KL}}(p_1 \| q_1) + (1-\lambda) D_{\text{KL}}(p_2 \| q_2)
$$

**Exercise 6.2.5** (★★)
For the chi-squared divergence $D_{\chi^2}(p \| q) = \sum_x \frac{(p(x) - q(x))^2}{q(x)}$:
1. Verify it's an f-divergence with generator $f(t) = (t-1)^2$
2. Show $D_{\chi^2}(p \| q) = \text{Var}_q\left[\frac{p(X)}{q(X)}\right]$
3. Prove $D_{\chi^2}(p \| q) \geq D_{\text{KL}}(p \| q)^2 / 2$ (using Taylor expansion)

**Exercise 6.2.6** (★★★)
**Data Processing Inequality for KL:** If $X \to Y \to Z$ forms a Markov chain (i.e., $p(z|x,y) = p(z|y)$), prove:
$$
D_{\text{KL}}(p(x,z) \| q(x,z)) \geq D_{\text{KL}}(p(x) \| q(x))
$$
and
$$
D_{\text{KL}}(p(x,z) \| q(x,z)) \geq D_{\text{KL}}(p(z) \| q(z))
$$
(Hint: Use chain rule and non-negativity of conditional KL.)

### Applied Exercises

**Exercise 6.2.7** (★★)
**Variational Autoencoder (VAE):**
The VAE loss is:
$$
\mathcal{L} = \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] - D_{\text{KL}}(q_\phi(z|x) \| p(z))
$$
where $p(z) = \mathcal{N}(0, I)$ and $q_\phi(z|x) = \mathcal{N}(\mu_\phi(x), \text{diag}(\sigma_\phi^2(x)))$.

1. Derive the closed-form KL term for this Gaussian case
2. Explain why the KL term acts as a regularizer
3. What happens if we use $D_{\text{KL}}(p(z) \| q_\phi(z|x))$ instead? (Consider mode-seeking vs mode-covering)

**Exercise 6.2.8** (★★★)
**Wasserstein Distance for Discrete Distributions:**
For distributions $p = [0.5, 0.5, 0]$ and $q = [0, 0.5, 0.5]$ on $\{1, 2, 3\}$ with Euclidean distance:
1. Set up the linear program to compute $W_1(p, q)$
2. Solve it (either analytically or numerically)
3. Compare the result to $D_{\text{KL}}(p \| q)$ and explain why one might prefer Wasserstein distance here

**Exercise 6.2.9** (★★★)
**MMD Gradient Flow:**
Consider training a generator $G_\theta: \mathcal{Z} \to \mathcal{X}$ by minimizing MMD:
$$
\min_\theta \text{MMD}^2(p_{\text{data}}, p_{G_\theta})
$$
1. Write the gradient $\nabla_\theta \text{MMD}^2$ using kernel evaluations
2. Discuss computational complexity compared to GAN training
3. What are the trade-offs between MMD-GAN and Wasserstein GAN?

---

## Related Topics

### Within This Textbook
- **Chapter 6.1**: Entropy and Information Theory (prerequisite)
- **Chapter 6.3**: Variational Inference and ELBO
- **Chapter 9.4**: Generative Adversarial Networks
- **Chapter 9.5**: Variational Autoencoders

### Advanced Topics
- **Rényi Divergence**: $D_\alpha(p \| q) = \frac{1}{\alpha - 1} \log \sum_x p(x)^\alpha q(x)^{1-\alpha}$ (generalizes KL)
- **Optimal Transport Theory**: $p$-Wasserstein distances, Kantorovich duality, Monge problem
- **Information Geometry**: Fisher information metric, natural gradients, exponential families
- **Rate-Distortion Theory**: Tradeoff between compression rate and reconstruction quality via $I(X;Z)$

### Further Reading
1. Cover, T. M., & Thomas, J. A. (2006). *Elements of Information Theory* (2nd ed.). Chapters 2, 8, 12.
2. Peyré, G., & Cuturi, M. (2019). *Computational Optimal Transport*. Foundations and Trends in ML.
3. Nowozin, S., et al. (2016). "f-GAN: Training Generative Neural Samplers using Variational Divergence Minimization." NeurIPS.
4. Arjovsky, M., et al. (2017). "Wasserstein Generative Adversarial Networks." ICML.
5. Gretton, A., et al. (2012). "A Kernel Two-Sample Test." JMLR.

---

*End of Chapter 6.2*
