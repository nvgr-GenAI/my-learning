# Bayesian Inference

Bayesian inference is the mathematical engine for learning from data under uncertainty. Where frequentist methods treat parameters as fixed unknowns, Bayesian inference treats them as random variables with distributions that update as evidence accumulates. This single idea — encoding beliefs as probability distributions and updating them via Bayes' rule — unifies regularization, uncertainty quantification, hyperparameter tuning, and exploration-exploitation tradeoffs. Every time a neural network uses dropout, L2 regularization, or a learned prior, it is performing approximate Bayesian inference.

---

## Prerequisites

- [Conditional Probability](conditional-probability.md) — Bayes' theorem, chain rule, independence
- [Distributions](distributions.md) — Beta, Gaussian, Dirichlet, and their properties
- [Expectation & Moments](expectation-and-moments.md) — mean, variance, integration over distributions
- [Joint Distributions](joint-distributions.md) — marginal and conditional distributions

---

## 1. The Bayesian Framework

**Definition 4.8.1 (Bayesian Inference).** Given observed data $D = \{x_1, x_2, \ldots, x_n\}$ and a parametric model with parameter $\theta$, *Bayesian inference* computes the posterior distribution:

$$P(\theta \mid D) = \frac{P(D \mid \theta) \, P(\theta)}{P(D)}$$

where:

- $P(\theta)$ is the **prior** — our belief about $\theta$ before observing data
- $P(D \mid \theta)$ is the **likelihood** — how probable the data is under parameter $\theta$
- $P(D) = \int P(D \mid \theta) P(\theta) \, d\theta$ is the **evidence** (marginal likelihood)
- $P(\theta \mid D)$ is the **posterior** — our updated belief after observing data

Since $P(D)$ is constant with respect to $\theta$, we often write:

$$P(\theta \mid D) \propto P(D \mid \theta) \, P(\theta) \qquad \text{(posterior} \propto \text{likelihood} \times \text{prior)}$$

```
THE BAYESIAN UPDATE CYCLE

  Prior P(θ)          Likelihood P(D|θ)          Posterior P(θ|D)
  ┌─────────┐         ┌─────────┐               ┌─────────┐
  │  ╱╲     │    ×    │    ╱╲   │    ∝          │   ╱╲    │
  │ ╱  ╲    │         │   ╱  ╲  │               │  ╱  ╲   │
  │╱    ╲   │         │  ╱    ╲ │               │ ╱    ╲  │
  │      ╲──│         │ ╱      ╲│               │╱      ╲ │
  └─────────┘         └─────────┘               └─────────┘
  (broad, uncertain)  (data points to           (narrower, shifted
                       specific θ values)         toward the data)

  More data ──▶ posterior concentrates ──▶ prior influence diminishes
  Less data ──▶ posterior ≈ prior       ──▶ prior dominates
```

**Example 4.9.1 (Medical Test — Full Bayesian Update).**
A disease affects $1$ in $1{,}000$ people. A test has $99\%$ sensitivity and $95\%$ specificity. A patient tests positive — what is the probability they have the disease?

- **Prior:** $P(\text{disease}) = 0.001$
- **Likelihood:** $P(+\mid \text{disease}) = 0.99$, $\;P(+\mid \text{no disease}) = 0.05$
- **Evidence:** $P(+) = 0.99 \times 0.001 + 0.05 \times 0.999 = 0.00099 + 0.04995 = 0.05094$
- **Posterior:** $P(\text{disease} \mid +) = \dfrac{0.99 \times 0.001}{0.05094} \approx 0.0194$

Despite the "accurate" test, there is only a $\approx 2\%$ chance of disease — the low prior dominates. This is why screening tests require confirmation.

**Theorem 4.8.1 (Sequential Updating).** Bayesian inference is sequential: the posterior from one batch of data becomes the prior for the next.

If $D = D_1 \cup D_2$ (independent batches), then:

$$P(\theta \mid D_1, D_2) \propto P(D_2 \mid \theta) \cdot \underbrace{P(\theta \mid D_1)}_{\text{posterior from } D_1}$$

*Proof.* $P(\theta \mid D_1, D_2) \propto P(D_1, D_2 \mid \theta)P(\theta) = P(D_2 \mid \theta)P(D_1 \mid \theta)P(\theta) \propto P(D_2 \mid \theta)P(\theta \mid D_1)$, where the last step uses $P(D_1 \mid \theta)P(\theta) \propto P(\theta \mid D_1)$. $\square$

**Example 4.9.2 (Sequential Updating — Coin Flips in Two Batches).**
Start with prior $\theta \sim \text{Beta}(1, 1)$ (uniform). Process data in two batches:

- **Batch 1:** Observe $3$ heads, $1$ tail. Posterior: $\text{Beta}(1+3,\; 1+1) = \text{Beta}(4, 2)$. Posterior mean $= 4/6 \approx 0.667$.
- **Batch 2:** Use $\text{Beta}(4, 2)$ as the new prior. Observe $2$ heads, $3$ tails. Posterior: $\text{Beta}(4+2,\; 2+3) = \text{Beta}(6, 5)$. Posterior mean $= 6/11 \approx 0.545$.

Processing all data at once: $\text{Beta}(1+5,\; 1+4) = \text{Beta}(6, 5)$. Same result — sequential updating is order-independent.

*ML connection:* Sequential updating is exactly online learning. In streaming applications, the model processes data in mini-batches, updating the posterior incrementally rather than re-processing the entire dataset. This is the theoretical foundation for continual learning and Bayesian online learning algorithms.

---

## 2. Prior Distributions

**Definition 4.8.2 (Informative Prior).** An *informative prior* encodes specific domain knowledge about $\theta$. For example, setting $P(\theta) = \mathcal{N}(0, 1)$ for a regression weight expresses the belief that weights are small and centered at zero.

**Definition 4.8.3 (Uninformative Prior).** An *uninformative* (or *diffuse*) prior aims to let the data dominate. Common choices:

| Prior Type | Form | Use Case |
|-----------|------|----------|
| Flat (uniform) | $P(\theta) \propto 1$ | No preference over $\theta$ range |
| Jeffreys prior | $P(\theta) \propto \sqrt{I(\theta)}$ | Invariant under reparameterization |
| Maximum entropy | Maximizes $H[P]$ given constraints | Least informative given constraints |

**Definition 4.8.4 (Improper Prior).** A prior $P(\theta)$ is *improper* if $\int P(\theta) \, d\theta = \infty$. For example, $P(\theta) \propto 1$ over $\mathbb{R}$ does not integrate to $1$. Improper priors are acceptable if the resulting posterior $P(\theta \mid D)$ is proper (integrates to $1$).

**Theorem 4.8.2 (Bernstein-von Mises).** Under regularity conditions, as $n \to \infty$ the posterior $P(\theta \mid D)$ converges to a Gaussian centered at the MLE $\hat{\theta}_{\text{MLE}}$ with variance $(nI(\hat{\theta}))^{-1}$, regardless of the prior. The prior is "washed out" by sufficient data.

*ML connection:* The Bernstein-von Mises theorem explains why Bayesian and frequentist methods often agree on large datasets. The prior matters most when data is scarce — exactly the regime where Bayesian methods provide the most value (few-shot learning, cold-start recommendations, small medical datasets).

**Example 4.9.3 (Prior Sensitivity — How Priors Affect the Posterior).**
Observe $k = 3$ heads in $n = 10$ coin flips ($\text{MLE} = 0.30$). Compare three priors:

| Prior | Posterior | Posterior Mean | Interpretation |
|---|---|---|---|
| $\text{Beta}(1, 1)$ — uniform | $\text{Beta}(4, 8)$ | $4/12 = 0.333$ | Data dominates |
| $\text{Beta}(5, 5)$ — fair-coin belief | $\text{Beta}(8, 12)$ | $8/20 = 0.400$ | Prior pulls toward $0.5$ |
| $\text{Beta}(50, 50)$ — strong fair-coin | $\text{Beta}(53, 57)$ | $53/110 = 0.482$ | Prior overwhelms $10$ observations |

With only $10$ data points, the strong prior ($\alpha + \beta = 100$ pseudo-observations) barely budges from $0.5$. With $n = 1{,}000$ observations, all three posteriors would converge near the MLE (Bernstein-von Mises).

---

## 3. Conjugate Priors

**Definition 4.8.5 (Conjugate Prior).** A prior $P(\theta)$ is *conjugate* to a likelihood $P(D \mid \theta)$ if the posterior $P(\theta \mid D)$ belongs to the same family as the prior.

**Theorem 4.8.3 (Conjugate Families).** The following are conjugate pairs:

| Likelihood | Conjugate Prior | Posterior | Interpretation |
|-----------|----------------|-----------|----------------|
| Binomial$(n, \theta)$ | Beta$(\alpha, \beta)$ | Beta$(\alpha + k, \beta + n - k)$ | $\alpha, \beta$ = pseudo-counts |
| Poisson$(\lambda)$ | Gamma$(\alpha, \beta)$ | Gamma$(\alpha + \sum x_i, \beta + n)$ | $\alpha$ = pseudo-events |
| $\mathcal{N}(\mu, \sigma^2_{\text{known}})$ | $\mathcal{N}(\mu_0, \sigma_0^2)$ | $\mathcal{N}(\mu_n, \sigma_n^2)$ | Precision-weighted average |
| Multinomial$(n, \boldsymbol{\theta})$ | Dirichlet$(\boldsymbol{\alpha})$ | Dirichlet$(\boldsymbol{\alpha} + \mathbf{c})$ | $\boldsymbol{\alpha}$ = pseudo-counts |

### 3.1 Beta-Binomial (Detailed Example)

Suppose we observe $k$ successes in $n$ Bernoulli trials with unknown probability $\theta$.

**Prior:** $\theta \sim \text{Beta}(\alpha, \beta)$, so $P(\theta) \propto \theta^{\alpha-1}(1-\theta)^{\beta-1}$.

**Likelihood:** $P(D \mid \theta) = \binom{n}{k}\theta^k(1-\theta)^{n-k}$.

**Posterior:**

$$P(\theta \mid D) \propto \theta^k(1-\theta)^{n-k} \cdot \theta^{\alpha-1}(1-\theta)^{\beta-1} = \theta^{(\alpha+k)-1}(1-\theta)^{(\beta+n-k)-1}$$

This is $\text{Beta}(\alpha + k, \beta + n - k)$.

*Proof.* The posterior is proportional to the product of the likelihood and prior. Collecting powers of $\theta$ gives $\alpha + k - 1$ and collecting powers of $(1-\theta)$ gives $\beta + n - k - 1$. Recognizing this as the kernel of a Beta distribution completes the proof. $\square$

**Posterior mean:**

$$\mathbb{E}[\theta \mid D] = \frac{\alpha + k}{\alpha + \beta + n} = \frac{\alpha + \beta}{\alpha + \beta + n} \cdot \underbrace{\frac{\alpha}{\alpha + \beta}}_{\text{prior mean}} + \frac{n}{\alpha + \beta + n} \cdot \underbrace{\frac{k}{n}}_{\text{MLE}}$$

This is a weighted average of the prior mean and the MLE, with the weight shifting toward the MLE as $n$ grows.

```
BETA-BINOMIAL CONJUGACY: Coin flips with unknown bias

  Prior: Beta(2, 2)        After 3H, 1T:           After 30H, 10T:
  (mild belief: fair)       Beta(5, 3)              Beta(32, 12)

  P(θ)                     P(θ|D)                  P(θ|D)
   ▲    ╱╲                  ▲     ╱╲                ▲      ╱╲
   │   ╱  ╲                 │    ╱  ╲               │     ╱  ╲
   │  ╱    ╲                │   ╱    ╲              │    ╱    ╲
   │ ╱      ╲               │ ╱       ╲             │   ╱      ╲
   │╱        ╲              │╱         ╲            │──╱        ╲
   └──────────▶ θ           └───────────▶ θ         └────────────▶ θ
   0   0.5    1             0   0.625   1           0   0.73    1

  Broad uncertainty          Shifting right           Concentrated near
                             (more heads than          the true proportion
                              tails observed)           (0.75)
```

**Example 4.9.4 (Beta-Binomial Conjugacy — Full Worked Computation).**
Prior: $\theta \sim \text{Beta}(2, 2)$ (mild belief the coin is fair). Observe $k = 7$ heads in $n = 10$ flips.

- **Posterior:** $\text{Beta}(2 + 7,\; 2 + 3) = \text{Beta}(9, 5)$
- **Posterior mean:** $\mathbb{E}[\theta \mid D] = \dfrac{9}{9 + 5} = \dfrac{9}{14} \approx 0.643$
- **MLE:** $\hat{\theta}_{\text{MLE}} = k/n = 7/10 = 0.700$
- **Prior mean:** $\alpha/(\alpha + \beta) = 2/4 = 0.500$

The posterior mean ($0.643$) lies between the prior mean ($0.500$) and the MLE ($0.700$), pulled toward $0.5$ by the prior's $\alpha + \beta = 4$ pseudo-observations.

**Example 4.9.5 (Posterior Mean as Weighted Average — Numerical Verification).**
Continuing Example 4.9.4, verify the weighted-average formula:

$$\mathbb{E}[\theta \mid D] = \frac{\alpha + \beta}{\alpha + \beta + n} \cdot \frac{\alpha}{\alpha + \beta} + \frac{n}{\alpha + \beta + n} \cdot \frac{k}{n} = \frac{4}{14} \cdot 0.500 + \frac{10}{14} \cdot 0.700$$

$$= 0.286 \times 0.500 + 0.714 \times 0.700 = 0.143 + 0.500 = 0.643 \;\checkmark$$

The data ($n = 10$) carries weight $10/14 \approx 71\%$ and the prior ($\alpha + \beta = 4$ pseudo-observations) carries weight $4/14 \approx 29\%$. With $n = 100$ observations of the same proportion, the data weight would be $100/104 \approx 96\%$.

### 3.2 Gaussian-Gaussian

**Prior:** $\mu \sim \mathcal{N}(\mu_0, \sigma_0^2)$.

**Likelihood:** $x_1, \ldots, x_n \overset{\text{iid}}{\sim} \mathcal{N}(\mu, \sigma^2)$ with $\sigma^2$ known.

**Posterior:** $\mu \mid D \sim \mathcal{N}(\mu_n, \sigma_n^2)$ where:

$$\mu_n = \frac{\frac{1}{\sigma_0^2}\mu_0 + \frac{n}{\sigma^2}\bar{x}}{\frac{1}{\sigma_0^2} + \frac{n}{\sigma^2}}, \qquad \frac{1}{\sigma_n^2} = \frac{1}{\sigma_0^2} + \frac{n}{\sigma^2}$$

The posterior precision (inverse variance) is the sum of prior precision and data precision.

**Example 4.9.6 (Gaussian-Gaussian — Estimating Average Temperature).**
Prior belief about today's average temperature: $\mu \sim \mathcal{N}(\mu_0 = 20, \sigma_0^2 = 9)$ (mean $20°$C, std $3°$C).
We take $n = 4$ sensor readings with known noise $\sigma^2 = 4$: readings are $\{23, 25, 22, 24\}$, so $\bar{x} = 23.5$.

- **Posterior precision:** $1/\sigma_n^2 = 1/9 + 4/4 = 0.111 + 1.000 = 1.111$, so $\sigma_n^2 = 0.900$
- **Posterior mean:** $\mu_n = \dfrac{(1/9)(20) + (4/4)(23.5)}{1.111} = \dfrac{2.222 + 23.500}{1.111} = \dfrac{25.722}{1.111} \approx 23.15$
- **Result:** $\mu \mid D \sim \mathcal{N}(23.15, \; 0.900)$, i.e., posterior std $\approx 0.95°$C

The prior ($20°$C) gets overridden by the data ($23.5°$C) because the data precision ($n/\sigma^2 = 1.0$) far exceeds the prior precision ($1/\sigma_0^2 \approx 0.11$).

*ML connection:* Gaussian-Gaussian conjugacy is the mathematical backbone of Kalman filters, used in robotics, autonomous vehicles, and time-series forecasting. The state estimate is the posterior mean $\mu_n$, and the uncertainty is the posterior variance $\sigma_n^2$, both updated recursively as new sensor readings arrive.

---

## 4. Maximum A Posteriori (MAP) Estimation

**Definition 4.8.6 (MAP Estimate).** The *maximum a posteriori* estimate is the mode of the posterior:

$$\hat{\theta}_{\text{MAP}} = \arg\max_\theta P(\theta \mid D) = \arg\max_\theta \left[\log P(D \mid \theta) + \log P(\theta)\right]$$

**Example 4.9.7 (MAP Estimate — Beta-Binomial).**
From Example 4.9.4: posterior is $\text{Beta}(9, 5)$. The mode of $\text{Beta}(\alpha, \beta)$ is $(\alpha - 1)/(\alpha + \beta - 2)$:

$$\hat{\theta}_{\text{MAP}} = \frac{9 - 1}{9 + 5 - 2} = \frac{8}{12} = 0.667$$

Compare the three point estimates:

- **MLE** $= k/n = 7/10 = 0.700$ (data only)
- **MAP** $= 8/12 = 0.667$ (mode of posterior — prior pulls it toward $0.5$)
- **Posterior mean** $= 9/14 = 0.643$ (mean of posterior — pulled further than MAP)

The MAP lies between the MLE and posterior mean. For symmetric posteriors they would coincide, but $\text{Beta}(9, 5)$ is slightly right-skewed, so the mode exceeds the mean.

**Theorem 4.8.4 (MAP and Regularization).** MAP estimation with specific priors is equivalent to regularized maximum likelihood:

| Prior | MAP Objective | Equivalent Regularization |
|-------|--------------|--------------------------|
| $\theta_j \sim \mathcal{N}(0, \tau^2)$ | $\log P(D \mid \theta) - \frac{1}{2\tau^2}\sum \theta_j^2$ | L2 regularization ($\lambda = \frac{1}{\tau^2}$) |
| $\theta_j \sim \text{Laplace}(0, b)$ | $\log P(D \mid \theta) - \frac{1}{b}\sum |\theta_j|$ | L1 regularization ($\lambda = \frac{1}{b}$) |
| Uniform on valid $\theta$ | $\log P(D \mid \theta)$ | No regularization (= MLE) |

*Proof (L2 case).* The log-posterior is $\log P(\theta \mid D) = \log P(D \mid \theta) + \sum_j \log P(\theta_j) + \text{const}$. With $P(\theta_j) = \frac{1}{\sqrt{2\pi}\tau}e^{-\theta_j^2/(2\tau^2)}$, we get $\log P(\theta_j) = -\frac{\theta_j^2}{2\tau^2} + \text{const}$. Maximizing the log-posterior is therefore equivalent to maximizing $\log P(D \mid \theta) - \frac{1}{2\tau^2}\|\theta\|_2^2$, which is L2-regularized MLE. $\square$

**Example 4.9.8 (MAP = Regularized MLE — Numerical).**
Suppose a single-parameter model with log-likelihood $\ell(\theta) = -2(\theta - 3)^2$ (maximized at $\hat{\theta}_{\text{MLE}} = 3$) and a Gaussian prior $\theta \sim \mathcal{N}(0, \tau^2 = 2)$.

The MAP objective is: $\ell(\theta) + \log P(\theta) = -2(\theta - 3)^2 - \theta^2/(2 \cdot 2) = -2(\theta - 3)^2 - \theta^2/4$.

Taking the derivative and setting to zero: $-4(\theta - 3) - \theta/2 = 0 \;\Rightarrow\; -4\theta + 12 - \theta/2 = 0 \;\Rightarrow\; \theta = 12/(4.5) \approx 2.667$.

$$\hat{\theta}_{\text{MAP}} = 2.667 < 3.0 = \hat{\theta}_{\text{MLE}}$$

The Gaussian prior ($\lambda = 1/\tau^2 = 0.5$) shrinks the estimate toward $0$, exactly like L2 regularization.

*ML connection:* This theorem gives a principled interpretation of regularization hyperparameters. The regularization strength $\lambda$ corresponds to the prior precision $1/\tau^2$. A strong prior (small $\tau^2$, large $\lambda$) pulls weights toward zero, preventing overfitting. A weak prior (large $\tau^2$, small $\lambda$) allows weights to fit the data more freely. Weight decay in neural networks is exactly L2 regularization, i.e., a Gaussian prior on the weights.

---

## 5. Posterior Predictive Distribution

**Definition 4.8.7 (Posterior Predictive).** The *posterior predictive distribution* for a new observation $x^*$ given training data $D$ is:

$$P(x^* \mid D) = \int P(x^* \mid \theta) \, P(\theta \mid D) \, d\theta$$

This marginalizes over all possible parameter values, weighted by the posterior. Unlike point estimates (MLE, MAP), the posterior predictive accounts for parameter uncertainty.

**Example 4.9.9 (Bayesian Prediction — Will the Next Coin Flip Be Heads?).**
From Example 4.9.4, the posterior is $\theta \mid D \sim \text{Beta}(9, 5)$. The predictive probability of heads on the next flip is:

$$P(x^* = 1 \mid D) = \int_0^1 \theta \cdot P(\theta \mid D) \, d\theta = \mathbb{E}[\theta \mid D] = \frac{9}{14} \approx 0.643$$

Compare: the MLE plug-in prediction would give $P(x^* = 1) = 0.700$. The Bayesian prediction is more conservative because it averages over parameter uncertainty, weighting lower values of $\theta$ that are still plausible under the posterior.

For predicting $m$ heads in the next $M = 5$ flips, the posterior predictive is Beta-Binomial:

$$P(m \mid D) = \binom{5}{m} \frac{B(9 + m,\; 5 + 5 - m)}{B(9, 5)}$$

This distribution has heavier tails than $\text{Binomial}(5, 0.643)$ because it accounts for uncertainty in $\theta$.

**Theorem 4.8.5 (Posterior Predictive Variance Decomposition).**

$$\text{Var}(x^* \mid D) = \underbrace{\mathbb{E}_{\theta|D}[\text{Var}(x^* \mid \theta)]}_{\text{aleatoric uncertainty}} + \underbrace{\text{Var}_{\theta|D}(\mathbb{E}[x^* \mid \theta])}_{\text{epistemic uncertainty}}$$

*Proof.* This is the law of total variance: $\text{Var}(X) = \mathbb{E}[\text{Var}(X \mid Y)] + \text{Var}(\mathbb{E}[X \mid Y])$, applied with $X = x^*$ and $Y = \theta$, conditioning on $D$. $\square$

```
POSTERIOR PREDICTIVE vs POINT ESTIMATE

  MAP/MLE prediction:                   Bayesian prediction:
  Single θ̂, single prediction           Integrate over posterior

       P(x*|θ̂)                             P(x*|D) = ∫ P(x*|θ)P(θ|D)dθ
        ▲                                   ▲
        │   ╱╲                              │   ╱──╲
        │  ╱  ╲                             │  ╱    ╲
        │ ╱    ╲                            │ ╱      ╲
        │╱      ╲                           │╱        ╲──
        └────────▶ x*                       └───────────▶ x*

  Overconfident (ignores                 Wider tails (accounts for
  parameter uncertainty)                 parameter uncertainty)
```

*ML connection:* The two-term decomposition separates *aleatoric uncertainty* (noise inherent in the data, irreducible) from *epistemic uncertainty* (uncertainty about the model, reducible with more data). In safety-critical applications (medical diagnosis, autonomous driving), distinguishing these is essential: the model should say "I don't know" (high epistemic uncertainty) rather than making a confident wrong prediction.

---

## 6. Bayesian vs Frequentist Comparison

**Theorem 4.8.6 (Equivalence in the Large-Sample Limit).** For well-specified models and regular priors, as $n \to \infty$:

$$\hat{\theta}_{\text{MAP}} \to \hat{\theta}_{\text{MLE}}, \quad P(\theta \mid D) \to \mathcal{N}\left(\hat{\theta}_{\text{MLE}}, \, [nI(\hat{\theta})]^{-1}\right)$$

The approaches diverge most when data is scarce. *Proof follows from the Bernstein-von Mises theorem (Theorem 4.8.2).* $\square$

| Aspect | Frequentist | Bayesian |
|--------|-------------|----------|
| Parameter $\theta$ | Fixed but unknown | Random variable with distribution |
| Probability | Long-run frequency | Degree of belief |
| Point estimate | MLE: $\arg\max P(D \mid \theta)$ | MAP: $\arg\max P(\theta \mid D)$ |
| Uncertainty | Confidence interval (covers $\theta$ in repeated experiments) | Credible interval ($P(\theta \in [a,b] \mid D) = 0.95$) |
| Regularization | Ad hoc penalty term | Emerges naturally from prior |
| Model comparison | Likelihood ratio test, AIC | Bayes factors, marginal likelihood |
| Small data | Can overfit or be undefined | Prior stabilizes estimates |
| Computation | Often closed-form or optimization | May require MCMC or variational methods |

```
CONFIDENCE vs CREDIBLE INTERVAL

  Frequentist 95% confidence interval:
  "If I repeated this experiment many times, 95% of intervals would contain θ"

  ──[────●────]──  ✓ contains θ
  ──[──●──────]──  ✓ contains θ
  ──────[──●──]──  ✓ contains θ
  [────●]────────  ✗ misses θ
                   │
                   θ (fixed, unknown)

  Bayesian 95% credible interval:
  "Given the data I observed, P(θ ∈ [a,b]) = 0.95"

        P(θ|D)
         ╱╲
        ╱  ╲         95% of the posterior
    ╱──╱    ╲──╲     mass lies in [a, b]
   ╱──╱──────╲──╲
  └───a───────b──▶ θ
```

**Example 4.9.10 (95% Credible Interval — Gaussian Posterior).**
From Example 4.9.6, the posterior is $\mu \mid D \sim \mathcal{N}(23.15, \; 0.900)$, so posterior std $= \sqrt{0.900} \approx 0.949$.

The $95\%$ equal-tailed credible interval is $\mu_n \pm 1.96 \cdot \sigma_n$:

$$[23.15 - 1.96 \times 0.949, \;\; 23.15 + 1.96 \times 0.949] = [21.29, \;\; 25.01]$$

Interpretation: given the prior and observed data, there is a $95\%$ posterior probability that the true average temperature lies in $[21.29, 25.01]°$C. This is a direct probability statement about $\mu$ — unlike a frequentist confidence interval, which is about the procedure, not the parameter.

---

## 7. Computational Methods

### 7.1 Exact Inference (Conjugate Families)

When the prior is conjugate to the likelihood, the posterior has a closed-form expression. This is computationally free but limited to the conjugate families listed in Section 3.

### 7.2 Markov Chain Monte Carlo (MCMC)

**Definition 4.8.8 (MCMC).** When the posterior has no closed form, MCMC constructs a Markov chain whose stationary distribution is $P(\theta \mid D)$. After sufficient iterations (burn-in), samples from the chain approximate posterior samples.

**Metropolis-Hastings (concept):** Propose $\theta' \sim q(\theta' \mid \theta_t)$, then accept with probability:

$$\alpha = \min\left(1, \; \frac{P(\theta' \mid D) \, q(\theta_t \mid \theta')}{P(\theta_t \mid D) \, q(\theta' \mid \theta_t)}\right)$$

Key insight: the evidence $P(D)$ cancels in the ratio, so we only need the unnormalized posterior.

**Gibbs Sampling (concept):** When the joint posterior is hard but the full conditionals $P(\theta_i \mid \theta_{-i}, D)$ are easy, sample each parameter in turn from its conditional. This is a special case of Metropolis-Hastings with acceptance probability $1$.

```
MCMC — EXPLORING THE POSTERIOR

  Posterior P(θ|D) in 2D parameter space

  θ₂ ▲
     │        ···
     │      ·· ╱╲ ··         Chain explores high-probability
     │    ·  ╱    ╲  ·       regions via random walk
     │   · ╱   ●   ╲ ·
     │    ╱  ↗ ╱↘    ╲      ● = current sample
     │   · ↗ ╱   ↘  · ·     → = proposed moves
     │    ╱ ╱  ↙  ╲         (accepted or rejected)
     │   · ·   ·   ·
     │    ·  · · ·
     └───────────────▶ θ₁

  After burn-in: histogram of samples ≈ posterior
```

### 7.3 Variational Inference

**Definition 4.8.9 (Variational Inference).** Approximate the intractable posterior $P(\theta \mid D)$ by finding the closest distribution $q(\theta)$ in a tractable family $\mathcal{Q}$:

$$q^*(\theta) = \arg\min_{q \in \mathcal{Q}} \text{KL}\big(q(\theta) \,\|\, P(\theta \mid D)\big)$$

**Definition 4.8.10 (Evidence Lower Bound).** Since $\text{KL} \geq 0$, minimizing the KL divergence is equivalent to maximizing the ELBO:

$$\text{ELBO}(q) = \mathbb{E}_{q}[\log P(D, \theta)] - \mathbb{E}_{q}[\log q(\theta)] = \log P(D) - \text{KL}(q \| P(\theta \mid D))$$

| Method | Pros | Cons |
|--------|------|------|
| Exact (conjugate) | No approximation, fast | Limited model families |
| MCMC | Asymptotically exact | Slow for high dimensions, hard to diagnose convergence |
| Variational inference | Fast, scalable | Approximate, may underestimate variance |

*ML connection:* Variational autoencoders (VAEs) are the flagship ML application of variational inference. The encoder network $q_\phi(z \mid x)$ approximates the posterior over latent variables, and training maximizes the ELBO. Stochastic variational inference scales to millions of data points, making Bayesian methods practical in deep learning.

---

## 8. Applications in Machine Learning

### 8.1 Bayesian Neural Networks

Standard neural networks produce point predictions. A Bayesian neural network places a prior $P(\mathbf{w})$ over all weights and computes the posterior $P(\mathbf{w} \mid D)$. Predictions average over weight uncertainty:

$$P(y^* \mid x^*, D) = \int P(y^* \mid x^*, \mathbf{w}) \, P(\mathbf{w} \mid D) \, d\mathbf{w}$$

In regions far from training data, the weight posterior is diffuse, producing high predictive variance — the network signals "I don't know."

### 8.2 Bayesian Optimization

For expensive-to-evaluate functions $f$ (e.g., neural network validation accuracy as a function of hyperparameters), Bayesian optimization maintains a Gaussian process posterior over $f$ and uses an acquisition function (Expected Improvement, Upper Confidence Bound) to decide where to evaluate next.

```
BAYESIAN OPTIMIZATION — HYPERPARAMETER TUNING

  f(θ)  ▲
  (val  │    ?              GP posterior mean ± 2σ
  acc.) │  ╱─╲  ╱ · · ╲         ╱──╲
        │ ╱   ╲╱        ╲      ╱    ╲
        │╱     ╲    ?    · · ·╱      ╲
        │       ╲       ╱              ╲
        └───●───●──●────────────●───────▶ θ (e.g., learning rate)
            │   │  │            │
         observed evaluations   next evaluation
                                (max acquisition fn)

  Few evaluations ──▶ high uncertainty ──▶ explore
  Many evaluations ──▶ low uncertainty ──▶ exploit near best
```

### 8.3 Thompson Sampling

In multi-armed bandits, Thompson sampling draws $\theta_a \sim P(\theta_a \mid D)$ for each arm $a$ and plays the arm with the highest sample. This naturally balances exploration (uncertain arms get lucky draws) and exploitation (arms with high posterior mean are likely sampled high).

### 8.4 Regularization as Prior

**Theorem 4.8.4** (Section 4) showed that L2 and L1 regularization are MAP estimation under Gaussian and Laplace priors respectively. This extends further:

| Regularization Technique | Bayesian Interpretation |
|-------------------------|------------------------|
| L2 (weight decay) | Gaussian prior $\mathcal{N}(0, \tau^2)$ on weights |
| L1 (Lasso) | Laplace prior on weights (promotes sparsity) |
| Elastic net | Mixture of Gaussian and Laplace priors |
| Early stopping | Implicit prior (limits posterior exploration) |
| Data augmentation | Implicit prior encoding invariances |

### 8.5 Dropout as Approximate Bayesian Inference

**Theorem 4.8.7 (Gal and Ghahramani, 2016).** Training a neural network with dropout applied before every weight layer and L2 regularization is mathematically equivalent to an approximation to the posterior in a deep Gaussian process model. At test time, running multiple forward passes with dropout active (MC Dropout) gives samples from the approximate posterior, and their variance estimates epistemic uncertainty.

```
MC DROPOUT — UNCERTAINTY FROM A STANDARD NETWORK

  Training: dropout as regularization (standard practice)

  Test time: run T forward passes with dropout ON

  Pass 1:  x ──▶ [  ▪ ○ ▪ ▪ ○ ] ──▶ ŷ₁ = 0.83
  Pass 2:  x ──▶ [ ○ ▪ ▪ ○ ▪  ] ──▶ ŷ₂ = 0.79
  Pass 3:  x ──▶ [  ▪ ▪ ○ ▪ ▪ ] ──▶ ŷ₃ = 0.81
    ⋮                                   ⋮
  Pass T:  x ──▶ [ ○ ▪ ▪ ▪ ○  ] ──▶ ŷ_T = 0.77

  Prediction: mean(ŷ₁..ŷ_T) = 0.80
  Uncertainty: var(ŷ₁..ŷ_T) = 0.0005  (low → confident)

  ▪ = active neuron    ○ = dropped neuron
```

*ML connection:* MC Dropout is the cheapest path to uncertainty quantification in deep learning. It requires no architectural changes — just keep dropout on at test time and run multiple passes. The variance across passes provides a calibrated uncertainty estimate, useful for active learning (query the most uncertain examples) and safety-critical deployment.

---

## 9. Summary of Key Results

| Result | Statement |
|--------|-----------|
| Bayes' rule for inference | $P(\theta \mid D) \propto P(D \mid \theta)P(\theta)$ |
| Sequential updating | Today's posterior is tomorrow's prior |
| Conjugate prior | Posterior stays in the same family as prior |
| Beta-Binomial posterior mean | Weighted average of prior mean and MLE |
| Gaussian posterior precision | Prior precision $+$ data precision |
| MAP = regularized MLE | Gaussian prior $\leftrightarrow$ L2, Laplace prior $\leftrightarrow$ L1 |
| Posterior predictive | $P(x^* \mid D) = \int P(x^* \mid \theta)P(\theta \mid D)d\theta$ |
| Variance decomposition | Total = aleatoric (irreducible) $+$ epistemic (reducible) |
| Bernstein-von Mises | Posterior $\to \mathcal{N}(\hat{\theta}_{\text{MLE}}, [nI(\hat\theta)]^{-1})$ as $n \to \infty$ |
| ELBO | $\log P(D) \geq \text{ELBO}(q) = \mathbb{E}_q[\log P(D,\theta)] - \mathbb{E}_q[\log q(\theta)]$ |

---

## Exercises

**★ Basic**

1. Suppose you have a coin with unknown bias $\theta$ and a $\text{Beta}(1, 1)$ prior (uniform). After observing $7$ heads and $3$ tails, write down the posterior distribution and compute the posterior mean and MAP estimate.

2. You believe a parameter $\mu$ is near $0$ and place a prior $\mu \sim \mathcal{N}(0, 4)$. You observe one data point $x = 3$ from $\mathcal{N}(\mu, 1)$. Compute the posterior mean and variance using the Gaussian-Gaussian conjugate update.

3. For L2-regularized linear regression with $\lambda = 0.1$, what is the corresponding Gaussian prior variance $\tau^2$? If you doubled $\lambda$, how would the prior change?

4. Explain in one sentence each: (a) why the posterior is narrower than the prior, (b) why the posterior mean lies between the prior mean and the MLE.

**★★ Intermediate**

5. Derive the posterior predictive distribution for the Beta-Binomial model: if $\theta \mid D \sim \text{Beta}(\alpha', \beta')$, show that $P(x^* = 1 \mid D) = \frac{\alpha'}{\alpha' + \beta'}$, the posterior mean.

6. Prove that MAP estimation with a uniform prior over a bounded interval $[a, b]$ reduces to maximum likelihood estimation.

7. A weather model gives $P(\text{rain} \mid \text{cloudy}) = 0.6$ and $P(\text{cloudy} \mid \text{rain}) = 0.9$. If $P(\text{rain}) = 0.2$, compute $P(\text{cloudy})$ and $P(\text{rain} \mid \text{cloudy})$ using Bayes' theorem. Compare with the naive estimate.

8. Show that the Gaussian posterior precision formula $1/\sigma_n^2 = 1/\sigma_0^2 + n/\sigma^2$ implies the posterior variance decreases as $O(1/n)$, matching the frequentist standard error scaling.

9. You have two models: $M_1$ with marginal likelihood $P(D \mid M_1) = 0.01$ and $M_2$ with $P(D \mid M_2) = 0.005$. With equal priors, compute the Bayes factor and posterior model probabilities. Which model does Bayesian model comparison prefer?

**★★★ Challenging**

10. Derive the ELBO for a model with latent variables $z$: starting from $\log P(D) = \log \int P(D, z) \, dz$, introduce $q(z)$, apply Jensen's inequality, and show $\log P(D) \geq \mathbb{E}_q[\log P(D, z)] - \mathbb{E}_q[\log q(z)]$.

11. Prove that the Dirichlet-Multinomial conjugacy holds: if $\boldsymbol{\theta} \sim \text{Dir}(\boldsymbol{\alpha})$ and $\mathbf{c} = (c_1, \ldots, c_K)$ are observed category counts, show the posterior is $\text{Dir}(\alpha_1 + c_1, \ldots, \alpha_K + c_K)$. Explain why Dirichlet priors with $\alpha_i < 1$ promote sparsity in topic models.

12. Consider a Bayesian linear regression model: $\mathbf{y} = X\mathbf{w} + \boldsymbol{\epsilon}$ with $\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \sigma^2 I)$ and prior $\mathbf{w} \sim \mathcal{N}(\mathbf{0}, \tau^2 I)$. Derive the posterior $P(\mathbf{w} \mid X, \mathbf{y})$ and show that the posterior mean equals the ridge regression solution $\hat{\mathbf{w}} = (X^TX + \frac{\sigma^2}{\tau^2}I)^{-1}X^T\mathbf{y}$.

13. In Thompson sampling for a 3-armed Bernoulli bandit with $\text{Beta}(1,1)$ priors, after observing arm rewards $\{1, 0, 1\}$ for arm 1, $\{0, 0\}$ for arm 2, and $\{1\}$ for arm 3: write the posterior for each arm, compute $P(\text{arm } i \text{ is best})$ analytically or via simulation, and explain why Thompson sampling would still sometimes explore arm 2.

---

## Related Topics

- [Conditional Probability](conditional-probability.md) — Bayes' theorem, the foundation of Bayesian inference
- [Distributions](distributions.md) — Beta, Gaussian, Dirichlet used as conjugate priors
- [Maximum Likelihood](maximum-likelihood.md) — the frequentist counterpart to Bayesian estimation
- [Limit Theorems](limit-theorems.md) — convergence results underlying Bernstein-von Mises
- [Information Theory](../information-theory/index.md) — KL divergence used in variational inference
- [Optimization](../optimization/index.md) — ELBO maximization and MAP optimization
