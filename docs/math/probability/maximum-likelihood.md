# Maximum Likelihood Estimation

Maximum likelihood estimation is the workhorse of statistical inference and machine learning. The idea is deceptively simple: given observed data, find the parameter values that make the data most probable. This single principle unifies logistic regression, neural network training, language models, and Gaussian mixture models. When you minimize cross-entropy loss, you are doing MLE. When you train a neural network with backpropagation, you are (usually) solving an MLE problem. Understanding MLE, its properties, and its limitations is essential for reasoning about why models behave the way they do.

---

## Prerequisites

- [Distributions](distributions.md) -- Bernoulli, Gaussian, and their PMFs/PDFs
- [Calculus -- Differentiation](../calculus/differentiation.md) -- partial derivatives, chain rule, optimization

---

## 1. The Likelihood Function

**Definition 4.9.1 (Likelihood Function).** Given independent observations $x_1, x_2, \ldots, x_n$ from a distribution with parameter $\theta$, the *likelihood function* is:

$$L(\theta;\, x_1, \ldots, x_n) = \prod_{i=1}^{n} P(x_i \mid \theta)$$

where $P(x_i \mid \theta)$ is the PMF (discrete) or PDF (continuous) evaluated at $x_i$.

The likelihood is **not** a probability over $\theta$. It is a function of $\theta$ that measures how well each parameter value explains the observed data.

```
THE LIKELIHOOD FUNCTION

  L(θ)
  │
  │         ╭──╮
  │        ╱    ╲
  │      ╱        ╲
  │    ╱            ╲
  │  ╱                ╲
  │╱                    ╲___
  └──────────────────────────── θ
              ^
              θ̂_MLE
         (parameter that makes
          data most probable)

  L(θ) = P(x₁|θ) · P(x₂|θ) · ... · P(xₙ|θ)
```

*ML connection:* Every supervised learning algorithm with a probabilistic interpretation defines a likelihood. For a classifier outputting $P(y \mid \mathbf{x}; \theta)$, the likelihood over a dataset $\{(\mathbf{x}_i, y_i)\}$ is $L(\theta) = \prod_i P(y_i \mid \mathbf{x}_i; \theta)$. Maximizing this likelihood is equivalent to minimizing cross-entropy loss.

**Example 4.10.1 (Likelihood for coin flips).** Suppose we flip a coin 5 times and observe $\text{H, H, T, H, T}$. Encoding $\text{H}=1, \text{T}=0$, the data is $(1,1,0,1,0)$. For a Bernoulli model with parameter $p$, the likelihood is:

$$L(p) = p \cdot p \cdot (1-p) \cdot p \cdot (1-p) = p^3(1-p)^2$$

Evaluating at a few candidate values:

| $p$ | $L(p) = p^3(1-p)^2$ |
|-----|----------------------|
| 0.3 | $0.027 \times 0.49 = 0.01323$ |
| 0.5 | $0.125 \times 0.25 = 0.03125$ |
| 0.6 | $0.216 \times 0.16 = 0.03456$ |
| 0.7 | $0.343 \times 0.09 = 0.03087$ |

The likelihood is highest near $p = 0.6$, which is $3/5$ -- the sample proportion.

---

## 2. Log-Likelihood

**Definition 4.9.2 (Log-Likelihood).** The *log-likelihood* is:

$$\ell(\theta) = \log L(\theta) = \sum_{i=1}^{n} \log P(x_i \mid \theta)$$

We use the log-likelihood instead of the likelihood for two reasons:

| Problem with $L(\theta)$ | How $\ell(\theta)$ fixes it |
|---------------------------|----------------------------|
| Product of $n$ small numbers $\to$ underflow ($10^{-300}$) | Sum of log-probabilities stays in a manageable range |
| Products are harder to differentiate | Sums are easy to differentiate term-by-term |
| No effect on the optimum | $\log$ is monotonically increasing, so $\arg\max L = \arg\max \ell$ |

```
LIKELIHOOD vs LOG-LIKELIHOOD

  L(θ)                               ℓ(θ) = log L(θ)
  │    ╭╮                              │     ╭───╮
  │   ╱  ╲                             │   ╱       ╲
  │  ╱    ╲                            │  ╱         ╲
  │ ╱      ╲___                        │╱             ╲
  │╱            ╲____                  │                ╲___
  └────────────────── θ               └──────────────────── θ
         ^                                    ^
      same argmax                          same argmax

  Numerical example (n = 1000):
  L(θ) = 0.3¹⁰⁰⁰ ≈ 10⁻⁵²³   (underflows to 0)
  ℓ(θ) = 1000 · log(0.3) ≈ −1204   (perfectly fine)
```

**Example 4.10.2 (Log-likelihood for coin flips).** Continuing Example 4.10.1 with data $\text{H, H, T, H, T}$ and $L(p) = p^3(1-p)^2$, the log-likelihood is:

$$\ell(p) = \log L(p) = 3\log p + 2\log(1-p)$$

At $p = 0.6$: $\ell(0.6) = 3\log(0.6) + 2\log(0.4) = 3(-0.5108) + 2(-0.9163) = -1.5324 - 1.8326 = -3.365$

At $p = 0.5$: $\ell(0.5) = 3\log(0.5) + 2\log(0.5) = 5\log(0.5) = 5(-0.6931) = -3.466$

Since $-3.365 > -3.466$, the log-likelihood confirms $p = 0.6$ is a better fit than $p = 0.5$, matching the likelihood comparison. Both functions agree on the ranking because $\log$ is monotonically increasing.

---

## 3. The MLE Definition and Computation

**Definition 4.9.3 (Maximum Likelihood Estimator).** The *maximum likelihood estimator* (MLE) is the parameter value that maximizes the likelihood (equivalently, the log-likelihood):

$$\hat{\theta}_{\text{MLE}} = \arg\max_\theta \, L(\theta) = \arg\max_\theta \, \ell(\theta)$$

### 3.1 The Score Equation

To find the MLE, we differentiate and set to zero.

**Definition 4.9.4 (Score Function).** The *score function* is the gradient of the log-likelihood:

$$s(\theta) = \frac{\partial \ell(\theta)}{\partial \theta}$$

The MLE satisfies the *score equation*:

$$s(\hat{\theta}) = \frac{\partial \ell}{\partial \theta}\bigg|_{\theta = \hat{\theta}} = 0$$

### 3.2 Second-Order Conditions

Setting $s(\theta) = 0$ finds critical points, but we need to verify we found a maximum:

$$\frac{\partial^2 \ell}{\partial \theta^2}\bigg|_{\theta = \hat{\theta}} < 0 \quad \text{(confirms a local maximum)}$$

For vector parameters $\boldsymbol{\theta} \in \mathbb{R}^k$, the Hessian $\nabla^2 \ell(\hat{\boldsymbol{\theta}})$ must be negative definite.

```
MLE COMPUTATION RECIPE

  Step 1: Write the likelihood
          L(θ) = ∏ P(xᵢ | θ)

  Step 2: Take the log
          ℓ(θ) = Σ log P(xᵢ | θ)

  Step 3: Differentiate and set to zero
          ∂ℓ/∂θ = 0   →   solve for θ̂

  Step 4: Check second-order condition
          ∂²ℓ/∂θ² < 0   →   confirms maximum

  Step 5: (If no closed form)
          Use gradient ascent: θ ← θ + η · ∂ℓ/∂θ
```

---

## 4. Worked Examples

### 4.1 Bernoulli Distribution

**Setup:** $x_1, \ldots, x_n \overset{\text{iid}}{\sim} \text{Bernoulli}(p)$, where $x_i \in \{0, 1\}$.

**Log-likelihood:**

$$\ell(p) = \sum_{i=1}^n \big[x_i \log p + (1 - x_i)\log(1-p)\big] = k\log p + (n-k)\log(1-p)$$

where $k = \sum x_i$ (number of successes).

**Score equation:**

$$\frac{d\ell}{dp} = \frac{k}{p} - \frac{n-k}{1-p} = 0$$

Solving: $k(1-p) = (n-k)p \implies k = np \implies \hat{p}_{\text{MLE}} = \frac{k}{n}$

**Theorem 4.9.1.** The MLE for the Bernoulli parameter is the sample proportion:

$$\hat{p}_{\text{MLE}} = \frac{1}{n}\sum_{i=1}^{n} x_i = \bar{x}$$

*Proof.* Derived above. The second derivative is $\frac{d^2\ell}{dp^2} = -\frac{k}{p^2} - \frac{n-k}{(1-p)^2} < 0$ for $0 < p < 1$, confirming a maximum. $\square$

**Example 4.10.3 (Bernoulli MLE with data).** Continuing the $\text{H, H, T, H, T}$ data from Example 4.10.1, we have $n = 5$ trials with $k = 3$ heads. Applying Theorem 4.9.1:

$$\hat{p}_{\text{MLE}} = \frac{k}{n} = \frac{3}{5} = 0.6$$

We can verify via the score equation: $\frac{d\ell}{dp} = \frac{3}{p} - \frac{2}{1-p}$. At $p = 0.6$: $\frac{3}{0.6} - \frac{2}{0.4} = 5 - 5 = 0$. The second derivative at $\hat{p} = 0.6$ is $-\frac{3}{0.36} - \frac{2}{0.16} = -8.33 - 12.5 = -20.83 < 0$, confirming a maximum.

*ML connection:* In binary classification, the output layer produces $\hat{p} = \sigma(\mathbf{w}^T\mathbf{x} + b)$. The cross-entropy loss $-[y\log\hat{p} + (1-y)\log(1-\hat{p})]$ is the negative log-likelihood of a Bernoulli. Minimizing cross-entropy = maximizing likelihood.

### 4.2 Gaussian Distribution

**Setup:** $x_1, \ldots, x_n \overset{\text{iid}}{\sim} \mathcal{N}(\mu, \sigma^2)$.

**Log-likelihood:**

$$\ell(\mu, \sigma^2) = -\frac{n}{2}\log(2\pi) - \frac{n}{2}\log(\sigma^2) - \frac{1}{2\sigma^2}\sum_{i=1}^{n}(x_i - \mu)^2$$

**Score equations:**

$$\frac{\partial \ell}{\partial \mu} = \frac{1}{\sigma^2}\sum_{i=1}^{n}(x_i - \mu) = 0 \implies \hat{\mu}_{\text{MLE}} = \frac{1}{n}\sum_{i=1}^n x_i = \bar{x}$$

$$\frac{\partial \ell}{\partial \sigma^2} = -\frac{n}{2\sigma^2} + \frac{1}{2(\sigma^2)^2}\sum_{i=1}^{n}(x_i - \mu)^2 = 0 \implies \hat{\sigma}^2_{\text{MLE}} = \frac{1}{n}\sum_{i=1}^{n}(x_i - \bar{x})^2$$

**Theorem 4.9.2.** The MLEs for the Gaussian parameters are:

$$\hat{\mu}_{\text{MLE}} = \bar{x} = \frac{1}{n}\sum_{i=1}^n x_i \qquad \hat{\sigma}^2_{\text{MLE}} = \frac{1}{n}\sum_{i=1}^{n}(x_i - \bar{x})^2$$

*Proof.* Derived above. The Hessian of $\ell$ with respect to $(\mu, \sigma^2)$ is negative definite at $(\hat{\mu}, \hat{\sigma}^2)$, confirming these are joint maxima. $\square$

**Example 4.10.4 (Normal MLE for $\mu$).** Given data $x_1 = 2, x_2 = 4, x_3 = 6, x_4 = 8$, the MLE for the mean is:

$$\hat{\mu}_{\text{MLE}} = \bar{x} = \frac{2 + 4 + 6 + 8}{4} = \frac{20}{4} = 5.0$$

**Example 4.10.5 (Normal MLE for $\sigma^2$).** Using the same data $(2, 4, 6, 8)$ with $\bar{x} = 5$:

$$\hat{\sigma}^2_{\text{MLE}} = \frac{1}{n}\sum_{i=1}^{n}(x_i - \bar{x})^2 = \frac{(2-5)^2 + (4-5)^2 + (6-5)^2 + (8-5)^2}{4} = \frac{9 + 1 + 1 + 9}{4} = \frac{20}{4} = 5.0$$

Note the MLE divides by $n = 4$. The unbiased estimate would divide by $n - 1 = 3$, giving $20/3 \approx 6.67$. The MLE underestimates the true variance by a factor of $(n-1)/n = 3/4$.

Note that $\hat{\sigma}^2_{\text{MLE}}$ divides by $n$, not $n-1$. It is a *biased* estimator of the true variance: $\mathbb{E}[\hat{\sigma}^2_{\text{MLE}}] = \frac{n-1}{n}\sigma^2$. The unbiased sample variance uses $n-1$ (Bessel's correction). This illustrates that MLE does not always produce unbiased estimators.

*ML connection:* Mean squared error (MSE) loss arises from assuming Gaussian noise. If $y = f_\theta(\mathbf{x}) + \epsilon$ with $\epsilon \sim \mathcal{N}(0, \sigma^2)$, the negative log-likelihood is $\frac{1}{2\sigma^2}\sum(y_i - f_\theta(\mathbf{x}_i))^2 + \text{const}$. Minimizing MSE = MLE under Gaussian noise.

---

## 5. Fisher Information

**Definition 4.9.5 (Fisher Information).** The *Fisher information* measures how much information the data carries about $\theta$:

$$I(\theta) = \mathbb{E}\left[\left(\frac{\partial \log P(X \mid \theta)}{\partial \theta}\right)^2\right] = -\mathbb{E}\left[\frac{\partial^2 \log P(X \mid \theta)}{\partial \theta^2}\right]$$

The two forms are equivalent under regularity conditions (the ability to exchange differentiation and integration).

For $n$ iid observations, the total Fisher information is $I_n(\theta) = n \cdot I(\theta)$.

**Example 4.10.6 (Fisher information for Bernoulli).** For a single $X \sim \text{Bernoulli}(p)$, the log-PMF is $\log P(X \mid p) = X\log p + (1-X)\log(1-p)$. The second derivative is:

$$\frac{\partial^2 \log P(X \mid p)}{\partial p^2} = -\frac{X}{p^2} - \frac{1-X}{(1-p)^2}$$

Taking the negative expectation (using $\mathbb{E}[X] = p$):

$$I(p) = -\mathbb{E}\left[-\frac{X}{p^2} - \frac{1-X}{(1-p)^2}\right] = \frac{p}{p^2} + \frac{1-p}{(1-p)^2} = \frac{1}{p} + \frac{1}{1-p} = \frac{1}{p(1-p)}$$

At $p = 0.5$: $I(0.5) = \frac{1}{0.25} = 4$ (maximum information -- most uncertainty about outcome).
At $p = 0.9$: $I(0.9) = \frac{1}{0.09} \approx 11.1$ (high information -- each observation is very informative about the small probability of tails).

For $n = 100$ observations: $I_{100}(0.5) = 100 \times 4 = 400$.

**Interpretation:** Fisher information measures the *curvature* of the log-likelihood at the true parameter. High curvature means the likelihood function is sharply peaked -- the data strongly constrains $\theta$. Low curvature means the likelihood is flat -- the data provides little information about $\theta$.

```
FISHER INFORMATION AS CURVATURE

  High Fisher Information:         Low Fisher Information:
  (data is informative)            (data is uninformative)

  ℓ(θ)                             ℓ(θ)
  │     ╭╮                          │    ╭────────╮
  │    ╱  ╲                          │  ╱            ╲
  │   ╱    ╲                         │╱                ╲
  │  ╱      ╲                        │
  │ ╱        ╲                       │
  └────────────── θ                 └──────────────────── θ
       sharp peak                        flat peak
    → small variance                  → large variance
      of estimator                      of estimator

  I(θ) = -E[∂²ℓ/∂θ²] = curvature at the peak
```

---

## 6. The Cramer-Rao Bound

**Theorem 4.9.3 (Cramer-Rao Lower Bound).** For any unbiased estimator $\hat{\theta}$ of $\theta$:

$$\text{Var}(\hat{\theta}) \geq \frac{1}{I_n(\theta)} = \frac{1}{n \cdot I(\theta)}$$

No unbiased estimator can have variance smaller than $1/(nI(\theta))$.

*Proof sketch.* Apply the Cauchy-Schwarz inequality to $\text{Cov}(\hat{\theta}, s(\theta))$ where $s(\theta) = \frac{\partial}{\partial\theta}\log L(\theta)$ is the score. Since $\hat{\theta}$ is unbiased, $\text{Cov}(\hat{\theta}, s) = 1$. Then $1 \leq \text{Var}(\hat{\theta}) \cdot \text{Var}(s) = \text{Var}(\hat{\theta}) \cdot I_n(\theta)$. $\square$

**Definition 4.9.6 (Efficient Estimator).** An unbiased estimator that achieves the Cramer-Rao bound (variance equals $1/I_n(\theta)$) is called *efficient*.

**Example 4.10.7 (Cramer-Rao bound for Bernoulli).** From Example 4.10.6, $I(p) = \frac{1}{p(1-p)}$. For $n = 100$ observations with true $p = 0.3$:

$$\text{Var}(\hat{p}) \geq \frac{1}{nI(p)} = \frac{1}{100 \cdot \frac{1}{0.3 \times 0.7}} = \frac{0.3 \times 0.7}{100} = \frac{0.21}{100} = 0.0021$$

The Bernoulli MLE $\hat{p} = k/n$ has exact variance $\text{Var}(\hat{p}) = p(1-p)/n = 0.0021$, which equals the bound. Therefore $\hat{p}$ is an **efficient** estimator -- no unbiased estimator can do better.

---

## 7. Properties of MLE

**Theorem 4.9.4 (Asymptotic Properties of MLE).** Under regularity conditions, as $n \to \infty$:

| Property | Statement | Meaning |
|----------|-----------|---------|
| **Consistency** | $\hat{\theta}_{\text{MLE}} \xrightarrow{p} \theta_0$ | MLE converges to the true parameter |
| **Asymptotic normality** | $\sqrt{n}(\hat{\theta}_{\text{MLE}} - \theta_0) \xrightarrow{d} \mathcal{N}(0, I(\theta_0)^{-1})$ | Distribution of MLE becomes Gaussian |
| **Asymptotic efficiency** | $\text{Var}(\hat{\theta}_{\text{MLE}}) \to \frac{1}{nI(\theta_0)}$ | MLE achieves the Cramer-Rao bound |

*Proof of consistency (sketch).* The normalized log-likelihood $\frac{1}{n}\ell(\theta)$ converges (by LLN) to $\mathbb{E}[\log P(X \mid \theta)]$. By Gibbs' inequality (or KL divergence non-negativity), this expectation is uniquely maximized at $\theta = \theta_0$. Since the MLE maximizes $\frac{1}{n}\ell(\theta)$, and the limiting function has a unique maximum, the MLE must converge to $\theta_0$. $\square$

**Theorem 4.9.5 (Invariance of MLE).** If $\hat{\theta}_{\text{MLE}}$ is the MLE of $\theta$, then for any function $g$:

$$\widehat{g(\theta)}_{\text{MLE}} = g(\hat{\theta}_{\text{MLE}})$$

The MLE of any transformation of $\theta$ is simply that transformation applied to the MLE.

**Example 4.10.10 (Invariance of MLE).** From Example 4.10.3, the MLE for a coin with data $\text{H, H, T, H, T}$ is $\hat{p} = 0.6$. Using invariance, we can immediately compute the MLE for any function of $p$:

- *Odds:* $g(p) = \frac{p}{1-p} \implies \widehat{\text{odds}} = \frac{0.6}{0.4} = 1.5$ (heads are 1.5x more likely than tails)
- *Log-odds:* $g(p) = \log\frac{p}{1-p} \implies \widehat{\text{log-odds}} = \log(1.5) \approx 0.405$
- *Variance:* $g(p) = p(1-p) \implies \widehat{\text{Var}} = 0.6 \times 0.4 = 0.24$

No re-derivation is needed. We do **not** re-maximize the likelihood in the transformed parameter space; we simply transform the MLE.

---

## 8. MLE vs MAP

**Definition 4.9.7 (Maximum a Posteriori Estimator).** The *MAP* estimator maximizes the posterior:

$$\hat{\theta}_{\text{MAP}} = \arg\max_\theta \, P(\theta \mid \mathbf{x}) = \arg\max_\theta \, \underbrace{P(\mathbf{x} \mid \theta)}_{\text{likelihood}} \cdot \underbrace{P(\theta)}_{\text{prior}}$$

**Theorem 4.9.6.** MLE is a special case of MAP with a uniform (flat) prior $P(\theta) \propto 1$:

$$\hat{\theta}_{\text{MAP}} \big|_{P(\theta) = \text{const}} = \arg\max_\theta P(\mathbf{x} \mid \theta) = \hat{\theta}_{\text{MLE}}$$

*Proof.* With $P(\theta) = c$ (constant), $\arg\max_\theta \, P(\mathbf{x} \mid \theta) \cdot c = \arg\max_\theta \, P(\mathbf{x} \mid \theta)$. $\square$

| Aspect | MLE | MAP |
|--------|-----|-----|
| Objective | $\max_\theta P(\mathbf{x} \mid \theta)$ | $\max_\theta P(\mathbf{x} \mid \theta) P(\theta)$ |
| Prior | None (or uniform) | Explicit prior $P(\theta)$ |
| Regularization | None | Prior acts as regularizer |
| With $n \to \infty$ | MLE = MAP | Likelihood dominates prior |
| Small $n$ | Can overfit | Prior prevents extreme estimates |
| Gaussian prior on $\theta$ | -- | Equivalent to $L_2$ regularization |
| Laplace prior on $\theta$ | -- | Equivalent to $L_1$ regularization |

**Example 4.10.9 (MLE vs MAP for small sample).** We flip a coin $n = 4$ times and observe $k = 4$ heads (all heads). We compare MLE and MAP with a $\text{Beta}(3, 3)$ prior (mildly favoring fair coins).

*MLE:* $\hat{p}_{\text{MLE}} = k/n = 4/4 = 1.0$ (the coin always lands heads -- an extreme estimate from sparse data).

*MAP with Beta(3,3) prior:* $\hat{p}_{\text{MAP}} = \frac{k + \alpha - 1}{n + \alpha + \beta - 2} = \frac{4 + 3 - 1}{4 + 3 + 3 - 2} = \frac{6}{8} = 0.75$

The prior pulls the estimate away from the boundary toward 0.5. With more data ($n = 100, k = 100$), the MAP gives $\frac{102}{104} \approx 0.981$, converging toward the MLE as the likelihood dominates the prior.

*ML connection:* $L_2$ regularization (weight decay) in neural networks corresponds to a Gaussian prior $\theta_j \sim \mathcal{N}(0, \tau^2)$ on each weight. The regularized loss $\mathcal{L}(\theta) + \lambda\|\theta\|^2$ is the negative log-posterior. So "MLE + regularization" = MAP estimation.

---

## 9. Applications in Machine Learning

### 9.1 Cross-Entropy Loss = Negative Log-Likelihood

For classification with $K$ classes, the model outputs $\hat{p}_k = P(y = k \mid \mathbf{x}; \theta)$. The negative log-likelihood for one example is:

$$-\log P(y \mid \mathbf{x}; \theta) = -\sum_{k=1}^K \mathbf{1}[y=k] \log \hat{p}_k = -\log \hat{p}_y$$

Averaged over the dataset, this is the cross-entropy loss. Minimizing cross-entropy = MLE.

### 9.2 Logistic Regression

In logistic regression, $P(y=1 \mid \mathbf{x}) = \sigma(\mathbf{w}^T\mathbf{x} + b)$. The log-likelihood:

$$\ell(\mathbf{w}, b) = \sum_{i=1}^n \big[y_i \log \sigma(\mathbf{w}^T\mathbf{x}_i + b) + (1-y_i)\log(1 - \sigma(\mathbf{w}^T\mathbf{x}_i + b))\big]$$

This is concave (no local maxima other than the global one), so gradient ascent always finds the MLE. No closed-form solution exists -- iterative methods (gradient descent, Newton's method) are required.

### 9.3 Neural Networks

Training a neural network with cross-entropy loss is MLE. The model defines a conditional distribution $P(y \mid \mathbf{x}; \theta)$, and SGD maximizes the log-likelihood:

$$\hat{\theta}_{\text{MLE}} = \arg\max_\theta \sum_{i=1}^n \log P(y_i \mid \mathbf{x}_i; \theta)$$

The loss landscape is non-convex, so SGD finds a local (not global) maximum of the likelihood. In practice, overparameterized networks have many near-equivalent local optima.

### 9.4 Gaussian Mixture Models and EM

For a mixture of $K$ Gaussians, the log-likelihood contains a $\log$ of a sum:

$$\ell(\theta) = \sum_{i=1}^n \log \left(\sum_{k=1}^K \pi_k \, \mathcal{N}(x_i \mid \mu_k, \sigma_k^2)\right)$$

Setting the derivative to zero has no closed form because the $\log$ and $\sum$ do not simplify. The Expectation-Maximization (EM) algorithm solves this by alternating:

- **E-step:** Compute "soft" cluster assignments using current parameters
- **M-step:** Update parameters by MLE given the soft assignments

EM is guaranteed to increase the likelihood at each step and converges to a local maximum.

### 9.5 Fisher Information in Practice

- **Confidence intervals:** $\hat{\theta} \pm z_{\alpha/2} / \sqrt{I_n(\hat{\theta})}$
- **Natural gradient descent:** Premultiplies the gradient by $I(\theta)^{-1}$ for reparameterization-invariant updates
- **Model comparison:** Observed information helps compute BIC

**Example 4.10.8 (Asymptotic confidence interval).** We flip a coin $n = 200$ times and observe $k = 130$ heads, so $\hat{p} = 130/200 = 0.65$. Using the asymptotic normality of the MLE, a 95% confidence interval is:

$$\hat{p} \pm z_{0.025} \cdot \frac{1}{\sqrt{nI(\hat{p})}} = 0.65 \pm 1.96 \cdot \sqrt{\frac{\hat{p}(1-\hat{p})}{n}} = 0.65 \pm 1.96 \cdot \sqrt{\frac{0.65 \times 0.35}{200}}$$

$$= 0.65 \pm 1.96 \times 0.0337 = 0.65 \pm 0.066 = (0.584,\; 0.716)$$

We are 95% confident that the true probability of heads lies between 0.584 and 0.716.

```
MLE ACROSS ML — UNIFIED VIEW

  ┌─────────────────────────────────────────────────────────┐
  │                    MLE Framework                        │
  │              θ̂ = argmax Σ log P(xᵢ | θ)                │
  └──────────────────────┬──────────────────────────────────┘
                         │
         ┌───────────────┼───────────────┐
         │               │               │
         ▼               ▼               ▼
  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
  │  Regression  │ │Classification│ │  Generative  │
  │              │ │              │ │   Models     │
  │ P(y|x) =    │ │ P(y|x) =    │ │ P(x) =      │
  │ N(f(x),σ²)  │ │ softmax(Wx)  │ │ Σπₖ N(μₖ,Σₖ)│
  │              │ │              │ │              │
  │ → MSE loss   │ │ → CE loss    │ │ → EM algo    │
  └──────────────┘ └──────────────┘ └──────────────┘

  + Gaussian prior on θ  →  L₂ regularization  →  MAP
  + Laplace prior on θ   →  L₁ regularization  →  MAP
```

---

## 10. Limitations and Pitfalls

| Pitfall | Description | Example |
|---------|-------------|---------|
| **Overfitting** | MLE can fit noise with small $n$ | Bernoulli with $n=1$: $\hat{p} = 0$ or $1$ |
| **Bias** | MLE is not always unbiased | Gaussian $\hat{\sigma}^2$ divides by $n$ not $n-1$ |
| **Boundary estimates** | MLE may land on parameter boundaries | Uniform$[0, \theta]$: $\hat{\theta} = \max(x_i)$ |
| **Multimodality** | Multiple local maxima exist for complex models | GMMs, neural networks |
| **Model misspecification** | MLE converges to "closest" distribution in KL sense, not truth | Wrong distributional assumption |

---

## Exercises

**★ Basic**

1. Flip a coin 10 times and observe 7 heads. What is the MLE of the probability of heads? Write the log-likelihood as a function of $p$ and verify your answer by differentiation.

2. Given observations $x_1 = 3, x_2 = 5, x_3 = 7, x_4 = 9, x_5 = 6$ from a $\mathcal{N}(\mu, \sigma^2)$, compute $\hat{\mu}_{\text{MLE}}$ and $\hat{\sigma}^2_{\text{MLE}}$.

3. Explain why $\arg\max_\theta L(\theta) = \arg\max_\theta \ell(\theta)$. For what kind of function does this argument work?

4. Compute the Fisher information $I(p)$ for a single Bernoulli($p$) observation. What happens as $p \to 0$ or $p \to 1$?

**★★ Intermediate**

5. For the Poisson distribution $P(X = k) = \frac{\lambda^k e^{-\lambda}}{k!}$, derive $\hat{\lambda}_{\text{MLE}} = \bar{x}$ from $n$ iid observations. Verify the second-order condition.

6. Show that the Fisher information for $\mathcal{N}(\mu, \sigma^2)$ with $\sigma^2$ known is $I(\mu) = 1/\sigma^2$. Then verify that the MLE $\hat{\mu} = \bar{x}$ achieves the Cramer-Rao bound exactly (i.e., it is efficient).

7. Prove the invariance property of MLE: if $\hat{\theta}$ maximizes $L(\theta)$ and $g$ is a one-to-one function, then $g(\hat{\theta})$ maximizes $L(g^{-1}(\phi))$ as a function of $\phi = g(\theta)$.

8. Consider a model where $P(x \mid \theta) = \theta e^{-\theta x}$ for $x > 0$ (exponential distribution). Derive the MLE from $n$ iid observations. What is the relationship between $\hat{\theta}_{\text{MLE}}$ and the sample mean?

**★★★ Challenging**

9. **(MLE vs MAP)** For Bernoulli data with $k$ successes in $n$ trials, suppose we place a Beta($\alpha, \beta$) prior on $p$. Show that $\hat{p}_{\text{MAP}} = \frac{k + \alpha - 1}{n + \alpha + \beta - 2}$. Verify that as $\alpha, \beta \to 1$ (uniform prior), MAP reduces to MLE. What happens when $n$ is very large?

10. **(Asymptotic normality)** For $n$ Bernoulli($p$) observations, the MLE is $\hat{p} = k/n$. Verify that $\text{Var}(\hat{p}) = p(1-p)/n = 1/(nI(p))$, confirming the MLE achieves the Cramer-Rao bound exactly for all $n$ (not just asymptotically).

11. **(EM motivation)** For a two-component Gaussian mixture $P(x) = \pi\mathcal{N}(x \mid \mu_1, 1) + (1-\pi)\mathcal{N}(x \mid \mu_2, 1)$, write the log-likelihood for observations $x_1, \ldots, x_n$. Show that setting $\partial\ell/\partial\mu_1 = 0$ leads to an equation that depends on the unknown cluster assignments. Explain why this motivates the EM algorithm.

12. **(Fisher information matrix)** For the multivariate Gaussian $\mathcal{N}(\boldsymbol{\mu}, \Sigma)$ in $\mathbb{R}^d$ with known $\Sigma$, show that the Fisher information matrix for $\boldsymbol{\mu}$ is $I(\boldsymbol{\mu}) = \Sigma^{-1}$. What does this say about estimating the mean in directions of high variance vs low variance?

---

## Related Topics

- [Distributions](distributions.md) -- the distributional families whose parameters MLE estimates
- [Bayesian Inference](bayesian-inference.md) -- MAP estimation generalizes MLE with priors
- [Limit Theorems](limit-theorems.md) -- CLT and LLN underpin MLE's asymptotic properties
- [Expectation & Moments](expectation-and-moments.md) -- bias, variance, and MSE of estimators
- [Optimization](../optimization/index.md) -- gradient methods for maximizing the likelihood
- [Information Theory](../information-theory/index.md) -- KL divergence and its connection to MLE
