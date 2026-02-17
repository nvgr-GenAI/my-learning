# Probability Distributions

Distributions are the vocabulary of machine learning. Every model assumption — "this noise is Gaussian," "these labels are Bernoulli," "these topics follow a Dirichlet" — is a choice of distribution. Understanding their shapes, parameters, and relationships lets you read any ML paper's model section fluently, choose the right likelihood for your data, and recognize when the central limit theorem is quietly making your method work.

---

## Prerequisites

- [Random Variables](random-variables.md) — PMF, PDF, CDF, transformations
- [Conditional Probability](conditional-probability.md) — Bayes' theorem, independence
- [Probability Foundations](probability-foundations.md) — axioms, probability spaces

---

## 1. Discrete Distributions

### Bernoulli Distribution

**Definition 4.4.1 (Bernoulli Distribution).** A random variable $X \sim \text{Bernoulli}(p)$ takes value $1$ with probability $p$ and value $0$ with probability $1-p$, for $p \in [0,1]$:

$$P(X = x) = p^x(1-p)^{1-x}, \quad x \in \{0, 1\}$$

- **Mean:** $\mathbb{E}[X] = p$
- **Variance:** $\text{Var}(X) = p(1-p)$

The variance is maximized at $p = 1/2$ (maximum uncertainty) and zero at $p \in \{0, 1\}$ (certainty).

**Example 4.5.1:** A biased coin lands heads with probability $p = 0.7$. Let $X = 1$ if heads, $X = 0$ if tails. Then $X \sim \text{Bernoulli}(0.7)$.

$$P(X = 1) = 0.7^1 \cdot 0.3^0 = 0.7, \quad P(X = 0) = 0.7^0 \cdot 0.3^1 = 0.3$$

$$\mathbb{E}[X] = 0.7, \quad \text{Var}(X) = 0.7 \times 0.3 = 0.21$$

Compare: a fair coin ($p = 0.5$) has $\text{Var}(X) = 0.25$ (maximum uncertainty), while this biased coin has lower variance because the outcome is more predictable.

*ML connection:* Logistic regression models $P(y=1 \mid \mathbf{x}) = \sigma(\mathbf{w}^T\mathbf{x} + b)$ where $\sigma$ is the sigmoid. Each label $y_i \sim \text{Bernoulli}(\sigma(\mathbf{w}^T\mathbf{x}_i))$. The binary cross-entropy loss is exactly the negative log-likelihood of this Bernoulli model: $\mathcal{L} = -\sum_i [y_i \log \hat{p}_i + (1-y_i)\log(1-\hat{p}_i)]$.

### Binomial Distribution

**Definition 4.4.2 (Binomial Distribution).** If $X_1, \ldots, X_n \stackrel{\text{iid}}{\sim} \text{Bernoulli}(p)$, then $Y = \sum_{i=1}^n X_i \sim \text{Binomial}(n, p)$:

$$P(Y = k) = \binom{n}{k} p^k (1-p)^{n-k}, \quad k = 0, 1, \ldots, n$$

- **Mean:** $\mathbb{E}[Y] = np$
- **Variance:** $\text{Var}(Y) = np(1-p)$

**Example 4.5.2:** A quality inspector tests $n = 5$ items, each independently defective with probability $p = 0.4$. Find $P(Y = 3)$, i.e., exactly 3 defective items.

$$P(Y = 3) = \binom{5}{3}(0.4)^3(0.6)^2 = 10 \times 0.064 \times 0.36 = 0.2304$$

The mean number of defectives is $\mathbb{E}[Y] = 5 \times 0.4 = 2$, and $\text{Var}(Y) = 5 \times 0.4 \times 0.6 = 1.2$.

**Theorem 4.4.1 (Normal Approximation).** For large $n$, $\text{Binomial}(n,p) \approx \mathcal{N}(np, np(1-p))$ when $np \geq 5$ and $n(1-p) \geq 5$. This is a direct consequence of the central limit theorem.

**Example 4.5.3:** A data pipeline processes $n = 100$ records, each succeeding independently with $p = 0.95$. Approximate $P(Y \geq 98)$ using the normal approximation.

$$\mu = np = 95, \quad \sigma = \sqrt{np(1-p)} = \sqrt{4.75} \approx 2.179$$

$$P(Y \geq 98) \approx P\!\left(Z \geq \frac{97.5 - 95}{2.179}\right) = P(Z \geq 1.148) \approx 1 - \Phi(1.15) \approx 0.125$$

(Using continuity correction: 97.5 instead of 98. Check: $np = 95 \geq 5$ and $n(1-p) = 5 \geq 5$, so the approximation is valid.)

*ML connection:* In a batch of $n$ predictions, the number of correct ones follows $\text{Binomial}(n, p_{\text{acc}})$. This gives confidence intervals for accuracy: $\hat{p} \pm z_{\alpha/2}\sqrt{\hat{p}(1-\hat{p})/n}$. With $n = 1000$ test samples and $\hat{p} = 0.95$, the 95% CI is $0.95 \pm 0.014$.

### Categorical Distribution

**Definition 4.4.3 (Categorical Distribution).** A random variable $X \sim \text{Categorical}(\mathbf{p})$ takes value $k \in \{1, \ldots, K\}$ with probability $p_k$, where $\sum_{k=1}^K p_k = 1$:

$$P(X = k) = p_k$$

This generalizes Bernoulli from 2 to $K$ outcomes. The one-hot encoding $\mathbf{e}_k \in \mathbb{R}^K$ represents outcome $k$.

*ML connection:* The softmax output layer produces $\mathbf{p} = \text{softmax}(\mathbf{z})$ where $p_k = e^{z_k}/\sum_j e^{z_j}$. Each label follows $y \sim \text{Categorical}(\mathbf{p})$. The categorical cross-entropy loss $\mathcal{L} = -\sum_k y_k \log p_k$ is the negative log-likelihood. This is the standard loss for multi-class classification (ImageNet, text classification, token prediction in language models).

### Poisson Distribution

**Definition 4.4.4 (Poisson Distribution).** $X \sim \text{Poisson}(\lambda)$ for rate $\lambda > 0$:

$$P(X = k) = \frac{\lambda^k e^{-\lambda}}{k!}, \quad k = 0, 1, 2, \ldots$$

- **Mean:** $\mathbb{E}[X] = \lambda$
- **Variance:** $\text{Var}(X) = \lambda$

The mean equals the variance — this is the defining property. When data has variance much larger than the mean, it is *overdispersed* and a negative binomial is more appropriate.

**Example 4.5.4:** A support inbox receives $\lambda = 3$ emails per hour on average. Find the probability of receiving exactly 2 emails in a given hour.

$$P(X = 2) = \frac{3^2 \, e^{-3}}{2!} = \frac{9 \times 0.04979}{2} = \frac{0.4481}{2} = 0.2240$$

For context: $P(X = 0) = e^{-3} \approx 0.0498$, $P(X = 1) \approx 0.1494$, $P(X = 3) \approx 0.2240$, $P(X = 4) \approx 0.1680$. The mode is at $k = 2$ and $k = 3$ (tied since $\lambda = 3$ is an integer).

**Theorem 4.4.2 (Poisson Limit).** $\text{Binomial}(n, p) \to \text{Poisson}(\lambda)$ as $n \to \infty$, $p \to 0$, with $np = \lambda$ held fixed. The Poisson models rare events in a large population.

*ML connection:* Poisson regression models count data: number of clicks per ad, hospital visits per patient, word occurrences in documents. The model is $\log \mathbb{E}[Y \mid \mathbf{x}] = \mathbf{w}^T\mathbf{x}$, giving $Y \mid \mathbf{x} \sim \text{Poisson}(e^{\mathbf{w}^T\mathbf{x}})$. In NLP, word counts in bag-of-words models are often assumed Poisson.

### Geometric Distribution

**Definition 4.4.5 (Geometric Distribution).** $X \sim \text{Geometric}(p)$ counts the number of independent Bernoulli trials until the first success:

$$P(X = k) = (1-p)^{k-1}p, \quad k = 1, 2, 3, \ldots$$

- **Mean:** $\mathbb{E}[X] = 1/p$
- **Variance:** $\text{Var}(X) = (1-p)/p^2$

The geometric distribution is the only discrete distribution with the *memoryless property*: $P(X > s + t \mid X > s) = P(X > t)$.

**Example 4.5.5:** A recruiter screens resumes where each candidate independently has a $p = 0.3$ chance of being qualified. What is the probability that the first qualified candidate is the 4th one reviewed?

$$P(X = 4) = (1 - 0.3)^{4-1} \times 0.3 = (0.7)^3 \times 0.3 = 0.343 \times 0.3 = 0.1029$$

The expected number of resumes to screen before finding one qualified candidate is $\mathbb{E}[X] = 1/0.3 \approx 3.33$, with variance $(0.7)/(0.3)^2 \approx 7.78$.

*ML connection:* In online learning and bandit algorithms, the number of rounds until an agent first selects the optimal arm follows a geometric-like distribution. The expected regret depends on $1/p$ where $p$ relates to the exploration probability.

```
DISCRETE DISTRIBUTION SHAPES

Bernoulli(0.3)         Binomial(10, 0.3)       Poisson(3)
P(X)                   P(X)                     P(X)
 │                      │                        │
0.7 ■                   │   ■                     │  ■ ■
 │                      │  ■ ■                    │ ■   ■
 │                      │ ■   ■                   │■     ■
0.3 ■                   │■     ■                  ■       ■
 │                      ■       ■■               ■         ■■
 └──────               └──────────────           └──────────────
  0  1                  0 1 2 3 4 5 6 7 8        0 1 2 3 4 5 6 7 8
```

---

## 2. Continuous Distributions

### Uniform Distribution

**Definition 4.4.6 (Continuous Uniform).** $X \sim \text{Uniform}(a, b)$ has constant density on $[a,b]$:

$$f(x) = \frac{1}{b-a}, \quad a \leq x \leq b$$

- **Mean:** $\mathbb{E}[X] = (a+b)/2$
- **Variance:** $\text{Var}(X) = (b-a)^2/12$

**Example 4.5.6:** A random number generator produces values uniformly on $[2, 8]$. Compute $P(3 < X < 5)$, the mean, and the variance.

$$P(3 < X < 5) = \frac{5 - 3}{8 - 2} = \frac{2}{6} = \frac{1}{3} \approx 0.333$$

$$\mathbb{E}[X] = \frac{2 + 8}{2} = 5, \quad \text{Var}(X) = \frac{(8-2)^2}{12} = \frac{36}{12} = 3$$

Note that probability is simply the fraction of the interval covered, since the density is constant at $f(x) = 1/6$.

*ML connection:* Uniform distributions appear in random initialization (Xavier/Glorot init uses $\text{Uniform}(-\sqrt{6/(n_{\text{in}}+n_{\text{out}})}, \sqrt{6/(n_{\text{in}}+n_{\text{out}})})$), in random search for hyperparameter tuning, and as an uninformative prior in Bayesian analysis.

### Gaussian (Normal) Distribution

**Definition 4.4.7 (Gaussian Distribution).** $X \sim \mathcal{N}(\mu, \sigma^2)$ has PDF:

$$f(x) = \frac{1}{\sigma\sqrt{2\pi}} \exp\left(-\frac{(x - \mu)^2}{2\sigma^2}\right)$$

- **Mean:** $\mathbb{E}[X] = \mu$
- **Variance:** $\text{Var}(X) = \sigma^2$

**Example 4.5.7:** IQ scores follow $X \sim \mathcal{N}(100, 15^2)$. What fraction of the population has IQ above 130?

$$z = \frac{130 - 100}{15} = 2.0$$

$$P(X > 130) = P(Z > 2) = 1 - \Phi(2) \approx 1 - 0.9772 = 0.0228$$

About 2.3% of the population. By the 68-95-99.7 rule, 130 is exactly $2\sigma$ above the mean, so roughly $\frac{1 - 0.954}{2} = 2.3\%$ lies above it.

This is the single most important distribution in machine learning. Section 3 below covers it in full detail.

### Exponential Distribution

**Definition 4.4.8 (Exponential Distribution).** $X \sim \text{Exponential}(\lambda)$ for rate $\lambda > 0$:

$$f(x) = \lambda e^{-\lambda x}, \quad x \geq 0$$

- **Mean:** $\mathbb{E}[X] = 1/\lambda$
- **Variance:** $\text{Var}(X) = 1/\lambda^2$

The exponential is the continuous analogue of the geometric: it is the only continuous distribution with the memoryless property $P(X > s + t \mid X > s) = P(X > t)$.

**Example 4.5.8:** A web server receives requests at rate $\lambda = 2$ per second, so the time between consecutive requests follows $X \sim \text{Exponential}(2)$. Find $P(X > 1.5)$, i.e., the probability of waiting more than 1.5 seconds for the next request.

$$P(X > t) = e^{-\lambda t} = e^{-2 \times 1.5} = e^{-3} \approx 0.0498$$

The expected wait time is $\mathbb{E}[X] = 1/2 = 0.5$ seconds. Waiting 1.5 seconds is $3\times$ the mean, and the exponential tail drops off fast: there is only about a 5% chance of such a long gap.

*ML connection:* In survival analysis and reliability modeling, the time until an event (customer churn, machine failure) is often modeled as exponential. The hazard rate is constant at $\lambda$. In Poisson processes, the inter-arrival times between events are $\text{Exponential}(\lambda)$.

### Beta Distribution

**Definition 4.4.9 (Beta Distribution).** $X \sim \text{Beta}(\alpha, \beta)$ for $\alpha, \beta > 0$:

$$f(x) = \frac{x^{\alpha-1}(1-x)^{\beta-1}}{B(\alpha, \beta)}, \quad 0 \leq x \leq 1$$

where $B(\alpha, \beta) = \frac{\Gamma(\alpha)\Gamma(\beta)}{\Gamma(\alpha+\beta)}$ is the beta function.

- **Mean:** $\mathbb{E}[X] = \frac{\alpha}{\alpha + \beta}$
- **Variance:** $\text{Var}(X) = \frac{\alpha\beta}{(\alpha+\beta)^2(\alpha+\beta+1)}$

**Example 4.5.9:** A website's click-through rate is modeled as $X \sim \text{Beta}(3, 7)$. Compute the mean and variance of the click-through rate.

$$\mathbb{E}[X] = \frac{3}{3 + 7} = \frac{3}{10} = 0.30$$

$$\text{Var}(X) = \frac{3 \times 7}{(3+7)^2(3+7+1)} = \frac{21}{100 \times 11} = \frac{21}{1100} \approx 0.0191$$

The mean click-through rate is 30%, with $\text{SD}(X) \approx 0.138$. The "effective sample size" is $\alpha + \beta = 10$, reflecting moderate uncertainty. Increasing to $\text{Beta}(30, 70)$ keeps the same mean but shrinks variance by $10\times$.

```
BETA DISTRIBUTION SHAPES

Beta(1,1) = Uniform    Beta(2,5)              Beta(5,5)
 f(x)                   f(x)                   f(x)
 │                       │                      │
1├─────────────          │■                     │      ■
 │                       │ ■                    │    ■   ■
 │                       │  ■■                  │   ■     ■
 │                       │    ■■■               │  ■       ■
 │                       │       ■■■■           │ ■         ■
 └──────────────         └──────────────        └──────────────
 0              1        0              1       0              1
 (uniform)               (skewed left)          (symmetric, peaked)

α > β: skewed right     α < β: skewed left     α = β: symmetric
Large α+β: concentrated around mean             α = β = 1: uniform
```

**Theorem 4.4.3 (Beta-Bernoulli Conjugacy).** If the prior is $p \sim \text{Beta}(\alpha, \beta)$ and we observe $k$ successes and $n-k$ failures from $\text{Bernoulli}(p)$ trials, then the posterior is:

$$p \mid \text{data} \sim \text{Beta}(\alpha + k, \beta + n - k)$$

*Proof.* By Bayes' theorem: $f(p \mid \text{data}) \propto P(\text{data} \mid p) \cdot f(p) \propto p^k(1-p)^{n-k} \cdot p^{\alpha-1}(1-p)^{\beta-1} = p^{(\alpha+k)-1}(1-p)^{(\beta+n-k)-1}$. This is the kernel of $\text{Beta}(\alpha+k, \beta+n-k)$. $\square$

**Example 4.5.10:** You start with a uniform prior $p \sim \text{Beta}(1, 1)$ on a coin's bias. You flip it $n = 10$ times and observe $k = 7$ heads. What is the posterior distribution and its mean?

$$p \mid \text{data} \sim \text{Beta}(1 + 7,\; 1 + 10 - 7) = \text{Beta}(8, 4)$$

$$\mathbb{E}[p \mid \text{data}] = \frac{8}{8+4} = \frac{8}{12} \approx 0.667$$

Note this is pulled slightly toward $0.5$ compared to the MLE of $\hat{p} = 7/10 = 0.70$. The prior acts as a "phantom" 2 extra trials (1 head, 1 tail), shrinking the estimate toward $0.5$.

*ML connection:* Beta priors are the workhorse of Bayesian A/B testing. Start with $\text{Beta}(1,1)$ (uniform prior on conversion rate), observe clicks and non-clicks, and the posterior $\text{Beta}(1+\text{clicks}, 1+\text{non-clicks})$ gives a full probability distribution over the true conversion rate. Thompson sampling for multi-armed bandits samples from these Beta posteriors to balance exploration and exploitation.

### Gamma Distribution

**Definition 4.4.10 (Gamma Distribution).** $X \sim \text{Gamma}(\alpha, \beta)$ for shape $\alpha > 0$ and rate $\beta > 0$:

$$f(x) = \frac{\beta^\alpha}{\Gamma(\alpha)} x^{\alpha-1} e^{-\beta x}, \quad x > 0$$

- **Mean:** $\mathbb{E}[X] = \alpha/\beta$
- **Variance:** $\text{Var}(X) = \alpha/\beta^2$

Special cases: $\text{Gamma}(1, \lambda) = \text{Exponential}(\lambda)$ and $\text{Gamma}(n/2, 1/2) = \chi^2(n)$.

**Example 4.5.11:** A call center models the total time for $\alpha = 3$ consecutive calls, each with an average duration of $1/\beta = 5$ minutes (rate $\beta = 0.2$). The total time follows $T \sim \text{Gamma}(3, 0.2)$.

$$\mathbb{E}[T] = \frac{\alpha}{\beta} = \frac{3}{0.2} = 15 \text{ minutes}, \quad \text{Var}(T) = \frac{\alpha}{\beta^2} = \frac{3}{0.04} = 75$$

This is the sum of 3 independent $\text{Exponential}(0.2)$ random variables. In general, $\text{Gamma}(\alpha, \beta)$ with integer $\alpha$ equals the sum of $\alpha$ independent $\text{Exponential}(\beta)$ waiting times (also called the Erlang distribution).

*ML connection:* The Gamma distribution serves as a conjugate prior for the precision (inverse variance) $\tau = 1/\sigma^2$ of a Gaussian. In Bayesian neural networks, Gamma priors on precision parameters encode beliefs about how tightly weights should be concentrated around zero — connecting to L2 regularization (weight decay).

### Dirichlet Distribution

**Definition 4.4.11 (Dirichlet Distribution).** $\mathbf{p} = (p_1, \ldots, p_K) \sim \text{Dirichlet}(\alpha_1, \ldots, \alpha_K)$ is a distribution over the $(K-1)$-simplex (probability vectors that sum to 1):

$$f(\mathbf{p}) = \frac{\Gamma(\sum_k \alpha_k)}{\prod_k \Gamma(\alpha_k)} \prod_{k=1}^K p_k^{\alpha_k - 1}, \quad p_k \geq 0, \quad \sum_k p_k = 1$$

- **Mean:** $\mathbb{E}[p_k] = \alpha_k / \alpha_0$ where $\alpha_0 = \sum_k \alpha_k$

The Dirichlet generalizes the Beta to $K$ dimensions: $\text{Beta}(\alpha, \beta) = \text{Dirichlet}(\alpha, \beta)$ for $K = 2$.

**Theorem 4.4.4 (Dirichlet-Categorical Conjugacy).** Prior $\mathbf{p} \sim \text{Dirichlet}(\boldsymbol{\alpha})$ with categorical observations $n_1, \ldots, n_K$ (counts per category) gives posterior:

$$\mathbf{p} \mid \text{data} \sim \text{Dirichlet}(\alpha_1 + n_1, \ldots, \alpha_K + n_K)$$

```
DIRICHLET ON THE 3-SIMPLEX (K = 3)

  Each point is a probability vector (p₁, p₂, p₃) with p₁+p₂+p₃ = 1

  Dir(1,1,1): Uniform       Dir(10,10,10): Concentrated    Dir(0.1,0.1,0.1): Sparse
  over the simplex           near center (1/3, 1/3, 1/3)    near corners

      p₃                        p₃                           p₃
      /\                         /\                            /\
     /  \                       /  \                          /  \
    / ·· \                     /    \                        /·   ·\
   / ···· \                   / ·██· \                      /      \
  / ······ \                 / ·████· \                    /        \
 /··········\               /··········\                  /·········\
 p₁─────────p₂             p₁─────────p₂                p₁─────────p₂

  (all mixtures equally      (prefer balanced             (prefer "pure"
   likely)                    mixtures)                    distributions)
```

*ML connection:* Latent Dirichlet Allocation (LDA) uses Dirichlet priors twice: $\text{Dir}(\boldsymbol{\alpha})$ generates topic distributions per document, and $\text{Dir}(\boldsymbol{\beta})$ generates word distributions per topic. With small $\alpha_k$ (e.g., 0.1), the model prefers documents that focus on a few topics — the sparsity-inducing property of the Dirichlet. This same prior appears in Bayesian mixture models and variational autoencoders with discrete latent variables.

---

## 3. The Gaussian Distribution in Detail

The Gaussian is so central to ML that it deserves extended treatment.

### 3.1 Properties of the Gaussian

**Theorem 4.4.5 (Closure Properties).** If $X \sim \mathcal{N}(\mu, \sigma^2)$, then:

1. **Linear transformation:** $aX + b \sim \mathcal{N}(a\mu + b, a^2\sigma^2)$
2. **Sum of independents:** $X_1 + X_2 \sim \mathcal{N}(\mu_1 + \mu_2, \sigma_1^2 + \sigma_2^2)$
3. **Standardization:** $Z = (X - \mu)/\sigma \sim \mathcal{N}(0, 1)$

**Example 4.5.12:** Suppose package weights are $X_1 \sim \mathcal{N}(10, 4)$ and $X_2 \sim \mathcal{N}(15, 9)$ (in kg), independently. What is the distribution of the total weight $X_1 + X_2$?

$$X_1 + X_2 \sim \mathcal{N}(10 + 15,\; 4 + 9) = \mathcal{N}(25, 13)$$

If we apply a linear transformation for a shipping surcharge of $Y = 2X_1 + 3$:

$$Y \sim \mathcal{N}(2 \times 10 + 3,\; 2^2 \times 4) = \mathcal{N}(23, 16)$$

These follow directly from the closure properties: sums of independent Gaussians are Gaussian, and linear transformations of Gaussians are Gaussian.

### 3.2 The 68-95-99.7 Rule

For $X \sim \mathcal{N}(\mu, \sigma^2)$:

- $P(\mu - \sigma \leq X \leq \mu + \sigma) \approx 0.683$ (68.3%)
- $P(\mu - 2\sigma \leq X \leq \mu + 2\sigma) \approx 0.954$ (95.4%)
- $P(\mu - 3\sigma \leq X \leq \mu + 3\sigma) \approx 0.997$ (99.7%)

```
THE GAUSSIAN (BELL CURVE) WITH STANDARD DEVIATIONS

                          ▄███▄
                        ▄██   ██▄
                       ██       ██
                      ██   68%   ██
                     ██           ██
                    ██    ┌───┐    ██
                   ██     │   │     ██
                  ██      │   │      ██
                ▄██  95%  │   │  95%  ██▄
              ▄██         │   │         ██▄
           ▄███  99.7%    │   │   99.7%  ███▄
      ▄▄████              │   │              ████▄▄
 ▄▄██████                 │   │                 ██████▄▄
 ─────┼──────┼──────┼─────┼───┼─────┼──────┼──────┼─────
    μ-3σ   μ-2σ    μ-σ   μ       μ+σ    μ+2σ   μ+3σ

 |◄────────── 68.3% ──────────►|
 |◄──────────────── 95.4% ────────────────►|
 |◄────────────────────── 99.7% ──────────────────────►|
```

### 3.3 Standard Normal and Z-Scores

**Definition 4.4.12 (Standard Normal).** The standard normal $Z \sim \mathcal{N}(0, 1)$ has PDF $\phi(z) = \frac{1}{\sqrt{2\pi}}e^{-z^2/2}$ and CDF $\Phi(z) = \int_{-\infty}^{z} \phi(t)\,dt$.

**Definition 4.4.13 (Z-Score).** The z-score of observation $x$ from $\mathcal{N}(\mu, \sigma^2)$ is $z = (x - \mu)/\sigma$. It measures how many standard deviations $x$ is from the mean.

Key values: $\Phi(1.96) \approx 0.975$ (used for 95% confidence intervals), $\Phi(2.576) \approx 0.995$ (99% CI).

**Example 4.5.13:** Exam scores follow $X \sim \mathcal{N}(72, 8^2)$. A student scores 85. What is their z-score, and what percentile are they in?

$$z = \frac{85 - 72}{8} = \frac{13}{8} = 1.625$$

$$\text{Percentile} = \Phi(1.625) \approx 0.9479$$

The student is at approximately the 95th percentile — their score is 1.625 standard deviations above the class mean. Equivalently, about 5.2% of students scored higher.

*ML connection:* Batch normalization transforms hidden layer activations to have zero mean and unit variance: $\hat{x}_i = (x_i - \mu_B)/\sqrt{\sigma_B^2 + \epsilon}$. This is z-score normalization applied per-batch. It stabilizes training by preventing internal covariate shift. Layer normalization does the same per-sample.

### 3.4 Why the Gaussian Appears Everywhere

**Theorem 4.4.6 (Central Limit Theorem — informal).** The sum (or average) of many independent random variables with finite variance converges to a Gaussian, regardless of the original distribution.

This explains why the Gaussian appears so frequently:

| Phenomenon | Why Gaussian |
|-----------|-------------|
| Measurement noise | Sum of many small independent errors |
| Heights, weights | Many genetic and environmental factors |
| Financial returns (short-term) | Aggregate of many small trades |
| SGD noise | Average of many per-sample gradients |
| Weight initialization | By design (Kaiming/He uses $\mathcal{N}(0, 2/n_{\text{in}})$) |
| Variational inference | Chosen for tractability (reparameterization trick) |

*ML connection:* The reparameterization trick in VAEs exploits the Gaussian's linear closure property. Instead of sampling $z \sim \mathcal{N}(\mu, \sigma^2)$ (which blocks gradient flow), write $z = \mu + \sigma \cdot \epsilon$ where $\epsilon \sim \mathcal{N}(0,1)$. Gradients now flow through $\mu$ and $\sigma$ while $\epsilon$ is treated as a constant random input.

### 3.5 Maximum Entropy Property

**Theorem 4.4.7.** Among all distributions on $\mathbb{R}$ with given mean $\mu$ and variance $\sigma^2$, the Gaussian $\mathcal{N}(\mu, \sigma^2)$ has the maximum entropy:

$$H[X] = \frac{1}{2}\ln(2\pi e\sigma^2)$$

This means: if you only know the mean and variance of your data, the Gaussian is the *least presumptuous* (most uncertain) distribution you can assume. Using any other distribution implicitly assumes more structure.

---

## 4. The Exponential Family

**Definition 4.4.14 (Exponential Family).** A parametric family of distributions belongs to the *exponential family* if the PDF/PMF can be written as:

$$f(x \mid \boldsymbol{\eta}) = h(x) \exp\left(\boldsymbol{\eta}^T \mathbf{T}(x) - A(\boldsymbol{\eta})\right)$$

where:

- $\boldsymbol{\eta}$ = natural (canonical) parameters
- $\mathbf{T}(x)$ = sufficient statistics
- $A(\boldsymbol{\eta})$ = log-partition function (normalizer)
- $h(x)$ = base measure

**Theorem 4.4.8 (Properties of the Log-Partition Function).**

$$\mathbb{E}[\mathbf{T}(X)] = \nabla_{\boldsymbol{\eta}} A(\boldsymbol{\eta}), \qquad \text{Cov}[\mathbf{T}(X)] = \nabla^2_{\boldsymbol{\eta}} A(\boldsymbol{\eta})$$

Since the Hessian of $A$ is a covariance matrix (hence PSD), $A(\boldsymbol{\eta})$ is convex. This is why maximum likelihood for exponential family models is a convex optimization problem.

| Distribution | Natural Parameter $\boldsymbol{\eta}$ | Sufficient Statistic $\mathbf{T}(x)$ |
|-------------|--------------------------------------|--------------------------------------|
| Bernoulli($p$) | $\log\frac{p}{1-p}$ (log-odds) | $x$ |
| Poisson($\lambda$) | $\log\lambda$ | $x$ |
| Gaussian($\mu, \sigma^2$) | $(\mu/\sigma^2,\; -1/(2\sigma^2))$ | $(x,\; x^2)$ |
| Exponential($\lambda$) | $-\lambda$ | $x$ |
| Gamma($\alpha, \beta$) | $(\alpha - 1,\; -\beta)$ | $(\log x,\; x)$ |

*ML connection:* Generalized linear models (GLMs) are defined by choosing an exponential family distribution and a link function. Logistic regression = Bernoulli + logit link. Poisson regression = Poisson + log link. The exponential family structure guarantees that the log-likelihood is concave in the natural parameters, so gradient descent finds the global optimum. Softmax regression is the categorical exponential family.

---

## 5. Distribution Reference Table

| Distribution | Parameters | Mean | Variance | Primary ML Usage |
|-------------|-----------|------|----------|-----------------|
| Bernoulli($p$) | $p \in [0,1]$ | $p$ | $p(1-p)$ | Binary classification, logistic regression |
| Binomial($n,p$) | $n \in \mathbb{N}, p \in [0,1]$ | $np$ | $np(1-p)$ | Confidence intervals for accuracy |
| Categorical($\mathbf{p}$) | $\mathbf{p} \in \Delta^{K-1}$ | $\mathbf{p}$ | $p_k(1-p_k)$ | Multi-class classification, LLM token prediction |
| Poisson($\lambda$) | $\lambda > 0$ | $\lambda$ | $\lambda$ | Count data, event modeling |
| Geometric($p$) | $p \in (0,1]$ | $1/p$ | $(1-p)/p^2$ | Trials until success, stopping times |
| Uniform($a,b$) | $a < b$ | $(a+b)/2$ | $(b-a)^2/12$ | Random init, hyperparameter search |
| Gaussian($\mu,\sigma^2$) | $\mu \in \mathbb{R}, \sigma^2 > 0$ | $\mu$ | $\sigma^2$ | Noise modeling, VAEs, weight init, everywhere |
| Exponential($\lambda$) | $\lambda > 0$ | $1/\lambda$ | $1/\lambda^2$ | Survival analysis, inter-arrival times |
| Beta($\alpha,\beta$) | $\alpha, \beta > 0$ | $\frac{\alpha}{\alpha+\beta}$ | $\frac{\alpha\beta}{(\alpha+\beta)^2(\alpha+\beta+1)}$ | Bayesian A/B testing, conjugate prior for Bernoulli |
| Gamma($\alpha,\beta$) | $\alpha, \beta > 0$ | $\alpha/\beta$ | $\alpha/\beta^2$ | Prior for precision, waiting times |
| Dirichlet($\boldsymbol{\alpha}$) | $\alpha_k > 0$ | $\alpha_k/\alpha_0$ | (see text) | Topic models (LDA), mixture models |

---

## 6. Relationships Between Distributions

```
DISTRIBUTION FAMILY TREE

                    ┌──────────────┐
                    │  Exponential  │
                    │    Family     │
                    └──────┬───────┘
           ┌───────┬───────┼───────┬──────────┐
           ▼       ▼       ▼       ▼          ▼
       Bernoulli Poisson Gaussian Gamma    Dirichlet
           │       ▲       │       │          │
     n iid │   n→∞,│p→0    │       │    K=2   │
     trials│   np=λ│       │     α=1│          │
           ▼       │       │       ▼          ▼
       Binomial────┘       │   Exponential   Beta
           │               │       │
     n→∞   │               │   sum of n│
           ▼               │       ▼
       Poisson             │    Erlang
                           │   (= Gamma, integer α)
                           │
                    ┌──────┴───────┐
                    │     CLT      │
                    │  Everything  │──▶ Gaussian
                    │  converges   │
                    └──────────────┘

KEY RELATIONSHIPS:
  Bernoulli(p)  ──sum of n──▶  Binomial(n,p)
  Binomial(n,p) ──n→∞, p→0──▶  Poisson(np)
  Binomial(n,p) ──n large────▶  Gaussian(np, np(1-p))      [CLT]
  Exponential(λ) ──sum of α──▶  Gamma(α, λ)
  Beta(α,β)     ──K dims────▶  Dirichlet(α₁,...,αK)
  Bernoulli(p)  ──conjugate──▶  Beta(α,β)
  Categorical(p) ─conjugate──▶  Dirichlet(α)
  Gaussian(μ,σ²) ─conjugate──▶  Gaussian (known σ²)
```

---

## Exercises

**★ Basic**

1. A spam classifier outputs $P(\text{spam}) = 0.8$. Model the label as Bernoulli. What is the variance? What does high variance mean for prediction confidence?

2. In 200 coin flips of a biased coin with $p = 0.6$, use the normal approximation to find $P(\text{heads} \geq 130)$. (Hint: standardize and use $\Phi$.)

3. A server receives 5 requests per minute on average. Using the Poisson distribution, find the probability of receiving exactly 8 requests in a given minute.

4. If $X \sim \mathcal{N}(100, 25)$, what fraction of values fall between 90 and 110? Convert to z-scores and use the 68-95-99.7 rule.

**★★ Intermediate**

5. Show that the Bernoulli distribution belongs to the exponential family. Identify $\eta$, $T(x)$, $A(\eta)$, and $h(x)$. Verify that $A'(\eta)$ gives the mean.

6. You run an A/B test. Variant A: 45 conversions out of 1000. Variant B: 52 out of 1000. Starting from $\text{Beta}(1,1)$ priors, compute the posterior distributions and calculate $P(p_B > p_A)$ conceptually. Why is the Beta-Bernoulli model preferred over a simple proportion comparison?

7. Prove that if $X \sim \text{Exponential}(\lambda)$, then $P(X > s + t \mid X > s) = P(X > t)$ (memoryless property). Why does this make the exponential both useful and unrealistic for modeling equipment failure?

8. A neural network has weight initialization $w \sim \mathcal{N}(0, 2/n)$ (He initialization). For a layer with $n = 512$ inputs, what is the probability that a single weight exceeds $0.1$ in absolute value? What fraction of weights fall within one standard deviation of zero?

**★★★ Challenging**

9. In LDA with $K = 5$ topics, explain why $\text{Dirichlet}(0.1, 0.1, 0.1, 0.1, 0.1)$ encourages documents to focus on few topics while $\text{Dirichlet}(100, 100, 100, 100, 100)$ encourages uniform topic mixtures. What is the expected topic distribution in each case, and how does the variance differ?

10. Prove the maximum entropy property (Theorem 4.4.7): among all distributions with mean $\mu$ and variance $\sigma^2$, the Gaussian maximizes the differential entropy $H[X] = -\int f(x)\log f(x)\,dx$. (Hint: use the method of Lagrange multipliers with constraints $\int f = 1$, $\int xf = \mu$, $\int x^2 f = \mu^2 + \sigma^2$.)

11. Derive the MLE for the Poisson distribution. Given observations $x_1, \ldots, x_n$, show that $\hat{\lambda}_{\text{MLE}} = \bar{x}$. Verify this is a global maximum using the exponential family convexity argument.

12. The softmax temperature trick modifies the categorical distribution as $p_k \propto \exp(z_k / T)$. Analyze: as $T \to 0$, the distribution approaches what? As $T \to \infty$? Relate this to the Boltzmann distribution in statistical mechanics and explain why it is used in language model sampling.

---

## Related Topics

- [Random Variables](random-variables.md) — the mathematical objects that distributions describe
- [Conditional Probability](conditional-probability.md) — Bayes' theorem for deriving posteriors
- [Expectation & Moments](expectation-and-moments.md) — computing means and variances of these distributions
- [Joint Distributions](joint-distributions.md) — multivariate Gaussian and joint discrete distributions
- [Limit Theorems](limit-theorems.md) — CLT explains Gaussian ubiquity, LLN justifies MLE
- [Bayesian Inference](bayesian-inference.md) — conjugate priors (Beta-Bernoulli, Dirichlet-Categorical)
- [Maximum Likelihood](maximum-likelihood.md) — exponential family MLE, Fisher information
