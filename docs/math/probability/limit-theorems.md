# Limit Theorems

Limit theorems are the bridge between a single random experiment and the predictable behavior of aggregates. They explain why averages stabilize (Law of Large Numbers), why the bell curve appears everywhere (Central Limit Theorem), and how tightly a random quantity concentrates around its mean (concentration inequalities). In machine learning, these results underpin SGD convergence, generalization bounds, sample complexity estimates, and the validity of Gaussian assumptions in large-sample regimes.

---

## Prerequisites

- [Expectation & Moments](expectation-and-moments.md) --- $\mathbb{E}[X]$, variance, moment generating functions
- [Distributions](distributions.md) --- Gaussian, Bernoulli, and their properties
- [Conditional Probability](conditional-probability.md) --- independence of random variables

---

## 1. Modes of Convergence

Before stating the major theorems, we need to make precise what it means for a sequence of random variables to "converge."

**Definition 4.7.1 (Convergence in Probability).** A sequence $X_1, X_2, \ldots$ converges *in probability* to $X$ if for every $\epsilon > 0$:

$$\lim_{n \to \infty} P(|X_n - X| > \epsilon) = 0$$

We write $X_n \xrightarrow{P} X$.

**Definition 4.7.2 (Almost Sure Convergence).** $X_n$ converges *almost surely* (a.s.) to $X$ if:

$$P\!\left(\lim_{n \to \infty} X_n = X\right) = 1$$

We write $X_n \xrightarrow{\text{a.s.}} X$.

**Definition 4.7.3 (Convergence in Distribution).** $X_n$ converges *in distribution* to $X$ if for every point $x$ where the CDF $F_X$ is continuous:

$$\lim_{n \to \infty} F_{X_n}(x) = F_X(x)$$

We write $X_n \xrightarrow{d} X$.

**Example 4.7.1 (Convergence in Probability vs. in Distribution).** Let $X \sim \mathcal{N}(0,1)$ and define $Y_n = X + \frac{1}{n}$ and $Z_n$ where each $Z_n \sim \mathcal{N}(0,1)$ independently of $X$.

- $Y_n \xrightarrow{P} X$ because $P(|Y_n - X| > \epsilon) = P(1/n > \epsilon) = 0$ for $n > 1/\epsilon$. The variables literally get close to $X$ on every outcome.
- $Z_n \xrightarrow{d} X$ because $F_{Z_n}(x) = \Phi(x) = F_X(x)$ for all $n$. But $Z_n$ does NOT converge in probability to $X$: since $Z_n - X \sim \mathcal{N}(0, 2)$, we have $P(|Z_n - X| > 1) \approx 0.48$ for every $n$.

The distinction: convergence in distribution only says the *shape* of the CDF matches; convergence in probability says the *actual values* get close.

**Definition 4.7.4 (Convergence in $r$-th Mean).** $X_n$ converges in *$r$-th mean* to $X$ if:

$$\lim_{n \to \infty} \mathbb{E}[|X_n - X|^r] = 0$$

For $r = 2$ this is *mean-square convergence*: $\mathbb{E}[(X_n - X)^2] \to 0$.

**Theorem 4.7.1 (Hierarchy of Convergence).** The modes are related by:

$$\text{a.s.} \implies \text{in probability} \implies \text{in distribution}$$

$$r\text{-th mean} \implies \text{in probability} \implies \text{in distribution}$$

No other implications hold in general.

```
CONVERGENCE HIERARCHY

    Almost sure ──────────┐
    (strongest path-      │
     wise notion)         ▼
                      In probability ──────▶ In distribution
    r-th mean ────────────┘                  (weakest; only
    (moment-based)                            about CDFs)

    Reverse implications FAIL in general.
    Exception: In distribution to a CONSTANT c
               implies convergence in probability to c.
```

*ML connection:* When we say "SGD converges," the precise statement matters. Under standard assumptions (bounded variance, decaying step sizes), SGD converges in mean-square to the optimum. Under stronger conditions, almost sure convergence holds. The mode of convergence determines what guarantees your trained model actually satisfies.

---

## 2. Law of Large Numbers

The LLN says that the sample mean $\bar{X}_n = \frac{1}{n}\sum_{i=1}^n X_i$ approaches the population mean $\mu = \mathbb{E}[X]$ as $n$ grows.

### 2.1 Weak Law of Large Numbers

**Theorem 4.7.2 (Weak LLN).** Let $X_1, X_2, \ldots$ be i.i.d. with $\mathbb{E}[X_i] = \mu$ and $\text{Var}(X_i) = \sigma^2 < \infty$. Then:

$$\bar{X}_n \xrightarrow{P} \mu$$

*Proof (via Chebyshev's inequality).* Since the $X_i$ are independent with common mean and variance:

$$\mathbb{E}[\bar{X}_n] = \mu, \qquad \text{Var}(\bar{X}_n) = \frac{\sigma^2}{n}$$

By Chebyshev's inequality (Theorem 4.7.7 below):

$$P(|\bar{X}_n - \mu| \geq \epsilon) \leq \frac{\text{Var}(\bar{X}_n)}{\epsilon^2} = \frac{\sigma^2}{n\epsilon^2} \xrightarrow{n \to \infty} 0 \quad \square$$

**Example 4.7.2 (Coin Flip Proportions).** Flip a fair coin ($p = 0.5$) and let $\bar{X}_n$ be the proportion of heads. Here $\mu = 0.5$, $\sigma^2 = p(1-p) = 0.25$. Chebyshev gives $P(|\bar{X}_n - 0.5| \geq 0.05) \leq \frac{0.25}{n \cdot 0.0025} = \frac{100}{n}$.

| $n$ | $\text{Var}(\bar{X}_n) = 0.25/n$ | Chebyshev bound for $\epsilon = 0.05$ | Typical observed $\bar{X}_n$ range |
|---|---|---|---|
| 10 | 0.025 | $\leq 10.0$ (trivial) | $0.2$ -- $0.8$ |
| 100 | 0.0025 | $\leq 1.0$ (trivial) | $0.42$ -- $0.58$ |
| 1,000 | 0.00025 | $\leq 0.10$ | $0.47$ -- $0.53$ |
| 10,000 | 0.000025 | $\leq 0.01$ | $0.49$ -- $0.51$ |

For small $n$ the bound is useless (exceeds 1), but the key point is that it goes to 0 --- the sample proportion converges to $0.5$.

### 2.2 Strong Law of Large Numbers

**Theorem 4.7.3 (Strong LLN --- Kolmogorov).** Let $X_1, X_2, \ldots$ be i.i.d. with $\mathbb{E}[|X_i|] < \infty$ and $\mathbb{E}[X_i] = \mu$. Then:

$$\bar{X}_n \xrightarrow{\text{a.s.}} \mu$$

Note the weaker requirement: only a finite first moment is needed (no finite variance required). The proof uses the Borel-Cantelli lemma and fourth-moment truncation arguments; we omit it here.

```
LAW OF LARGE NUMBERS --- SAMPLE MEAN CONVERGENCE

  Sample mean X_n vs. n     (μ = 0, σ² = 1)

  X_n
   1.0 ┤ *
       │  *  *
   0.5 ┤     *  *
       │         * *
   0.0 ┤─ ─ ─ ─ ─ ─*─*──*──*──*──*──*──*──── μ = 0
       │                *
  -0.5 ┤
       │
  -1.0 ┤
       └──────────────────────────────────── n
        1   5   10   20   50  100  200  500

  The sample mean oscillates wildly for small n,
  then settles toward μ as n grows.
  Weak LLN: P(|X_n - μ| > ε) → 0
  Strong LLN: the PATH itself converges (with probability 1)
```

*ML connection:* The LLN is why mini-batch SGD works. A mini-batch gradient $\hat{g} = \frac{1}{B}\sum_{i=1}^B \nabla \ell(x_i; \theta)$ is a sample mean of per-example gradients. By the LLN, as $B$ increases, $\hat{g}$ converges to the true gradient $\nabla \mathcal{L}(\theta) = \mathbb{E}[\nabla \ell(X; \theta)]$. Larger batches give less noisy gradient estimates, but with diminishing returns (variance decreases as $\sigma^2 / B$).

---

## 3. Central Limit Theorem

### 3.1 Statement and Interpretation

**Theorem 4.7.4 (Central Limit Theorem).** Let $X_1, X_2, \ldots$ be i.i.d. with $\mathbb{E}[X_i] = \mu$ and $0 < \text{Var}(X_i) = \sigma^2 < \infty$. Then:

$$\frac{\bar{X}_n - \mu}{\sigma / \sqrt{n}} \xrightarrow{d} \mathcal{N}(0, 1)$$

Equivalently, $\sqrt{n}(\bar{X}_n - \mu) \xrightarrow{d} \mathcal{N}(0, \sigma^2)$.

The CLT says: no matter the shape of the original distribution, the standardized sample mean approaches a Gaussian as $n$ grows.

**Example 4.7.3 (Approximating a Binomial with the Normal).** Let $X \sim \text{Binomial}(n=100, p=0.3)$, so $X = \sum_{i=1}^{100} X_i$ where $X_i \sim \text{Bernoulli}(0.3)$. Then $\mu = np = 30$ and $\sigma = \sqrt{np(1-p)} = \sqrt{21} \approx 4.58$. By the CLT, $X \approx \mathcal{N}(30, 21)$. Approximate $P(X \geq 35)$:

$$P(X \geq 35) = P\!\left(\frac{X - 30}{4.58} \geq \frac{35 - 30}{4.58}\right) \approx P(Z \geq 1.09) = 1 - \Phi(1.09) \approx 0.138$$

The exact binomial value is $P(X \geq 35) \approx 0.132$. The normal approximation is already good at $n = 100$.

**Example 4.7.4 (CLT for Dice Averages).** Roll a fair die $n = 50$ times. Each roll has $\mu = 3.5$ and $\sigma^2 = 35/12 \approx 2.917$. What is $P(\bar{X}_{50} > 3.8)$?

$$P(\bar{X}_{50} > 3.8) = P\!\left(\frac{\bar{X}_{50} - 3.5}{\sqrt{2.917/50}} > \frac{3.8 - 3.5}{0.2415}\right) \approx P(Z > 1.24) = 1 - \Phi(1.24) \approx 0.107$$

So there is roughly an 11% chance the sample mean exceeds 3.8 --- the CLT lets us compute this without knowing the exact distribution of the sum of 50 discrete uniform variables.

### 3.2 Why the Gaussian Appears Everywhere

The Gaussian is not "chosen" --- it is *forced* by mathematics. Any quantity that arises as the sum of many small, independent contributions will be approximately Gaussian. This explains:

- Measurement noise (sum of many tiny independent error sources)
- Heights in a population (sum of many genetic and environmental factors)
- Financial returns over moderate time scales
- Thermal fluctuations in physics

The key insight: the CLT is a statement about the *sum's distribution*, not about the individual summands. The individual $X_i$ can follow any distribution with finite variance.

### 3.3 Berry-Esseen Bound

The CLT tells us *that* convergence happens but not *how fast*. The Berry-Esseen theorem quantifies the rate.

**Theorem 4.7.5 (Berry-Esseen).** Under the CLT conditions, if additionally $\mathbb{E}[|X_i|^3] = \rho < \infty$, then:

$$\sup_x \left| P\!\left(\frac{\bar{X}_n - \mu}{\sigma/\sqrt{n}} \leq x\right) - \Phi(x) \right| \leq \frac{C \rho}{\sigma^3 \sqrt{n}}$$

where $\Phi$ is the standard normal CDF and $C \leq 0.4748$ (best known constant).

The convergence rate is $O(1/\sqrt{n})$: to halve the approximation error, quadruple the sample size.

**Example 4.7.5 (Berry-Esseen for Coin Flips).** Let $X_i \sim \text{Bernoulli}(0.5)$. Then $\mu = 0.5$, $\sigma^2 = 0.25$, $\sigma = 0.5$, and $\rho = \mathbb{E}[|X_i - 0.5|^3] = 0.5 \cdot 0.5^3 + 0.5 \cdot 0.5^3 = 0.125$. The Berry-Esseen bound gives:

$$\sup_x \left| P\!\left(\frac{\bar{X}_n - 0.5}{0.5/\sqrt{n}} \leq x\right) - \Phi(x) \right| \leq \frac{0.4748 \times 0.125}{(0.5)^3 \sqrt{n}} = \frac{0.4748}{\sqrt{n}}$$

| $n$ | Berry-Esseen bound |
|---|---|
| 25 | $\leq 0.095$ |
| 100 | $\leq 0.047$ |
| 400 | $\leq 0.024$ |
| 10,000 | $\leq 0.0047$ |

To guarantee the CDF error is below 0.01, we need $n \geq (0.4748/0.01)^2 \approx 2{,}254$.

```
CENTRAL LIMIT THEOREM --- HISTOGRAM APPROACHING BELL CURVE

  n = 1 (Uniform[0,1])    n = 5 (sum of 5)       n = 30 (sum of 30)
  ┌──────────────┐         ┌──────────────┐        ┌──────────────┐
  │              │         │     ██       │        │      █       │
  │              │         │    ████      │        │     ███      │
  │██████████████│         │   ██████     │        │    █████     │
  │              │         │  ████████    │        │   ███████    │
  │              │         │ ██████████   │        │  █████████   │
  └──────────────┘         └──────────────┘        └──────────────┘
   Flat (uniform)          Roughly bell-shaped     Nearly Gaussian

  The sum of n i.i.d. variables, regardless of original distribution,
  approaches a Gaussian shape. Rate: O(1/√n) by Berry-Esseen.
```

*ML connection:* The CLT justifies using Gaussian-based confidence intervals and hypothesis tests when evaluating model performance. If you compute accuracy over $n$ test examples, the average accuracy $\hat{p}$ satisfies $\sqrt{n}(\hat{p} - p)/\sqrt{p(1-p)} \xrightarrow{d} \mathcal{N}(0,1)$, giving the standard confidence interval $\hat{p} \pm z_{\alpha/2}\sqrt{\hat{p}(1-\hat{p})/n}$. The CLT also underlies the Gaussian assumptions in Bayesian deep learning: with enough data, the posterior over parameters is approximately Gaussian (Bernstein-von Mises theorem).

---

## 4. Concentration Inequalities

Concentration inequalities give *finite-sample* tail bounds --- they say how unlikely it is that a random variable deviates far from its expectation. Unlike the LLN (asymptotic), these give usable bounds for a *fixed* sample size $n$.

### 4.1 Markov's Inequality

**Theorem 4.7.6 (Markov's Inequality).** For a non-negative random variable $X$ and $a > 0$:

$$P(X \geq a) \leq \frac{\mathbb{E}[X]}{a}$$

*Proof.* $\mathbb{E}[X] = \mathbb{E}[X \cdot \mathbf{1}_{X \geq a}] + \mathbb{E}[X \cdot \mathbf{1}_{X < a}] \geq \mathbb{E}[X \cdot \mathbf{1}_{X \geq a}] \geq a \cdot P(X \geq a)$. $\square$

Markov is the weakest bound but requires the least: only $\mathbb{E}[X] < \infty$ and $X \geq 0$. It is the foundation for all stronger concentration inequalities.

**Example 4.7.6 (Markov for a Die Roll).** Let $X$ be the outcome of rolling a fair die, so $\mathbb{E}[X] = 3.5$ and $X \geq 0$. Bound $P(X \geq 5)$:

$$P(X \geq 5) \leq \frac{\mathbb{E}[X]}{5} = \frac{3.5}{5} = 0.70$$

The exact value is $P(X \geq 5) = P(X=5) + P(X=6) = 2/6 \approx 0.333$. Markov's bound of 0.70 is loose but valid. For $P(X \geq 2)$: Markov gives $\leq 3.5/2 = 1.75$ (trivial --- exceeds 1), while the exact probability is $5/6 \approx 0.833$. Markov is most useful when $a$ is much larger than $\mathbb{E}[X]$.

### 4.2 Chebyshev's Inequality

**Theorem 4.7.7 (Chebyshev's Inequality).** For any random variable $X$ with $\mathbb{E}[X] = \mu$ and $\text{Var}(X) = \sigma^2 < \infty$:

$$P(|X - \mu| \geq k\sigma) \leq \frac{1}{k^2}$$

Equivalently, $P(|X - \mu| \geq \epsilon) \leq \sigma^2 / \epsilon^2$.

*Proof.* Apply Markov to the non-negative variable $(X - \mu)^2$:

$$P(|X - \mu| \geq \epsilon) = P((X - \mu)^2 \geq \epsilon^2) \leq \frac{\mathbb{E}[(X - \mu)^2]}{\epsilon^2} = \frac{\sigma^2}{\epsilon^2} \quad \square$$

**Example 4.7.7 (Chebyshev for Specific $k$).** Suppose exam scores have $\mu = 70$ and $\sigma = 10$. Chebyshev bounds the probability of being far from the mean:

| Deviation | Condition | Chebyshev bound $\leq 1/k^2$ |
|---|---|---|
| $k=2$ ($\pm 20$ points) | $P(\|X - 70\| \geq 20)$ | $\leq 1/4 = 0.25$ |
| $k=3$ ($\pm 30$ points) | $P(\|X - 70\| \geq 30)$ | $\leq 1/9 \approx 0.111$ |
| $k=4$ ($\pm 40$ points) | $P(\|X - 70\| \geq 40)$ | $\leq 1/16 = 0.0625$ |

So at least $1 - 1/k^2 = 75\%$ of students score between 50 and 90 (within $2\sigma$), and at least $89\%$ score between 40 and 100 (within $3\sigma$) --- regardless of the score distribution's shape.

### 4.3 Hoeffding's Inequality

**Theorem 4.7.8 (Hoeffding's Inequality).** Let $X_1, \ldots, X_n$ be independent with $a_i \leq X_i \leq b_i$ almost surely. Then for $t > 0$:

$$P\!\left(\bar{X}_n - \mathbb{E}[\bar{X}_n] \geq t\right) \leq \exp\!\left(-\frac{2n^2t^2}{\sum_{i=1}^n (b_i - a_i)^2}\right)$$

For i.i.d. variables bounded in $[a, b]$:

$$P(|\bar{X}_n - \mu| \geq t) \leq 2\exp\!\left(-\frac{2nt^2}{(b-a)^2}\right)$$

The bound decays *exponentially* in $n$ --- vastly stronger than Chebyshev's $1/n$ decay for the sample mean.

**Example 4.7.8 (Hoeffding for Coin Flips).** Flip a fair coin $n = 200$ times. Each $X_i \in [0, 1]$, $\mu = 0.5$. What is the probability the sample proportion deviates from 0.5 by more than $t = 0.05$?

$$P(|\bar{X}_{200} - 0.5| \geq 0.05) \leq 2\exp\!\left(-\frac{2 \cdot 200 \cdot 0.05^2}{(1-0)^2}\right) = 2\exp(-1.0) = 2 \cdot 0.368 \approx 0.736$$

Compare with Chebyshev: $P(|\bar{X}_{200} - 0.5| \geq 0.05) \leq \frac{0.25}{200 \cdot 0.0025} = 0.50$. Here Chebyshev is actually tighter for small $n$. But for $n = 2{,}000$:

- Hoeffding: $2\exp(-10) \approx 0.0000907$
- Chebyshev: $0.25/(2000 \cdot 0.0025) = 0.05$

The exponential decay of Hoeffding dominates for larger $n$.

*Proof sketch.* The proof uses Hoeffding's lemma: if $a \leq X \leq b$ and $\mathbb{E}[X] = 0$, then $\mathbb{E}[e^{sX}] \leq \exp(s^2(b-a)^2/8)$. Applying the Chernoff method (exponential Markov) to $\bar{X}_n - \mu$ and optimizing over $s$ yields the result. $\square$

### 4.4 Chernoff Bound

**Theorem 4.7.9 (Chernoff Bound --- General Method).** For any random variable $X$ and $t > 0$:

$$P(X \geq a) = P(e^{tX} \geq e^{ta}) \leq \frac{\mathbb{E}[e^{tX}]}{e^{ta}} \quad \text{for all } t > 0$$

The tightest bound comes from optimizing over $t$: $P(X \geq a) \leq \inf_{t > 0} e^{-ta} M_X(t)$, where $M_X(t) = \mathbb{E}[e^{tX}]$ is the moment generating function.

**Theorem 4.7.10 (Chernoff Bound --- Bernoulli Sum).** Let $X = \sum_{i=1}^n X_i$ where $X_i \sim \text{Bernoulli}(p_i)$ independently, and $\mu = \mathbb{E}[X] = \sum p_i$. Then for $\delta > 0$:

$$P(X \geq (1+\delta)\mu) \leq \left(\frac{e^\delta}{(1+\delta)^{(1+\delta)}}\right)^\mu$$

For small $\delta$, this simplifies to $P(X \geq (1+\delta)\mu) \leq \exp(-\mu\delta^2/3)$.

**Example 4.7.9 (Chernoff for Ad Clicks).** A website gets $n = 1{,}000$ visitors, each clicking an ad independently with $p = 0.04$. Let $X$ = total clicks, so $\mu = np = 40$. Bound $P(X \geq 60)$, i.e., 50% more clicks than expected ($\delta = 0.5$):

$$P(X \geq 60) = P(X \geq (1 + 0.5) \cdot 40) \leq \exp\!\left(-\frac{40 \cdot 0.5^2}{3}\right) = \exp(-10/3) = e^{-3.33} \approx 0.036$$

Using the tighter form: $P(X \geq 60) \leq \left(\frac{e^{0.5}}{1.5^{1.5}}\right)^{40} = \left(\frac{1.649}{1.837}\right)^{40} = (0.8977)^{40} \approx 0.014$.

The CLT approximation gives $P(X \geq 60) \approx P(Z \geq (60 - 40)/\sqrt{38.4}) \approx P(Z \geq 3.23) \approx 0.0006$, which is much smaller --- showing Chernoff is conservative but valid without requiring normality.

### 4.5 Comparison of Tail Bounds

| Inequality | Tail Decay | Requirements | Tightness |
|------------|-----------|--------------|-----------|
| Markov | $O(1/a)$ | $X \geq 0$, finite $\mathbb{E}[X]$ | Very loose |
| Chebyshev | $O(1/\epsilon^2)$ | Finite variance | Loose |
| Hoeffding | $\exp(-Cn)$ | Independent, bounded | Moderate |
| Chernoff | $\exp(-Cn)$ | Independent, MGF exists | Tight (optimized) |

```
TAIL BOUND COMPARISON

  P(|X_n - μ| ≥ ε)
    │
  1 ┤ ▓▓▓
    │ ▓▓▓▓
    │ ▓▓▓▓▓▓
    │ ▓▓▓▓▓▓▓▓
    │ ▓▓▓▓▓▓▓▓▓▓▓          ─── Chebyshev: σ²/(nε²)
    │ ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓    ─── Hoeffding: 2·exp(-2nε²)
    │ ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
  0 ┤─────────────────────────── n
    1     10     50    100   200

  Chebyshev (polynomial): useful for quick arguments
  Hoeffding (exponential): useful for finite-sample bounds
  For the SAME ε, Hoeffding gives a much smaller probability
  of deviation once n is moderately large.
```

---

## 5. Applications in Machine Learning

### 5.1 SGD Convergence and Mini-Batch Gradients

Stochastic gradient descent replaces the true gradient with a noisy estimate:

$$\theta_{t+1} = \theta_t - \eta_t \hat{g}_t, \qquad \hat{g}_t = \frac{1}{B}\sum_{i=1}^B \nabla \ell(x_i; \theta_t)$$

**LLN connection:** By the LLN, $\hat{g}_t \xrightarrow{P} \nabla \mathcal{L}(\theta_t)$ as $B \to \infty$. The mini-batch gradient is an unbiased estimator: $\mathbb{E}[\hat{g}_t] = \nabla \mathcal{L}(\theta_t)$.

**CLT connection:** The estimation error is approximately Gaussian for large $B$:

$$\hat{g}_t - \nabla\mathcal{L}(\theta_t) \approx \mathcal{N}\!\left(0, \frac{\Sigma_g}{B}\right)$$

where $\Sigma_g$ is the gradient covariance. This justifies analyses that model SGD noise as Gaussian.

**Variance reduction:** $\text{Var}(\hat{g}_t) = \sigma_g^2 / B$. Doubling the batch size halves the variance but doubles compute per step --- explaining the practical trade-off between batch size and convergence speed.

### 5.2 Generalization Bounds via Hoeffding

Hoeffding's inequality directly yields PAC (Probably Approximately Correct) learning bounds.

**Setup:** A hypothesis $h$ has true error $\text{err}(h) = P(h(X) \neq Y)$ and empirical error $\hat{\text{err}}(h) = \frac{1}{n}\sum_{i=1}^n \mathbf{1}[h(x_i) \neq y_i]$. Each indicator is bounded in $[0, 1]$.

By Hoeffding:

$$P(|\hat{\text{err}}(h) - \text{err}(h)| \geq \epsilon) \leq 2\exp(-2n\epsilon^2)$$

For a finite hypothesis class $|\mathcal{H}|$, union bounding over all hypotheses:

$$P\!\left(\exists h \in \mathcal{H}: |\hat{\text{err}}(h) - \text{err}(h)| \geq \epsilon\right) \leq 2|\mathcal{H}|\exp(-2n\epsilon^2)$$

Setting the right side to $\delta$ and solving for $\epsilon$:

$$\text{err}(h) \leq \hat{\text{err}}(h) + \sqrt{\frac{\ln(2|\mathcal{H}|/\delta)}{2n}}$$

with probability at least $1 - \delta$. This is a foundational result in learning theory.

**Example 4.7.10 (PAC Bound in Practice).** A classifier selected from $|\mathcal{H}| = 200$ hypotheses achieves $\hat{\text{err}}(h) = 0.05$ on $n = 2{,}000$ test examples. With confidence $1 - \delta = 0.95$ (so $\delta = 0.05$):

$$\text{err}(h) \leq 0.05 + \sqrt{\frac{\ln(2 \cdot 200 / 0.05)}{2 \cdot 2000}} = 0.05 + \sqrt{\frac{\ln(8000)}{4000}} = 0.05 + \sqrt{\frac{8.987}{4000}} \approx 0.05 + 0.047 = 0.097$$

So with 95% confidence, the true error is at most 9.7%. Doubling the test set to $n = 8{,}000$ would tighten this to $0.05 + 0.024 = 0.074$ --- the bound improves as $1/\sqrt{n}$.

### 5.3 Sample Complexity

Inverting the PAC bound answers: *how many samples do we need?*

To ensure that $|\hat{\text{err}}(h) - \text{err}(h)| \leq \epsilon$ for all $h \in \mathcal{H}$ with probability $\geq 1 - \delta$:

$$n \geq \frac{\ln(2|\mathcal{H}|/\delta)}{2\epsilon^2}$$

| Desired accuracy $\epsilon$ | Desired confidence $1-\delta$ | Hypothesis class $|\mathcal{H}|$ | Required $n$ |
|----|---|---|---|
| 0.05 | 95% | 100 | 3,684 |
| 0.01 | 99% | 1,000 | 88,029 |
| 0.05 | 99% | $10^6$ | 6,365 |

Key insight: $n$ scales as $1/\epsilon^2$ (quadratic in desired precision) and logarithmically in $|\mathcal{H}|$ and $1/\delta$. Doubling precision requires four times the data.

### 5.4 Confidence Intervals for Model Evaluation

When reporting test accuracy $\hat{p}$ from $n$ examples, the CLT gives:

$$\hat{p} \pm z_{\alpha/2}\sqrt{\frac{\hat{p}(1-\hat{p})}{n}}$$

| Test set size $n$ | 95% CI half-width (at $\hat{p} = 0.90$) |
|---|---|
| 100 | $\pm 0.059$ |
| 1,000 | $\pm 0.019$ |
| 10,000 | $\pm 0.006$ |

A model reporting "92% accuracy" on 100 test examples has a 95% CI of roughly $[0.86, 0.98]$ --- too wide to distinguish it from a model with 88% or 96% accuracy. This is why large test sets matter.

### 5.5 CLT Justifies Gaussian Assumptions

Many ML methods assume Gaussian distributions. The CLT explains when this is valid:

- **Batch normalization** standardizes layer activations. Each activation is a sum of many weighted inputs, so by the CLT it is approximately Gaussian --- making the standardization natural.
- **Gaussian Processes** model function values as jointly Gaussian. In wide neural networks, the Central Limit Theorem implies that pre-activation values converge to a Gaussian process as width approaches infinity (the *neural network--GP correspondence*).
- **Bayesian posteriors** become approximately Gaussian around the MAP estimate for large datasets (Bernstein--von Mises theorem), which justifies Laplace approximation.

---

## 6. Summary of Key Results

| Result | Statement | Rate / Strength |
|--------|-----------|-----------------|
| Weak LLN | $\bar{X}_n \xrightarrow{P} \mu$ | Requires finite variance |
| Strong LLN | $\bar{X}_n \xrightarrow{\text{a.s.}} \mu$ | Requires only finite mean |
| CLT | $\sqrt{n}(\bar{X}_n - \mu)/\sigma \xrightarrow{d} \mathcal{N}(0,1)$ | $O(1/\sqrt{n})$ by Berry-Esseen |
| Markov | $P(X \geq a) \leq \mathbb{E}[X]/a$ | Polynomial, very general |
| Chebyshev | $P(\|X - \mu\| \geq \epsilon) \leq \sigma^2/\epsilon^2$ | Polynomial, needs variance |
| Hoeffding | $P(\|\bar{X}_n - \mu\| \geq t) \leq 2e^{-2nt^2/(b-a)^2}$ | Exponential, needs bounded |
| Chernoff | $P(X \geq a) \leq \inf_t e^{-ta}M_X(t)$ | Exponential, needs MGF |

---

## Exercises

**★ Basic**

1. Let $X_1, \ldots, X_n$ be i.i.d. $\text{Bernoulli}(p)$. Compute $\mathbb{E}[\bar{X}_n]$ and $\text{Var}(\bar{X}_n)$. Verify that Chebyshev gives $P(|\bar{X}_n - p| \geq \epsilon) \leq p(1-p)/(n\epsilon^2)$.

2. A fair die is rolled $n = 1000$ times. Use the Weak LLN to explain why the sample mean of the rolls is close to $3.5$. Use Chebyshev to bound $P(|\bar{X}_{1000} - 3.5| \geq 0.1)$.

3. You flip a fair coin 100 times. Using the CLT, approximate $P(\text{number of heads} \geq 60)$. Compare with the exact binomial probability.

4. Apply Markov's inequality to bound $P(X \geq 10)$ when $X \sim \text{Exponential}(\lambda = 1)$. Compare with the exact value $e^{-10}$.

**★★ Intermediate**

5. **(Berry-Esseen in practice)** Let $X_i \sim \text{Uniform}[0, 1]$. Compute $\mu$, $\sigma^2$, and $\rho = \mathbb{E}[|X - \mu|^3]$. How large must $n$ be for the Berry-Esseen bound to guarantee the CDF of $\sqrt{n}(\bar{X}_n - \mu)/\sigma$ is within $0.01$ of $\Phi$?

6. **(PAC bound)** A learning algorithm selects from $|\mathcal{H}| = 500$ hypotheses. You want the empirical error to be within $\epsilon = 0.05$ of the true error with probability $\geq 0.95$. How many training examples are needed?

7. **(Chernoff for Bernoulli sums)** A website has $n = 10{,}000$ visitors per day, each clicking an ad with probability $p = 0.02$. Use the Chernoff bound to bound $P(\text{clicks} \geq 250)$. Compare with the Gaussian approximation from the CLT.

8. Prove that convergence in probability does NOT imply almost sure convergence by constructing a counterexample. (Hint: consider random variables on $[0, 1]$ with $P(X_n \neq 0) = 1/n$ but where infinitely many $X_n$ are nonzero with probability 1.)

**★★★ Challenging**

9. **(Generalization gap)** A neural network achieves 98% training accuracy on $n = 5{,}000$ examples. Using Hoeffding's inequality, give an upper bound on the true error with confidence $1 - \delta = 0.99$. Discuss why this bound is pessimistic for neural networks and what tools (Rademacher complexity, PAC-Bayes) give tighter bounds.

10. **(SGD variance analysis)** Consider SGD with batch size $B$ on a loss with per-example gradient variance $\sigma_g^2$. Show that the variance of the mini-batch gradient is $\sigma_g^2 / B$. If doubling $B$ from 64 to 128 halves the gradient variance, but each step now costs twice as much, argue using the CLT that the "statistical efficiency" (progress per unit compute) is roughly constant. When might larger batches still be preferable?

11. **(Proof of CLT via characteristic functions)** Outline the proof of the CLT using the fact that if characteristic functions $\phi_{Z_n}(t) \to \phi_Z(t)$ pointwise, then $Z_n \xrightarrow{d} Z$. Show that for $Z_n = \sqrt{n}(\bar{X}_n - \mu)/\sigma$, we have $\phi_{Z_n}(t) \to e^{-t^2/2}$, the characteristic function of $\mathcal{N}(0,1)$. (Use the Taylor expansion $\phi_X(t) = 1 + it\mu - t^2(\mu^2 + \sigma^2)/2 + o(t^2)$ and the limit $(1 + a/n)^n \to e^a$.)

12. **(McDiarmid's inequality)** State McDiarmid's bounded differences inequality and show that Hoeffding's inequality is a special case. Apply McDiarmid to prove a generalization bound for $k$-nearest neighbors, where changing one training example can change the classifier's prediction on at most a bounded number of test points.

---

## Related Topics

- [Expectation & Moments](expectation-and-moments.md) --- variance and MGFs used throughout
- [Distributions](distributions.md) --- the Gaussian distribution that the CLT converges to
- [Bayesian Inference](bayesian-inference.md) --- posterior concentration uses Bernstein-von Mises (CLT for posteriors)
- [Maximum Likelihood](maximum-likelihood.md) --- asymptotic normality of the MLE is a CLT result
- [Stochastic Processes](stochastic-processes.md) --- martingale convergence theorems extend these ideas
- [Optimization](../optimization/index.md) --- SGD convergence proofs rely on concentration inequalities
