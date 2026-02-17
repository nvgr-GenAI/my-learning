# Stochastic Processes

A stochastic process is a collection of random variables indexed by time (or space), providing the mathematical framework for modeling systems that evolve with inherent randomness. Where a single random variable captures uncertainty at one moment, a stochastic process captures how uncertainty unfolds over time. Markov chains model state transitions in MCMC sampling and PageRank. Brownian motion underpins diffusion models that generate images from noise. Stochastic gradient descent is itself a stochastic process navigating the loss landscape. Understanding stochastic processes connects static probability theory to the dynamic algorithms at the heart of modern ML.

---

## Prerequisites

- [Conditional Probability](conditional-probability.md) -- Bayes' theorem, conditional expectation
- [Limit Theorems](limit-theorems.md) -- law of large numbers, convergence concepts
- [Linear Algebra - Matrices](../linear-algebra/matrices.md) -- matrix multiplication, eigenvalues, stochastic matrices

---

## 1. Definitions and Basic Framework

**Definition 4.10.1 (Stochastic Process).** A *stochastic process* is a family of random variables $\{X_t\}_{t \in T}$ defined on a probability space $(\Omega, \mathcal{F}, P)$, where $T$ is an *index set* (interpreted as time or space) and each $X_t$ takes values in a *state space* $S$.

- **Discrete-time:** $T = \{0, 1, 2, \ldots\}$, written $X_0, X_1, X_2, \ldots$
- **Continuous-time:** $T = [0, \infty)$, written $X(t)$ or $X_t$
- **Discrete state space:** $S$ is finite or countable (e.g., $S = \{s_1, s_2, \ldots, s_k\}$)
- **Continuous state space:** $S = \mathbb{R}$ or $\mathbb{R}^d$

```
TAXONOMY OF STOCHASTIC PROCESSES

                        State Space
                   Discrete        Continuous
              ┌──────────────┬──────────────────┐
   Discrete   │ Markov chain │ Random walk on R  │
   Time       │ (e.g., HMM)  │ (e.g., SGD path) │
Index         ├──────────────┼──────────────────┤
 Set   Cont.  │ Poisson      │ Brownian motion   │
   Time       │ process      │ (Wiener process)  │
              └──────────────┴──────────────────┘
```

**Definition 4.10.2 (Sample Path / Realization).** For a fixed outcome $\omega \in \Omega$, the function $t \mapsto X_t(\omega)$ is called a *sample path* or *realization* of the process. A stochastic process can be viewed as a random function: each outcome $\omega$ selects an entire trajectory.

*ML connection:* A single training run of SGD produces one sample path $\theta_0, \theta_1, \theta_2, \ldots$ through parameter space. Different random seeds yield different realizations of the same stochastic process. Analyzing SGD requires understanding the *distribution over paths*, not just individual runs.

---

## 2. Discrete-Time Markov Chains

### 2.1 The Markov Property

**Definition 4.10.3 (Markov Chain).** A discrete-time stochastic process $\{X_n\}_{n \geq 0}$ with countable state space $S$ is a *Markov chain* if it satisfies the **Markov property** (memorylessness):

$$P(X_{n+1} = j \mid X_n = i, X_{n-1} = i_{n-1}, \ldots, X_0 = i_0) = P(X_{n+1} = j \mid X_n = i)$$

for all $n \geq 0$ and all states $i, j, i_0, \ldots, i_{n-1} \in S$.

The future depends on the present, but not the past. All relevant history is encoded in the current state.

**Example 4.8.1 (Markov Property — concrete illustration).**
Consider a weather chain with states $\{S, R\}$ (Sunny, Rainy) and the observed sequence $X_0 = S,\; X_1 = S,\; X_2 = R,\; X_3 = \;?$

The Markov property says:

$$P(X_3 = S \mid X_2 = R,\, X_1 = S,\, X_0 = S) = P(X_3 = S \mid X_2 = R)$$

Only the current state $X_2 = R$ matters for predicting $X_3$. The earlier sunny days $X_0, X_1$ contribute no additional information.

### 2.2 Transition Matrix

**Definition 4.10.4 (Transition Matrix).** For a *time-homogeneous* Markov chain (transition probabilities do not depend on $n$), define the *transition matrix* $P \in \mathbb{R}^{|S| \times |S|}$ by:

$$P_{ij} = P(X_{n+1} = j \mid X_n = i)$$

The matrix $P$ is *row-stochastic*: entries are non-negative and each row sums to 1.

**Theorem 4.10.1 (Chapman-Kolmogorov).** The $n$-step transition probabilities are given by matrix powers:

$$P(X_{m+n} = j \mid X_m = i) = (P^n)_{ij}$$

*Proof.* By induction. For $n = 1$, this is the definition. For the inductive step, condition on the intermediate state:

$$P(X_{m+n+1} = j \mid X_m = i) = \sum_{k \in S} P(X_{m+n+1} = j \mid X_{m+n} = k) \cdot P(X_{m+n} = k \mid X_m = i) = \sum_k P_{kj}(P^n)_{ik} = (P^{n+1})_{ij}$$

$\square$

```
MARKOV CHAIN STATE DIAGRAM

  States: {Sunny, Cloudy, Rainy}

           0.6                 0.3
    ┌──────────┐       ┌──────────┐
    │          ▼       │          ▼
    │   ┌─────────┐    │   ┌─────────┐
    └───│  Sunny  │────┘   │ Cloudy  │
        └─────────┘  0.3   └─────────┘
             │ 0.1              │ 0.4
             ▼                  ▼
        ┌─────────┐       ┌─────────┐
        │  Rainy  │──────▶│ Cloudy  │
        └─────────┘  0.5  └─────────┘
             │ 0.3              ▲
             └──────────────────┘
                    0.2

  Transition matrix P:
              Sunny  Cloudy  Rainy
    Sunny   [  0.6    0.3    0.1  ]
    Cloudy  [  0.3    0.4    0.3  ]
    Rainy   [  0.2    0.5    0.3  ]

  Row sums = 1 (row-stochastic)
```

**Example 4.8.2 (Transition matrix — two-step probability via Chapman-Kolmogorov).**
Consider a 2-state chain $\{A, B\}$ with transition matrix:

$$P = \begin{pmatrix} 0.8 & 0.2 \\ 0.5 & 0.5 \end{pmatrix}$$

Compute $P(X_2 = B \mid X_0 = A)$, i.e., $(P^2)_{AB}$:

$$P^2 = P \cdot P = \begin{pmatrix} 0.8 & 0.2 \\ 0.5 & 0.5 \end{pmatrix}\begin{pmatrix} 0.8 & 0.2 \\ 0.5 & 0.5 \end{pmatrix} = \begin{pmatrix} 0.74 & 0.26 \\ 0.65 & 0.35 \end{pmatrix}$$

So $P(X_2 = B \mid X_0 = A) = 0.26$. This sums over both intermediate paths: $A \to A \to B$ (probability $0.8 \times 0.2 = 0.16$) and $A \to B \to B$ (probability $0.2 \times 0.5 = 0.10$), giving $0.16 + 0.10 = 0.26$. $\checkmark$

### 2.3 Stationary Distribution

**Definition 4.10.5 (Stationary Distribution).** A probability distribution $\boldsymbol{\pi} = (\pi_1, \pi_2, \ldots)$ over states is *stationary* (or *invariant*) if:

$$\boldsymbol{\pi} P = \boldsymbol{\pi}, \qquad \sum_i \pi_i = 1, \qquad \pi_i \geq 0$$

If the chain starts in $\boldsymbol{\pi}$, it remains in $\boldsymbol{\pi}$ at all future times. The stationary distribution is a left eigenvector of $P$ corresponding to eigenvalue 1.

**Example 4.8.3 (Solving for the stationary distribution of a 2×2 chain).**
Let $P = \begin{pmatrix} 0.8 & 0.2 \\ 0.5 & 0.5 \end{pmatrix}$ with states $\{A, B\}$. Solve $\boldsymbol{\pi}P = \boldsymbol{\pi}$:

$$(\pi_A,\; \pi_B)\begin{pmatrix} 0.8 & 0.2 \\ 0.5 & 0.5 \end{pmatrix} = (\pi_A,\; \pi_B)$$

From the first component: $0.8\pi_A + 0.5\pi_B = \pi_A \;\Rightarrow\; 0.5\pi_B = 0.2\pi_A \;\Rightarrow\; \pi_B = \tfrac{2}{5}\pi_A$.

Using the normalization constraint $\pi_A + \pi_B = 1$:

$$\pi_A + \tfrac{2}{5}\pi_A = 1 \;\Rightarrow\; \tfrac{7}{5}\pi_A = 1 \;\Rightarrow\; \pi_A = \tfrac{5}{7} \approx 0.714, \quad \pi_B = \tfrac{2}{7} \approx 0.286$$

In the long run, the chain spends about 71.4% of time in state $A$ and 28.6% in state $B$.

### 2.4 Convergence and Classification

**Definition 4.10.6 (Irreducibility, Aperiodicity, Ergodicity).**

- **Irreducible:** Every state is reachable from every other state (the chain has a single communicating class)
- **Aperiodic:** For every state $i$, $\gcd\{n \geq 1 : P(X_n = i \mid X_0 = i) > 0\} = 1$ (the chain does not cycle deterministically)
- **Ergodic:** Both irreducible and aperiodic (for finite state spaces)

**Theorem 4.10.2 (Fundamental Theorem of Markov Chains).** If a finite-state Markov chain is irreducible and aperiodic, then:

1. A unique stationary distribution $\boldsymbol{\pi}$ exists
2. For any initial distribution, $P^n$ converges to the matrix with all rows equal to $\boldsymbol{\pi}$:

$$\lim_{n \to \infty} (P^n)_{ij} = \pi_j \quad \text{for all } i, j$$

3. The time-average converges: $\frac{1}{N}\sum_{n=0}^{N-1} \mathbf{1}_{X_n = j} \to \pi_j$ almost surely

The rate of convergence depends on the *spectral gap* $1 - |\lambda_2|$, where $\lambda_2$ is the second-largest eigenvalue of $P$ in absolute value. A larger spectral gap means faster mixing.

*ML connection:* **MCMC (Markov Chain Monte Carlo)** constructs a Markov chain whose stationary distribution equals the desired posterior $p(\theta \mid \text{data})$. The Metropolis-Hastings algorithm and Gibbs sampling are specific constructions that guarantee the chain converges to the posterior. The spectral gap determines how many samples we need before the chain has "mixed" and samples are approximately drawn from the target.

**Example 4.8.4 (Absorbing states and absorption probabilities).**
Consider a 3-state chain $\{1, 2, 3\}$ where state 3 is *absorbing* ($P_{33} = 1$):

$$P = \begin{pmatrix} 0.5 & 0.5 & 0 \\ 0 & 0.4 & 0.6 \\ 0 & 0 & 1 \end{pmatrix}$$

State 3 is absorbing because once entered, the chain never leaves. What is the probability of eventual absorption into state 3, starting from state 1?

Let $a_i = P(\text{eventually reach state } 3 \mid X_0 = i)$. Clearly $a_3 = 1$. Setting up first-step equations:

$$a_1 = 0.5\,a_1 + 0.5\,a_2 \qquad a_2 = 0.4\,a_2 + 0.6\,a_3 = 0.4\,a_2 + 0.6$$

From the second equation: $0.6\,a_2 = 0.6$, so $a_2 = 1$.
Substituting: $a_1 = 0.5\,a_1 + 0.5(1)$, so $0.5\,a_1 = 0.5$, giving $a_1 = 1$.

Both transient states are absorbed into state 3 with probability 1.

---

## 3. Random Walks

**Definition 4.10.7 (Simple Random Walk).** The *simple symmetric random walk* on $\mathbb{Z}$ is:

$$X_n = X_0 + \sum_{i=1}^{n} Z_i$$

where $Z_1, Z_2, \ldots$ are i.i.d. with $P(Z_i = +1) = P(Z_i = -1) = \frac{1}{2}$.

```
SAMPLE PATH OF A RANDOM WALK (1D)

  Position
    3 │                              ·
    2 │              · ·        · ·
    1 │         · ·     ·    ·
    0 │─ · · ·           · ·              ── start
   -1 │
   -2 │
      └──────────────────────────────── Time
       0  1  2  3  4  5  6  7  8  9 10

  Each step: +1 or -1 with equal probability
  After n steps: E[Xₙ] = 0, Var(Xₙ) = n
  Typical displacement: |Xₙ| ~ √n
```

**Theorem 4.10.3 (Properties of the Simple Random Walk).**

1. $\mathbb{E}[X_n] = X_0$ (the walk is a martingale)
2. $\text{Var}(X_n) = n$ (variance grows linearly with time)
3. $X_n / \sqrt{n} \xrightarrow{d} \mathcal{N}(0, 1)$ as $n \to \infty$ (by the Central Limit Theorem)

*Proof of (2).* $\text{Var}(X_n) = \text{Var}\left(\sum_{i=1}^n Z_i\right) = \sum_{i=1}^n \text{Var}(Z_i) = n \cdot 1 = n$, using independence. $\square$

Property (3) connects the discrete random walk to its continuous counterpart: Brownian motion is the scaling limit of the random walk.

*ML connection:* SGD iterates $\theta_{t+1} = \theta_t - \eta \nabla \tilde{L}(\theta_t)$ behave like a biased random walk in parameter space. The stochastic gradient noise acts as the random increment. The variance of the noise influences whether SGD escapes sharp minima (favoring flatter, more generalizable solutions) -- this is the implicit regularization effect of SGD.

**Example 4.8.5 (Simple random walk — computing probabilities).**
A symmetric random walk starts at $X_0 = 0$. Each step is $+1$ or $-1$ with probability $\frac{1}{2}$ each.

*What is $P(X_3 = 1)$?* After 3 steps, the position $X_3 = Z_1 + Z_2 + Z_3$. To reach $X_3 = 1$, we need exactly 2 ups and 1 down (net $+1$). The number of ways is $\binom{3}{2} = 3$, each with probability $(\tfrac{1}{2})^3 = \tfrac{1}{8}$:

$$P(X_3 = 1) = \binom{3}{2}\left(\tfrac{1}{2}\right)^3 = \frac{3}{8} = 0.375$$

*Verification of properties:* $\mathbb{E}[X_3] = 0$ (martingale), $\text{Var}(X_3) = 3$, and the standard deviation is $\sqrt{3} \approx 1.73$, so $X_3 = 1$ is within one standard deviation of the mean — consistent with having relatively high probability.

---

## 4. Poisson Process

**Definition 4.10.8 (Poisson Process).** A *Poisson process* with rate $\lambda > 0$ is a continuous-time counting process $\{N(t)\}_{t \geq 0}$ satisfying:

1. $N(0) = 0$
2. **Independent increments:** $N(t_2) - N(t_1)$ and $N(t_4) - N(t_3)$ are independent for $t_1 < t_2 \leq t_3 < t_4$
3. **Poisson-distributed counts:** $N(t+s) - N(t) \sim \text{Poisson}(\lambda s)$ for all $t, s \geq 0$

**Theorem 4.10.4 (Memoryless Inter-arrival Times).** The time between successive events (inter-arrival times) are i.i.d. $\text{Exponential}(\lambda)$. This follows from the memoryless property of the exponential distribution: $P(T > t + s \mid T > t) = P(T > s)$.

*ML connection:* Poisson processes model event arrivals in temporal point process models -- used for predicting user activity (clicks, purchases), earthquake aftershocks, and social media cascades. Neural temporal point processes learn the intensity function $\lambda(t)$ with an RNN or transformer, generalizing the constant-rate Poisson process to time-varying and history-dependent rates.

**Example 4.8.6 (Poisson process — event counts and inter-arrival times).**
A web server receives requests as a Poisson process with rate $\lambda = 10$ requests/minute.

**(a)** Probability of exactly 3 requests in a 30-second ($= 0.5$ min) window:

$$N(0.5) \sim \text{Poisson}(\lambda \cdot 0.5) = \text{Poisson}(5)$$

$$P(N(0.5) = 3) = \frac{e^{-5} \cdot 5^3}{3!} = \frac{e^{-5} \cdot 125}{6} \approx \frac{0.00674 \times 125}{6} \approx 0.1404$$

**(b)** Expected time until the next request (inter-arrival time):

$$T \sim \text{Exponential}(\lambda = 10), \quad \mathbb{E}[T] = \frac{1}{\lambda} = \frac{1}{10} \text{ min} = 6 \text{ seconds}$$

**(c)** Probability the next request takes more than 12 seconds ($= 0.2$ min):

$$P(T > 0.2) = e^{-\lambda \cdot 0.2} = e^{-2} \approx 0.1353$$

---

## 5. Brownian Motion (Wiener Process)

**Definition 4.10.9 (Standard Brownian Motion).** A *standard Brownian motion* (Wiener process) $\{W(t)\}_{t \geq 0}$ is a continuous-time stochastic process satisfying:

1. $W(0) = 0$
2. **Independent increments:** $W(t) - W(s)$ is independent of $\{W(u) : u \leq s\}$ for $s < t$
3. **Gaussian increments:** $W(t) - W(s) \sim \mathcal{N}(0, t - s)$ for $0 \leq s < t$
4. **Continuous paths:** $t \mapsto W(t)$ is continuous almost surely

```
BROWNIAN MOTION AS SCALING LIMIT OF RANDOM WALK

  Random walk (n steps)           Brownian motion (continuous)
                                  (rescale time by 1/n, space by 1/√n)
  Position                        W(t)
    │    ╱╲                         │     ╱╲
    │   ╱  ╲  ╱╲                    │    ╱  ╲  ╱╲
    │  ╱    ╲╱  ╲                   │   ╱    ╲╱  ╲
  0 │─╱          ╲───              0│──╱          ╲~~~~~
    │              ╲╱               │               ╲╱
    └──────────────────             └──────────────────
      0  1  2  ...  n                0             1   t

  Donsker's Theorem:
  X_⌊nt⌋ / √n  ──→  W(t)  as n → ∞
  (convergence in distribution over paths)
```

**Theorem 4.10.5 (Properties of Brownian Motion).**

1. $\mathbb{E}[W(t)] = 0$ for all $t$
2. $\text{Cov}(W(s), W(t)) = \min(s, t)$
3. $W(t)$ is a martingale (and so is $W(t)^2 - t$)
4. Paths are continuous but *nowhere differentiable* almost surely

*Proof of (2).* Assume $s \leq t$. Then $\text{Cov}(W(s), W(t)) = \text{Cov}(W(s), W(s) + [W(t) - W(s)]) = \text{Var}(W(s)) + \text{Cov}(W(s), W(t) - W(s))$. By independent increments, the cross-covariance vanishes, giving $\text{Cov}(W(s), W(t)) = \text{Var}(W(s)) = s = \min(s, t)$. $\square$

*ML connection:* **Diffusion models** (DDPM, score-based models) are directly built on Brownian motion. The forward process adds Gaussian noise progressively: $dx = -\frac{1}{2}\beta(t)x\,dt + \sqrt{\beta(t)}\,dW(t)$, which is a stochastic differential equation (SDE) driven by Brownian motion. The neural network learns to reverse this process, denoising step by step to generate data from pure noise.

**Example 4.8.7 (Brownian motion — mean, variance, and covariance).**
Let $W(t)$ be standard Brownian motion. Compute key quantities at $t = 4$:

$$\mathbb{E}[W(4)] = 0, \qquad \text{Var}(W(4)) = 4, \qquad \text{Std}(W(4)) = 2$$

So at time $t = 4$, the process is centered at 0 with about 95% of realizations in $[-4, 4]$ (within $\pm 2$ standard deviations).

*Covariance computation:* $\text{Cov}(W(3), W(7)) = \min(3, 7) = 3$.

*Increment variance:* $W(7) - W(3) \sim \mathcal{N}(0, 7 - 3) = \mathcal{N}(0, 4)$, so $\text{Var}(W(7) - W(3)) = 4$ and $\text{Std} = 2$.

**Example 4.8.8 (Product of Brownian motion values).**
Compute $\mathbb{E}[W(2) \cdot W(5)]$:

$$\mathbb{E}[W(2) \cdot W(5)] = \text{Cov}(W(2), W(5)) + \mathbb{E}[W(2)] \cdot \mathbb{E}[W(5)] = \min(2,5) + 0 \cdot 0 = 2$$

This uses the fact that for zero-mean random variables, $\mathbb{E}[XY] = \text{Cov}(X,Y)$.

---

## 6. Martingales

**Definition 4.10.10 (Martingale).** A stochastic process $\{M_n\}_{n \geq 0}$ (adapted to a filtration $\{\mathcal{F}_n\}$) is a *martingale* if:

1. $\mathbb{E}[|M_n|] < \infty$ for all $n$
2. $\mathbb{E}[M_{n+1} \mid \mathcal{F}_n] = M_n$ for all $n$

Intuitively, a martingale is a "fair game": the expected future value, given the present, equals the present value. No systematic drift up or down.

**Key examples:**

| Process | Why it is a martingale |
|---------|----------------------|
| Simple random walk $X_n$ | $\mathbb{E}[X_{n+1} \mid X_n] = X_n + \mathbb{E}[Z_{n+1}] = X_n$ |
| Brownian motion $W(t)$ | $\mathbb{E}[W(t) \mid W(s)] = W(s)$ for $s < t$ |
| $W(t)^2 - t$ | Compensated square has zero drift |
| Likelihood ratio $\prod_{i=1}^n \frac{p(X_i)}{q(X_i)}$ | Under $q$, this is a martingale |

**Theorem 4.10.6 (Optional Stopping Theorem -- simplified).** If $\{M_n\}$ is a martingale and $\tau$ is a bounded stopping time (i.e., $\tau \leq N$ for some fixed $N$), then $\mathbb{E}[M_\tau] = \mathbb{E}[M_0]$.

*ML connection:* Martingale theory provides the foundation for analyzing online learning algorithms and sequential hypothesis testing. In bandit problems, the regret of certain algorithms can be bounded using martingale concentration inequalities (Azuma-Hoeffding). The training loss of SGD, viewed appropriately, satisfies supermartingale properties that guarantee convergence.

**Example 4.8.9 (Verifying the martingale property for a random walk).**
Let $X_n$ be a simple symmetric random walk with $X_0 = 3$. Verify the martingale property at step $n = 2$:

$$\mathbb{E}[X_3 \mid X_2 = 5] = \frac{1}{2}(5 + 1) + \frac{1}{2}(5 - 1) = \frac{1}{2}(6) + \frac{1}{2}(4) = 5 = X_2 \;\; \checkmark$$

This holds for *any* value of $X_2$. The conditional expectation of the next step always equals the current position — the hallmark of a fair game. Consequently, $\mathbb{E}[X_n] = \mathbb{E}[X_0] = 3$ for all $n$.

**Example 4.8.10 (Optional Stopping — Gambler's Ruin setup).**
A gambler starts with \$5 and bets \$1 on a fair coin flip each round (a martingale). She stops when she reaches \$0 (ruin) or \$10 (target). Let $\tau$ be the stopping time.

By the Optional Stopping Theorem: $\mathbb{E}[X_\tau] = \mathbb{E}[X_0] = 5$.

Since $X_\tau \in \{0, 10\}$, let $p = P(X_\tau = 10)$. Then:

$$10p + 0(1 - p) = 5 \;\Rightarrow\; p = \frac{1}{2}$$

The gambler has a 50% chance of reaching \$10 and a 50% chance of going bankrupt — exactly what fairness predicts from her starting position midway between the two barriers.

---

## 7. Applications in ML and AI

### 7.1 MCMC: Sampling via Markov Chains

The core idea: construct a Markov chain whose stationary distribution is the target posterior $p(\theta \mid \mathcal{D})$.

```
MCMC SAMPLING SCHEME

  Target: posterior p(θ|data)  ←  intractable to sample directly

  Strategy: Build Markov chain  θ₀ → θ₁ → θ₂ → ... → θₙ
            whose stationary distribution = p(θ|data)

  ┌─────────────────────────────────────────────┐
  │  Metropolis-Hastings at step n:             │
  │                                             │
  │  1. Propose θ* ~ q(θ*|θₙ)                  │
  │  2. Accept with probability:                │
  │                                             │
  │     α = min(1, p(θ*|D)q(θₙ|θ*) )           │
  │                  ─────────────              │
  │                  p(θₙ|D)q(θ*|θₙ)            │
  │                                             │
  │  3. θₙ₊₁ = θ* if accepted, else θₙ        │
  └─────────────────────────────────────────────┘

  Burn-in period          Stationary samples
  ◀──────────────▶◀──────────────────────────▶
  θ₀  θ₁  ...  θₖ  θₖ₊₁  θₖ₊₂  ...  θₙ
  (discard)         (use for posterior estimates)
```

The chain satisfies *detailed balance*: $\pi_i P_{ij} = \pi_j P_{ji}$, which guarantees $\boldsymbol{\pi}$ is stationary.

### 7.2 PageRank as Stationary Distribution

Google's PageRank models a "random surfer" browsing the web:

- With probability $\alpha \approx 0.85$: follow a random outgoing link from the current page
- With probability $1 - \alpha$: jump to a uniformly random page

The PageRank vector $\boldsymbol{\pi}$ is the stationary distribution of this Markov chain:

$$\boldsymbol{\pi} = \boldsymbol{\pi}\left[\alpha S + (1-\alpha)\frac{1}{n}\mathbf{1}\mathbf{1}^T\right]$$

where $S$ is the column-normalized link matrix. The damping factor ensures irreducibility and aperiodicity (guaranteeing a unique $\boldsymbol{\pi}$ by Theorem 4.10.2). Computed via power iteration on billions of pages.

### 7.3 Diffusion Models

Diffusion models define a *forward* stochastic process that gradually destroys data, and learn a *reverse* process that reconstructs it.

```
DIFFUSION MODEL: FORWARD AND REVERSE PROCESSES

  Forward process (fixed): add noise gradually
  ─────────────────────────────────────────▶

  x₀          x₁          x₂    ...    xT
  (clean)                               (noise)
  🖼️ ───▶  slightly ───▶  more  ───▶  ≈ N(0,I)
           noisy        noisy

  ◀─────────────────────────────────────────
  Reverse process (learned): denoise step by step

  At each step t:
    Forward:  q(xₜ|xₜ₋₁) = N(xₜ; √(1-βₜ) xₜ₋₁, βₜI)
    Reverse:  pθ(xₜ₋₁|xₜ) = N(xₜ₋₁; μθ(xₜ,t), σₜ²I)
              └── neural network predicts mean ──┘

  Mathematical foundation:
    Forward SDE:  dx = -½ β(t)x dt + √β(t) dW
    Reverse SDE:  dx = [-½ β(t)x - β(t)∇ₓ log pₜ(x)] dt + √β(t) dW̃
                                    └── score function (learned) ──┘
```

The forward process is a specific instance of an Ornstein-Uhlenbeck process (a type of SDE driven by Brownian motion). The reverse-time SDE result (Anderson, 1982) guarantees that the reverse process exists and depends on the *score function* $\nabla_x \log p_t(x)$, which the neural network learns.

### 7.4 SGD as a Stochastic Process

Stochastic gradient descent with mini-batch noise:

$$\theta_{t+1} = \theta_t - \eta \nabla \tilde{L}(\theta_t) = \theta_t - \eta [\nabla L(\theta_t) + \varepsilon_t]$$

where $\varepsilon_t$ is the gradient noise. This is a discrete-time stochastic process with continuous state space. In the continuous-time limit (small $\eta$), SGD approximates the SDE:

$$d\theta = -\nabla L(\theta)\,dt + \sqrt{\eta\, \Sigma(\theta)}\,dW(t)$$

The noise term acts as implicit regularization, biasing SGD toward flat minima with better generalization.

### 7.5 Hidden Markov Models (HMMs)

An HMM separates the *hidden* Markov chain of states from *observed* emissions:

- Hidden states: $Z_1 \to Z_2 \to \cdots \to Z_T$ (Markov chain with transition matrix $A$)
- Observations: $X_t \sim p(X_t \mid Z_t = k) = B_k$ (emission distribution)

Three fundamental problems: (1) likelihood via the forward algorithm, (2) decoding via Viterbi, (3) learning via Baum-Welch (EM). Used in speech recognition, gene finding, and NLP before the deep learning era.

### 7.6 Reinforcement Learning: MDPs

A *Markov Decision Process* (MDP) extends a Markov chain with actions and rewards:

$$(S, A, P, R, \gamma): \quad P(s' \mid s, a), \quad R(s, a)$$

The agent selects actions to maximize cumulative discounted reward $\sum_{t=0}^\infty \gamma^t R(s_t, a_t)$. Under a fixed policy $\pi(a \mid s)$, the state sequence is a Markov chain with transition kernel $P^\pi(s' \mid s) = \sum_a \pi(a \mid s) P(s' \mid s, a)$. The value function $V^\pi(s) = \mathbb{E}^\pi[\sum_{t=0}^\infty \gamma^t R_t \mid s_0 = s]$ satisfies the Bellman equation -- a fixed-point condition on this controlled Markov chain.

---

## Exercises

**★ Basic**

1. A Markov chain on $\{A, B\}$ has transition matrix $P = \begin{pmatrix} 0.7 & 0.3 \\ 0.4 & 0.6 \end{pmatrix}$. Find the stationary distribution by solving $\boldsymbol{\pi}P = \boldsymbol{\pi}$.

2. Starting from state $A$, compute the probability of being in state $B$ after exactly 2 steps (i.e., find $(P^2)_{AB}$).

3. A simple random walk starts at $X_0 = 0$. Compute $\mathbb{E}[X_{10}]$ and $\text{Var}(X_{10})$.

4. Events arrive as a Poisson process with rate $\lambda = 3$ per hour. What is the probability of exactly 5 events in a 2-hour window?

**★★ Intermediate**

5. Prove that a Markov chain on states $\{1, 2, 3\}$ with transition matrix $P = \begin{pmatrix} 0 & 1 & 0 \\ 0 & 0 & 1 \\ 1 & 0 & 0 \end{pmatrix}$ is irreducible but periodic. What is its period? Does it have a stationary distribution?

6. Let $W(t)$ be standard Brownian motion. Show that $\text{Var}(W(t) - W(s)) = |t - s|$ and compute $\mathbb{E}[W(3)W(5)]$ using the covariance formula.

7. Verify that the Metropolis-Hastings acceptance rule satisfies detailed balance: $\pi_i P_{ij} = \pi_j P_{ji}$, where $P_{ij}$ is the chain's transition probability and $\pi$ is the target distribution.

8. For the HMM with two hidden states and Gaussian emissions, write the forward recursion for computing $P(X_1, X_2, \ldots, X_T)$.

**★★★ Challenging**

9. **(Mixing time)** A Markov chain has transition matrix with eigenvalues $1, 0.9, 0.5, -0.3$. Estimate the mixing time (number of steps until the chain is approximately stationary). How would the mixing time change if the second eigenvalue were $0.99$ instead?

10. **(Diffusion models)** In the forward process $q(x_t \mid x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t}\, x_0, (1 - \bar{\alpha}_t)I)$ where $\bar{\alpha}_t = \prod_{s=1}^t (1 - \beta_s)$, show that as $T \to \infty$ with appropriate $\beta_t$ schedule, $q(x_T \mid x_0) \to \mathcal{N}(0, I)$ regardless of $x_0$. Explain why this is essential for the generative model.

11. **(Martingale convergence)** Let $\{M_n\}$ be a non-negative martingale. Use the optional stopping theorem to prove that a gambler with finite wealth playing a fair game will eventually go bankrupt with probability 1 (the gambler's ruin problem).

12. **(SGD as SDE)** Starting from the SGD update $\theta_{t+1} = \theta_t - \eta[\nabla L(\theta_t) + \varepsilon_t]$ where $\varepsilon_t$ has mean zero and covariance $\Sigma(\theta_t)$, derive the continuous-time SDE approximation and explain how the noise covariance $\eta \Sigma(\theta)$ acts as an implicit regularizer favoring flat minima.

---

## Related Topics

- [Conditional Probability](conditional-probability.md) -- Bayes' theorem, the foundation of MCMC target distributions
- [Distributions](distributions.md) -- Poisson, exponential, and Gaussian distributions used throughout
- [Limit Theorems](limit-theorems.md) -- CLT connects random walks to Brownian motion via Donsker's theorem
- [Bayesian Inference](bayesian-inference.md) -- MCMC as a computational tool for posterior sampling
- [Linear Algebra - Eigenvalues](../linear-algebra/eigenvalues.md) -- spectral gap and convergence of Markov chains
