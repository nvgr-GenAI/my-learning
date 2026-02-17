# Chapter 5.4: Stochastic Optimization

## Prerequisites

- **Required:**
  - Gradient descent and optimization fundamentals (Chapter 5.1)
  - Probability theory: expectation, variance, unbiased estimators
  - Convex analysis basics
  - Linear algebra: norms, inner products

- **Recommended:**
  - Convergence analysis of iterative algorithms
  - Statistical estimation theory
  - Large-scale optimization

## Overview

Stochastic optimization methods form the backbone of modern machine learning, enabling training on datasets too large to fit in memory. Unlike deterministic gradient descent which computes exact gradients over the entire dataset, stochastic methods use noisy gradient estimates computed on random subsets of data. This introduces variance but dramatically reduces computational cost per iteration.

The fundamental trade-off in stochastic optimization:
- **Computational efficiency:** Each iteration is cheap (small batch)
- **Statistical efficiency:** Convergence requires more iterations due to noise

This chapter develops the mathematical theory of stochastic gradient methods, variance reduction techniques, adaptive learning rates, and convergence guarantees that underpin all modern deep learning optimization.

---

## 5.4.1 Stochastic Gradient Descent

### The Empirical Risk Minimization Problem

In supervised learning, we minimize the expected loss over the data distribution:

$$\min_{\theta \in \mathbb{R}^d} \mathcal{L}(\theta) = \mathbb{E}_{(x,y) \sim \mathcal{D}}[\ell(\theta; x, y)]$$

In practice, we only have a finite training set $S = \{(x_i, y_i)\}_{i=1}^n$, so we minimize the empirical risk:

$$\min_{\theta \in \mathbb{R}^d} \mathcal{L}(\theta) = \frac{1}{n} \sum_{i=1}^n \ell(\theta; x_i, y_i)$$

**Definition 5.4.1 (Gradient Estimators):**

1. **Full-batch gradient:** $g(\theta) = \frac{1}{n} \sum_{i=1}^n \nabla \ell(\theta; x_i, y_i)$
2. **Mini-batch gradient:** $g_B(\theta) = \frac{1}{|B|} \sum_{i \in B} \nabla \ell(\theta; x_i, y_i)$ where $B \subset \{1,\ldots,n\}$
3. **Stochastic gradient:** $g_i(\theta) = \nabla \ell(\theta; x_i, y_i)$ for random $i$

**Theorem 5.4.1 (Unbiased Gradient Estimator):**

Let $B$ be a random subset of $\{1,\ldots,n\}$ with each element sampled uniformly. Then:

$$\mathbb{E}_B[g_B(\theta)] = \nabla \mathcal{L}(\theta)$$

*Proof:* By linearity of expectation:
$$\mathbb{E}_B\left[\frac{1}{|B|} \sum_{i \in B} \nabla \ell(\theta; x_i, y_i)\right] = \frac{1}{|B|} \sum_{i \in B} \mathbb{E}[\nabla \ell(\theta; x_i, y_i)] = \frac{1}{n} \sum_{i=1}^n \nabla \ell(\theta; x_i, y_i)$$

**Example 5.4.1 (Stochastic vs Full Gradient):**

Consider $\mathcal{L}(\theta) = \frac{1}{4}\sum_{i=1}^{4} (\theta - y_i)^2$ with $y = [1, 3, 5, 7]$ and current $\theta = 2$.

Per-sample gradients: $\nabla \ell_i = 2(\theta - y_i)$, so $\nabla \ell_1 = 2$, $\nabla \ell_2 = -2$, $\nabla \ell_3 = -6$, $\nabla \ell_4 = -10$.

- **Full gradient:** $g = \frac{1}{4}(2 + (-2) + (-6) + (-10)) = -4$
- **Stochastic (sample $i=1$):** $g_1 = 2$ (points *away* from optimum!)
- **Mini-batch ($B = \{1,3\}$):** $g_B = \frac{1}{2}(2 + (-6)) = -2$ (correct direction, wrong magnitude)

All are unbiased: $\mathbb{E}[g_i] = \frac{1}{4}(2 - 2 - 6 - 10) = -4 = g$, but individual estimates can be very noisy.

### The SGD Algorithm

**Algorithm 5.4.1 (Stochastic Gradient Descent):**

```
Input: Initial point θ₀, learning rate schedule {ηₖ}, batch size b
For k = 0, 1, 2, ...:
    1. Sample mini-batch Bₖ ⊂ {1,...,n} with |Bₖ| = b
    2. Compute gradient estimate: gₖ = (1/b) Σᵢ∈Bₖ ∇ℓ(θₖ; xᵢ, yᵢ)
    3. Update: θₖ₊₁ = θₖ - ηₖ gₖ
```

**Example 5.4.2 (One SGD Update Step):**

Loss: $\ell(\theta; x_i, y_i) = \frac{1}{2}(\theta^T x_i - y_i)^2$ with $\theta_0 = [1.0, 0.5]^T$, $\eta = 0.1$, batch $B = \{1,2\}$.

Data: $(x_1, y_1) = ([2, 1], 3)$ and $(x_2, y_2) = ([1, 3], 2)$.

Step 1 -- Per-sample gradients $\nabla \ell_i = (\theta^T x_i - y_i) x_i$:

$$\nabla \ell_1 = (1\cdot2 + 0.5\cdot1 - 3)[2, 1]^T = (-0.5)[2, 1]^T = [-1.0, -0.5]^T$$

$$\nabla \ell_2 = (1\cdot1 + 0.5\cdot3 - 2)[1, 3]^T = (0.5)[1, 3]^T = [0.5, 1.5]^T$$

Step 2 -- Mini-batch gradient: $g_B = \frac{1}{2}([-1.0, -0.5]^T + [0.5, 1.5]^T) = [-0.25, 0.5]^T$

Step 3 -- Update: $\theta_1 = [1.0, 0.5]^T - 0.1 \cdot [-0.25, 0.5]^T = [1.025, 0.45]^T$

**Computational Comparison:**

```
Iteration Cost per Dataset Pass:
─────────────────────────────────────────────────
Method      | Cost/Iter | Iters/Epoch | Total
─────────────────────────────────────────────────
Full-batch  |    n      |      1      |   n
Mini-batch  |    b      |    n/b      |   n
  (b=32)    |   32      |   n/32      |   n
─────────────────────────────────────────────────

Example: n = 1,000,000, b = 32
- Full-batch: 1 update per 1M samples
- Mini-batch: 31,250 updates per 1M samples
```

**Definition 5.4.2 (Epoch):**
One epoch is a complete pass through the training dataset. With batch size $b$ and dataset size $n$, one epoch consists of $\lceil n/b \rceil$ SGD iterations.

---

## 5.4.2 Variance Reduction

### Sources of Variance in SGD

The mini-batch gradient $g_B(\theta)$ is an unbiased but noisy estimator of $\nabla \mathcal{L}(\theta)$. The variance comes from random sampling of the mini-batch.

**Theorem 5.4.2 (Variance of Mini-batch Gradient):**

For a mini-batch of size $b$ sampled uniformly without replacement:

$$\text{Var}[g_B(\theta)] = \frac{1}{b} \cdot \frac{n-b}{n-1} \cdot \sigma^2(\theta)$$

where $\sigma^2(\theta) = \frac{1}{n} \sum_{i=1}^n \|\nabla \ell(\theta; x_i, y_i) - \nabla \mathcal{L}(\theta)\|^2$ is the gradient variance.

**Example 5.4.3 (Mini-batch Variance vs Batch Size):**

Dataset: $n = 4$ samples with per-sample gradients $\nabla \ell_i = \{-3, -1, 1, 3\}$ (scalar case).

Full gradient: $\nabla \mathcal{L} = \frac{1}{4}(-3 - 1 + 1 + 3) = 0$.

Gradient variance: $\sigma^2 = \frac{1}{4}(9 + 1 + 1 + 9) = 5$.

Using the formula $\text{Var}[g_B] = \frac{1}{b} \cdot \frac{n - b}{n - 1} \cdot \sigma^2$:

| Batch size $b$ | $\text{Var}[g_B]$ | Computation |
| --- | --- | --- |
| 1 | $\frac{1}{1} \cdot \frac{3}{3} \cdot 5 = 5.0$ | Noisiest |
| 2 | $\frac{1}{2} \cdot \frac{2}{3} \cdot 5 = 1.67$ | 3x reduction |
| 4 | $\frac{1}{4} \cdot \frac{0}{3} \cdot 5 = 0$ | Exact gradient |

Doubling batch size from 1 to 2 cuts variance by 3x, not just 2x, due to the finite-population correction $\frac{n-b}{n-1}$.

**Key observations:**
1. Variance decreases as $O(1/b)$ with batch size
2. For $b \ll n$: $\text{Var}[g_B(\theta)] \approx \sigma^2(\theta)/b$
3. For $b = n$: $\text{Var}[g_B(\theta)] = 0$ (exact gradient)

```
Variance vs Batch Size:
  Var
   │
σ² │●
   │ ●
   │  ●
   │   ●
   │    ●
   │      ●──●──●──────────●
   └─────────────────────────── b
   1    4    16   64   256   n

Trade-off:
- Small b: high variance, fast iterations
- Large b: low variance, slow iterations, better parallelism
```

### Importance Sampling

**Definition 5.4.3 (Importance Sampling for SGD):**

Sample index $i$ with probability $p_i \propto \|\nabla \ell(\theta; x_i, y_i)\|$ and use gradient estimator:

$$g_{IS}(\theta) = \frac{1}{n p_i} \nabla \ell(\theta; x_i, y_i)$$

This remains unbiased but can have lower variance by focusing on "important" samples.

**Challenge:** Computing $\|\nabla \ell(\theta; x_i, y_i)\|$ for all samples defeats the purpose of SGD. Practical approaches use:
- Cached approximate norms from previous iterations
- Loss values as proxies for gradient norms
- Adaptive sampling based on training dynamics

---

## 5.4.3 Adaptive Methods

Fixed learning rates require careful tuning and may be suboptimal when different parameters need different step sizes. Adaptive methods automatically adjust per-parameter learning rates based on gradient history.

### SGD with Momentum

Before exploring fully adaptive methods, note that plain SGD can be augmented with a *momentum* term. The update becomes:

$$v_k = \beta v_{k-1} + g_k, \quad \theta_{k+1} = \theta_k - \eta \, v_k$$

where $\beta \in [0, 1)$ is the momentum coefficient and $v_0 = 0$.

**Example 5.4.4 (Vanilla SGD vs SGD with Momentum):**

Minimize $f(\theta) = \theta^2$ with $\theta_0 = 10$, $\eta = 0.1$. Gradient: $\nabla f = 2\theta$.

**Vanilla SGD** ($\beta = 0$):

| Step | $\theta_k$ | $g_k = 2\theta_k$ | $\theta_{k+1} = \theta_k - 0.1 \cdot g_k$ |
| --- | --- | --- | --- |
| 0 | 10.0 | 20.0 | $10.0 - 2.0 = 8.0$ |
| 1 | 8.0 | 16.0 | $8.0 - 1.6 = 6.4$ |

**SGD with Momentum** ($\beta = 0.9$, $v_0 = 0$):

| Step | $\theta_k$ | $g_k$ | $v_k = 0.9 v_{k-1} + g_k$ | $\theta_{k+1} = \theta_k - 0.1 v_k$ |
| --- | --- | --- | --- | --- |
| 0 | 10.0 | 20.0 | $0 + 20.0 = 20.0$ | $10.0 - 2.0 = 8.0$ |
| 1 | 8.0 | 16.0 | $18.0 + 16.0 = 34.0$ | $8.0 - 3.4 = 4.6$ |

After 2 steps: vanilla SGD reaches $6.4$, momentum reaches $4.6$. Momentum accelerates convergence by accumulating velocity in consistent gradient directions.

### AdaGrad: Adaptive Gradient Algorithm

**Algorithm 5.4.2 (AdaGrad):**

$$\begin{align}
g_k &= \nabla \mathcal{L}(θ_k) \\
G_k &= G_{k-1} + g_k \odot g_k \quad \text{(accumulate squared gradients)} \\
θ_{k+1} &= θ_k - \frac{\eta}{\sqrt{G_k + \epsilon}} \odot g_k
\end{align}$$

where $\odot$ is element-wise product, division and square root are element-wise, and $\epsilon \approx 10^{-8}$ prevents division by zero.

**Definition 5.4.4 (Per-parameter Learning Rate):**

AdaGrad effectively uses learning rate $\eta_k^{(i)} = \frac{\eta}{\sqrt{\sum_{t=0}^k (g_t^{(i)})^2 + \epsilon}}$ for parameter $\theta^{(i)}$.

**Properties:**
- Parameters with large cumulative gradients get small learning rates
- Parameters with small cumulative gradients get large learning rates
- Learning rate decays as $O(1/\sqrt{k})$ automatically
- **Drawback:** Accumulator $G_k$ grows monotonically, eventually making learning rates infinitesimally small

### RMSProp: Root Mean Square Propagation

**Algorithm 5.4.3 (RMSProp):**

$$\begin{align}
g_k &= \nabla \mathcal{L}(θ_k) \\
v_k &= \beta v_{k-1} + (1-\beta) g_k \odot g_k \quad \text{(exponential moving average)} \\
θ_{k+1} &= θ_k - \frac{\eta}{\sqrt{v_k + \epsilon}} \odot g_k
\end{align}$$

Typical hyperparameter: $\beta = 0.9$

**Key difference from AdaGrad:** Uses exponential moving average instead of cumulative sum, preventing learning rate from vanishing. Effectively uses a sliding window of recent gradients.

### Adam: Adaptive Moment Estimation

**Algorithm 5.4.4 (Adam):**

$$\begin{align}
g_k &= \nabla \mathcal{L}(θ_k) \\
m_k &= \beta_1 m_{k-1} + (1-\beta_1) g_k \quad \text{(first moment, momentum)} \\
v_k &= \beta_2 v_{k-1} + (1-\beta_2) g_k \odot g_k \quad \text{(second moment, RMSProp)} \\
\hat{m}_k &= \frac{m_k}{1-\beta_1^k} \quad \text{(bias correction)} \\
\hat{v}_k &= \frac{v_k}{1-\beta_2^k} \quad \text{(bias correction)} \\
θ_{k+1} &= θ_k - \frac{\eta}{\sqrt{\hat{v}_k} + \epsilon} \odot \hat{m}_k
\end{align}$$

**Standard hyperparameters:** $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\eta = 10^{-3}$, $\epsilon = 10^{-8}$

**Bias correction:** Since $m_0 = v_0 = 0$, early estimates are biased toward zero. The correction factors $\frac{1}{1-\beta^k}$ compensate for this initialization bias.

**Example 5.4.5 (One Complete Adam Step):**

Scalar case: $\theta_0 = 5.0$, $m_0 = 0$, $v_0 = 0$, $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\eta = 0.01$, $\epsilon = 10^{-8}$.

Suppose gradient at step $k=1$: $g_1 = -4.0$.

Step 1 -- First moment: $m_1 = 0.9 \cdot 0 + 0.1 \cdot (-4.0) = -0.4$

Step 2 -- Second moment: $v_1 = 0.999 \cdot 0 + 0.001 \cdot (-4.0)^2 = 0.016$

Step 3 -- Bias correction: $\hat{m}_1 = \frac{-0.4}{1 - 0.9^1} = \frac{-0.4}{0.1} = -4.0$, $\quad \hat{v}_1 = \frac{0.016}{1 - 0.999^1} = \frac{0.016}{0.001} = 16.0$

Step 4 -- Update: $\theta_1 = 5.0 - \frac{0.01}{\sqrt{16.0} + 10^{-8}} \cdot (-4.0) = 5.0 + \frac{0.04}{4.0} = 5.01$

Note: Bias correction at $k=1$ is dramatic -- it amplifies $m_1$ by $10\times$ and $v_1$ by $1000\times$. Without it, the effective step would be near-zero due to initialization at $m_0 = v_0 = 0$.

### Comparison Table

```
┌─────────────┬──────────────┬───────────────────┬─────────────────────┬─────────────────┐
│ Method      │ Memory       │ LR per Parameter  │ Use Momentum        │ Best For        │
├─────────────┼──────────────┼───────────────────┼─────────────────────┼─────────────────┤
│ SGD         │ O(d)         │ No                │ Optional (SGDM)     │ Vision tasks    │
│             │              │                   │                     │ with tuning     │
├─────────────┼──────────────┼───────────────────┼─────────────────────┼─────────────────┤
│ AdaGrad     │ O(d)         │ Yes (decaying)    │ No                  │ Sparse features │
│             │              │                   │                     │ (NLP, ads)      │
├─────────────┼──────────────┼───────────────────┼─────────────────────┼─────────────────┤
│ RMSProp     │ O(d)         │ Yes (adaptive)    │ No                  │ RNNs (historic) │
│             │              │                   │                     │                 │
├─────────────┼──────────────┼───────────────────┼─────────────────────┼─────────────────┤
│ Adam        │ O(2d)        │ Yes (adaptive)    │ Yes (built-in)      │ Default choice, │
│             │              │                   │                     │ transformers    │
└─────────────┴──────────────┴───────────────────┴─────────────────────┴─────────────────┘
```

### Optimization Paths: Fixed vs Adaptive Learning Rates

```
Loss landscape cross-section:

         ╱╲                     ╱╲
        ╱  ╲                   ╱  ╲
       ╱    ╲                 ╱    ╲
      ╱      ╲_______________╱      ╲
     ╱            ★                  ╲
    ╱          (optimum)              ╲

SGD with fixed LR:
    →  →  →  →  → →→→ →  →  →
    (constant step size, may overshoot)

Adam with adaptive LR:
    →  → → →→⟶⇒⟹⟶→ → → ·
    (adapts to curvature, slows near optimum)
```

**ML Connection:** Adam is the default optimizer for most deep learning tasks (transformers, diffusion models), while SGD with momentum remains popular for computer vision where careful tuning is feasible and generalization is critical.

---

## 5.4.4 Learning Rate Schedules

### Motivation

Even with adaptive methods, global learning rate scheduling improves convergence. Key principles:
1. **Early training:** Large learning rates for rapid progress
2. **Late training:** Small learning rates for fine-tuning
3. **Exploration:** Periodic increases can escape local minima

**Definition 5.4.5 (Learning Rate Schedule):**
A function $\eta: \mathbb{N} \to \mathbb{R}^+$ that maps iteration number to learning rate: $\eta_k = \eta(k)$.

### Step Decay

$$\eta_k = \eta_0 \cdot \gamma^{\lfloor k/s \rfloor}$$

where $\gamma \in (0,1)$ is decay factor and $s$ is step size.

**Example:** $\eta_0 = 0.1$, $\gamma = 0.1$, $s = 30$ epochs
- Epochs 0-29: $\eta = 0.1$
- Epochs 30-59: $\eta = 0.01$
- Epochs 60-89: $\eta = 0.001$

```
Step Decay Schedule:
  η
  │
η₀│████████████░░░░░░░░▒▒▒▒
  │            ░░░░░░░░▒▒▒▒
  │                    ▒▒▒▒
  │                    ▒▒▒▒
  └────────────────────────── k
  0        s       2s      3s
```

### Exponential Decay

$$\eta_k = \eta_0 \cdot e^{-\lambda k}$$

Smooth continuous decay. More gradual than step decay.

**Example 5.4.6 (Learning Rate Schedules Over 4 Epochs):**

Starting learning rate $\eta_0 = 0.1$. Compare three schedules over epochs $k = 0, 1, 2, 3, 4$:

| Epoch $k$ | Step decay ($\gamma=0.5, s=2$) | Exponential ($\lambda=0.5$) | $1/\sqrt{k+1}$ schedule |
| --- | --- | --- | --- |
| 0 | $0.1 \cdot 0.5^0 = 0.100$ | $0.1 \cdot e^{0} = 0.100$ | $0.1 / 1.00 = 0.100$ |
| 1 | $0.1 \cdot 0.5^0 = 0.100$ | $0.1 \cdot e^{-0.5} = 0.061$ | $0.1 / 1.41 = 0.071$ |
| 2 | $0.1 \cdot 0.5^1 = 0.050$ | $0.1 \cdot e^{-1.0} = 0.037$ | $0.1 / 1.73 = 0.058$ |
| 3 | $0.1 \cdot 0.5^1 = 0.050$ | $0.1 \cdot e^{-1.5} = 0.022$ | $0.1 / 2.00 = 0.050$ |
| 4 | $0.1 \cdot 0.5^2 = 0.025$ | $0.1 \cdot e^{-2.0} = 0.014$ | $0.1 / 2.24 = 0.045$ |

Step decay holds the rate constant within each interval then drops sharply. Exponential decay is the most aggressive. The $1/\sqrt{k}$ schedule decays most gently -- preferred by theory but rarely used in practice.

### Cosine Annealing

$$\eta_k = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})\left(1 + \cos\left(\frac{k}{T}\pi\right)\right)$$

for $k \in [0, T]$ where $T$ is total iterations.

```
Cosine Annealing:
    η
    │
ηₘₐₓ│●─╮
    │   ╲
    │    ╲
    │     ╲
    │      ╲
    │       ╲
    │        ╲
ηₘᵢₙ│         ╰──●
    └────────────── k
    0          T

Smooth decay, commonly used for transformers
```

### Warmup

Start with very small learning rate and gradually increase to target value over $k_w$ iterations:

$$\eta_k = \begin{cases}
\frac{k}{k_w} \eta_0 & \text{if } k \leq k_w \\
\eta_0 & \text{if } k > k_w
\end{cases}$$

**Rationale:** Large learning rates early in training can be unstable when parameters are randomly initialized. Warmup allows the network to "settle" before aggressive optimization.

```
Warmup Schedule (then constant):
  η
  │         ███████████████
η₀│       ╱
  │     ╱
  │   ╱
  │ ╱
  └────────────────────────── k
  0     kᵥᵥ

Critical for transformer training!
```

### Cyclical Learning Rates

**Definition 5.4.6 (Cyclical LR):**

$$\eta_k = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})\left(1 + \cos\left(\frac{2\pi k}{c}\right)\right)$$

where $c$ is cycle length. Repeatedly oscillate between $\eta_{\min}$ and $\eta_{\max}$.

```
Cyclical LR (multiple restarts):
  η
  │  ╱╲    ╱╲    ╱╲
  │ ╱  ╲  ╱  ╲  ╱  ╲
  │╱    ╲╱    ╲╱    ╲
  └────────────────────── k

Can help escape saddle points and local minima
```

**ML Connection:** Warmup is essential for transformer training (BERT, GPT use 10k step warmup). Cosine annealing is standard for vision models (ResNet, ViT). Cyclical learning rates used in super-convergence techniques.

---

## 5.4.5 Convergence of SGD

### Assumptions for Convergence Analysis

**Assumption 5.4.1 (L-smooth):**
$\mathcal{L}$ is $L$-smooth: $\|\nabla \mathcal{L}(\theta) - \nabla \mathcal{L}(\theta')\| \leq L\|\theta - \theta'\|$ for all $\theta, \theta'$.

**Assumption 5.4.2 (Bounded Variance):**
The variance of the gradient estimator is bounded: $\mathbb{E}[\|g_k - \nabla \mathcal{L}(\theta_k)\|^2] \leq \sigma^2$ for all $k$.

**Assumption 5.4.3 (Unbiased Gradient):**
$\mathbb{E}[g_k | \theta_k] = \nabla \mathcal{L}(\theta_k)$

### Convergence for Convex Functions

**Theorem 5.4.3 (SGD Convergence Rate, Convex Case):**

Suppose $\mathcal{L}$ is convex and $L$-smooth, and use constant learning rate $\eta_k = \eta$. Then after $K$ iterations:

$$\mathbb{E}[\mathcal{L}(\bar{\theta}_K)] - \mathcal{L}(\theta^*) \leq \frac{\|\theta_0 - \theta^*\|^2}{2\eta K} + \frac{\eta L \sigma^2}{2}$$

where $\bar{\theta}_K = \frac{1}{K}\sum_{k=0}^{K-1} \theta_k$ is the average iterate.

**Implications:**
1. **First term:** Decreases as $O(1/K)$ - optimization error
2. **Second term:** Constant - statistical error from gradient noise
3. **Optimal fixed learning rate:** $\eta^* = \frac{\|\theta_0 - \theta^*\|}{L\sigma\sqrt{K}}$ gives rate $O(1/\sqrt{K})$
4. To achieve $\epsilon$-accuracy: need $K = O(1/\epsilon^2)$ iterations

**Example 5.4.7 (Convergence: SGD $O(1/\sqrt{K})$ vs GD $O(1/K)$):**

Let initial distance $\|\theta_0 - \theta^*\|^2 = 100$, $L = 1$, noise $\sigma^2 = 10$.

For **full-batch GD** (convex, $L$-smooth): error $\leq \frac{L \|\theta_0 - \theta^*\|^2}{2K} = \frac{100}{2K}$

For **SGD** with optimal $\eta$: error $\leq \frac{\|\theta_0 - \theta^*\| \sigma \sqrt{L}}{\sqrt{K}} = \frac{10\sqrt{10}}{\sqrt{K}} \approx \frac{31.6}{\sqrt{K}}$

| Iterations $K$ | GD bound ($O(1/K)$) | SGD bound ($O(1/\sqrt{K})$) |
| --- | --- | --- |
| 100 | $0.50$ | $3.16$ |
| 1,000 | $0.05$ | $1.00$ |
| 10,000 | $0.005$ | $0.316$ |

GD converges faster *per iteration*, but each SGD iteration is $n/b$ times cheaper. If $n = 10{,}000$ and $b = 32$, SGD does $\sim 312$ updates for the same compute as one GD step -- making SGD vastly faster in wall-clock time.

### Convergence for Non-convex Functions

**Theorem 5.4.4 (SGD for Non-convex, Finding Stationary Points):**

Under Assumptions 5.4.1-5.4.3, with diminishing learning rate $\eta_k = \frac{\eta_0}{\sqrt{k+1}}$:

$$\min_{k=0,\ldots,K-1} \mathbb{E}[\|\nabla \mathcal{L}(\theta_k)\|^2] \leq \frac{2(\mathcal{L}(\theta_0) - \mathcal{L}_{\inf})}{\eta_0 \sqrt{K}} + \frac{\eta_0 L \sigma^2}{\sqrt{K}}$$

After $K = O(1/\epsilon^4)$ iterations, we find $\theta_k$ with $\mathbb{E}[\|\nabla \mathcal{L}(\theta_k)\|^2] \leq \epsilon$.

**Note:** This only guarantees finding stationary points ($\nabla \mathcal{L}(\theta) = 0$), which could be local minima or saddle points. Deep learning practice suggests SGD often finds "good" local minima.

### Learning Rate Requirements for Convergence

**Theorem 5.4.5 (Robbins-Monro Conditions):**

For almost-sure convergence of SGD to a stationary point, the learning rate sequence must satisfy:

$$\sum_{k=0}^{\infty} \eta_k = \infty \quad \text{and} \quad \sum_{k=0}^{\infty} \eta_k^2 < \infty$$

**Examples:**
- $\eta_k = \frac{1}{k}$: Satisfies both (but converges very slowly in practice)
- $\eta_k = \frac{1}{\sqrt{k}}$: Satisfies first, violates second (but better empirical performance)
- $\eta_k = \eta_0$ (constant): Violates second (converges to neighborhood of optimum)

```
Learning Rate Decay Comparison:

  ηₖ
   │
η₀ │●──────────────────────  Constant (no convergence guarantee)
   │ ●
   │  ●─────────────────     1/√k (theory optimal)
   │   ●●
   │     ●●──────           1/k (too slow)
   │       ●●●●──
   └──────────────────────── k
   1   10   100   1000

Practical ML: constant or step decay, ignore theory!
```

**ML Practice vs Theory:** Most practical deep learning uses constant learning rates with manual decay, violating convergence theory but achieving excellent empirical results. Theory-optimal $O(1/\sqrt{k})$ schedules are too aggressive initially and too conservative later.

---

## 5.4.6 Gradient Clipping

### Exploding Gradients Problem

In deep networks, especially RNNs, gradients can grow exponentially through backpropagation:

$$\frac{\partial \mathcal{L}}{\partial \theta_1} = \frac{\partial \mathcal{L}}{\partial h_T} \cdot \frac{\partial h_T}{\partial h_{T-1}} \cdots \frac{\partial h_2}{\partial h_1} \cdot \frac{\partial h_1}{\partial \theta_1}$$

If $\|\frac{\partial h_t}{\partial h_{t-1}}\| > 1$, gradients explode. This causes:
- Numerical overflow (NaN values)
- Catastrophic parameter updates
- Training instability

**Definition 5.4.7 (Gradient Clipping):**

Methods to bound gradient magnitudes before parameter updates:

1. **Clipping by Value:**
   $$g_k^{(i)} \leftarrow \max(-c, \min(c, g_k^{(i)}))$$
   Clip each gradient component to $[-c, c]$.

2. **Clipping by Norm:**
   $$g_k \leftarrow \begin{cases}
   g_k & \text{if } \|g_k\| \leq c \\
   \frac{c}{\|g_k\|} g_k & \text{if } \|g_k\| > c
   \end{cases}$$
   Rescale gradient to have maximum norm $c$.

**Example 5.4.8 (Gradient Clipping by Norm):**

Gradient $g = [3.0, 4.0]^T$ with clip threshold $c = 2.5$.

Step 1 -- Compute norm: $\|g\| = \sqrt{3^2 + 4^2} = \sqrt{25} = 5.0$

Step 2 -- Since $\|g\| = 5.0 > c = 2.5$, rescale:

$$\tilde{g} = \frac{c}{\|g\|} g = \frac{2.5}{5.0} [3.0, 4.0]^T = [1.5, 2.0]^T$$

Step 3 -- Verify: $\|\tilde{g}\| = \sqrt{1.5^2 + 2.0^2} = \sqrt{6.25} = 2.5 = c$ $\checkmark$

The direction is preserved ($\tilde{g} \propto g$) but the magnitude is halved. With $\eta = 0.1$, the parameter step shrinks from $0.5$ to $0.25$, preventing a destabilizing large update.

```
Gradient Clipping Visualization:

By Norm (spherical constraint):
     ĝ
     │    /g (clipped to circle)
     │   /
     │  ● ← original g
     │ /│\
     │/ │ \c
     ●──┼──●────
       │

Preserves direction, scales magnitude

By Value (box constraint):
     ĝ
     │    ●─── clipped
     │    │
     │    │● ← original g
     │    │
     │────┼────

Clips each dimension independently
```

**Theorem 5.4.6 (Effect on Convergence):**

Gradient clipping preserves convergence guarantees if:
$$c \geq \sqrt{d} \cdot \max_{\theta} \|\nabla \mathcal{L}(\theta)\|$$

where $d$ is dimension. In practice, $c = 1.0$ to $5.0$ is common.

**ML Connection:** Gradient clipping is essential for training RNNs and transformers. By-norm clipping ($c=1.0$) is standard for sequence models. Prevents exploding gradients while allowing vanishing gradients (addressed by architecture design like LSTM/GRU).

---

## 5.4.7 Distributed SGD

### Parallel Mini-batch SGD

With $N$ workers, compute gradients on different mini-batches in parallel, then aggregate.

```
Distributed Training Architecture:

Worker 1: [Data B₁] → [Compute g₁] ─┐
                                      │
Worker 2: [Data B₂] → [Compute g₂] ─┼→ [Aggregate] → [Update θ] → [Broadcast θ]
                                      │      ḡ
Worker N: [Data Bₙ] → [Compute gₙ] ─┘

Synchronous: All workers wait for all gradients
Asynchronous: Workers proceed independently
```

### Synchronous vs Asynchronous SGD

**Synchronous Data Parallel (SDP):**

$$\bar{g}_k = \frac{1}{N} \sum_{i=1}^N g_k^{(i)}, \quad \theta_{k+1} = \theta_k - \eta_k \bar{g}_k$$

All workers synchronize at each step.

**Properties:**
- ✓ Equivalent to large batch SGD (batch size $= N \cdot b$)
- ✓ Deterministic, reproducible
- ✗ Bottlenecked by slowest worker ("straggler problem")
- ✗ Requires fast interconnect for frequent communication

**Asynchronous SGD:**

Each worker $i$ independently:
1. Reads current parameters $\theta$
2. Computes gradient $g^{(i)}$
3. Updates shared parameters: $\theta \leftarrow \theta - \eta g^{(i)}$

**Properties:**
- ✓ No waiting for stragglers, high throughput
- ✓ Tolerant to hardware failures
- ✗ Stale gradients (computed at old parameters)
- ✗ Can diverge without careful learning rate tuning
- ✗ Non-deterministic

### Gradient Compression

Communication cost dominates when $d$ (parameter count) is large. Compress gradients before transmission.

**Definition 5.4.8 (Gradient Compression Methods):**

1. **Quantization:** Reduce precision from FP32 to INT8 or lower
   $$\tilde{g} = Q(g) = \text{round}\left(\frac{g - g_{\min}}{g_{\max} - g_{\min}} \cdot 255\right)$$

2. **Top-k Sparsification:** Send only $k$ largest magnitude components
   $$\tilde{g}^{(i)} = \begin{cases} g^{(i)} & \text{if } |g^{(i)}| \in \text{top-}k \\ 0 & \text{otherwise} \end{cases}$$

3. **Random Sparsification:** Send each component with probability $p$
   $$\tilde{g}^{(i)} = \begin{cases} g^{(i)}/p & \text{with probability } p \\ 0 & \text{with probability } 1-p \end{cases}$$

**Error Compensation:** Accumulate compression errors and add to next iteration:

$$e_{k+1} = g_k - \tilde{g}_k + e_k$$

This ensures $\sum_{k=0}^K \tilde{g}_k = \sum_{k=0}^K g_k$ (unbiased in expectation over iterations).

### Communication Efficiency

**Theorem 5.4.7 (Communication Complexity):**

Per iteration communication cost:
- **Uncompressed:** $O(d)$ per worker ($2d$ total: broadcast + reduce)
- **Top-k:** $O(k)$ per worker with $k \ll d$
- **Quantized:** $O(d/r)$ where $r$ is compression ratio (e.g., $r=4$ for FP32→INT8)

For modern transformers with $d \approx 10^{11}$ parameters (GPT-3 scale), compression is essential.

**Gradient Accumulation:** Simulate large batch on single GPU by accumulating gradients over multiple mini-batches before updating:

```
For k = 0, 1, 2, ...:
    g_acc ← 0
    For i = 1 to M:
        Sample mini-batch Bᵢ
        g_acc ← g_acc + (1/M) ∇L_Bᵢ(θ)
    θ ← θ - η g_acc
```

Effective batch size is $M \cdot b$. Memory footprint remains $O(b)$.

**ML Connection:** Large-scale training uses synchronous data parallelism (PyTorch DDP, DeepSpeed). Gradient accumulation enables training large models (LLaMA, GPT) on consumer hardware. Mixed-precision (FP16/BF16) provides 2x compression with minimal accuracy loss.

---

## Key Takeaways

1. **SGD Foundation:** Mini-batch gradients are unbiased but noisy estimators with variance $O(1/b)$

2. **Adaptive Methods:**
   - AdaGrad: Per-parameter learning rates, good for sparse features
   - RMSProp: Exponential averaging prevents vanishing learning rates
   - Adam: Combines momentum + adaptive learning rates, default for most tasks

3. **Learning Rate Schedules:**
   - Cosine annealing: Smooth decay for vision/transformers
   - Warmup: Essential for transformer stability
   - Step decay: Simple and effective for convex problems

4. **Convergence Theory:**
   - Convex: $O(1/\sqrt{K})$ with optimal learning rate
   - Non-convex: $O(1/\epsilon^4)$ iterations to find $\epsilon$-stationary point
   - Practice ignores theory: constant LR works best empirically

5. **Stability:**
   - Gradient clipping prevents exploding gradients (critical for RNNs)
   - Norm clipping preserves direction, value clipping per-dimension

6. **Scale:**
   - Distributed SGD: synchronous (deterministic) vs asynchronous (fast)
   - Compression: Top-k sparsification, quantization, error compensation
   - Gradient accumulation simulates large batches

---

## Exercises

### Computational (★)

**Exercise 5.4.1:** For a dataset with $n = 10,000$ samples and mini-batch size $b = 64$:
(a) How many SGD iterations per epoch?
(b) If each gradient evaluation costs $10^{-3}$ seconds, how long is one epoch?
(c) Compare to full-batch GD time per iteration.

**Exercise 5.4.2:** Implement SGD with momentum on $f(x,y) = x^2 + 10y^2$ starting from $(10, 10)$:
(a) Use learning rate $\eta = 0.1$, momentum $\beta = 0.9$
(b) Plot first 20 iterations
(c) Compare convergence to vanilla SGD

**Exercise 5.4.3:** Given gradient estimates $g_1 = [2.1, -1.9]$, $g_2 = [1.8, -2.2]$, $g_3 = [2.3, -1.7]$:
(a) Compute AdaGrad updates with $\eta = 1.0$, $\epsilon = 10^{-8}$ starting from $\theta_0 = [0, 0]$
(b) Compute RMSProp updates with $\eta = 0.1$, $\beta = 0.9$
(c) Which parameter has larger learning rate after 3 iterations?

### Theoretical (★★)

**Exercise 5.4.4:** Prove that mini-batch gradient with replacement is unbiased: $\mathbb{E}[g_B] = \nabla \mathcal{L}$.

**Exercise 5.4.5:** Show that for Adam's bias correction, $\lim_{k \to \infty} \frac{1}{1 - \beta^k} = 1$.

**Exercise 5.4.6:** Prove variance reduction: if $\mathbb{E}[g] = \nabla \mathcal{L}$ and we average $m$ independent gradient estimates $\bar{g} = \frac{1}{m}\sum_{i=1}^m g_i$, then $\text{Var}[\bar{g}] = \frac{1}{m}\text{Var}[g]$.

**Exercise 5.4.7:** Show that learning rate $\eta_k = \frac{1}{k}$ satisfies Robbins-Monro conditions but $\eta_k = \frac{1}{k^2}$ does not.

**Exercise 5.4.8:** For gradient clipping by norm with threshold $c$, prove that the clipped gradient $\tilde{g}$ satisfies $\tilde{g}^T \nabla \mathcal{L} \geq 0$ (descent direction preserved).

### Advanced (★★★)

**Exercise 5.4.9:** Consider strongly convex $\mathcal{L}$ with condition number $\kappa = L/\mu$.
(a) Derive the optimal constant learning rate for SGD
(b) Show that iteration complexity is $O(\kappa/\epsilon)$ to reach $\epsilon$-accuracy
(c) Compare to GD complexity $O(\kappa \log(1/\epsilon))$

**Exercise 5.4.10:** Analyze Adam's convergence:
(a) Show that without bias correction, $\mathbb{E}[m_k]$ is biased in early iterations
(b) Prove that bias correction gives $\mathbb{E}[\hat{m}_k] = \mathbb{E}[\nabla \mathcal{L}(\theta_k)]$ under stationary gradient distribution
(c) Discuss why Adam can fail to converge (AMSGrad fixes this - research!)

**Exercise 5.4.11:** Variance reduction via control variates:
(a) Given full gradient $\nabla \mathcal{L}(\theta_0)$ at initial point, construct estimator $\tilde{g}_k = g_k - g_k^0 + \nabla \mathcal{L}(\theta_0)$ where $g_k^0$ is stochastic gradient at $\theta_0$
(b) Prove this is unbiased: $\mathbb{E}[\tilde{g}_k] = \nabla \mathcal{L}(\theta_k)$
(c) Show variance can be lower than standard SGD if $\theta_k$ is close to $\theta_0$

**Exercise 5.4.12:** Distributed SGD with $N$ workers and asynchronous updates:
(a) Model staleness: worker sees parameters $\theta_{k-\tau}$ where $\tau$ is delay
(b) Show expected update is $\mathbb{E}[\theta_{k+1} - \theta_k] \approx -\eta (\nabla \mathcal{L}(\theta_k) + O(\tau \eta L))$
(c) Derive learning rate condition $\eta = O(1/\tau)$ for convergence

---

## Related Topics

**Connections to Other Chapters:**
- **Chapter 5.1 (Gradient Descent):** Deterministic optimization foundation
- **Chapter 5.2 (Convex Optimization):** Convergence guarantees for convex case
- **Chapter 5.3 (Constrained Optimization):** Projected SGD, prox-operators with SGD
- **Chapter 6 (Probability Theory):** Variance, expectation, unbiased estimators

**Advanced Topics:**
- **Variance-reduced methods:** SVRG, SAGA, control variates
- **Second-order stochastic methods:** Stochastic L-BFGS, natural gradient
- **Non-smooth optimization:** Subgradient methods, proximal SGD
- **Online learning:** Regret bounds, online-to-batch conversion
- **Federated learning:** Privacy-preserving distributed optimization
- **Meta-learning:** Learning learning rates and optimization algorithms

**ML Applications:**
- **Deep learning:** Neural network training, backpropagation + SGD
- **Reinforcement learning:** Policy gradient methods use stochastic optimization
- **Generative models:** GAN training uses simultaneous SGD on two objectives
- **Large language models:** Mixed-precision training, ZeRO optimizer, pipeline parallelism

**Further Reading:**
- Bottou, L. (2010). "Large-Scale Machine Learning with Stochastic Gradient Descent"
- Kingma & Ba (2014). "Adam: A Method for Stochastic Optimization"
- Robbins & Monro (1951). "A Stochastic Approximation Method"
- Ruder, S. (2016). "An Overview of Gradient Descent Optimization Algorithms"
