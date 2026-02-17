# Chapter 5.5: Variational Methods

## Prerequisites

- **Probability Theory**: Conditional probability, Bayes' theorem, expectations
- **Information Theory**: KL divergence, entropy, mutual information
- **Calculus**: Partial derivatives, gradients, Lagrange multipliers
- **Linear Algebra**: Matrix operations, positive definite matrices
- **Optimization**: Gradient descent, coordinate ascent
- **Previous Chapters**: Maximum likelihood estimation (5.1), gradient-based optimization (5.2)

## Learning Objectives

After completing this chapter, you should be able to:

1. Understand the motivation for variational inference in intractable probabilistic models
2. Derive and interpret the Evidence Lower Bound (ELBO)
3. Apply mean-field approximation to factorize complex posteriors
4. Explain the Variational Autoencoder (VAE) framework and reparameterization trick
5. Compare forward and reverse KL divergence in variational optimization
6. Connect Expectation-Maximization to variational principles
7. Understand amortized inference using neural networks

---

## 5.5.1 Introduction to Variational Inference

**Definition 5.5.1 (Variational Inference)**
Variational inference is a deterministic optimization approach to approximate intractable probability distributions by finding the closest tractable distribution within a restricted family.

### The Inference Problem

Given observed data $\mathcal{D} = \{x_1, \ldots, x_n\}$ and a probabilistic model with latent variables $z$ and parameters $\theta$, we want to compute the posterior distribution:

$$p(\theta, z | \mathcal{D}) = \frac{p(\mathcal{D} | \theta, z) p(\theta, z)}{p(\mathcal{D})}$$

**The Challenge**: Computing $p(\mathcal{D}) = \int \int p(\mathcal{D} | \theta, z) p(\theta, z) \, dz \, d\theta$ requires integrating over all possible values of latent variables and parameters, which is often **intractable** for complex models.

### Variational Approach

Instead of computing $p(\theta, z | \mathcal{D})$ directly, we:

1. Define a family of tractable distributions $\mathcal{Q}$
2. Find $q^*(\theta, z) \in \mathcal{Q}$ that is "closest" to $p(\theta, z | \mathcal{D})$
3. Use $q^*(\theta, z)$ as a proxy for the true posterior

```
True Posterior          Variational Approximation
p(θ,z|D)                      q*(θ,z)
  (complex)                   (simple)

     |                           |
     v                           v
Intractable         KL Minimization      Tractable
Integration    ←─────────────────────→   Expectation
```

**Definition 5.5.2 (Variational Family)**
A variational family $\mathcal{Q}$ is a set of tractable probability distributions parameterized by variational parameters $\lambda$:

$$\mathcal{Q} = \{q_\lambda(z) : \lambda \in \Lambda\}$$

Common choices include Gaussian families, mean-field families, and neural network-parameterized distributions.

---

## 5.5.2 Evidence Lower Bound (ELBO)

### Derivation from KL Divergence

Our goal is to minimize the KL divergence between $q(z)$ and $p(z|x)$:

$$\text{KL}(q(z) \| p(z|x)) = \mathbb{E}_{q(z)}\left[\log \frac{q(z)}{p(z|x)}\right]$$

Expanding using Bayes' theorem $p(z|x) = \frac{p(x,z)}{p(x)}$:

$$\begin{align}
\text{KL}(q(z) \| p(z|x)) &= \mathbb{E}_{q(z)}\left[\log \frac{q(z)}{p(x,z)/p(x)}\right] \\
&= \mathbb{E}_{q(z)}[\log q(z)] - \mathbb{E}_{q(z)}[\log p(x,z)] + \log p(x) \\
&= -\left(\mathbb{E}_{q(z)}[\log p(x,z)] - \mathbb{E}_{q(z)}[\log q(z)]\right) + \log p(x)
\end{align}$$

Rearranging:

$$\log p(x) = \underbrace{\mathbb{E}_{q(z)}[\log p(x,z)] - \mathbb{E}_{q(z)}[\log q(z)]}_{\text{ELBO}(\lambda)} + \text{KL}(q(z) \| p(z|x))$$

**Definition 5.5.3 (Evidence Lower Bound)**
The Evidence Lower Bound (ELBO) is defined as:

$$\mathcal{L}(q) = \mathbb{E}_{q(z)}[\log p(x,z)] - \mathbb{E}_{q(z)}[\log q(z)]$$

Equivalently:

$$\mathcal{L}(q) = \mathbb{E}_{q(z)}[\log p(x|z)] - \text{KL}(q(z) \| p(z))$$

**Theorem 5.5.1 (ELBO as Lower Bound)**
For any distribution $q(z)$:

$$\log p(x) \geq \mathcal{L}(q)$$

with equality if and only if $q(z) = p(z|x)$.

*Proof*: Since $\text{KL}(q(z) \| p(z|x)) \geq 0$ with equality iff $q(z) = p(z|x)$ almost everywhere. $\square$

**Example 5.5.1 (ELBO Computation for a Simple Model)**
Consider a model where $z \sim \mathcal{N}(0, 1)$ and $x | z \sim \mathcal{N}(z, 1)$, with observed $x = 1$. Use variational distribution $q(z) = \mathcal{N}(\mu_q, 1)$.

The joint is $\log p(x, z) = \log p(x|z) + \log p(z) = -\frac{1}{2}(x - z)^2 - \frac{1}{2}z^2 + \text{const}$.

$$\text{ELBO} = \mathbb{E}_{q(z)}[\log p(x, z)] - \mathbb{E}_{q(z)}[\log q(z)]$$

Computing each term with $q(z) = \mathcal{N}(\mu_q, 1)$:

$$\mathbb{E}_q[\log p(x, z)] = -\frac{1}{2}\mathbb{E}_q[(1-z)^2] - \frac{1}{2}\mathbb{E}_q[z^2] + C_1 = -\frac{1}{2}[(1-\mu_q)^2 + 1] - \frac{1}{2}[\mu_q^2 + 1] + C_1$$

The entropy $-\mathbb{E}_q[\log q(z)] = \frac{1}{2}\log(2\pi e)$ is constant in $\mu_q$.

Maximizing over $\mu_q$: $\frac{\partial}{\partial \mu_q}\text{ELBO} = (1 - \mu_q) - \mu_q = 0 \implies \mu_q^* = \frac{1}{2}$.

The true posterior is $p(z|x=1) = \mathcal{N}(\frac{1}{2}, \frac{1}{2})$, so $q^*$ matches the posterior mean exactly (but not the variance, since we fixed $\sigma_q^2 = 1$).

**Example 5.5.2 (ELBO $\leq$ log p(x) Numerically)**
Continuing Example 5.5.1, the marginal is $p(x) = \mathcal{N}(x | 0, 2)$, so:

$$\log p(x = 1) = -\frac{1}{2}\log(4\pi) - \frac{1}{4} \approx -1.58$$

At the optimal $q^*(z) = \mathcal{N}(\frac{1}{2}, 1)$:

$$\text{ELBO}(q^*) = -\frac{1}{2}[(1 - \tfrac{1}{2})^2 + 1] - \frac{1}{2}[(\tfrac{1}{2})^2 + 1] + \frac{1}{2}\log(2\pi e) + C_1$$

After evaluation: $\text{ELBO}(q^*) \approx -1.74$. The gap is $\text{KL}(q^* \| p(z|x)) \approx 0.16 > 0$, confirming $\text{ELBO} = -1.74 < -1.58 = \log p(x)$. Equality would require $q(z) = \mathcal{N}(\frac{1}{2}, \frac{1}{2})$, matching the true posterior.

### Interpretation of ELBO

```
log p(x) (Evidence)
     ^
     |
     |  KL(q||p)        ← Gap to minimize
     |    (≥ 0)
     |
     +-------------------
     |
     |  ELBO(q)          ← Lower bound to maximize
     |
     +-------------------
```

The ELBO decomposes into two terms:

$$\mathcal{L}(q) = \underbrace{\mathbb{E}_{q(z)}[\log p(x|z)]}_{\text{Expected likelihood}} - \underbrace{\text{KL}(q(z) \| p(z))}_{\text{Regularization}}$$

- **Expected likelihood**: How well does $q$ explain the data?
- **Regularization**: How close is $q$ to the prior $p(z)$?

**Maximizing ELBO** is equivalent to **minimizing KL divergence** between $q(z)$ and $p(z|x)$.

---

## 5.5.3 Mean-Field Approximation

**Definition 5.5.4 (Mean-Field Variational Family)**
The mean-field approximation assumes the latent variables factorize:

$$q(z) = \prod_{i=1}^m q_i(z_i)$$

where $z = (z_1, \ldots, z_m)$ and each $q_i(z_i)$ is optimized independently.

### Coordinate Ascent Variational Inference (CAVI)

**Theorem 5.5.2 (Mean-Field Optimal Solution)**
Under mean-field approximation, the optimal distribution for factor $j$ is:

$$\log q_j^*(z_j) = \mathbb{E}_{q_{-j}}[\log p(x, z)] + \text{const}$$

where $q_{-j} = \prod_{i \neq j} q_i(z_i)$ denotes all factors except $j$.

*Proof sketch*: Set $\frac{\partial \mathcal{L}}{\partial q_j} = 0$ under the constraint $\int q_j(z_j) dz_j = 1$ using calculus of variations. $\square$

### CAVI Algorithm

```
Algorithm: Coordinate Ascent Variational Inference
─────────────────────────────────────────────────
Input: Data x, model p(x,z), tolerance ε
Initialize: q_i(z_i) for all i

Repeat until convergence:
    For j = 1 to m:
        1. Compute expectation over q_{-j}:
           log q_j*(z_j) = E_{q_{-j}}[log p(x,z)] + const

        2. Normalize to get valid distribution:
           q_j*(z_j) ← exp(E_{q_{-j}}[log p(x,z)]) / Z_j

    3. Compute ELBO(q)

    If |ELBO_new - ELBO_old| < ε:
        break

Output: q*(z) = ∏_i q_i*(z_i)
```

**Example 5.5.3 (Gaussian Mean-Field)**
For a Gaussian graphical model, if the true posterior is:

$$p(z|x) \propto \exp\left(-\frac{1}{2} z^\top \Sigma^{-1} z + \mu^\top \Sigma^{-1} z\right)$$

The mean-field approximation $q(z) = \prod_{i=1}^m \mathcal{N}(z_i | \mu_i, \sigma_i^2)$ yields diagonal covariance, losing correlations between variables.

**Example 5.5.4 (Mean-Field Factorization — Concrete 2D Case)**
Suppose the true posterior over $(z_1, z_2)$ is bivariate Gaussian:

$$p(z_1, z_2 | x) = \mathcal{N}\left(\begin{pmatrix} 1 \\ 2 \end{pmatrix}, \begin{pmatrix} 1 & 0.8 \\ 0.8 & 1 \end{pmatrix}\right)$$

Mean-field assumes $q(z_1, z_2) = q_1(z_1) q_2(z_2)$. Using CAVI:

$$\log q_1^*(z_1) = \mathbb{E}_{q_2}[\log p(z_1, z_2 | x)] + \text{const}$$

For this Gaussian, the optimal mean-field factors are $q_1^*(z_1) = \mathcal{N}(1, 1)$ and $q_2^*(z_2) = \mathcal{N}(2, 1)$. The marginal means are recovered exactly, but the correlation $\rho = 0.8$ is completely lost — the mean-field approximation treats $z_1$ and $z_2$ as independent. This underestimates the posterior variance in correlated directions.

---

## 5.5.4 Variational Autoencoders (VAEs)

### Problem Setup

Given data $x \in \mathbb{R}^d$ generated from latent code $z \in \mathbb{R}^k$:

- **Generative model**: $p_\theta(x|z)$ (decoder), $p(z)$ (prior, typically $\mathcal{N}(0, I)$)
- **Inference model**: $q_\phi(z|x)$ (encoder, approximate posterior)

**Definition 5.5.5 (Variational Autoencoder)**
A VAE consists of:

1. **Encoder**: Neural network parameterizing $q_\phi(z|x) = \mathcal{N}(z | \mu_\phi(x), \Sigma_\phi(x))$
2. **Decoder**: Neural network parameterizing $p_\theta(x|z)$

### VAE Architecture

```
                    Variational Autoencoder
                    ═══════════════════════

Input x              Latent Space           Reconstructed x̃
  │                                              ▲
  │                                              │
  v                                              │
┌─────────────┐                           ┌─────────────┐
│   Encoder   │                           │   Decoder   │
│   Network   │                           │   Network   │
│             │                           │             │
│  (NN: φ)    │                           │  (NN: θ)    │
└─────────────┘                           └─────────────┘
  │        │                                     ▲
  │        │                                     │
  v        v                                     │
 μ(x)    σ(x)         z ~ q_φ(z|x)             │
  │        │          ═══════════               │
  │        │               │                    │
  │        └───────┐       │                    │
  │                │       v                    │
  │                │   ┌───────┐                │
  │                └──>│Sample │                │
  │                    │  z    │────────────────┘
  └───────────────────>│ε~N(0,I)│
                       └───────┘
                    Reparameterization
                    z = μ(x) + σ(x)⊙ε
```

### The ELBO for VAEs

For a single data point $x$:

$$\mathcal{L}(\theta, \phi; x) = \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] - \text{KL}(q_\phi(z|x) \| p(z))$$

- **Reconstruction term**: $\mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)]$ encourages accurate reconstruction
- **Regularization term**: $\text{KL}(q_\phi(z|x) \| p(z))$ keeps latent codes close to prior

### The Reparameterization Trick

**Problem**: Cannot backpropagate through sampling operation $z \sim q_\phi(z|x)$.

**Definition 5.5.6 (Reparameterization Trick)**
Express $z$ as a deterministic function of $\phi$ and noise $\epsilon$:

$$z = \mu_\phi(x) + \sigma_\phi(x) \odot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

where $\odot$ is element-wise multiplication.

**Before** (non-differentiable):
```
x → [Encoder] → z ~ N(μ,σ²) → [Decoder] → x̃
                    ↑
               Cannot backprop!
```

**After** (differentiable):
```
x → [Encoder] → μ,σ → z = μ + σ⊙ε → [Decoder] → x̃
                         ↑ε~N(0,I)
                    Can backprop through μ,σ!
```

**Theorem 5.5.3 (Reparameterization Gradient)**
The gradient of an expectation can be rewritten as:

$$\nabla_\phi \mathbb{E}_{q_\phi(z|x)}[f(z)] = \mathbb{E}_{\epsilon \sim p(\epsilon)}\left[\nabla_\phi f(g(\phi, \epsilon, x))\right]$$

where $z = g(\phi, \epsilon, x)$ is the reparameterization.

**Example 5.5.5 (Reparameterization Trick — Specific Case)**
Let the encoder output $\mu_\phi(x) = 3$ and $\sigma_\phi(x) = 0.5$ for a 1D latent space. We draw $\epsilon \sim \mathcal{N}(0, 1)$.

Suppose $\epsilon = 1.2$ (one sample). Then:

$$z = \mu + \sigma \cdot \epsilon = 3 + 0.5 \times 1.2 = 3.6$$

The gradients flow through $\mu$ and $\sigma$:

$$\frac{\partial z}{\partial \mu} = 1, \quad \frac{\partial z}{\partial \sigma} = \epsilon = 1.2$$

If the loss at this $z$ is $f(z) = z^2$, then $\nabla_\mu f = 2z \cdot 1 = 7.2$ and $\nabla_\sigma f = 2z \cdot \epsilon = 8.64$. Without reparameterization, we could not compute these gradients because sampling $z \sim \mathcal{N}(3, 0.25)$ is non-differentiable.

### VAE Training Algorithm

```
Algorithm: Train Variational Autoencoder
─────────────────────────────────────────
Input: Dataset {x_1,...,x_n}, learning rate α

Initialize: θ, φ randomly

For each epoch:
    For each minibatch {x_i}:
        1. Encode: compute μ_φ(x_i), σ_φ(x_i)

        2. Sample: ε ~ N(0,I)
                   z_i = μ_φ(x_i) + σ_φ(x_i) ⊙ ε

        3. Decode: compute p_θ(x_i|z_i)

        4. Compute ELBO:
           L = E_q[log p_θ(x|z)] - KL(q_φ(z|x)||p(z))

        5. Update parameters:
           θ ← θ + α∇_θ L
           φ ← φ + α∇_φ L

Output: Trained encoder q_φ, decoder p_θ
```

**Closed-form KL for Gaussians**:
When $q_\phi(z|x) = \mathcal{N}(\mu, \sigma^2 I)$ and $p(z) = \mathcal{N}(0, I)$:

$$\text{KL}(q_\phi(z|x) \| p(z)) = \frac{1}{2} \sum_{j=1}^k \left(\mu_j^2 + \sigma_j^2 - \log \sigma_j^2 - 1\right)$$

**Example 5.5.6 (KL Divergence Between Two Gaussians — Numerical)**
Let $q(z) = \mathcal{N}(z | \mu = 2, \sigma^2 = 0.5)$ and $p(z) = \mathcal{N}(0, 1)$ (standard normal prior), with $k = 1$ dimension.

$$\text{KL}(q \| p) = \frac{1}{2}\left(\mu^2 + \sigma^2 - \log \sigma^2 - 1\right) = \frac{1}{2}\left(4 + 0.5 - \log 0.5 - 1\right)$$

$$= \frac{1}{2}(4 + 0.5 + 0.693 - 1) = \frac{1}{2}(4.193) = 2.097$$

This means $q$ is far from the prior — the mean shift ($\mu = 2$) dominates. If the encoder instead output $\mu = 0, \sigma^2 = 1$, then $\text{KL} = \frac{1}{2}(0 + 1 - 0 - 1) = 0$, confirming $q = p$ gives zero divergence.

For a 2D latent space with $\mu = (1, -1)$ and $\sigma^2 = (0.5, 2.0)$:

$$\text{KL} = \frac{1}{2}\left[(1 + 0.5 + 0.693 - 1) + (1 + 2 - 0.693 - 1)\right] = \frac{1}{2}[1.193 + 1.307] = 1.25$$

---

## 5.5.5 KL Divergence Minimization

### Forward vs. Reverse KL

**Definition 5.5.7 (KL Divergence Directions)**

- **Forward KL**: $\text{KL}(p \| q) = \mathbb{E}_{p(z)}[\log p(z) - \log q(z)]$
- **Reverse KL**: $\text{KL}(q \| p) = \mathbb{E}_{q(z)}[\log q(z) - \log p(z)]$

In variational inference, we minimize **reverse KL**: $\text{KL}(q(z) \| p(z|x))$ because:
1. We can compute expectations under $q(z)$ (we control it)
2. We don't need to normalize $p(z|x)$ (use unnormalized $p(x,z)$ instead)

**Example 5.5.7 (Forward vs. Reverse KL — Numerical Comparison)**
Let $p(z) = \mathcal{N}(0, 1)$ and $q(z) = \mathcal{N}(1, 4)$ (shifted, wider Gaussian).

**Forward KL** $\text{KL}(p \| q)$: expectations under $p(z) = \mathcal{N}(0, 1)$:

$$\text{KL}(p \| q) = \frac{1}{2}\left[\frac{\sigma_p^2}{\sigma_q^2} + \frac{(\mu_q - \mu_p)^2}{\sigma_q^2} - 1 + \log\frac{\sigma_q^2}{\sigma_p^2}\right] = \frac{1}{2}\left[\frac{1}{4} + \frac{1}{4} - 1 + \log 4\right] = \frac{1}{2}(0.886) = 0.443$$

**Reverse KL** $\text{KL}(q \| p)$: expectations under $q(z) = \mathcal{N}(1, 4)$:

$$\text{KL}(q \| p) = \frac{1}{2}\left[\frac{\sigma_q^2}{\sigma_p^2} + \frac{(\mu_p - \mu_q)^2}{\sigma_p^2} - 1 + \log\frac{\sigma_p^2}{\sigma_q^2}\right] = \frac{1}{2}\left[4 + 1 - 1 + \log\frac{1}{4}\right] = \frac{1}{2}(2.614) = 1.307$$

Note $\text{KL}(p \| q) \neq \text{KL}(q \| p)$ — KL divergence is asymmetric. The reverse KL is larger because $q$ places mass in regions where $p$ has low density, incurring heavy penalty.

### Mode-Seeking vs. Mode-Covering

```
True Distribution p(z)        Forward KL: KL(p||q)
(multimodal)                  (mode-averaging)

    *   *                         ___
   * * * *                       /   \
  * * * * *       ≈            /     \
   * * * *       q(z)         /       \
    *   *                    /         \
                            /___________\

                            Covers both modes
                            (but fits poorly)


True Distribution p(z)        Reverse KL: KL(q||p)
(multimodal)                  (mode-seeking)

    *   *                        *
   * * * *                      ***
  * * * * *       ≈            *****
   * * * *       q(z)           ***
    *   *                        *

                            Locks onto one mode
                            (fits well locally)
```

**Theorem 5.5.4 (Behavior of KL Divergence)**

1. **Reverse KL** $\text{KL}(q \| p)$: **Mode-seeking**
   - When $p(z) > 0$ but $q(z) \approx 0$: small penalty
   - When $q(z) > 0$ but $p(z) \approx 0$: **infinite penalty**
   - Result: $q$ avoids regions where $p$ has low density

2. **Forward KL** $\text{KL}(p \| q)$: **Mode-covering**
   - When $p(z) > 0$ but $q(z) \approx 0$: **infinite penalty**
   - When $q(z) > 0$ but $p(z) \approx 0$: small penalty
   - Result: $q$ spreads mass to cover all modes of $p$

**Practical Implications**:

- **Reverse KL** (used in variational inference): May miss modes of true posterior, but provides sharp approximations
- **Forward KL** (used in expectation propagation): Covers all modes but may include regions with low probability

---

## 5.5.6 Expectation-Maximization Algorithm

### The EM Framework

**Definition 5.5.8 (Expectation-Maximization)**
Given observed data $x$ and latent variables $z$, EM alternates between:

- **E-step**: Compute posterior $p(z|x, \theta^{(t)})$ given current parameters
- **M-step**: Maximize expected complete log-likelihood:

$$\theta^{(t+1)} = \arg\max_\theta \mathbb{E}_{p(z|x,\theta^{(t)})}[\log p(x, z | \theta)]$$

### Connection to Variational Inference

**Theorem 5.5.5 (EM as Coordinate Ascent on ELBO)**
EM can be viewed as coordinate ascent on the ELBO with respect to $q(z)$ and $\theta$:

$$\mathcal{L}(q, \theta) = \mathbb{E}_{q(z)}[\log p(x, z | \theta)] - \mathbb{E}_{q(z)}[\log q(z)]$$

- **E-step**: $q^{(t+1)} = \arg\max_q \mathcal{L}(q, \theta^{(t)})$ → optimal $q^{(t+1)} = p(z|x, \theta^{(t)})$
- **M-step**: $\theta^{(t+1)} = \arg\max_\theta \mathcal{L}(q^{(t+1)}, \theta)$

```
EM Algorithm Flow
═════════════════

Initialize θ⁽⁰⁾
     │
     v
┌────────────────────┐
│   E-Step           │
│ q⁽ᵗ⁺¹⁾ = p(z|x,θ⁽ᵗ⁾)│  ← Exact posterior
└────────────────────┘   (not approximation!)
     │
     v
┌────────────────────┐
│   M-Step           │
│ θ⁽ᵗ⁺¹⁾ = argmax L  │  ← Maximize ELBO
└────────────────────┘
     │
     v
  Converged? ───No──→ Repeat
     │
    Yes
     v
  Output θ*
```

**Key Difference from Variational Inference**:

- **EM**: E-step computes **exact posterior** $p(z|x,\theta)$ (tractable in specific models)
- **Variational Inference**: Uses **approximate posterior** $q(z)$ from restricted family (when exact is intractable)

### Example: Gaussian Mixture Model

For GMM with $K$ components, latent indicators $z_i \in \{1, \ldots, K\}$:

**E-step**: Compute responsibilities
$$\gamma_{ik} = p(z_i = k | x_i, \theta^{(t)}) = \frac{\pi_k \mathcal{N}(x_i | \mu_k, \Sigma_k)}{\sum_{j=1}^K \pi_j \mathcal{N}(x_i | \mu_j, \Sigma_j)}$$

**M-step**: Update parameters
$$\mu_k = \frac{\sum_{i=1}^n \gamma_{ik} x_i}{\sum_{i=1}^n \gamma_{ik}}, \quad \pi_k = \frac{1}{n}\sum_{i=1}^n \gamma_{ik}$$

**Example 5.5.8 (Variational EM — One Complete Iteration)**
Consider a 1D GMM with $K = 2$ components. Data: $x = \{1, 2, 5, 6\}$, $n = 4$.

**Initial parameters**: $\mu_1^{(0)} = 0, \mu_2^{(0)} = 4, \sigma_1^2 = \sigma_2^2 = 2, \pi_1 = \pi_2 = 0.5$.

**E-step**: Compute responsibilities $\gamma_{ik} = \frac{\pi_k \mathcal{N}(x_i | \mu_k, \sigma_k^2)}{\sum_j \pi_j \mathcal{N}(x_i | \mu_j, \sigma_j^2)}$.

For $x_1 = 1$: $\mathcal{N}(1|0,2) = 0.242$, $\mathcal{N}(1|4,2) = 0.033$, so $\gamma_{1,1} = \frac{0.242}{0.242 + 0.033} = 0.88$, $\gamma_{1,2} = 0.12$.

For $x_3 = 5$: $\mathcal{N}(5|0,2) = 0.0001$, $\mathcal{N}(5|4,2) = 0.242$, so $\gamma_{3,1} = 0.0004$, $\gamma_{3,2} = 0.9996$.

(Similarly for $x_2$ and $x_4$.)

**M-step**: Update means:

$$\mu_1^{(1)} = \frac{0.88 \cdot 1 + 0.61 \cdot 2 + 0.0004 \cdot 5 + 0.0001 \cdot 6}{0.88 + 0.61 + 0.0004 + 0.0001} \approx \frac{2.10}{1.49} \approx 1.41$$

$$\mu_2^{(1)} = \frac{0.12 \cdot 1 + 0.39 \cdot 2 + 0.9996 \cdot 5 + 0.9999 \cdot 6}{0.12 + 0.39 + 0.9996 + 0.9999} \approx \frac{11.90}{2.51} \approx 4.74$$

The means moved toward the natural clusters $\{1, 2\}$ and $\{5, 6\}$. Iterating converges to $\mu_1^* \approx 1.5$ and $\mu_2^* \approx 5.5$.

---

## 5.5.7 Amortized Inference

**Definition 5.5.9 (Amortized Inference)**
Amortized inference uses a parametric function (e.g., neural network) to directly map observations to variational parameters:

$$x \xrightarrow{f_\phi} \text{parameters of } q_\phi(z|x)$$

instead of optimizing variational parameters separately for each data point.

### Comparison with Classical Variational Inference

**Classical Variational Inference**:
- Optimize separate $\lambda_i$ for each data point $x_i$
- Computational cost: $O(n \cdot |\lambda|)$ parameters for $n$ data points
- Each inference requires optimization loop

**Amortized Inference**:
- Learn single function $f_\phi$ mapping $x \to \lambda$
- Computational cost: $O(|\phi|)$ parameters (independent of $n$)
- Inference is single forward pass: $\lambda_i = f_\phi(x_i)$

```
Classical VI                 Amortized VI
════════════                 ═══════════

x₁ → optimize λ₁             x₁ ──┐
x₂ → optimize λ₂             x₂ ──┤
x₃ → optimize λ₃             x₃ ──┼──→ f_φ → λ
...                          ... ──┤   (shared)
xₙ → optimize λₙ             xₙ ──┘

n inference problems         1 learned function
```

### Benefits of Amortization

**Theorem 5.5.6 (Amortization Gap)**
Let $\mathcal{L}_{\text{opt}}(x)$ be the ELBO with optimal variational parameters, and $\mathcal{L}_{\text{amort}}(x)$ be the ELBO with amortized parameters. The amortization gap is:

$$\Delta(x) = \mathcal{L}_{\text{opt}}(x) - \mathcal{L}_{\text{amort}}(x) \geq 0$$

Trade-off:
- **Amortization gap**: May not achieve optimal $q$ for each $x$
- **Generalization**: Learns to infer $z$ from $x$ across entire data distribution
- **Speed**: Fast inference for new data (no optimization required)

### Neural Network Parameterization

For Gaussian $q_\phi(z|x) = \mathcal{N}(\mu_\phi(x), \Sigma_\phi(x))$:

$$\mu_\phi(x) = W_\mu h + b_\mu, \quad \log \sigma_\phi^2(x) = W_\sigma h + b_\sigma$$

where $h = \text{ReLU}(W_1 x + b_1)$ is a hidden representation.

**Why log-variance?** Ensures $\sigma^2 > 0$ without constraints.

### Applications Beyond VAEs

1. **Bayesian Neural Networks**: Approximate posterior over weights $q_\phi(w|x)$
2. **Normalizing Flows**: Use invertible neural networks to define flexible $q_\phi(z|x)$
3. **Inverse Graphics**: Infer 3D scene parameters from 2D images
4. **Probabilistic Programming**: Automatic inference in complex generative models

---

## 5.5.8 ML Connections and Extensions

### Variational Methods in Machine Learning

**Latent Dirichlet Allocation (LDA)**:
- Topic modeling with variational inference
- Mean-field approximation: $q(\theta, z, \beta) = q(\theta) q(z) q(\beta)$
- CAVI updates for document-topic and topic-word distributions

**Bayesian Deep Learning**:
- Variational inference over neural network weights
- Dropout as approximate variational inference (Gal & Ghahramani, 2016)
- Uncertainty quantification in predictions

**Generative Models**:
- VAEs for image, text, audio generation
- Conditional VAEs: $q_\phi(z|x, y)$, $p_\theta(x|z, y)$ with labels $y$
- Hierarchical VAEs: Multiple stochastic layers

### Extensions of Variational Families

**Definition 5.5.10 (Normalizing Flow)**
A normalizing flow transforms a simple base distribution $q_0(z_0)$ through a sequence of invertible mappings $f_1, \ldots, f_K$:

$$z_K = f_K \circ \cdots \circ f_1(z_0), \quad q_K(z_K) = q_0(z_0) \left|\det \frac{\partial f^{-1}}{\partial z_K}\right|$$

Allows flexible variational families while maintaining tractable densities.

**Structured Variational Inference**:
- Beyond mean-field: capture correlations
- Copula variational inference
- Mixture of variational posteriors

**Black-Box Variational Inference**:
- Score function gradient estimator (REINFORCE)
- Applicable when reparameterization trick unavailable
- Higher variance, requires control variates

### Practical Considerations

**Posterior Collapse in VAEs**:
- Decoder ignores latent code, $q(z|x) \approx p(z)$
- Solutions: KL annealing, stronger decoders, auxiliary losses

**Optimization Challenges**:
- ELBO is non-convex in $\theta, \phi$
- Local optima, sensitivity to initialization
- Careful learning rate scheduling

**Evaluating Variational Approximations**:
- ELBO only provides lower bound
- Importance-weighted bounds: $\log p(x) \approx \log \frac{1}{K}\sum_{k=1}^K \frac{p(x, z_k)}{q(z_k)}$
- Compare to ground truth when available (synthetic data)

---

## Summary

**Key Takeaways**:

1. **Variational inference** approximates intractable posteriors by optimization over tractable distributions
2. **ELBO** decomposes into reconstruction and regularization terms, lower bounds the marginal likelihood
3. **Mean-field approximation** factorizes posterior, enabling coordinate ascent but losing correlations
4. **VAEs** combine neural networks with variational inference via the reparameterization trick
5. **Reverse KL** is mode-seeking, leading to potentially missed modes but sharper approximations
6. **EM algorithm** is coordinate ascent on ELBO with exact E-step (when tractable)
7. **Amortized inference** learns a single inference network, trading optimality for speed and generalization

**Connections**:
- Optimization: ELBO maximization as constrained optimization
- Information theory: KL divergence as optimization objective
- Deep learning: Neural networks as flexible function approximators for $q_\phi(z|x)$
- Bayesian inference: Approximating intractable posteriors

---

## Exercises

### Conceptual Understanding (★)

**Exercise 5.5.1**: Explain why we cannot directly minimize $\text{KL}(q(z) \| p(z|x))$ but can maximize the ELBO.

**Exercise 5.5.2**: Consider a multimodal posterior with two separated modes. Draw a sketch showing how forward KL vs. reverse KL would approximate this distribution.

**Exercise 5.5.3**: Why is the ELBO called a "lower bound"? What is it a lower bound of?

**Exercise 5.5.4**: In the mean-field approximation $q(z) = \prod_i q_i(z_i)$, what information about the true posterior is lost?

### Derivations and Proofs (★★)

**Exercise 5.5.5**: Derive the ELBO starting from $\log p(x) = \log \int p(x,z) dz$ by introducing $q(z)$ and using Jensen's inequality.

**Exercise 5.5.6**: Show that for Gaussian distributions $q(z|x) = \mathcal{N}(\mu, \sigma^2)$ and $p(z) = \mathcal{N}(0, 1)$:
$$\text{KL}(q(z|x) \| p(z)) = \frac{1}{2}\left(\mu^2 + \sigma^2 - \log \sigma^2 - 1\right)$$

**Exercise 5.5.7**: Prove that maximizing ELBO is equivalent to minimizing $\text{KL}(q(z) \| p(z|x))$.

**Exercise 5.5.8**: For mean-field approximation, derive the update equation for $q_j(z_j)$ using calculus of variations.

### Implementation and Applications (★★★)

**Exercise 5.5.9**: Implement a simple VAE for MNIST digits:
- Encoder: $784 \to 400 \to 20$ (10-dim $\mu$, 10-dim $\log\sigma^2$)
- Decoder: $10 \to 400 \to 784$
- Train with binary cross-entropy reconstruction loss
- Visualize latent space by plotting 2D projections colored by digit class

**Exercise 5.5.10**: Implement CAVI for Bayesian Gaussian mixture model:
- Variational factors: $q(\mu_k)$, $q(\Sigma_k)$, $q(\pi)$, $q(z_i)$
- Derive and implement coordinate ascent updates
- Compare convergence to standard EM algorithm

**Exercise 5.5.11**: Investigate posterior collapse:
- Train a VAE and monitor $\text{KL}(q(z|x) \| p(z))$ per data point
- Implement KL annealing: $\beta \cdot \text{KL}(q \| p)$ with $\beta: 0 \to 1$
- Compare reconstruction quality and latent code usage

**Exercise 5.5.12**: Implement a simple normalizing flow:
- Use planar flows: $f(z) = z + u \cdot \tanh(w^\top z + b)$
- Stack 10 flow layers in a VAE encoder
- Compare to Gaussian $q(z|x)$ in terms of ELBO and sample quality

---

## Related Topics

### Within This Textbook
- **Chapter 3.4**: Probability distributions and exponential families
- **Chapter 4.2**: Information theory and KL divergence
- **Chapter 5.1**: Maximum likelihood estimation
- **Chapter 5.2**: Gradient-based optimization
- **Chapter 6.3**: Neural network architectures (for VAE implementation)

### Advanced Topics
- **Normalizing Flows**: Invertible neural networks for flexible distributions
- **Importance-Weighted Autoencoders (IWAE)**: Tighter bounds with multiple samples
- **Adversarial Variational Bayes**: Using discriminators for flexible inference
- **Structured Variational Inference**: Capturing posterior correlations
- **Variational Inference in Probabilistic Programming**: Automatic inference (Pyro, Edward)

### Applications
- **Computer Vision**: Image generation, disentangled representations, style transfer
- **Natural Language Processing**: Text generation, sentence embeddings, neural machine translation
- **Computational Biology**: Single-cell RNA sequencing (scVI), protein folding
- **Reinforcement Learning**: Model-based RL with learned world models

---

## Further Reading

**Foundational Papers**:
- Jordan et al. (1999): "An Introduction to Variational Methods for Graphical Models"
- Kingma & Welling (2014): "Auto-Encoding Variational Bayes" (VAE)
- Rezende et al. (2014): "Stochastic Backpropagation and Approximate Inference" (reparameterization)

**Textbooks**:
- Bishop (2006): *Pattern Recognition and Machine Learning*, Chapter 10
- Murphy (2022): *Probabilistic Machine Learning: Advanced Topics*, Chapters 8-10
- Blei et al. (2017): "Variational Inference: A Review for Statisticians"

**Recent Advances**:
- Kingma et al. (2016): "Improved Variational Inference with Inverse Autoregressive Flow"
- Higgins et al. (2017): "β-VAE: Learning Basic Visual Concepts with a Constrained VAE"
- Tomczak & Welling (2018): "VAE with a VampPrior"

---

**End of Chapter 5.5**
