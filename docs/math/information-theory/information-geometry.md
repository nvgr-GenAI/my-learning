# 6.4 Information Geometry

## Prerequisites

- **Required:**
  - Differential calculus and gradients (Chapter 3.1)
  - Probability distributions and expectations (Chapter 4.1)
  - KL divergence (Chapter 6.3)
  - Basic manifold concepts (tangent spaces, metrics)

- **Recommended:**
  - Riemannian geometry fundamentals
  - Maximum likelihood estimation
  - Gradient descent optimization (Chapter 5.2)

## Overview

Information geometry studies probability distributions as points on a smooth manifold, where the geometry is determined by information-theoretic quantities. This perspective reveals deep connections between statistics, optimization, and differential geometry, with profound implications for machine learning.

The key insight: the space of probability distributions has intrinsic geometric structure, and this geometry matters for optimization and inference.

---

## 6.4.1 Statistical Manifolds

**Definition 6.4.1 (Statistical Manifold)**
A *statistical manifold* is a smooth manifold $\mathcal{M}$ where each point $\theta \in \mathcal{M}$ represents a probability distribution $p_\theta(x)$ from a parametric family. The manifold has dimension equal to the number of parameters.

**Example 6.4.1 (Gaussian Manifold)**
The family of univariate Gaussians forms a 2-dimensional manifold:
$$\mathcal{M} = \{p_{\mu,\sigma}(x) = \mathcal{N}(x|\mu,\sigma^2) : \mu \in \mathbb{R}, \sigma > 0\}$$

Each point $(\mu, \sigma)$ represents a distinct Gaussian distribution.

**Example 6.4.6 (Bernoulli Manifold)**
The family of Bernoulli distributions forms a 1-dimensional manifold:
$$\mathcal{M} = \{p_\theta(x) = \theta^x(1-\theta)^{1-x} : \theta \in (0,1)\}, \quad x \in \{0,1\}$$
This manifold is the open interval $(0,1)$ — a curve. Each point $\theta$ is a coin with bias $\theta$:

- $\theta = 0.5$ is the fair coin
- $\theta = 0.9$ is a heavily biased coin
- The boundary points $\theta = 0$ and $\theta = 1$ (degenerate distributions) are excluded, keeping the manifold smooth

The tangent space at each $\theta$ is 1-dimensional, spanned by the score:
$$\frac{\partial \log p_\theta(x)}{\partial \theta} = \frac{x}{\theta} - \frac{1-x}{1-\theta} = \frac{x - \theta}{\theta(1-\theta)}$$

**Tangent Space Interpretation**
At each point $\theta \in \mathcal{M}$, the tangent space $T_\theta\mathcal{M}$ consists of infinitesimal directions of change in the distribution. A tangent vector can be represented by the score function:
$$v_i = \frac{\partial \log p_\theta(x)}{\partial \theta_i}$$

This links differential geometry to statistical estimation.

**ASCII Diagram: Statistical Manifold**
```
      Statistical Manifold M

      p_θ₃ ●
         /  \
        /    \
    p_θ₁●----●p_θ₂     Each point is a distribution
       |      |
       |      |         Paths = transformations
       ●------●         between distributions
     p_θ₄    p_θ₅

  Tangent space at θ:
  T_θM = span{∂log p/∂θᵢ}
```

---

## 6.4.2 Fisher Information Matrix

**Definition 6.4.2 (Fisher Information Matrix)**
The *Fisher information matrix* (FIM) for a parametric family $p_\theta(x)$ is defined as:
$$I(\theta)_{ij} = \mathbb{E}_{x \sim p_\theta}\left[\frac{\partial \log p_\theta(x)}{\partial \theta_i}\frac{\partial \log p_\theta(x)}{\partial \theta_j}\right]$$

Equivalently, using the second derivative form:
$$I(\theta)_{ij} = -\mathbb{E}_{x \sim p_\theta}\left[\frac{\partial^2 \log p_\theta(x)}{\partial \theta_i \partial \theta_j}\right]$$

The equivalence holds under regularity conditions.

**Example 6.4.7 (Fisher Information for Bernoulli)**
For $p(x|\theta) = \theta^x(1-\theta)^{1-x}$, compute $I(\theta)$ step by step:

1. Log-likelihood: $\log p(x|\theta) = x\log\theta + (1-x)\log(1-\theta)$
2. Score: $\frac{\partial}{\partial\theta}\log p = \frac{x}{\theta} - \frac{1-x}{1-\theta} = \frac{x - \theta}{\theta(1-\theta)}$
3. Fisher information: $I(\theta) = \mathbb{E}\!\left[\left(\frac{x-\theta}{\theta(1-\theta)}\right)^2\right] = \frac{\text{Var}(x)}{\theta^2(1-\theta)^2} = \frac{\theta(1-\theta)}{\theta^2(1-\theta)^2} = \frac{1}{\theta(1-\theta)}$

**Numerical check:** At $\theta = 0.3$: $I(0.3) = \frac{1}{0.3 \times 0.7} = \frac{1}{0.21} \approx 4.76$. At $\theta = 0.5$ (fair coin): $I(0.5) = \frac{1}{0.25} = 4$ — the minimum, reflecting that a fair coin is hardest to distinguish from its neighbors.

**Theorem 6.4.1 (Properties of Fisher Information)**
For a parametric family $p_\theta$, the Fisher information matrix satisfies:

1. **Positive Semi-Definite:** $I(\theta) \succeq 0$ for all $\theta$
2. **Invariance Under Sufficient Statistics:** If $T(x)$ is sufficient for $\theta$, then $I_{\text{full}}(\theta) = I_{\text{sufficient}}(\theta)$
3. **Reparametrization:** Under $\phi = g(\theta)$, Fisher transforms as:
   $$I_\phi = J^T I_\theta J \quad \text{where } J_{ij} = \frac{\partial \theta_i}{\partial \phi_j}$$

**Example 6.4.2 (Fisher Information for Gaussian)**
For $p(x|\mu,\sigma^2) = \mathcal{N}(x|\mu,\sigma^2)$:
$$I(\mu,\sigma^2) = \begin{pmatrix}
\frac{1}{\sigma^2} & 0 \\
0 & \frac{1}{2\sigma^4}
\end{pmatrix}$$

Note the diagonal structure (parameters are orthogonal) and that uncertainty increases as $\sigma^2$ increases.

**Example 6.4.8 (Scalar Fisher Information for Gaussian Mean)**
For $p(x|\mu) = \mathcal{N}(x|\mu, \sigma^2)$ with $\sigma^2$ known, compute $I(\mu)$ via the second-derivative method:

1. Log-likelihood: $\log p(x|\mu) = -\frac{1}{2}\log(2\pi\sigma^2) - \frac{(x-\mu)^2}{2\sigma^2}$
2. Second derivative: $\frac{\partial^2}{\partial\mu^2}\log p = -\frac{1}{\sigma^2}$ (constant — does not depend on $x$)
3. Fisher information: $I(\mu) = -\mathbb{E}\!\left[-\frac{1}{\sigma^2}\right] = \frac{1}{\sigma^2}$

**Numerical check:** With $\sigma^2 = 4$: $I(\mu) = 0.25$. With $\sigma^2 = 0.01$: $I(\mu) = 100$. Smaller noise means each observation carries more information about $\mu$.

**Interpretation: Local Curvature**
The Fisher information measures the curvature of the log-likelihood surface. High Fisher information at $\theta_0$ means:
- Small changes in $\theta$ lead to large changes in the distribution
- Data is highly informative about $\theta$ near $\theta_0$
- The log-likelihood is sharply peaked

---

## 6.4.3 Fisher Information as Hessian of KL Divergence

**Theorem 6.4.2 (Fisher-KL Connection)**
The Fisher information matrix is the Hessian of the KL divergence:
$$D_{KL}(p_\theta \| p_{\theta+d\theta}) \approx \frac{1}{2}d\theta^T I(\theta) d\theta + O(\|d\theta\|^3)$$

This shows that the Fisher information defines the local geometry of the distribution space.

**Proof Sketch:**
Starting with the KL divergence:
$$D_{KL}(p_\theta \| p_{\theta+\delta}) = \mathbb{E}_{x \sim p_\theta}\left[\log \frac{p_\theta(x)}{p_{\theta+\delta}(x)}\right]$$

Taylor expand $\log p_{\theta+\delta}(x)$ around $\theta$:
$$\log p_{\theta+\delta}(x) = \log p_\theta(x) + \sum_i \frac{\partial \log p}{\partial \theta_i}\delta_i + \frac{1}{2}\sum_{i,j} \frac{\partial^2 \log p}{\partial \theta_i \partial \theta_j}\delta_i\delta_j + O(\|\delta\|^3)$$

Substituting and using $\mathbb{E}[\partial \log p / \partial \theta_i] = 0$:
$$D_{KL}(p_\theta \| p_{\theta+\delta}) = -\frac{1}{2}\mathbb{E}\left[\sum_{i,j} \frac{\partial^2 \log p}{\partial \theta_i \partial \theta_j}\delta_i\delta_j\right] = \frac{1}{2}\delta^T I(\theta)\delta$$

**Example 6.4.9 (Fisher-KL Connection for Bernoulli)**
Verify the quadratic approximation $D_{KL}(p_\theta \| p_{\theta+\delta}) \approx \frac{1}{2}I(\theta)\delta^2$ at $\theta = 0.5$ with $\delta = 0.05$:

1. Exact KL: $D_{KL}(\text{Ber}(0.5)\|\text{Ber}(0.55)) = 0.5\log\frac{0.5}{0.55} + 0.5\log\frac{0.5}{0.45} = 0.5(-0.0953) + 0.5(0.1054) = 0.00503$
2. Fisher approximation: $I(0.5) = \frac{1}{0.5 \times 0.5} = 4$, so $\frac{1}{2}\cdot 4 \cdot (0.05)^2 = 0.005$
3. The approximation ($0.005$) matches the exact value ($0.00503$) to within $0.6\%$

For larger $\delta = 0.2$: exact $D_{KL}(\text{Ber}(0.5)\|\text{Ber}(0.7)) = 0.0849$, approximation $= \frac{1}{2}\cdot 4 \cdot 0.04 = 0.08$. Error grows to $\sim 6\%$, showing the quadratic approximation is local.

**Geometric Meaning**
The Fisher information is the Riemannian metric tensor on the statistical manifold. It defines:
- **Distance:** $d_{\text{Fisher}}(\theta_1, \theta_2) = \inf_{\gamma} \int_0^1 \sqrt{\dot{\gamma}(t)^T I(\gamma(t)) \dot{\gamma}(t)} dt$
- **Angles:** Between tangent vectors
- **Volume:** On the manifold

This is the *natural* geometry for probability distributions.

---

## 6.4.4 Natural Gradient

**Definition 6.4.3 (Natural Gradient)**
The *natural gradient* of a function $L(\theta)$ with respect to the Fisher metric is:
$$\tilde{\nabla}_\theta L = I(\theta)^{-1} \nabla_\theta L$$

This is the direction of steepest descent in the space of distributions (Fisher geometry), not in parameter space (Euclidean geometry).

**Motivation**
Standard gradient descent uses Euclidean geometry:
$$\theta_{t+1} = \theta_t - \alpha \nabla_\theta L$$

But parameter space is arbitrary. A reparametrization $\phi = g(\theta)$ changes the gradient direction. Natural gradient is invariant to such reparametrizations.

**Example 6.4.10 (Natural vs Euclidean Gradient for Bernoulli)**
Suppose we minimize $L(\theta) = -\log p(x=1|\theta) = -\log\theta$ over $\text{Ber}(\theta)$. At $\theta = 0.1$:

1. Euclidean gradient: $\nabla_\theta L = -1/\theta = -10$
2. Fisher information: $I(\theta) = 1/[\theta(1-\theta)] = 1/(0.1\times 0.9) = 11.11$
3. Natural gradient: $\tilde{\nabla}_\theta L = I(\theta)^{-1}\nabla_\theta L = 0.09 \times (-10) = -0.9$

At $\theta = 0.9$ for the same loss:

1. Euclidean gradient: $-1/0.9 = -1.11$
2. Natural gradient: $0.9\times 0.1 \times (-1.11) = -0.1$

The Euclidean gradient is $10\times$ larger at $\theta=0.1$ than at $\theta=0.9$, but the natural gradient adapts: it takes smaller steps near the boundary where the manifold curves sharply, preventing overshooting past $\theta=0$ or $\theta=1$.

**ASCII Diagram: Euclidean vs Natural Gradient**
```
Distribution Space (Statistical Manifold)

     p₀  ────────────────────→  p*
     ●                           ●  (target)
      \                         /
       \                       /
        ↘ Euclidean          ↙ Natural
         \ gradient         / gradient
          \               /
           \             /
            ●-----------●
           p₁         p₂

Euclidean path:        Natural path:
- Straight in          - Follows geodesic
  parameter space        in Fisher geometry
- May take            - Efficient in
  inefficient            distribution space
  route in dist.      - Fewer steps
  space
- Many steps          Parameter-invariant

In parameter space θ:      In distribution space:
∇θ points here            ∇̃θ = I⁻¹∇θ points here
(arbitrary)               (intrinsic)
```

**Theorem 6.4.3 (Natural Gradient Properties)**
The natural gradient has the following properties:

1. **Reparametrization Invariance:** For any invertible transformation $\phi = g(\theta)$:
   $$\tilde{\nabla}_\phi L = \frac{\partial \phi}{\partial \theta} \tilde{\nabla}_\theta L$$

2. **Steepest Descent in Fisher Geometry:** It minimizes $L$ along the direction that changes the distribution least (in KL sense)

3. **Connection to Second-Order Methods:** When $L = -\log p(D|\theta)$ (negative log-likelihood), natural gradient resembles Newton's method

**Example 6.4.3 (Natural Gradient for Gaussian Mean)**
For $p(x|\mu) = \mathcal{N}(x|\mu, \sigma^2)$ with known $\sigma^2$:
- Fisher information: $I(\mu) = 1/\sigma^2$
- Standard gradient: $\nabla_\mu L = \nabla_\mu L$
- Natural gradient: $\tilde{\nabla}_\mu L = \sigma^2 \nabla_\mu L$

The natural gradient automatically scales the learning rate by the noise level.

---

## 6.4.5 Cramér-Rao Lower Bound

**Theorem 6.4.4 (Cramér-Rao Bound)**
For any unbiased estimator $\hat{\theta}(x)$ of parameter $\theta$ based on data $x \sim p_\theta$:
$$\text{Cov}(\hat{\theta}) \succeq I(\theta)^{-1}$$

where $A \succeq B$ means $A - B$ is positive semi-definite. For scalar parameters:
$$\text{Var}(\hat{\theta}) \geq \frac{1}{I(\theta)}$$

**Proof Sketch:**
For unbiased estimators, $\mathbb{E}[\hat{\theta}] = \theta$. Differentiating with respect to $\theta$:
$$1 = \frac{\partial}{\partial \theta}\mathbb{E}[\hat{\theta}] = \mathbb{E}\left[\hat{\theta} \frac{\partial \log p}{\partial \theta}\right]$$

By Cauchy-Schwarz inequality:
$$1 = \mathbb{E}[\hat{\theta} \cdot s]^2 \leq \text{Var}(\hat{\theta}) \cdot \mathbb{E}[s^2] = \text{Var}(\hat{\theta}) \cdot I(\theta)$$

where $s = \partial \log p / \partial \theta$ is the score function.

**Definition 6.4.4 (Efficient Estimator)**
An estimator $\hat{\theta}$ is *efficient* if it achieves the Cramér-Rao bound:
$$\text{Var}(\hat{\theta}) = \frac{1}{I(\theta)}$$

Efficient estimators have minimum variance among all unbiased estimators.

**Example 6.4.11 (Cramér-Rao Bound for Bernoulli)**
Given $n = 100$ i.i.d. draws from $\text{Ber}(\theta)$ with true $\theta = 0.3$. Consider the estimator $\hat{\theta} = \bar{x} = \frac{1}{n}\sum x_i$:

1. Fisher information (single observation): $I(\theta) = \frac{1}{\theta(1-\theta)} = \frac{1}{0.21} \approx 4.76$
2. Fisher information ($n$ observations): $I_n(\theta) = \frac{n}{\theta(1-\theta)} = \frac{100}{0.21} \approx 476.2$
3. Cramér-Rao lower bound: $\text{Var}(\hat{\theta}) \geq \frac{1}{I_n(\theta)} = \frac{\theta(1-\theta)}{n} = \frac{0.21}{100} = 0.0021$
4. Actual variance of $\bar{x}$: $\text{Var}(\bar{x}) = \frac{\theta(1-\theta)}{n} = 0.0021$

The bound is achieved exactly — the sample mean is an efficient estimator for the Bernoulli parameter. The standard error is $\sqrt{0.0021} \approx 0.046$, so a $95\%$ confidence interval has width $\approx \pm 0.09$.

**Example 6.4.4 (Efficient Estimator)**
For $x_1, \ldots, x_n \sim \mathcal{N}(\mu, \sigma^2)$ with known $\sigma^2$:
- Fisher information: $I_n(\mu) = n/\sigma^2$
- Sample mean: $\bar{x} = \frac{1}{n}\sum_{i=1}^n x_i$
- Variance: $\text{Var}(\bar{x}) = \sigma^2/n = 1/I_n(\mu)$

The sample mean is efficient.

**Connection to Fisher Information**
The inverse Fisher information sets the fundamental limit on estimation accuracy. This connects:
- **Geometry:** Curvature of the manifold
- **Statistics:** Precision of estimation
- **Information:** Amount learned from data

---

## 6.4.6 Connections to Riemannian Geometry

**The Fisher Metric as Riemannian Metric**
The Fisher information matrix defines a Riemannian metric $g_{ij}(\theta) = I(\theta)_{ij}$ on the statistical manifold. This induces:

1. **Inner Product:** For tangent vectors $u, v \in T_\theta\mathcal{M}$:
   $$\langle u, v \rangle_\theta = u^T I(\theta) v$$

2. **Length of Curves:** For a path $\gamma(t)$ in $\mathcal{M}$:
   $$L(\gamma) = \int_0^1 \sqrt{\dot{\gamma}(t)^T I(\gamma(t)) \dot{\gamma}(t)} dt$$

3. **Geodesics:** Shortest paths between distributions, satisfying:
   $$\ddot{\gamma}^k + \Gamma^k_{ij}\dot{\gamma}^i\dot{\gamma}^j = 0$$
   where $\Gamma^k_{ij}$ are Christoffel symbols

**Example 6.4.5 (Geodesics in Gaussian Manifold)**
In the manifold of univariate Gaussians $\mathcal{N}(\mu, \sigma^2)$, geodesics connecting two points are *not* straight lines in $(\mu, \sigma)$ coordinates due to the curved geometry induced by the Fisher metric.

**ASCII Diagram: Geodesics on Statistical Manifold**
```
     Statistical Manifold (curved)

        p₂ ●
          /│\
         / │ \
        /  │  \        Straight line (Euclidean)
       /   │   \       vs
      /    │    \      Geodesic (Fisher)
     /   ╱─│─╲   \
    /  ╱   │   ╲  \    Geodesic minimizes
   / ╱     │     ╲ \   Fisher distance
  /╱       │       ╲\
 ●─────────●─────────●
p₁     (θ-space)     p₃

Geodesic properties:
- Shortest path in Fisher geometry
- Parallel transports vectors
- Locally minimizes KL divergence
```

**Example 6.4.12 (Geodesic Distance Between Two Bernoulli Distributions)**
Compute the Fisher-Rao geodesic distance between $\text{Ber}(\theta_1 = 0.2)$ and $\text{Ber}(\theta_2 = 0.8)$:

1. Use the reparametrization $\phi = \arcsin(\sqrt{\theta})$, which makes the metric Euclidean (constant Fisher information in $\phi$-coordinates)
2. $\phi_1 = \arcsin(\sqrt{0.2}) = \arcsin(0.4472) = 0.4636$ rad
3. $\phi_2 = \arcsin(\sqrt{0.8}) = \arcsin(0.8944) = 1.1071$ rad
4. Geodesic distance: $d(\theta_1, \theta_2) = 2|\phi_2 - \phi_1| = 2|1.1071 - 0.4636| = 2 \times 0.6435 = 1.287$

**Comparison:** Between $\text{Ber}(0.5)$ and $\text{Ber}(0.6)$: $d = 2|\arcsin(\sqrt{0.6}) - \arcsin(\sqrt{0.5})| = 2|0.8861 - 0.7854| = 0.201$. The distance between $0.2$ and $0.8$ is much larger than $6\times$ the distance between $0.5$ and $0.6$, reflecting the curved geometry — distributions near the boundary are farther apart in information-theoretic terms.

**Curvature and Connections**
The Riemann curvature tensor captures how the manifold curves. For exponential families, the curvature is related to higher-order cumulants of the sufficient statistics.

---

## 6.4.7 Exponential and Mixture Families

**Exponential Families and Duality**

**Definition 6.4.5 (Exponential Family)**
A parametric family is an *exponential family* if:
$$p(x|\theta) = h(x) \exp\left(\theta^T T(x) - A(\theta)\right)$$

where $\theta$ are natural parameters, $T(x)$ are sufficient statistics, and $A(\theta)$ is the log-partition function.

**Dual Coordinates**
Exponential families admit two natural parametrizations:
1. **Natural parameters** $\theta$ (canonical)
2. **Expectation parameters** $\eta = \mathbb{E}[T(x)] = \nabla A(\theta)$

These are related by the Legendre transformation:
$$A(\theta) = \sup_\eta (\theta^T \eta - A^*(\eta))$$

**Definition 6.4.6 (e-flat and m-flat)**
In an exponential family:
- The family is *e-flat* (exponentially flat) in natural parameters $\theta$
- The family is *m-flat* (mixture flat) in expectation parameters $\eta$

This means geodesics are straight lines in these respective coordinate systems.

**ASCII Diagram: Dual Geometries**
```
e-flat geometry (θ coords)    m-flat geometry (η coords)

p₁●─────────●p₂              p₁●╲
  │         │                   │ ╲
  │  rect   │                   │  ╲ curved
  │  grid   │                   │   ╲
p₃●─────────●p₄              p₃●────●p₂
                                    ╱│
Geodesics = straight         Geodesics ╱ │
lines in θ-space            = curved ╱  │
                            in η-space  ●p₄

e-connection ∇(e)            m-connection ∇(m)
- Natural for exponential    - Natural for mixtures
- θᵢ are affine coords       - η = E[T(x)] are coords
```

**Theorem 6.4.5 (Pythagorean Theorem in Information Geometry)**
For exponential families, if $q$ is the projection of $r$ onto a submanifold $\mathcal{S}$ containing $p$:
$$D_{KL}(r \| p) = D_{KL}(r \| q) + D_{KL}(q \| p)$$

This is analogous to the Pythagorean theorem in Euclidean geometry, but with KL divergence replacing squared distance.

**Example 6.4.13 (Information-Geometric Pythagorean Theorem)**
Consider three Bernoulli distributions: $r = \text{Ber}(0.6)$, $q = \text{Ber}(0.5)$, and $p = \text{Ber}(0.4)$. Let $q$ be the e-projection of $r$ onto a submanifold containing $p$ (here the "submanifold" is chosen so the orthogonality condition holds). Verify:

1. $D_{KL}(r \| p) = 0.6\log\frac{0.6}{0.4} + 0.4\log\frac{0.4}{0.6} = 0.6(0.4055) + 0.4(-0.4055) = 0.0811$
2. $D_{KL}(r \| q) = 0.6\log\frac{0.6}{0.5} + 0.4\log\frac{0.4}{0.5} = 0.6(0.1823) + 0.4(-0.2231) = 0.0201$
3. $D_{KL}(q \| p) = 0.5\log\frac{0.5}{0.4} + 0.5\log\frac{0.5}{0.6} = 0.5(0.2231) + 0.5(-0.1823) = 0.0204$
4. Check: $D_{KL}(r\|q) + D_{KL}(q\|p) = 0.0201 + 0.0204 = 0.0405 \neq 0.0811$

The Pythagorean identity does **not** hold here because $q$ is not the true e-projection of $r$ onto the submanifold through $p$. The theorem requires the specific geometric orthogonality condition. When the projection is computed correctly (minimizing $D_{KL}(r\|q)$ over the submanifold), the identity holds exactly — this illustrates that the "right angle" condition is essential, just as in Euclidean geometry.

**Mixture Families**
A *mixture family* has form:
$$p(x|\eta) = \sum_{i} \eta_i p_i(x) \quad \text{with } \sum_i \eta_i = 1$$

Mixture families are m-flat, meaning geodesics are straight lines in the mixing proportions $\eta$.

**Duality Table**

| Property | e-flat (Exponential) | m-flat (Mixture) |
|----------|---------------------|------------------|
| Parameters | Natural $\theta$ | Expectation $\eta$ |
| Geodesics | Straight in $\theta$ | Straight in $\eta$ |
| Examples | Exponential families | Mixture models |
| Connection | e-connection $\nabla^{(e)}$ | m-connection $\nabla^{(m)}$ |
| Projection | e-projection (I-projection) | m-projection (M-projection) |

---

## 6.4.8 Machine Learning Applications

### 1. Natural Gradient Descent

**Algorithm 6.4.1 (Natural Gradient Descent)**
```
Input: Loss L(θ), initial θ₀, learning rate α
for t = 0, 1, 2, ... do
  Compute gradient: g = ∇_θ L(θₜ)
  Compute Fisher: I(θₜ) or approximation
  Natural gradient: g̃ = I(θₜ)⁻¹ g
  Update: θₜ₊₁ = θₜ - α g̃
end
```

**Challenges:**
- Computing $I(\theta)$ is $O(d^2)$ for $d$ parameters
- Inverting $I(\theta)$ is $O(d^3)$

**Practical Approximations:**

1. **K-FAC (Kronecker-Factored Approximate Curvature):**
   Approximates Fisher as Kronecker product:
   $$I(\theta) \approx A \otimes B$$

   For neural networks, $A$ captures activations, $B$ captures gradients. Inversion becomes $O(d^{3/2})$.

2. **Block-Diagonal Approximation:**
   Treat parameters in each layer independently:
   $$I(\theta) \approx \text{diag}(I_1, I_2, \ldots, I_L)$$

3. **Empirical Fisher:**
   Use model's predictions instead of true labels:
   $$\hat{I}(\theta) = \frac{1}{n}\sum_{i=1}^n \nabla \log p(y_i|x_i;\theta) \nabla \log p(y_i|x_i;\theta)^T$$

### 2. Trust Region Policy Optimization (TRPO)

In reinforcement learning, TRPO constrains policy updates using KL divergence:
$$\begin{aligned}
\max_\theta \quad & \mathbb{E}[A^{\pi_{\theta_{\text{old}}}}(s,a) \frac{\pi_\theta(a|s)}{\pi_{\theta_{\text{old}}}(a|s)}] \\
\text{s.t.} \quad & D_{KL}(\pi_{\theta_{\text{old}}} \| \pi_\theta) \leq \delta
\end{aligned}$$

The constraint is approximately:
$$D_{KL}(\pi_{\theta_{\text{old}}} \| \pi_\theta) \approx \frac{1}{2}(\theta - \theta_{\text{old}})^T I(\theta_{\text{old}})(\theta - \theta_{\text{old}})$$

This uses the Fisher metric to ensure safe policy updates.

### 3. Model Pruning and Compression

The Fisher information identifies important parameters:
$$\text{Importance}(w_i) = I(w)_{ii}$$

Parameters with low Fisher information can be pruned with minimal impact on the distribution.

### 4. Adam as Approximate Natural Gradient

Adam optimizer:
$$m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t$$
$$v_t = \beta_2 v_{t-1} + (1-\beta_2)g_t^2$$
$$\theta_{t+1} = \theta_t - \alpha \frac{m_t}{\sqrt{v_t} + \epsilon}$$

The diagonal scaling $1/\sqrt{v_t}$ approximates $I(\theta)^{-1}$ when the Fisher is diagonal. This provides some parameter-space invariance without computing the full Fisher matrix.

### 5. Information Geometry of Neural Networks

Neural networks define a statistical manifold where:
- Each weight configuration $w$ defines a distribution $p(y|x;w)$
- The Fisher information measures sensitivity to weight changes
- Training trajectories are curves on this manifold

**Loss Surface Geometry:**
The Fisher information affects the loss surface:
- High curvature directions: Hard to optimize, require small steps
- Low curvature directions: Easy to optimize, allow large steps

Natural gradient adapts step size to local geometry.

### 6. Bayesian Deep Learning

In variational inference, we minimize:
$$D_{KL}(q_\phi \| p(w|D)) = \mathbb{E}_{q_\phi}[\log q_\phi(w)] - \mathbb{E}_{q_\phi}[\log p(w|D)]$$

Natural gradients on $\phi$ use the Fisher metric of $q_\phi$, leading to more efficient inference.

---

## Summary

Information geometry unifies statistics, differential geometry, and optimization:

1. **Statistical Manifolds:** Probability distributions form smooth manifolds
2. **Fisher Information:** Defines the natural Riemannian metric, connects to KL divergence
3. **Natural Gradient:** Steepest descent in distribution space, parameter-invariant
4. **Cramér-Rao Bound:** Fisher information bounds estimation efficiency
5. **Dual Geometries:** Exponential families have e-flat and m-flat structures
6. **ML Applications:** Natural gradient methods, TRPO, network pruning, understanding Adam

The key insight: geometry matters. Optimizing in parameter space (Euclidean geometry) differs fundamentally from optimizing in distribution space (Fisher geometry). Natural gradient methods respect this intrinsic geometry.

---

## Exercises

**Exercise 6.4.1 (Fisher Information Calculation) ★**
Compute the Fisher information matrix for the Bernoulli distribution $p(x|\theta) = \theta^x(1-\theta)^{1-x}$ where $x \in \{0,1\}$.

**Exercise 6.4.2 (Natural Gradient for Softmax) ★★**
For a softmax classifier with parameters $W \in \mathbb{R}^{K \times d}$ and loss $L(W)$:
1. Write the Fisher information matrix $I(W)$
2. Show that the natural gradient has a simpler form than the standard gradient
3. Explain why this improves optimization

**Exercise 6.4.3 (Cramér-Rao Bound) ★**
For $n$ i.i.d. samples from $\text{Exponential}(\lambda)$:
1. Compute the Fisher information $I_n(\lambda)$
2. Find the Cramér-Rao lower bound for estimating $\lambda$
3. Show that $\hat{\lambda} = 1/\bar{x}$ achieves this bound asymptotically

**Exercise 6.4.4 (KL-Fisher Connection) ★★**
Verify that $D_{KL}(p_\theta \| p_{\theta+d\theta}) \approx \frac{1}{2}d\theta^T I(\theta)d\theta$ for the Gaussian distribution $\mathcal{N}(\mu, \sigma^2)$ by:
1. Computing the KL divergence exactly
2. Taylor expanding to second order in $d\mu$ and $d\sigma$
3. Comparing to the Fisher information matrix

**Exercise 6.4.5 (Geodesics in Gaussian Manifold) ★★★**
Consider the manifold of zero-mean Gaussians $\mathcal{N}(0, \sigma^2)$ parametrized by $\sigma > 0$:
1. Compute the Fisher information $I(\sigma)$
2. Write the geodesic equation
3. Find the geodesic connecting $\sigma_1$ to $\sigma_2$
4. Compare to the straight line $\sigma(t) = (1-t)\sigma_1 + t\sigma_2$

**Exercise 6.4.6 (K-FAC Approximation) ★★★**
For a single layer $y = Wx$ with Gaussian likelihood:
1. Write the exact Fisher information matrix
2. Derive the Kronecker-factored approximation $I \approx A \otimes B$
3. Show that inversion complexity reduces from $O(d^3)$ to $O(d^{3/2})$
4. Analyze the approximation error

**Exercise 6.4.7 (e-flat and m-flat) ★★**
For the exponential family $p(x|\theta) = \exp(\theta x - \log(1 + e^\theta))$ (logistic):
1. Identify the natural parameter $\theta$ and sufficient statistic $T(x)$
2. Compute the expectation parameter $\eta = \mathbb{E}[x]$
3. Show that geodesics are straight lines in $\theta$-space
4. Derive the relationship $\theta(\eta)$

**Exercise 6.4.8 (Information Geometry in Practice) ★★★**
Implement natural gradient descent for a simple neural network:
1. Train a 2-layer network on MNIST using standard gradient descent
2. Implement natural gradient with diagonal Fisher approximation
3. Compare convergence speed and final performance
4. Analyze computation time vs. iteration efficiency trade-off

---

## Related Topics

- **Chapter 5.3:** Second-Order Optimization Methods (Newton's method, quasi-Newton)
- **Chapter 6.3:** KL Divergence and f-Divergences
- **Chapter 7.2:** Maximum Likelihood Estimation
- **Chapter 8.4:** Variational Inference
- **Chapter 9.5:** Trust Region Methods in Reinforcement Learning
- **Advanced:** Dual connections, α-connections, Amari-Chentsov tensor

---

## Further Reading

1. **Amari, S. (2016).** *Information Geometry and Its Applications.* Springer. (Comprehensive treatment)
2. **Martens, J. (2020).** "New Insights and Perspectives on the Natural Gradient Method." *Journal of Machine Learning Research.* (Modern ML perspective)
3. **Amari, S., & Nagaoka, H. (2000).** *Methods of Information Geometry.* AMS. (Mathematical foundations)
4. **Pascanu, R., & Bengio, Y. (2014).** "Revisiting Natural Gradient for Deep Networks." *ICLR.* (Practical aspects)
5. **Martens, J., & Grosse, R. (2015).** "Optimizing Neural Networks with Kronecker-factored Approximate Curvature." *ICML.* (K-FAC algorithm)

---

**Previous:** [6.2 Divergence Measures](./divergence-measures.md)
