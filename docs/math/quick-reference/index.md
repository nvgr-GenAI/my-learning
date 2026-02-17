# Quick Reference: Math for AI/ML

A single-page formula reference covering the essential mathematics used across machine learning, deep learning, and AI systems. Organized by topic with direct links to full chapters.

---

## 1. Linear Algebra

> Full coverage: [Part 2: Linear Algebra](../linear-algebra/index.md)

### Vectors & Norms

$$\|\mathbf{x}\|_1 = \sum_{i=1}^n |x_i| \qquad \|\mathbf{x}\|_2 = \sqrt{\sum_{i=1}^n x_i^2} \qquad \|\mathbf{x}\|_\infty = \max_i |x_i|$$

$$\|\mathbf{x}\|_p = \left(\sum_{i=1}^n |x_i|^p\right)^{1/p}$$

**Dot product:** $\mathbf{u} \cdot \mathbf{v} = \sum_{i=1}^n u_i v_i = \|\mathbf{u}\| \|\mathbf{v}\| \cos\theta$

**Cosine similarity:** $\text{sim}(\mathbf{u}, \mathbf{v}) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\| \|\mathbf{v}\|}$

### Matrix Operations

$$\text{Matrix multiply: } (AB)_{ij} = \sum_k A_{ik} B_{kj}$$

$$\text{Transpose: } (A^T)_{ij} = A_{ji} \qquad (AB)^T = B^T A^T$$

$$\text{Inverse: } AA^{-1} = I \qquad (AB)^{-1} = B^{-1}A^{-1}$$

$$\text{Trace: } \text{tr}(A) = \sum_i A_{ii} \qquad \text{tr}(AB) = \text{tr}(BA)$$

**Rank:** $\text{rank}(A) = \dim(\text{Col}(A)) \leq \min(m, n)$

**Frobenius norm:** $\|A\|_F = \sqrt{\text{tr}(A^T A)} = \sqrt{\sum_{ij} a_{ij}^2}$

### Special Matrix Types

| Type | Definition | Key Property | ML Use |
|---|---|---|---|
| Symmetric | $A^T = A$ | Real eigenvalues, orthogonal eigenvectors | Covariance matrices, kernel matrices |
| Orthogonal | $Q^T Q = I$ | Preserves norms and angles, $Q^{-1} = Q^T$ | Orthogonal init (RNNs), QR decomposition |
| Diagonal | $D_{ij} = 0$ for $i \neq j$ | Scales each dimension independently | Eigenvalues ($\Lambda$), variance |
| Positive Definite | $\mathbf{x}^T A \mathbf{x} > 0$ | All eigenvalues $> 0$, Cholesky exists | Hessian at minima, covariance, kernels |
| Positive Semi-Def | $\mathbf{x}^T A \mathbf{x} \geq 0$ | All eigenvalues $\geq 0$ | Gram matrices, kernel matrices |
| Upper Triangular | $a_{ij} = 0$ for $i > j$ | $\det = \prod_i a_{ii}$ | Masked attention (GPT), LU decomposition |

### Matrix-Vector Product

$A\mathbf{x} = \sum_{j=1}^n x_j \mathbf{a}_j$ (linear combination of columns of $A$)

**Outer product:** $\mathbf{u}\mathbf{v}^T \in \mathbb{R}^{m \times n}$, always rank 1

**Key identity:** $AB = \sum_{k=1}^p \mathbf{a}_k \mathbf{b}_k^T$ (sum of outer products)

### Eigenvalues & Decompositions

$$A\mathbf{v} = \lambda \mathbf{v} \qquad \det(A - \lambda I) = 0$$

**Spectral theorem** (symmetric $A$): $A = Q\Lambda Q^T$ where $Q$ orthogonal, $\Lambda$ diagonal

**SVD:** $A = U\Sigma V^T$ where $U, V$ orthogonal, $\Sigma$ diagonal (singular values)

**Low-rank approximation (PCA):** $A_k = \sum_{i=1}^k \sigma_i \mathbf{u}_i \mathbf{v}_i^T$

| Decomposition | When to Use | ML Application |
|---|---|---|
| Eigendecomposition | Square, symmetric | PCA, PageRank |
| SVD | Any matrix | PCA, recommenders, compression |
| Cholesky ($A = LL^T$) | Positive definite | Gaussian sampling, covariance |
| QR ($A = QR$) | Any matrix | Least squares, Gram-Schmidt |

---

## 2. Trigonometry

> Full coverage: [Trigonometry](../foundations/trigonometry.md)

### Core Functions

$$\sin \theta = y\text{-coordinate on unit circle} \qquad \cos \theta = x\text{-coordinate on unit circle}$$

$$\tan \theta = \frac{\sin \theta}{\cos \theta} \qquad \sec \theta = \frac{1}{\cos \theta} \qquad \csc \theta = \frac{1}{\sin \theta}$$

### Key Angle Values

| $\theta$ | $0$ | $\pi/6$ | $\pi/4$ | $\pi/3$ | $\pi/2$ | $\pi$ |
|---|---|---|---|---|---|---|
| $\sin$ | $0$ | $1/2$ | $\sqrt{2}/2$ | $\sqrt{3}/2$ | $1$ | $0$ |
| $\cos$ | $1$ | $\sqrt{3}/2$ | $\sqrt{2}/2$ | $1/2$ | $0$ | $-1$ |

### Fundamental Identities

**Pythagorean:** $\sin^2 \theta + \cos^2 \theta = 1 \qquad 1 + \tan^2 \theta = \sec^2 \theta$

**Angle addition:**

$$\sin(\alpha \pm \beta) = \sin \alpha \cos \beta \pm \cos \alpha \sin \beta$$

$$\cos(\alpha \pm \beta) = \cos \alpha \cos \beta \mp \sin \alpha \sin \beta$$

**Double angle:** $\sin 2\theta = 2\sin\theta\cos\theta \qquad \cos 2\theta = \cos^2\theta - \sin^2\theta$

**Half angle:** $\sin^2(\theta/2) = \frac{1 - \cos\theta}{2} \qquad \cos^2(\theta/2) = \frac{1 + \cos\theta}{2}$

### Derivatives

$$\frac{d}{d\theta}\sin\theta = \cos\theta \qquad \frac{d}{d\theta}\cos\theta = -\sin\theta \qquad \frac{d}{d\theta}\tan\theta = \sec^2\theta$$

$$\frac{d}{dx}\arcsin x = \frac{1}{\sqrt{1-x^2}} \qquad \frac{d}{dx}\arctan x = \frac{1}{1+x^2}$$

### Hyperbolic Functions

$$\sinh x = \frac{e^x - e^{-x}}{2} \qquad \cosh x = \frac{e^x + e^{-x}}{2} \qquad \tanh x = \frac{e^x - e^{-x}}{e^x + e^{-x}}$$

**Hyperbolic Pythagorean:** $\cosh^2 x - \sinh^2 x = 1$

**Derivatives:** $(\sinh x)' = \cosh x \qquad (\cosh x)' = \sinh x \qquad (\tanh x)' = 1 - \tanh^2 x$

**Complex connection (Euler):** $e^{i\theta} = \cos\theta + i\sin\theta \implies \cos\theta = \cosh(i\theta), \quad \sin\theta = -i\sinh(i\theta)$

### ML Applications

| Function | Where It Appears |
|---|---|
| $\tanh(x)$ | LSTM/GRU gates, activation function (range $(-1,1)$) |
| $\sin/\cos$ at multiple frequencies | Transformer positional encoding: $PE_{(pos,2i)} = \sin(pos / 10000^{2i/d})$ |
| Rotation via $\cos(\alpha-\beta)$ | RoPE (Rotary Position Embeddings) — relative position from angle subtraction |
| Fourier series: $f(x) = \sum a_n\cos(nx) + b_n\sin(nx)$ | Fourier features, signal processing, frequency analysis |
| $e^{i\theta}$ | Quantum gates, DFT, phase encoding |

---

## 3. Calculus & Optimization

> Full coverage: [Part 3: Calculus](../calculus/index.md)

### Differentiation

$$f'(x) = \lim_{h \to 0} \frac{f(x+h) - f(x)}{h}$$

**Chain rule:** $\frac{d}{dx}f(g(x)) = f'(g(x)) \cdot g'(x)$

**Product rule:** $(fg)' = f'g + fg'$

### Common Derivatives in ML

| Function | Derivative | Used In |
|---|---|---|
| $\sigma(x) = \frac{1}{1+e^{-x}}$ | $\sigma(x)(1-\sigma(x))$ | Sigmoid activation, logistic regression |
| $\tanh(x)$ | $1 - \tanh^2(x)$ | LSTM gates, activation |
| $\text{ReLU}(x) = \max(0,x)$ | $\begin{cases} 1 & x > 0 \\ 0 & x < 0 \end{cases}$ | Default activation |
| $\text{softmax}(x)_i = \frac{e^{x_i}}{\sum_j e^{x_j}}$ | $s_i(\delta_{ij} - s_j)$ | Classification output |
| $\log x$ | $1/x$ | Cross-entropy loss |
| $x^n$ | $nx^{n-1}$ | Polynomial features |

### Multivariable Calculus

**Gradient:** $\nabla f = \left(\frac{\partial f}{\partial x_1}, \ldots, \frac{\partial f}{\partial x_n}\right)$

**Jacobian:** $J_{ij} = \frac{\partial f_i}{\partial x_j}$ — maps input perturbations to output perturbations

**Hessian:** $H_{ij} = \frac{\partial^2 f}{\partial x_i \partial x_j}$ — curvature of loss landscape

**Taylor expansion (1st order):** $f(\mathbf{x} + \mathbf{h}) \approx f(\mathbf{x}) + \nabla f(\mathbf{x})^T \mathbf{h}$

**Taylor expansion (2nd order):** $f(\mathbf{x} + \mathbf{h}) \approx f(\mathbf{x}) + \nabla f^T \mathbf{h} + \frac{1}{2}\mathbf{h}^T H \mathbf{h}$

### Gradient Descent

$$\theta_{t+1} = \theta_t - \eta \nabla \mathcal{L}(\theta_t)$$

| Variant | Update Rule | Key Property |
|---|---|---|
| SGD | $\theta \leftarrow \theta - \eta \nabla \mathcal{L}_{\text{batch}}$ | Noisy but fast |
| Momentum | $v \leftarrow \beta v + \nabla \mathcal{L}; \quad \theta \leftarrow \theta - \eta v$ | Smooths updates |
| Adam | $m, v$ exponential averages; $\theta \leftarrow \theta - \eta \frac{m}{\sqrt{v}+\epsilon}$ | Adaptive learning rate |

### Backpropagation

Chain rule through computation graph:

$$\frac{\partial \mathcal{L}}{\partial w_{ij}^{(\ell)}} = \frac{\partial \mathcal{L}}{\partial a^{(\ell)}} \cdot \frac{\partial a^{(\ell)}}{\partial z^{(\ell)}} \cdot \frac{\partial z^{(\ell)}}{\partial w_{ij}^{(\ell)}}$$

where $z^{(\ell)} = W^{(\ell)} a^{(\ell-1)} + b^{(\ell)}$ and $a^{(\ell)} = \sigma(z^{(\ell)})$.

---

## 4. Probability & Statistics

> Full coverage: [Part 4: Probability](../probability/index.md)

### Core Rules

$$P(A \cup B) = P(A) + P(B) - P(A \cap B)$$

$$P(A \mid B) = \frac{P(B \mid A) P(A)}{P(B)} \quad \text{(Bayes' theorem)}$$

$$P(A, B) = P(A \mid B) P(B) = P(B \mid A) P(A) \quad \text{(chain rule)}$$

**Independence:** $P(A \cap B) = P(A)P(B)$

### Expectation & Variance

$$\mathbb{E}[X] = \sum_x x \, P(X=x) \quad \text{or} \quad \int x \, f(x) \, dx$$

$$\text{Var}(X) = \mathbb{E}[X^2] - (\mathbb{E}[X])^2$$

$$\text{Cov}(X,Y) = \mathbb{E}[XY] - \mathbb{E}[X]\mathbb{E}[Y]$$

**Linearity:** $\mathbb{E}[aX + bY] = a\mathbb{E}[X] + b\mathbb{E}[Y]$ (always)

**Variance:** $\text{Var}(aX + b) = a^2 \text{Var}(X)$

### Key Distributions

| Distribution | PDF/PMF | Mean | Variance | ML Use |
|---|---|---|---|---|
| Bernoulli($p$) | $p^k(1-p)^{1-k}$ | $p$ | $p(1-p)$ | Binary classification |
| Gaussian($\mu, \sigma^2$) | $\frac{1}{\sqrt{2\pi}\sigma}e^{-(x-\mu)^2/2\sigma^2}$ | $\mu$ | $\sigma^2$ | Everywhere |
| Categorical($\mathbf{p}$) | $\prod p_i^{[x=i]}$ | — | — | Multi-class |
| Poisson($\lambda$) | $\frac{\lambda^k e^{-\lambda}}{k!}$ | $\lambda$ | $\lambda$ | Count data |
| Exponential($\lambda$) | $\lambda e^{-\lambda x}$ | $1/\lambda$ | $1/\lambda^2$ | Time between events |
| Beta($\alpha, \beta$) | $\propto x^{\alpha-1}(1-x)^{\beta-1}$ | $\frac{\alpha}{\alpha+\beta}$ | — | Prior for probabilities |
| Dirichlet($\boldsymbol{\alpha}$) | $\propto \prod x_i^{\alpha_i - 1}$ | $\frac{\alpha_i}{\sum \alpha_j}$ | — | Prior for categorical |

**Multivariate Gaussian:**

$$\mathcal{N}(\mathbf{x} \mid \boldsymbol{\mu}, \Sigma) = \frac{1}{(2\pi)^{d/2}|\Sigma|^{1/2}} \exp\left(-\frac{1}{2}(\mathbf{x}-\boldsymbol{\mu})^T \Sigma^{-1}(\mathbf{x}-\boldsymbol{\mu})\right)$$

### Limit Theorems

**Law of Large Numbers:** $\bar{X}_n \xrightarrow{p} \mu$ as $n \to \infty$

**Central Limit Theorem:** $\frac{\bar{X}_n - \mu}{\sigma/\sqrt{n}} \xrightarrow{d} \mathcal{N}(0,1)$

### Maximum Likelihood Estimation

$$\hat{\theta}_{\text{MLE}} = \arg\max_\theta \sum_{i=1}^n \log p(x_i \mid \theta)$$

**MAP estimate:** $\hat{\theta}_{\text{MAP}} = \arg\max_\theta \left[\sum_{i=1}^n \log p(x_i \mid \theta) + \log p(\theta)\right]$

| Prior | MAP = MLE + ... | Regularization |
|---|---|---|
| Gaussian prior | L2 penalty $\|\theta\|_2^2$ | Ridge / Weight decay |
| Laplace prior | L1 penalty $\|\theta\|_1$ | Lasso / Sparsity |

---

## 5. Information Theory

> Full coverage: [Part 6: Information Theory](../information-theory/index.md)

### Entropy

$$H(X) = -\sum_x p(x) \log p(x) = \mathbb{E}[-\log p(X)]$$

**Continuous (differential entropy):** $h(X) = -\int f(x) \log f(x) \, dx$

### Cross-Entropy & KL Divergence

$$H(p, q) = -\sum_x p(x) \log q(x) \quad \text{(cross-entropy)}$$

$$D_{\text{KL}}(p \| q) = \sum_x p(x) \log \frac{p(x)}{q(x)} = H(p, q) - H(p)$$

**Key identity:** Minimizing cross-entropy $H(p, q)$ = minimizing KL divergence $D_{\text{KL}}(p \| q)$ when $p$ is fixed

**Properties:** $D_{\text{KL}} \geq 0$ (Gibbs' inequality), $D_{\text{KL}}(p \| q) \neq D_{\text{KL}}(q \| p)$

### Mutual Information

$$I(X; Y) = H(X) - H(X \mid Y) = H(X) + H(Y) - H(X, Y)$$

$$I(X; Y) = D_{\text{KL}}(p(x,y) \| p(x)p(y))$$

---

## 6. Loss Functions

| Loss | Formula | When to Use |
|---|---|---|
| MSE | $\frac{1}{n}\sum(y_i - \hat{y}_i)^2$ | Regression |
| MAE | $\frac{1}{n}\sum \|y_i - \hat{y}_i\|$ | Robust regression |
| Cross-Entropy | $-\sum y_i \log \hat{y}_i$ | Classification |
| Binary CE | $-[y\log\hat{y} + (1-y)\log(1-\hat{y})]$ | Binary classification |
| Hinge | $\max(0, 1 - y\hat{y})$ | SVM |
| Huber | $\begin{cases} \frac{1}{2}x^2 & \|x\| \leq \delta \\ \delta\|x\| - \frac{1}{2}\delta^2 & \text{else} \end{cases}$ | Robust regression |
| KL Divergence | $\sum p \log(p/q)$ | VAE, distribution matching |
| Contrastive | $-\log \frac{e^{s_{pos}/\tau}}{\sum e^{s_i/\tau}}$ | Self-supervised learning |
| Focal | $-\alpha(1-\hat{y})^\gamma \log \hat{y}$ | Imbalanced classification |

---

## 7. Matrix Calculus (for Deep Learning)

### Scalar-by-Vector Derivatives

$$\frac{\partial}{\partial \mathbf{x}}(\mathbf{a}^T\mathbf{x}) = \mathbf{a} \qquad \frac{\partial}{\partial \mathbf{x}}(\mathbf{x}^T A\mathbf{x}) = (A + A^T)\mathbf{x}$$

$$\frac{\partial}{\partial \mathbf{x}}\|\mathbf{x}\|_2^2 = 2\mathbf{x} \qquad \frac{\partial}{\partial \mathbf{x}}\|\mathbf{x} - \mathbf{a}\|_2^2 = 2(\mathbf{x} - \mathbf{a})$$

### Common Layer Gradients

| Layer | Forward: $\mathbf{y} = $ | Backward: $\frac{\partial \mathcal{L}}{\partial \mathbf{x}} = $ |
|---|---|---|
| Linear | $W\mathbf{x} + \mathbf{b}$ | $W^T \frac{\partial \mathcal{L}}{\partial \mathbf{y}}$ |
| ReLU | $\max(0, \mathbf{x})$ | $\frac{\partial \mathcal{L}}{\partial \mathbf{y}} \odot \mathbf{1}_{x > 0}$ |
| Softmax + CE | $-\log \text{softmax}(\mathbf{x})_y$ | $\hat{\mathbf{y}} - \mathbf{e}_y$ |
| Batch Norm | $\gamma \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta$ | (complex, see [Calculus](../calculus/multivariable-calculus.md)) |

---

## 8. Dimensionality Reduction

**PCA:** Eigendecompose covariance $C = \frac{1}{n}X^TX$, project onto top-$k$ eigenvectors

$$\mathbf{z} = W_k^T(\mathbf{x} - \boldsymbol{\mu}) \quad \text{where } W_k = [\mathbf{v}_1, \ldots, \mathbf{v}_k]$$

**Explained variance ratio:** $\frac{\lambda_i}{\sum_j \lambda_j}$

**t-SNE objective:** Minimize $D_{\text{KL}}(P \| Q)$ where $P$ = pairwise similarities in high-dim, $Q$ = in low-dim (Student-t)

---

## 9. Bayesian ML Formulas

**Posterior:** $p(\theta \mid D) = \frac{p(D \mid \theta) p(\theta)}{p(D)}$

**Evidence (marginal likelihood):** $p(D) = \int p(D \mid \theta) p(\theta) \, d\theta$

**Predictive distribution:** $p(y^* \mid x^*, D) = \int p(y^* \mid x^*, \theta) p(\theta \mid D) \, d\theta$

**ELBO (Variational Inference):**

$$\log p(x) \geq \mathbb{E}_{q(\theta)}[\log p(x \mid \theta)] - D_{\text{KL}}(q(\theta) \| p(\theta))$$

**VAE loss:** $\mathcal{L} = -\mathbb{E}_{q(z|x)}[\log p(x|z)] + D_{\text{KL}}(q(z|x) \| p(z))$

---

## 10. Attention & Transformers

**Scaled dot-product attention:**

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

**Multi-head attention:**

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)W^O$$

$$\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$

**Positional encoding:**

$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d}}\right), \quad PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d}}\right)$$

**RoPE (Rotary):** $q_m^T k_n$ depends on relative position $(m-n)$ via rotation matrices

---

## 11. Evaluation Metrics

### Classification

$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$

$$\text{Precision} = \frac{TP}{TP + FP} \qquad \text{Recall} = \frac{TP}{TP + FN}$$

$$F_1 = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}} = \frac{2TP}{2TP + FP + FN}$$

$$\text{AUC-ROC:} \quad P(\hat{y}_{pos} > \hat{y}_{neg})$$

### Regression

$$\text{MSE} = \frac{1}{n}\sum(y_i - \hat{y}_i)^2 \qquad \text{RMSE} = \sqrt{\text{MSE}}$$

$$R^2 = 1 - \frac{\sum(y_i - \hat{y}_i)^2}{\sum(y_i - \bar{y})^2}$$

---

## 12. Complexity & Bounds

**Bias-Variance Decomposition:**

$$\mathbb{E}[(\hat{f}(x) - y)^2] = \text{Bias}^2 + \text{Variance} + \text{Noise}$$

**VC Dimension bound:** With probability $1 - \delta$:

$$R(\hat{f}) \leq \hat{R}(\hat{f}) + O\left(\sqrt{\frac{d_{\text{VC}} \log(n/d_{\text{VC}})}{n}}\right)$$

**PAC bound:** To learn with error $\leq \epsilon$ and probability $\geq 1-\delta$:

$$n \geq \frac{1}{\epsilon}\left(\log|H| + \log\frac{1}{\delta}\right)$$

---

## 13. Notation Reference

| Symbol | Meaning |
|---|---|
| $\mathbf{x}, \mathbf{v}$ | Column vectors (bold lowercase) |
| $A, W, X$ | Matrices (bold uppercase) |
| $\theta$ | Parameters |
| $\mathcal{L}$ | Loss function |
| $\nabla f$ | Gradient |
| $\mathbb{E}[\cdot]$ | Expectation |
| $\mathcal{N}(\mu, \sigma^2)$ | Gaussian distribution |
| $\propto$ | Proportional to |
| $\sim$ | Distributed as |
| $\odot$ | Element-wise product (Hadamard) |
| $\otimes$ | Tensor/Kronecker product |
| $\|x\|_p$ | $L^p$ norm |
| $\mathbb{R}^n$ | $n$-dimensional real space |
| $\mathbb{1}[\cdot]$ | Indicator function |
| $\arg\max$ | Argument that maximizes |
