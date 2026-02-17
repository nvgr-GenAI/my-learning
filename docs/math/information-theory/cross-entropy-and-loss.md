# Chapter 6.3: Cross-Entropy and Loss Functions

## Prerequisites

- **Probability Theory**: Discrete and continuous distributions, expectation
- **Information Theory Basics**: Entropy $H(p)$, KL divergence $D_{KL}(p\|q)$ (Chapter 6.1, 6.2)
- **Calculus**: Partial derivatives, chain rule, logarithmic differentiation
- **Linear Algebra**: Vector operations, matrix multiplication
- **Maximum Likelihood Estimation**: Log-likelihood, Bernoulli/categorical distributions

## Introduction

Cross-entropy is a fundamental concept bridging information theory and machine learning. While entropy $H(p)$ measures the inherent uncertainty in a distribution $p$, cross-entropy $H(p,q)$ measures the cost of encoding data from distribution $p$ using a code optimized for distribution $q$. This asymmetry makes cross-entropy the natural loss function for training classifiers, where $p$ represents true labels and $q$ represents model predictions.

**Key Insight**: Minimizing cross-entropy loss is equivalent to maximizing likelihood and minimizing KL divergence when the true distribution is fixed.

---

## 6.3.1 Cross-Entropy: Definition and Properties

**Definition 6.3.1 (Cross-Entropy)**
Let $p$ and $q$ be two probability distributions over a discrete sample space $\mathcal{X}$. The **cross-entropy** of $p$ with respect to $q$ is:

$$H(p, q) = -\sum_{x \in \mathcal{X}} p(x) \log q(x) = -\mathbb{E}_{x \sim p}[\log q(x)]$$

For continuous distributions with densities $p(x)$ and $q(x)$:

$$H(p, q) = -\int_{-\infty}^{\infty} p(x) \log q(x) \, dx$$

**Interpretation**: Cross-entropy is the expected number of bits needed to encode samples from $p$ using an optimal code designed for $q$.

**Example 6.3.1:** Let $p = (1, 0, 0)$ (one-hot for class 1) and $q = (0.7, 0.2, 0.1)$ (model prediction). Compute $H(p, q)$:

$$H(p, q) = -\sum_{k=1}^{3} p_k \log q_k = -[1 \cdot \log(0.7) + 0 \cdot \log(0.2) + 0 \cdot \log(0.1)]$$

$$= -\log(0.7) = -(-0.3567) = 0.357 \text{ nats} \quad (\approx 0.515 \text{ bits})$$

Only the true class contributes since $p_2 = p_3 = 0$. A perfect model with $q = (1, 0, 0)$ would give $H(p, q) = -\log(1) = 0$, so 0.357 nats represents the cost of the model's imperfect prediction.

### Information-Theoretic Interpretation

```
True Distribution p         Model Distribution q

Event: x₁ (p=0.7)          Assigns: q(x₁)=0.5
     ↓                           ↓
Occurs 70% of time         Uses -log(0.5)=1 bit code
     ↓                           ↓
Expected cost: 0.7 × 1 = 0.7 bits (inefficient!)

Optimal code for p: -log(0.7)=0.51 bits
Penalty: 0.49 extra bits per sample
```

**Theorem 6.3.1 (Cross-Entropy Lower Bound)**
For any distributions $p$ and $q$ over the same space:

$$H(p, q) \geq H(p)$$

with equality if and only if $p = q$ almost everywhere.

**Proof**: Direct consequence of Gibbs' inequality. By definition of KL divergence:

$$D_{KL}(p \| q) = \sum p(x) \log \frac{p(x)}{q(x)} \geq 0$$

Expanding:

$$\sum p(x) \log p(x) - \sum p(x) \log q(x) \geq 0$$
$$-H(p) + H(p, q) \geq 0$$
$$H(p, q) \geq H(p)$$

Equality holds when $D_{KL}(p \| q) = 0$, i.e., $p = q$. ∎

---

## 6.3.2 Relationship to KL Divergence

**Theorem 6.3.2 (Cross-Entropy Decomposition)**
For any distributions $p$ and $q$:

$$H(p, q) = H(p) + D_{KL}(p \| q)$$

**Proof**: By definition:

$$H(p, q) = -\sum p(x) \log q(x)$$
$$= -\sum p(x) \log p(x) + \sum p(x) \log p(x) - \sum p(x) \log q(x)$$
$$= H(p) + \sum p(x) \log \frac{p(x)}{q(x)} = H(p) + D_{KL}(p \| q)$$

∎

**Example 6.3.2:** Verify the decomposition for $p = (0.8, 0.2)$ and $q = (0.6, 0.4)$:

**Step 1 — Entropy** $H(p)$:
$$H(p) = -[0.8 \ln(0.8) + 0.2 \ln(0.2)] = -[0.8(-0.2231) + 0.2(-1.6094)] = 0.1785 + 0.3219 = 0.5004 \text{ nats}$$

**Step 2 — KL divergence** $D_{KL}(p \| q)$:
$$D_{KL}(p \| q) = 0.8 \ln\!\frac{0.8}{0.6} + 0.2 \ln\!\frac{0.2}{0.4} = 0.8(0.2877) + 0.2(-0.6931) = 0.2302 - 0.1386 = 0.0916 \text{ nats}$$

**Step 3 — Cross-entropy** $H(p, q)$ (direct):
$$H(p, q) = -[0.8 \ln(0.6) + 0.2 \ln(0.4)] = -[0.8(-0.5108) + 0.2(-0.9163)] = 0.4087 + 0.1833 = 0.5920 \text{ nats}$$

**Verify**: $H(p) + D_{KL}(p \| q) = 0.5004 + 0.0916 = 0.5920 = H(p, q)$ ✓

**Corollary 6.3.1 (Optimization Equivalence)**
When the true distribution $p$ is fixed, the following are equivalent:

1. Minimize $H(p, q)$ over $q$
2. Minimize $D_{KL}(p \| q)$ over $q$
3. Maximize $\mathbb{E}_{x \sim p}[\log q(x)]$ over $q$

**Significance for ML**: In supervised learning, true labels define $p$ (fixed), so cross-entropy loss directly measures how well the model $q$ approximates $p$.

---

## 6.3.3 Cross-Entropy Loss for Classification

### Binary Cross-Entropy

**Definition 6.3.2 (Binary Cross-Entropy Loss)**
For binary classification with true label $y \in \{0, 1\}$ and predicted probability $\hat{y} = P(y=1)$:

$$\mathcal{L}_{BCE}(y, \hat{y}) = -[y \log \hat{y} + (1-y) \log(1-\hat{y})]$$

**Derivation from Maximum Likelihood**:
Model output as Bernoulli distribution: $P(y | \hat{y}) = \hat{y}^y (1-\hat{y})^{1-y}$

Log-likelihood for single sample:
$$\log P(y | \hat{y}) = y \log \hat{y} + (1-y) \log(1-\hat{y})$$

Negative log-likelihood (loss):
$$\mathcal{L}_{BCE} = -\log P(y | \hat{y})$$

**Example 6.3.3:** Compute the binary cross-entropy loss for $y = 1$ and $\hat{y} = 0.8$:

$$\mathcal{L}_{BCE} = -[y \log \hat{y} + (1-y) \log(1-\hat{y})]$$

$$= -[1 \cdot \ln(0.8) + 0 \cdot \ln(0.2)] = -\ln(0.8) = -(-0.2231) = 0.2231 \text{ nats}$$

Compare: if the model were less confident at $\hat{y} = 0.6$: $\mathcal{L} = -\ln(0.6) = 0.5108$ nats (2.3x higher loss). If more confident at $\hat{y} = 0.95$: $\mathcal{L} = -\ln(0.95) = 0.0513$ nats (4.3x lower).

**Asymmetry**: Binary CE penalizes wrong predictions differently:

```
True y=1, predict ŷ=0.1:  -log(0.1) = 2.30 bits
True y=0, predict ŷ=0.9:  -log(0.1) = 2.30 bits  (symmetric)

True y=1, predict ŷ=0.99: -log(0.99) = 0.01 bits (low loss)
True y=1, predict ŷ=0.01: -log(0.01) = 4.61 bits (high loss)
```

**Example 6.3.4:** Compute the gradient $\frac{\partial \mathcal{L}_{BCE}}{\partial \hat{y}}$ for $y = 1$, $\hat{y} = 0.8$:

$$\frac{\partial \mathcal{L}_{BCE}}{\partial \hat{y}} = -\frac{y}{\hat{y}} + \frac{1-y}{1-\hat{y}} = -\frac{1}{0.8} + \frac{0}{0.2} = -1.25$$

The negative gradient means the loss decreases as $\hat{y}$ increases (model should become more confident). For $y = 0$, $\hat{y} = 0.8$:

$$\frac{\partial \mathcal{L}_{BCE}}{\partial \hat{y}} = -\frac{0}{0.8} + \frac{1}{0.2} = +5.0$$

The large positive gradient strongly pushes $\hat{y}$ down -- the model is confidently wrong ($\hat{y} = 0.8$ but true $y = 0$), so the gradient signal is 4x stronger than the correctly-classified case.

### Categorical Cross-Entropy

**Definition 6.3.3 (Categorical Cross-Entropy Loss)**
For $K$-class classification with one-hot encoded true label $\mathbf{y} = [y_1, \ldots, y_K]$ and predicted probabilities $\hat{\mathbf{y}} = [\hat{y}_1, \ldots, \hat{y}_K]$:

$$\mathcal{L}_{CCE}(\mathbf{y}, \hat{\mathbf{y}}) = -\sum_{k=1}^K y_k \log \hat{y}_k$$

Since $y_k \in \{0, 1\}$ with $\sum_k y_k = 1$ (one-hot), if $y_c = 1$ (true class $c$):

$$\mathcal{L}_{CCE} = -\log \hat{y}_c$$

**Derivation from Categorical Distribution**:
Model output as categorical: $P(\mathbf{y} | \hat{\mathbf{y}}) = \prod_{k=1}^K \hat{y}_k^{y_k}$

Log-likelihood:
$$\log P(\mathbf{y} | \hat{\mathbf{y}}) = \sum_{k=1}^K y_k \log \hat{y}_k$$

Negative log-likelihood gives categorical cross-entropy.

**Example**: 3-class classification

```
True:      y = [0, 1, 0]         (class 2)
Predicted: ŷ = [0.1, 0.7, 0.2]
Loss:      -log(0.7) = 0.36 bits

Bad prediction: ŷ' = [0.6, 0.1, 0.3]
Loss:              -log(0.1) = 2.30 bits (6× higher)
```

**Example 6.3.5:** 4-class classification (e.g., cat/dog/bird/fish). True class is "bird" ($c = 3$):

$$\mathbf{y} = [0, 0, 1, 0], \quad \hat{\mathbf{y}} = [0.1, 0.15, 0.6, 0.15]$$

$$\mathcal{L}_{CCE} = -\sum_{k=1}^{4} y_k \log \hat{y}_k = -[0 \cdot \ln(0.1) + 0 \cdot \ln(0.15) + 1 \cdot \ln(0.6) + 0 \cdot \ln(0.15)]$$

$$= -\ln(0.6) = 0.5108 \text{ nats}$$

For a **batch of 3 samples**, the average loss is:

| Sample | True class | $\hat{y}_c$ | Loss $-\ln(\hat{y}_c)$ |
| ------ | ---------- | ----------- | ---------------------- |
| 1      | bird (k=3) | 0.60        | 0.5108                 |
| 2      | cat (k=1)  | 0.85        | 0.1625                 |
| 3      | fish (k=4) | 0.30        | 1.2040                 |

$$\mathcal{L}_{batch} = \frac{1}{3}(0.5108 + 0.1625 + 1.2040) = 0.6258 \text{ nats}$$

---

## 6.3.4 Softmax and Cross-Entropy

### The Softmax Function

**Definition 6.3.4 (Softmax)**
Given logits $\mathbf{z} = [z_1, \ldots, z_K] \in \mathbb{R}^K$, the softmax function produces probabilities:

$$\text{softmax}(\mathbf{z})_k = \frac{e^{z_k}}{\sum_{j=1}^K e^{z_j}}$$

**Properties**:
- $\hat{y}_k \in (0, 1)$ for all $k$
- $\sum_{k=1}^K \hat{y}_k = 1$ (valid probability distribution)
- Differentiable everywhere
- Monotone: higher logit → higher probability

**Example 6.3.6:** Compute softmax for logits $\mathbf{z} = (2.0, 1.0, 0.1)$:

**Step 1 — Exponentiate** each logit:
$$e^{2.0} = 7.389, \quad e^{1.0} = 2.718, \quad e^{0.1} = 1.105$$

**Step 2 — Sum**: $\sum_j e^{z_j} = 7.389 + 2.718 + 1.105 = 11.212$

**Step 3 — Normalize**:
$$\hat{y}_1 = \frac{7.389}{11.212} = 0.659, \quad \hat{y}_2 = \frac{2.718}{11.212} = 0.242, \quad \hat{y}_3 = \frac{1.105}{11.212} = 0.099$$

Result: $\text{softmax}(2.0, 1.0, 0.1) = (0.659, 0.242, 0.099)$. Note the highest logit (2.0) gets the largest probability, and the outputs sum to 1.

### Numerical Stability: Log-Sum-Exp Trick

**Problem**: For large logits, $e^{z_k}$ overflows; for negative logits, underflows to 0.

**Solution**: Subtract max logit before exponentiation:

$$\text{softmax}(\mathbf{z})_k = \frac{e^{z_k - \max_j z_j}}{\sum_{j=1}^K e^{z_j - \max_j z_j}}$$

**Theorem 6.3.3 (Translation Invariance)**
For any constant $c \in \mathbb{R}$:
$$\text{softmax}(\mathbf{z} + c\mathbf{1}) = \text{softmax}(\mathbf{z})$$

where $\mathbf{1} = [1, \ldots, 1]^T$.

**Proof**:
$$\frac{e^{z_k + c}}{\sum_j e^{z_j + c}} = \frac{e^c \cdot e^{z_k}}{e^c \sum_j e^{z_j}} = \frac{e^{z_k}}{\sum_j e^{z_j}}$$
∎

### Gradient of Softmax + Cross-Entropy

**Theorem 6.3.4 (Softmax-CrossEntropy Gradient)**
Let $\mathbf{z} \in \mathbb{R}^K$ be logits, $\hat{\mathbf{y}} = \text{softmax}(\mathbf{z})$, and $\mathbf{y}$ be one-hot true labels. The gradient of the cross-entropy loss is:

$$\frac{\partial \mathcal{L}_{CCE}}{\partial z_k} = \hat{y}_k - y_k$$

**Proof**: Loss is $\mathcal{L} = -\sum_i y_i \log \hat{y}_i$ where $\hat{y}_i = \frac{e^{z_i}}{\sum_j e^{z_j}}$.

First, compute $\frac{\partial \hat{y}_i}{\partial z_k}$:

For $i = k$:
$$\frac{\partial \hat{y}_k}{\partial z_k} = \frac{e^{z_k} \sum_j e^{z_j} - e^{z_k} \cdot e^{z_k}}{(\sum_j e^{z_j})^2} = \hat{y}_k(1 - \hat{y}_k)$$

For $i \neq k$:
$$\frac{\partial \hat{y}_i}{\partial z_k} = \frac{0 - e^{z_i} \cdot e^{z_k}}{(\sum_j e^{z_j})^2} = -\hat{y}_i \hat{y}_k$$

Now compute loss gradient:
$$\frac{\partial \mathcal{L}}{\partial z_k} = -\sum_i y_i \frac{1}{\hat{y}_i} \frac{\partial \hat{y}_i}{\partial z_k}$$

$$= -y_k \frac{1}{\hat{y}_k} \cdot \hat{y}_k(1 - \hat{y}_k) - \sum_{i \neq k} y_i \frac{1}{\hat{y}_i} \cdot (-\hat{y}_i \hat{y}_k)$$

$$= -y_k(1 - \hat{y}_k) + \sum_{i \neq k} y_i \hat{y}_k$$

$$= -y_k + y_k \hat{y}_k + \hat{y}_k \sum_{i \neq k} y_i$$

Since $\sum_i y_i = 1$ (one-hot):
$$= -y_k + \hat{y}_k(y_k + \sum_{i \neq k} y_i) = -y_k + \hat{y}_k$$

$$= \hat{y}_k - y_k$$

∎

**Significance**: The gradient has an elegant form — prediction error for each class. This makes backpropagation through softmax + cross-entropy efficient.

**Example 6.3.7:** Using the softmax output from Example 6.3.6, suppose the true class is $k=1$, so $\mathbf{y} = (1, 0, 0)$ and $\hat{\mathbf{y}} = (0.659, 0.242, 0.099)$. The gradient with respect to each logit is:

$$\frac{\partial \mathcal{L}}{\partial z_1} = \hat{y}_1 - y_1 = 0.659 - 1 = -0.341$$

$$\frac{\partial \mathcal{L}}{\partial z_2} = \hat{y}_2 - y_2 = 0.242 - 0 = +0.242$$

$$\frac{\partial \mathcal{L}}{\partial z_3} = \hat{y}_3 - y_3 = 0.099 - 0 = +0.099$$

The negative gradient for $z_1$ pushes the true-class logit **up**, while positive gradients for $z_2, z_3$ push competing logits **down**. Note the gradients sum to zero: $-0.341 + 0.242 + 0.099 = 0$.

---

## 6.3.5 Label Smoothing

**Motivation**: Hard one-hot labels encourage overconfident predictions ($\hat{y}_k \to 1$), which can lead to:
- Poor calibration (predicted probabilities don't match true frequencies)
- Overfitting to noisy labels
- Large logit magnitudes (numerical issues)

**Definition 6.3.5 (Label Smoothing)**
Replace hard target $\mathbf{y}$ with smoothed target $\mathbf{y}^{LS}$:

$$y_k^{LS} = \begin{cases}
1 - \epsilon + \frac{\epsilon}{K} & \text{if } k = c \text{ (true class)} \\
\frac{\epsilon}{K} & \text{otherwise}
\end{cases}$$

where $\epsilon \in [0, 1)$ is the smoothing parameter (commonly $\epsilon = 0.1$) and $K$ is the number of classes.

**Example**: 3-class problem, true class 2, $\epsilon = 0.1$

```
Hard label:      y = [0.0,   1.0,   0.0  ]
Smoothed label:  y = [0.033, 0.933, 0.033]
```

**Theorem 6.3.5 (Label Smoothing as Regularization)**
Label smoothing adds a regularization term to the standard cross-entropy:

$$\mathcal{L}_{LS} = (1 - \epsilon) \mathcal{L}_{CCE} + \epsilon \mathcal{L}_{uniform}$$

where $\mathcal{L}_{uniform} = -\frac{1}{K} \sum_{k=1}^K \log \hat{y}_k$ encourages uniform predictions.

**Proof**: Expand smoothed loss:
$$\mathcal{L}_{LS} = -\sum_k y_k^{LS} \log \hat{y}_k$$

For true class $c$:
$$= -\left(1 - \epsilon + \frac{\epsilon}{K}\right) \log \hat{y}_c - \sum_{k \neq c} \frac{\epsilon}{K} \log \hat{y}_k$$

$$= -(1 - \epsilon) \log \hat{y}_c - \frac{\epsilon}{K} \sum_{k=1}^K \log \hat{y}_k$$

$$= (1 - \epsilon)(-\log \hat{y}_c) + \epsilon \left(-\frac{1}{K} \sum_{k=1}^K \log \hat{y}_k\right)$$

$$= (1 - \epsilon) \mathcal{L}_{CCE} + \epsilon \mathcal{L}_{uniform}$$

∎

**Benefits**:
- Prevents overconfident predictions
- Improves model calibration
- Reduces overfitting (acts as regularization)
- Better generalization in practice

---

## 6.3.6 Focal Loss

**Motivation**: In object detection, class imbalance is severe (background vs objects ~1000:1). Standard cross-entropy gives equal weight to easy negatives, overwhelming the gradient signal from rare positives.

**Definition 6.3.6 (Focal Loss)**
For binary classification:

$$\mathcal{L}_{focal}(y, \hat{y}) = -\alpha_y (1 - \hat{y}_t)^\gamma \log(\hat{y}_t)$$

where:
- $\hat{y}_t = \begin{cases} \hat{y} & \text{if } y = 1 \\ 1 - \hat{y} & \text{if } y = 0 \end{cases}$ (probability of true class)
- $\gamma \geq 0$ is the focusing parameter (commonly $\gamma = 2$)
- $\alpha_y \in [0, 1]$ is the class weight (commonly $\alpha_1 = 0.25$)

**Multiclass Extension**:
$$\mathcal{L}_{focal} = -\alpha_c (1 - \hat{y}_c)^\gamma \log(\hat{y}_c)$$

where $c$ is the true class.

### Down-Weighting Easy Examples

**Key Mechanism**: The term $(1 - \hat{y}_t)^\gamma$ modulates the loss:

```
Prediction Quality    (1 - ŷₜ)²   Standard CE   Focal Loss (γ=2)
────────────────────────────────────────────────────────────────
Easy   (ŷₜ = 0.9)     0.01        0.11          0.001   (÷100)
Medium (ŷₜ = 0.7)     0.09        0.36          0.032   (÷11)
Hard   (ŷₜ = 0.5)     0.25        0.69          0.173   (÷4)
Very Hard (ŷₜ = 0.1)  0.81        2.30          1.863   (÷1.2)
```

**Effect of $\gamma$**:
- $\gamma = 0$: Focal loss = standard cross-entropy
- $\gamma = 1$: Linear down-weighting
- $\gamma = 2$: Quadratic down-weighting (common choice)
- $\gamma = 5$: Aggressive focus on hard examples

**Comparison to Standard Cross-Entropy**:

| Aspect | Standard CE | Focal Loss |
|--------|-------------|------------|
| Easy examples | Full loss contribution | Heavily down-weighted |
| Hard examples | Full loss contribution | Dominant loss contribution |
| Balanced data | Works well | Unnecessary overhead |
| Imbalanced data (1:1000) | Swamped by easy negatives | Focuses on hard cases |
| Detection (RetinaNet) | mAP ~31% | mAP ~39% (+8 points) |

**Theorem 6.3.6 (Focal Loss Gradient)**
The gradient with respect to logit $z$ (where $\hat{y} = \sigma(z)$) is:

$$\frac{\partial \mathcal{L}_{focal}}{\partial z} = \alpha(1 - \hat{y}_t)^\gamma \left[\gamma \hat{y}_t \log(\hat{y}_t) + \hat{y}_t - y\right]$$

**Interpretation**: Gradient magnitude scales with $(1 - \hat{y}_t)^\gamma$, suppressing updates from easy examples.

---

## 6.3.7 Connection to Coding Theory

Cross-entropy has a precise interpretation in Shannon's coding theory.

**Theorem 6.3.7 (Cross-Entropy as Expected Code Length)**
Let $p$ be the true distribution and $q$ be a model distribution. If we design a prefix-free code with codeword length $-\log q(x)$ for symbol $x$, the expected code length for data from $p$ is:

$$L_{\text{avg}} = \mathbb{E}_{x \sim p}[-\log q(x)] = H(p, q)$$

The inefficiency compared to the optimal code (designed for $p$) is exactly the KL divergence:

$$L_{\text{avg}} - L_{\text{optimal}} = H(p, q) - H(p) = D_{KL}(p \| q)$$

**Example**: Weather prediction coding

```
True weather distribution p:
  Sunny:  p = 0.7  →  Optimal code: 1 bit   (code: "0")
  Rainy:  p = 0.3  →  Optimal code: 2 bits  (code: "10")

Model distribution q (poor):
  Sunny:  q = 0.5  →  Uses 1 bit code
  Rainy:  q = 0.5  →  Uses 1 bit code

Expected code length under p using q's code:
  H(p,q) = -[0.7·log(0.5) + 0.3·log(0.5)] = 1.0 bit/symbol

Optimal (using p's code):
  H(p) = -[0.7·log(0.7) + 0.3·log(0.3)] = 0.88 bits/symbol

Inefficiency: 0.12 bits/symbol wasted
```

**Cross-Entropy in Compression**:
- Arithmetic coding uses model $q(x)$ to assign code lengths
- Cross-entropy $H(p, q)$ predicts compression ratio on test data from $p$
- Minimizing cross-entropy = designing better compression codes

---

## 6.3.8 Temperature Scaling

**Motivation**: Softmax sharpness affects:
- Model calibration (confidence vs accuracy)
- Knowledge distillation (transferring soft targets)
- Exploration in reinforcement learning
- LLM text generation diversity

**Definition 6.3.7 (Temperature Softmax)**
Given logits $\mathbf{z}$ and temperature $T > 0$:

$$\text{softmax}_T(\mathbf{z})_k = \frac{e^{z_k / T}}{\sum_{j=1}^K e^{z_j / T}}$$

### Effect of Temperature

**Cold ($T < 1$)**: Sharpens distribution, amplifies differences

```
Logits:    z = [2.0, 1.0, 0.5]

T = 0.5:   ŷ = [0.76, 0.18, 0.06]  (very confident)
T = 1.0:   ŷ = [0.58, 0.24, 0.18]  (standard)
T = 2.0:   ŷ = [0.42, 0.31, 0.27]  (soft)
T → 0:     ŷ → [1.0, 0.0, 0.0]     (argmax, one-hot)
T → ∞:     ŷ → [0.33, 0.33, 0.33]  (uniform)
```

**Theorem 6.3.8 (Temperature Limiting Behavior)**
1. As $T \to 0^+$: $\text{softmax}_T(\mathbf{z}) \to \mathbf{e}_{\arg\max_k z_k}$ (one-hot at argmax)
2. As $T \to \infty$: $\text{softmax}_T(\mathbf{z}) \to \mathbf{u}$ where $u_k = 1/K$ (uniform)

**Proof Sketch**:
For (1), as $T \to 0$, let $z_{\max} = \max_k z_k$:
$$\frac{e^{z_k/T}}{e^{z_{\max}/T}} = e^{(z_k - z_{\max})/T} \to \begin{cases} 1 & k = \arg\max \\ 0 & \text{otherwise} \end{cases}$$

For (2), as $T \to \infty$: $z_k/T \to 0$ for all $k$, so $e^{z_k/T} \to 1$, giving uniform distribution. ∎

### Applications

**1. Knowledge Distillation**
Train student network to match teacher's soft predictions with $T > 1$:

$$\mathcal{L}_{KD} = T^2 \cdot D_{KL}\left(\text{softmax}_T(\mathbf{z}^{teacher}) \| \text{softmax}_T(\mathbf{z}^{student})\right)$$

The $T^2$ factor compensates for gradient scaling. Higher temperature transfers more information about class relationships.

**2. Model Calibration**
Post-training, adjust temperature to match predicted confidence to accuracy:

$$\text{Calibration Error} = \mathbb{E}[|\text{Accuracy} - \text{Confidence}|]$$

Find optimal $T^*$ on validation set to minimize calibration error.

**3. LLM Sampling**
Control text generation diversity:
- $T = 0$: Greedy (deterministic)
- $T = 0.7$: Focused, coherent (common for factual tasks)
- $T = 1.0$: Balanced creativity
- $T = 1.5$: High diversity (creative writing)

---

## Machine Learning Connections

### Cross-Entropy as THE Classification Loss

**Why Cross-Entropy Dominates**:

1. **Probabilistic Foundation**: Directly derived from maximum likelihood estimation
2. **Proper Scoring Rule**: Encourages honest probability estimates
3. **Convex (when combined with log-linear models)**: Efficient optimization
4. **Unbounded Penalty**: Heavily penalizes confident wrong predictions

**Comparison to Alternatives**:

| Loss Function | Formula | Problem |
|---------------|---------|---------|
| 0-1 Loss | $\mathbb{1}[\hat{y} \neq y]$ | Non-differentiable, no gradient signal |
| Squared Error | $(\hat{y} - y)^2$ | Saturates gradients for sigmoids, not proper scoring |
| Hinge Loss | $\max(0, 1 - y\hat{y})$ | For SVMs, not probabilistic |
| Cross-Entropy | $-\log(\hat{y}_c)$ | Standard, well-calibrated, smooth gradients |

**Example 6.3.8:** Compare MSE vs cross-entropy for binary classification with $y = 1$:

| $\hat{y}$ | MSE: $(\hat{y} - 1)^2$ | CE: $-\ln(\hat{y})$ | MSE gradient: $2(\hat{y}-1)$ | CE gradient: $-1/\hat{y}$ |
| ---------- | ----------------------- | -------------------- | ---------------------------- | -------------------------- |
| 0.95       | 0.0025                  | 0.0513               | -0.10                        | -1.05                      |
| 0.80       | 0.0400                  | 0.2231               | -0.40                        | -1.25                      |
| 0.50       | 0.2500                  | 0.6931               | -1.00                        | -2.00                      |
| 0.10       | 0.8100                  | 2.3026               | -1.80                        | -10.00                     |
| 0.01       | 0.9801                  | 4.6052               | -1.98                        | -100.00                    |

At $\hat{y} = 0.01$ (confidently wrong), the CE gradient is $-100$ while the MSE gradient is only $-1.98$. Cross-entropy produces a **50x stronger** learning signal for catastrophically wrong predictions, which is why it is preferred for classification.

### Connection to Maximum Likelihood Estimation

**Fundamental Identity**:
$$\text{Minimize Cross-Entropy} \equiv \text{Maximize Likelihood}$$

For dataset $\mathcal{D} = \{(\mathbf{x}_i, y_i)\}_{i=1}^N$ with model $P_\theta(y|\mathbf{x})$:

$$\arg\max_\theta \prod_{i=1}^N P_\theta(y_i | \mathbf{x}_i) = \arg\min_\theta -\frac{1}{N} \sum_{i=1}^N \log P_\theta(y_i | \mathbf{x}_i)$$

The right side is average cross-entropy loss.

### Modern Training Practices

**Label Smoothing** (Szegedy et al., 2016):
- Used in Inception-v3, ResNet training
- Improves top-1 accuracy by 0.2-0.5%
- Better calibration on out-of-distribution data

**Focal Loss** (Lin et al., 2017):
- Enabled one-stage object detectors (RetinaNet)
- State-of-the-art performance without two-stage complexity
- Standard in dense prediction tasks (detection, segmentation)

**Temperature Scaling**:
- Knowledge distillation (Hinton et al., 2015): $T = 3-5$ typical
- LLM sampling: User-adjustable hyperparameter
- Post-hoc calibration: Platt scaling generalization

---

## Summary

**Key Concepts**:

1. **Cross-Entropy**: $H(p,q) = -\sum p(x) \log q(x)$ measures encoding cost when using wrong distribution
2. **Decomposition**: $H(p,q) = H(p) + D_{KL}(p\|q)$ — minimizing CE = minimizing KL when $p$ fixed
3. **Classification Loss**: Negative log-likelihood of true class under model distribution
4. **Softmax Gradient**: $\nabla_z \mathcal{L} = \hat{\mathbf{y}} - \mathbf{y}$ — elegantly simple error signal
5. **Label Smoothing**: Regularization preventing overconfidence, $(1-\epsilon)$ on true class
6. **Focal Loss**: $(1-\hat{y}_t)^\gamma$ down-weights easy examples for imbalanced data
7. **Temperature**: Controls distribution sharpness, $T<1$ (sharp), $T>1$ (soft)

**Practical Significance**:
- Cross-entropy is the universal loss for classification
- Derived from principled maximum likelihood estimation
- Admits efficient computation and stable gradients
- Extensions (focal, smoothing, temperature) address real-world challenges

---

## Related Topics

- **Chapter 6.1**: Entropy and Information — foundation for cross-entropy
- **Chapter 6.2**: KL Divergence and Mutual Information — cross-entropy decomposition
- **Chapter 7.4**: Maximum Likelihood Estimation — probabilistic interpretation of CE loss
- **Chapter 9.3**: Neural Network Training — practical use of CE loss with backpropagation
- **Chapter 11.2**: Calibration and Uncertainty — temperature scaling for better confidence
- **Chapter 12.5**: Multi-Task Learning — balancing multiple cross-entropy losses

---

## Exercises

### ★ Basic Understanding

**Exercise 6.3.1**: Compute cross-entropy $H(p,q)$ for:
- $p = [0.5, 0.3, 0.2]$, $q = [0.4, 0.4, 0.2]$
- Verify $H(p,q) \geq H(p)$

**Exercise 6.3.2**: For binary classification with $y=1$, compute BCE loss for:
- $\hat{y} = 0.9$
- $\hat{y} = 0.1$
- $\hat{y} = 0.5$

**Exercise 6.3.3**: Apply softmax to logits $\mathbf{z} = [2, 1, 0.5]$ with $T = 0.5, 1, 2$. Observe sharpening/smoothing.

### ★★ Intermediate Problems

**Exercise 6.3.4**: Prove that binary cross-entropy is symmetric in misclassification cost: show that the loss for predicting $\hat{y}$ when $y=1$ equals the loss for predicting $1-\hat{y}$ when $y=0$.

**Exercise 6.3.5**: Derive the gradient $\frac{\partial \mathcal{L}_{BCE}}{\partial z}$ where $\hat{y} = \sigma(z)$ and $\sigma$ is the sigmoid function. Show it simplifies to $\hat{y} - y$.

**Exercise 6.3.6**: Implement label smoothing with $\epsilon=0.1$ for a 5-class problem with true class $c=3$. Compute smoothed loss for predictions $\hat{\mathbf{y}} = [0.05, 0.1, 0.7, 0.1, 0.05]$.

**Exercise 6.3.7**: For focal loss with $\gamma=2$, $\alpha=0.25$, compute loss for:
- Easy positive: $y=1$, $\hat{y}=0.9$
- Hard positive: $y=1$, $\hat{y}=0.6$
- Compare to standard BCE.

### ★★★ Advanced Problems

**Exercise 6.3.8**: Prove that for fixed $p$, the distribution $q^* = p$ uniquely minimizes $H(p,q)$ over all distributions $q$.

**Exercise 6.3.9**: Show that the Hessian of binary cross-entropy is:
$$\frac{\partial^2 \mathcal{L}_{BCE}}{\partial z^2} = \hat{y}(1-\hat{y})$$
Explain why this causes vanishing gradients when $\hat{y} \to 0$ or $\hat{y} \to 1$.

**Exercise 6.3.10**: For temperature scaling, prove that:
$$\frac{\partial \text{softmax}_T(\mathbf{z})_k}{\partial T} = \frac{1}{T^2} \text{softmax}_T(\mathbf{z})_k \left(z_k - \sum_j z_j \text{softmax}_T(\mathbf{z})_j\right)$$
Interpret this gradient for model calibration.

**Exercise 6.3.11**: Derive the optimal temperature $T^*$ for knowledge distillation by minimizing:
$$\mathbb{E}_{(\mathbf{x},y) \sim \mathcal{D}}\left[D_{KL}(\text{softmax}_T(\mathbf{z}^{teacher}) \| \text{softmax}_T(\mathbf{z}^{student}))\right]$$
assuming Gaussian noise in logits.

**Exercise 6.3.12**: Prove that focal loss with $\gamma > 0$ is not a proper scoring rule (i.e., true probabilities don't always minimize expected loss). Discuss implications for probabilistic interpretation.

---

## Further Reading

**Classic Papers**:
- Shannon (1948): "A Mathematical Theory of Communication" — foundation of information theory
- Hinton et al. (2015): "Distilling the Knowledge in a Neural Network" — temperature scaling
- Szegedy et al. (2016): "Rethinking the Inception Architecture" — label smoothing
- Lin et al. (2017): "Focal Loss for Dense Object Detection" — focal loss for imbalanced data

**Modern Surveys**:
- Guo et al. (2017): "On Calibration of Modern Neural Networks" — temperature scaling for calibration
- Zhang et al. (2021): "A Survey on Loss Functions" — comprehensive comparison

**Textbooks**:
- Cover & Thomas (2006): "Elements of Information Theory" (Ch. 2) — rigorous treatment
- Murphy (2022): "Probabilistic Machine Learning" (Ch. 5) — ML perspective
- Goodfellow et al. (2016): "Deep Learning" (Ch. 3, 6) — practical implementation
