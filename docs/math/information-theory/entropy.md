# Chapter 6.1: Entropy

## Prerequisites

- **Probability Theory**: Random variables, probability distributions, expectation (Chapter 3)
- **Logarithms**: Properties of logarithms, different bases (Chapter 1.3)
- **Calculus**: Integration, differentiation, Lagrange multipliers (Chapters 2.1-2.4)
- **Optimization**: Constrained optimization, KKT conditions (Chapter 5.2)

## Introduction

Entropy is a fundamental concept in information theory that quantifies the average amount of uncertainty or information content in a random variable. Introduced by Claude Shannon in 1948, entropy provides a mathematical framework for measuring information and has profound implications across machine learning, from decision tree construction to uncertainty quantification in probabilistic models.

The key insight is that entropy measures the "average surprise" when observing outcomes from a distribution: rare events are more surprising (carry more information) than common ones.

---

## 6.1.1 Shannon Entropy

### Definition 6.1.1 (Shannon Entropy - Discrete Case)

Let $X$ be a discrete random variable taking values in a finite set $\mathcal{X}$ with probability mass function $p(x) = P(X = x)$. The **Shannon entropy** of $X$ is defined as:

$$H(X) = -\sum_{x \in \mathcal{X}} p(x) \log p(x)$$

By convention, we set $0 \log 0 = 0$ (justified by $\lim_{p \to 0^+} p \log p = 0$).

**Units**: The unit of entropy depends on the logarithm base:
- Base 2: bits (binary digits)
- Base $e$: nats (natural units)
- Base 10: hartleys (decimal digits)

In machine learning, base 2 is most common. Throughout this chapter, $\log$ denotes $\log_2$ unless specified otherwise.

### Interpretation

Entropy can be interpreted in three equivalent ways:

1. **Average Surprise**: The expected value of the "surprise" $-\log p(x)$ when observing $X$
   $$H(X) = \mathbb{E}[-\log p(X)]$$

2. **Average Information Content**: The average number of bits needed to encode outcomes of $X$ optimally

3. **Uncertainty Measure**: A quantification of how unpredictable $X$ is

**Example 6.1.1** (Fair Coin):
Let $X \sim \text{Bernoulli}(0.5)$. Then:
$$H(X) = -0.5 \log_2(0.5) - 0.5 \log_2(0.5) = 1 \text{ bit}$$

**Example 6.1.2** (Biased Coin):
Let $X \sim \text{Bernoulli}(0.9)$. Then:
$$H(X) = -0.9 \log_2(0.9) - 0.1 \log_2(0.1) \approx 0.469 \text{ bits}$$

The biased coin has lower entropy than the fair coin, reflecting reduced uncertainty.

**Example 6.1.3** (Deterministic Variable):
If $P(X = x_0) = 1$ for some $x_0$, then:
$$H(X) = -1 \cdot \log_2(1) = 0$$

No uncertainty means zero entropy.

**Example 6.1.4** (Fair Die):
Let $X$ be uniform on $\{1, 2, 3, 4, 5, 6\}$, so $p(x) = 1/6$ for each outcome. Then:
$$H(X) = -\sum_{i=1}^{6} \frac{1}{6} \log_2 \frac{1}{6} = -6 \cdot \frac{1}{6} \log_2 \frac{1}{6} = \log_2 6 \approx 2.585 \text{ bits}$$

A fair die requires about 2.585 bits to encode each outcome, compared to 1 bit for a fair coin. More equally likely outcomes means higher entropy.

---

## 6.1.2 Properties of Shannon Entropy

### Theorem 6.1.1 (Fundamental Properties)

Let $X$ be a discrete random variable with alphabet $\mathcal{X}$ of size $|\mathcal{X}| = n$. Then:

1. **Non-negativity**: $H(X) \geq 0$, with equality iff $X$ is deterministic
2. **Upper bound**: $H(X) \leq \log n$, with equality iff $X$ is uniform on $\mathcal{X}$
3. **Continuity**: $H(X)$ is a continuous function of $p(x)$
4. **Symmetry**: $H(X)$ depends only on the probabilities, not on the actual values of $X$

**Proof of Property 2**:
We prove that the uniform distribution maximizes entropy. Let $u(x) = 1/n$ be the uniform distribution. By Gibbs' inequality:
$$-\sum_{x} p(x) \log p(x) \leq -\sum_{x} p(x) \log u(x) = \log n$$

Equality holds iff $p(x) = u(x)$ for all $x$. $\square$

**Example 6.1.5** (Maximum Entropy with 3 Outcomes):
Let $\mathcal{X} = \{a, b, c\}$. The upper bound gives $H(X) \leq \log_2 3 \approx 1.585$ bits.

Verify: the uniform distribution $p(a) = p(b) = p(c) = 1/3$ achieves this:
$$H(X) = -3 \cdot \frac{1}{3}\log_2 \frac{1}{3} = \log_2 3 \approx 1.585 \text{ bits} \checkmark$$

Now compare with a non-uniform distribution $p(a) = 0.5, p(b) = 0.3, p(c) = 0.2$:
$$H(X) = -0.5\log_2 0.5 - 0.3\log_2 0.3 - 0.2\log_2 0.2 = 0.5 + 0.521 + 0.464 = 1.485 \text{ bits}$$

Indeed $1.485 < 1.585$, confirming the uniform distribution maximizes entropy.

### Visualization of Entropy for Binary Distributions

For $X \sim \text{Bernoulli}(p)$, entropy as a function of $p$:

```
H(X)
 1.0 |           ***
     |         **   **
     |        *       *
 0.5 |      *           *
     |    *               *
     |  *                   *
 0.0 |*                       *
     +--+----+----+----+----+----+-> p
     0  0.2  0.4  0.6  0.8  1.0
```

Maximum entropy (1 bit) occurs at $p = 0.5$. Entropy is symmetric around $p = 0.5$ and reaches 0 at $p \in \{0, 1\}$.

**Example 6.1.6** (Binary Entropy Function — Key Values):
The binary entropy function $h(p) = -p\log_2 p - (1-p)\log_2(1-p)$ has these key values:

| $p$ | $h(p)$ (bits) | Interpretation |
| --- | ------------- | -------------- |
| 0.0 | 0.000 | Deterministic: always tails |
| 0.1 | 0.469 | Highly biased coin |
| 0.2 | 0.722 | Moderately biased |
| 0.3 | 0.881 | Slightly biased |
| 0.5 | 1.000 | Maximum uncertainty (fair coin) |
| 0.7 | 0.881 | Symmetric with $p=0.3$ |
| 0.9 | 0.469 | Symmetric with $p=0.1$ |
| 1.0 | 0.000 | Deterministic: always heads |

Note the symmetry: $h(p) = h(1-p)$ for all $p$.

### Theorem 6.1.2 (Chain Rule for Entropy)

For random variables $X$ and $Y$:
$$H(X, Y) = H(X) + H(Y|X)$$

More generally, for $X_1, \ldots, X_n$:
$$H(X_1, \ldots, X_n) = \sum_{i=1}^{n} H(X_i | X_1, \ldots, X_{i-1})$$

**Proof**:
$$\begin{align}
H(X, Y) &= -\sum_{x,y} p(x,y) \log p(x,y) \\
&= -\sum_{x,y} p(x,y) \log[p(x) p(y|x)] \\
&= -\sum_{x,y} p(x,y) \log p(x) - \sum_{x,y} p(x,y) \log p(y|x) \\
&= -\sum_{x} p(x) \log p(x) - \sum_{x} p(x) \sum_{y} p(y|x) \log p(y|x) \\
&= H(X) + H(Y|X) \quad \square
\end{align}$$

**Example 6.1.7** (Chain Rule — Numerical Verification):
Let $X \in \{0, 1\}$ and $Y \in \{0, 1\}$ with joint distribution:

| | $Y=0$ | $Y=1$ |
| --- | --- | --- |
| $X=0$ | 1/4 | 1/4 |
| $X=1$ | 1/4 | 1/4 |

**Step 1 — Joint entropy** $H(X,Y)$: All four outcomes have probability $1/4$:
$$H(X,Y) = -4 \cdot \frac{1}{4}\log_2 \frac{1}{4} = 2 \text{ bits}$$

**Step 2 — Marginal** $H(X)$: $p(X=0) = p(X=1) = 1/2$, so $H(X) = 1$ bit.

**Step 3 — Conditional** $H(Y|X)$: Given $X=0$, $Y$ is uniform on $\{0,1\}$, so $H(Y|X=0) = 1$ bit. Similarly $H(Y|X=1) = 1$ bit. Thus:
$$H(Y|X) = \frac{1}{2}(1) + \frac{1}{2}(1) = 1 \text{ bit}$$

**Verify**: $H(X,Y) = H(X) + H(Y|X) \Rightarrow 2 = 1 + 1$ $\checkmark$

(Here $X$ and $Y$ are independent, so conditioning does not reduce entropy.)

---

## 6.1.3 Conditional Entropy

### Definition 6.1.2 (Conditional Entropy)

The **conditional entropy** of $Y$ given $X$ is:
$$H(Y|X) = \sum_{x \in \mathcal{X}} p(x) H(Y|X=x) = -\sum_{x,y} p(x,y) \log p(y|x)$$

Equivalently:
$$H(Y|X) = \mathbb{E}_X[H(Y|X=x)]$$

### Interpretation

$H(Y|X)$ measures the average uncertainty in $Y$ after observing $X$. It represents the expected entropy of $Y$ over all possible values of $X$, weighted by their probabilities.

### Theorem 6.1.3 (Conditioning Reduces Entropy)

For any random variables $X$ and $Y$:
$$H(Y|X) \leq H(Y)$$

with equality iff $X$ and $Y$ are independent.

**Proof**:
Using the chain rule and non-negativity of mutual information (covered in Chapter 6.2):
$$H(Y) = H(Y|X) + I(X;Y)$$

Since $I(X;Y) \geq 0$, we have $H(Y|X) \leq H(Y)$. Equality holds iff $I(X;Y) = 0$, which occurs iff $X$ and $Y$ are independent. $\square$

**Intuition**: Observing $X$ can only reduce (or maintain) our uncertainty about $Y$; it cannot increase uncertainty.

**Example 6.1.8** (Weather and Clothing):
Let $X$ = weather (sunny/rainy) and $Y$ = clothing choice. Knowing the weather reduces uncertainty about clothing choice: $H(Y|X) < H(Y)$.

**Example 6.1.9** (Conditional Entropy — Worked Computation):
Let $X \in \{0, 1\}$ (coin flip) and $Y \in \{a, b\}$ with joint distribution:

| | $Y=a$ | $Y=b$ |
| --- | --- | --- |
| $X=0$ | 1/2 | 0 |
| $X=1$ | 1/8 | 3/8 |

Marginals: $p(X=0) = 1/2$, $p(X=1) = 1/2$.

**Conditional distributions**:

- Given $X=0$: $p(Y=a|X=0) = 1$, $p(Y=b|X=0) = 0$, so $H(Y|X=0) = 0$ bits
- Given $X=1$: $p(Y=a|X=1) = 1/4$, $p(Y=b|X=1) = 3/4$, so $H(Y|X=1) = -\frac{1}{4}\log_2\frac{1}{4} - \frac{3}{4}\log_2\frac{3}{4} = 0.5 + 0.311 = 0.811$ bits

Therefore:
$$H(Y|X) = \frac{1}{2}(0) + \frac{1}{2}(0.811) = 0.406 \text{ bits}$$

Compare with $H(Y) = -\frac{5}{8}\log_2\frac{5}{8} - \frac{3}{8}\log_2\frac{3}{8} \approx 0.954$ bits. Indeed $H(Y|X) = 0.406 < 0.954 = H(Y)$, confirming that conditioning reduces entropy.

---

## 6.1.4 Joint Entropy

### Definition 6.1.3 (Joint Entropy)

The **joint entropy** of random variables $X$ and $Y$ is:
$$H(X,Y) = -\sum_{x \in \mathcal{X}} \sum_{y \in \mathcal{Y}} p(x,y) \log p(x,y)$$

### Relationships

```
       H(X,Y)
      /      \
    H(X)    H(Y)
      \      /
       I(X;Y)
         |
    [Mutual Information]
```

Key relationships:
1. **Chain rule**: $H(X,Y) = H(X) + H(Y|X) = H(Y) + H(X|Y)$
2. **Subadditivity**: $H(X,Y) \leq H(X) + H(Y)$, with equality iff $X$ and $Y$ are independent
3. **Mutual information**: $I(X;Y) = H(X) + H(Y) - H(X,Y) = H(Y) - H(Y|X)$

### Theorem 6.1.4 (Subadditivity)

For random variables $X$ and $Y$:
$$H(X,Y) \leq H(X) + H(Y)$$

**Proof**:
$$\begin{align}
H(X,Y) &= H(X) + H(Y|X) \quad \text{(chain rule)} \\
&\leq H(X) + H(Y) \quad \text{(conditioning reduces entropy)} \quad \square
\end{align}$$

**Visualization of Entropy Relations**:

```
   +----------------+
   |     H(X)       |
   |  +----------+  |
   |  |   I(X;Y) |  |
   |  +----------+  |
   +----------------+
        +----------+
        | H(Y|X)   |
        +----------+

H(X,Y) = H(X) + H(Y|X) = H(Y) + H(X|Y)
H(X) = I(X;Y) + H(X|Y)
H(Y) = I(X;Y) + H(Y|X)
```

**Example 6.1.10** (Joint Entropy and Mutual Information -- Full Worked Example):
Let $X \in \{0, 1\}$ and $Y \in \{0, 1\}$ with joint distribution:

| | $Y=0$ | $Y=1$ |
| --- | --- | --- |
| $X=0$ | 3/8 | 1/8 |
| $X=1$ | 1/8 | 3/8 |

**Joint entropy**:
$$H(X,Y) = -2\cdot\frac{3}{8}\log_2\frac{3}{8} - 2\cdot\frac{1}{8}\log_2\frac{1}{8} = 2(0.531) + 2(0.375) = 1.811 \text{ bits}$$

**Marginals**: $p(X=0)=p(X=1)=1/2$ and $p(Y=0)=p(Y=1)=1/2$, so $H(X) = H(Y) = 1$ bit.

**Mutual information**:
$$I(X;Y) = H(X) + H(Y) - H(X,Y) = 1 + 1 - 1.811 = 0.189 \text{ bits}$$

**Interpretation**: Knowing $X$ gives us 0.189 bits of information about $Y$. The variables are not independent ($I > 0$): when $X=0$, $Y=0$ is three times more likely than $Y=1$.

**Verify chain rule**: $H(Y|X) = H(Y) - I(X;Y) = 1 - 0.189 = 0.811$ bits, and:
$$H(X,Y) = H(X) + H(Y|X) = 1 + 0.811 = 1.811 \text{ bits} \checkmark$$

---

## 6.1.5 Differential Entropy

### Definition 6.1.4 (Differential Entropy)

For a continuous random variable $X$ with probability density function $f(x)$, the **differential entropy** is:
$$h(X) = -\int_{\mathcal{X}} f(x) \log f(x) \, dx$$

provided the integral exists.

### Important Distinctions from Discrete Entropy

1. **Can be negative**: Unlike $H(X) \geq 0$, differential entropy can be negative
2. **Not invariant under change of variables**: $h(X) \neq h(g(X))$ in general
3. **Depends on coordinate system**: Differential entropy changes under coordinate transformations

**Example 6.1.11** (Uniform Distribution):
Let $X \sim \text{Uniform}(0, a)$ with $f(x) = 1/a$ for $x \in [0, a]$.
$$h(X) = -\int_0^a \frac{1}{a} \log \frac{1}{a} dx = \log a$$

When $a < 1$, we have $h(X) < 0$ (negative differential entropy).

**Example 6.1.12** (Uniform -- Concrete Values):
Compute $h(X)$ for three uniform distributions using $\log_2$:

- $X \sim \text{Uniform}(0, 8)$: $h(X) = \log_2 8 = 3$ bits
- $X \sim \text{Uniform}(0, 1)$: $h(X) = \log_2 1 = 0$ bits
- $X \sim \text{Uniform}(0, 0.5)$: $h(X) = \log_2 0.5 = -1$ bit (negative!)

Doubling the interval width adds exactly 1 bit of differential entropy.

**Example 6.1.13** (Exponential Distribution):
Let $X \sim \text{Exp}(\lambda)$ with $f(x) = \lambda e^{-\lambda x}$ for $x \geq 0$.
$$h(X) = \log \frac{e}{\lambda} \text{ nats} = 1 - \log \lambda \text{ nats}$$

**Example 6.1.14** (Gaussian Differential Entropy):
Let $X \sim \mathcal{N}(\mu, \sigma^2)$. The differential entropy is $h(X) = \frac{1}{2}\log_2(2\pi e \sigma^2)$.

Compute for specific variances (using $\log_2$, in bits):

- $\sigma^2 = 1$: $h(X) = \frac{1}{2}\log_2(2\pi e) = \frac{1}{2}\log_2(17.08) \approx 2.047$ bits
- $\sigma^2 = 4$: $h(X) = \frac{1}{2}\log_2(2\pi e \cdot 4) = \frac{1}{2}\log_2(68.33) \approx 3.047$ bits
- $\sigma^2 = 1/4$: $h(X) = \frac{1}{2}\log_2(2\pi e \cdot 0.25) = \frac{1}{2}\log_2(4.27) \approx 1.047$ bits

Doubling $\sigma$ (quadrupling $\sigma^2$) adds exactly 1 bit, since $\frac{1}{2}\log_2(4) = 1$. Note that $h(X)$ depends only on $\sigma$, not on $\mu$.

### Theorem 6.1.5 (Transformation of Differential Entropy)

If $Y = g(X)$ where $g$ is a differentiable bijection, then:
$$h(Y) = h(X) + \mathbb{E}\left[\log\left|g'(X)\right|\right]$$

This shows that differential entropy is not coordinate-free.

---

## 6.1.6 Maximum Entropy Principle

### Theorem 6.1.6 (Maximum Entropy Distribution - Gaussian)

Among all continuous distributions on $\mathbb{R}$ with fixed mean $\mu$ and variance $\sigma^2$, the Gaussian distribution $\mathcal{N}(\mu, \sigma^2)$ has the maximum differential entropy:
$$h_{\max} = \frac{1}{2}\log(2\pi e \sigma^2)$$

**Proof Sketch**:
We use the calculus of variations with Lagrange multipliers. The optimization problem is:
$$\max_{f} -\int f(x) \log f(x) dx$$
subject to:
$$\int f(x) dx = 1, \quad \int x f(x) dx = \mu, \quad \int (x-\mu)^2 f(x) dx = \sigma^2$$

Using Lagrange multipliers $\lambda_0, \lambda_1, \lambda_2$:
$$\mathcal{L} = -\int f(x) \log f(x) dx + \lambda_0\left(\int f(x) dx - 1\right) + \lambda_1\left(\int x f(x) dx - \mu\right) + \lambda_2\left(\int (x-\mu)^2 f(x) dx - \sigma^2\right)$$

Taking the functional derivative and setting to zero:
$$-\log f(x) - 1 + \lambda_0 + \lambda_1 x + \lambda_2(x-\mu)^2 = 0$$

Solving yields:
$$f(x) = C \exp\left(\lambda_2(x-\mu)^2\right)$$

From the variance constraint, we find $\lambda_2 = -1/(2\sigma^2)$, giving:
$$f(x) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)$$

This is the Gaussian density. Computing its differential entropy:
$$h(X) = \frac{1}{2}\log(2\pi e \sigma^2) \quad \square$$

### Maximum Entropy Principle in Machine Learning

**Principle**: Given constraints (e.g., empirical means), choose the distribution that maximizes entropy. This represents the least committal distribution consistent with the constraints.

**Applications**:
- **Maximum entropy models**: Used in natural language processing and structured prediction
- **Gaussian assumption**: Justified when only mean and variance are known
- **Exponential family**: Distributions that maximize entropy subject to moment constraints

**Example 6.1.15** (Discrete Maximum Entropy):
Given expected value constraint $\mathbb{E}[X] = \mu$ for $X \in \{x_1, \ldots, x_n\}$, the maximum entropy distribution is:
$$p(x_i) = \frac{e^{\lambda x_i}}{\sum_j e^{\lambda x_j}}$$
where $\lambda$ is chosen to satisfy the constraint.

**Example 6.1.16** (Max Entropy Verification -- No Constraints):
For $X \in \{1, 2, 3\}$ with no constraints (besides probabilities summing to 1), the maximum entropy distribution is uniform: $p(1) = p(2) = p(3) = 1/3$.

Verify by comparing with two other valid distributions:

| Distribution | $p(1)$ | $p(2)$ | $p(3)$ | $H(X)$ (bits) |
| --- | --- | --- | --- | --- |
| Uniform | 1/3 | 1/3 | 1/3 | $\log_2 3 \approx 1.585$ |
| Peaked | 0.6 | 0.3 | 0.1 | $-0.6\log_2 0.6 - 0.3\log_2 0.3 - 0.1\log_2 0.1 \approx 1.295$ |
| Near-deterministic | 0.9 | 0.05 | 0.05 | $-0.9\log_2 0.9 - 2(0.05\log_2 0.05) \approx 0.569$ |

The uniform distribution achieves the maximum $\log_2 3 \approx 1.585$ bits, as guaranteed by Theorem 6.1.1.

---

## 6.1.7 Rényi Entropy

### Definition 6.1.5 (Rényi Entropy)

For a discrete random variable $X$ with distribution $p$, the **Rényi entropy** of order $\alpha$ (where $\alpha \geq 0$ and $\alpha \neq 1$) is:
$$H_\alpha(X) = \frac{1}{1-\alpha} \log \sum_{x} p(x)^\alpha$$

For $\alpha = 1$, Rényi entropy is defined as the limit:
$$H_1(X) = \lim_{\alpha \to 1} H_\alpha(X) = H(X) \text{ (Shannon entropy)}$$

### Special Cases

- **$\alpha = 0$**: $H_0(X) = \log |\text{supp}(p)|$ (Hartley entropy, counts non-zero probabilities)
- **$\alpha = 1$**: $H_1(X) = H(X)$ (Shannon entropy)
- **$\alpha = 2$**: $H_2(X) = -\log \sum_x p(x)^2$ (Collision entropy)
- **$\alpha \to \infty$**: $H_\infty(X) = -\log \max_x p(x)$ (Min-entropy)

### Properties

1. **Monotonicity in $\alpha$**: For $0 \leq \alpha < \beta$, we have $H_\alpha(X) \geq H_\beta(X)$
2. **Generalization**: Rényi entropy generalizes Shannon entropy and provides a family of uncertainty measures
3. **Additivity**: For independent $X, Y$: $H_\alpha(X,Y) = H_\alpha(X) + H_\alpha(Y)$

**Example 6.1.17** (Comparison of Entropies):
For $X \sim \text{Bernoulli}(0.7)$:
- $H_0(X) = \log 2 \approx 1.0$ bits
- $H_1(X) = H(X) \approx 0.881$ bits
- $H_2(X) \approx 0.844$ bits
- $H_\infty(X) = -\log 0.7 \approx 0.515$ bits

Note: $H_0(X) \geq H_1(X) \geq H_2(X) \geq H_\infty(X)$

### Theorem 6.1.7 (Shannon Entropy as Limit)

$$\lim_{\alpha \to 1} H_\alpha(X) = H(X)$$

**Proof Sketch**:
Using L'Hôpital's rule:
$$\lim_{\alpha \to 1} \frac{\log \sum_x p(x)^\alpha}{1-\alpha} = \lim_{\alpha \to 1} \frac{\sum_x p(x)^\alpha \log p(x)}{\sum_x p(x)^\alpha} = -\sum_x p(x) \log p(x) = H(X) \quad \square$$

---

## 6.1.8 Machine Learning Connections

### Decision Trees and Information Gain

**Information gain** is the reduction in entropy achieved by splitting on a feature:
$$\text{IG}(Y, X) = H(Y) - H(Y|X) = I(X;Y)$$

Decision tree algorithms (ID3, C4.5, CART) use information gain to select the best splitting feature at each node.

**Example**: For binary classification with 50/50 class split:
- Before split: $H(Y) = 1$ bit
- After perfect split: $H(Y|X) = 0$ bits
- Information gain: $\text{IG} = 1$ bit

**Algorithm pseudocode**:
```
function BestSplit(data, features):
    current_entropy = H(labels)
    best_feature = null
    max_gain = 0

    for feature in features:
        gain = current_entropy - H(labels | feature)
        if gain > max_gain:
            max_gain = gain
            best_feature = feature

    return best_feature
```

### Uncertainty Quantification

Entropy quantifies uncertainty in probabilistic predictions:
- **High entropy**: Model is uncertain (uniform predictions)
- **Low entropy**: Model is confident (peaked predictions)

For a softmax output $p(y|x)$ over $K$ classes:
$$H(Y|X=x) = -\sum_{k=1}^K p(y_k|x) \log p(y_k|x)$$

This is used in:
- **Active learning**: Select samples with highest prediction entropy
- **Ensemble methods**: Measure disagreement via prediction entropy
- **Calibration**: Well-calibrated models should have entropy matching accuracy

### Maximum Entropy Models

Maximum entropy models choose the distribution with maximum entropy subject to constraints from training data. For feature functions $f_i(x, y)$:
$$p(y|x) = \frac{1}{Z(x)} \exp\left(\sum_i \lambda_i f_i(x,y)\right)$$

where $\lambda_i$ are learned to match empirical feature expectations. Used in:
- Natural language processing (MaxEnt classifiers)
- Structured prediction
- Generalized linear models

### Entropy Regularization in Reinforcement Learning

In policy gradient methods, entropy regularization encourages exploration:
$$J(\theta) = \mathbb{E}_{\pi_\theta}[R] + \beta H(\pi_\theta)$$

where $H(\pi_\theta) = -\sum_a \pi_\theta(a|s) \log \pi_\theta(a|s)$ is the policy entropy. Higher $\beta$ encourages more exploratory (higher entropy) policies.

**Soft Actor-Critic (SAC)** explicitly maximizes entropy-augmented returns:
$$J(\pi) = \sum_{t=0}^T \mathbb{E}_{(s_t,a_t) \sim \rho_\pi}[r(s_t, a_t) + \alpha H(\pi(\cdot|s_t))]$$

### Variational Inference

In variational autoencoders (VAEs), the ELBO objective includes an entropy term:
$$\mathcal{L}(\theta, \phi) = \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] - D_{KL}(q_\phi(z|x) \| p(z))$$

The KL term can be decomposed as:
$$D_{KL}(q \| p) = -H(q) + \mathbb{E}_q[\log q] - \mathbb{E}_q[\log p]$$

Higher entropy in the variational distribution $q$ (more spread out) is penalized against matching the prior.

---

## Summary

**Key Concepts**:

1. **Shannon Entropy**: $H(X) = -\sum p(x) \log p(x)$ measures average uncertainty
2. **Properties**: Non-negative, maximized by uniform distribution, chain rule
3. **Conditional Entropy**: $H(Y|X)$ measures remaining uncertainty in $Y$ after observing $X$
4. **Joint Entropy**: $H(X,Y) = H(X) + H(Y|X)$
5. **Differential Entropy**: Continuous analog, can be negative
6. **Maximum Entropy**: Gaussian maximizes entropy for fixed mean/variance
7. **Rényi Entropy**: Family of entropy measures, Shannon as special case ($\alpha = 1$)

**ML Applications**:
- Decision trees (information gain for splitting)
- Uncertainty quantification (prediction confidence)
- Maximum entropy models (least committal inference)
- RL entropy regularization (exploration bonus)

---

## Exercises

### Basic Concepts ★

**Exercise 6.1.1**: Compute the entropy $H(X)$ for:
(a) $X \sim \text{Bernoulli}(0.3)$
(b) $X$ uniform on $\{1, 2, 3, 4\}$
(c) $X$ with distribution $p(1) = 0.5, p(2) = 0.25, p(3) = 0.25$

**Exercise 6.1.2**: Show that for a deterministic function $Y = g(X)$:
$$H(Y|X) = 0$$

**Exercise 6.1.3**: Prove that $H(X,Y) = H(Y,X)$ (symmetry of joint entropy).

**Exercise 6.1.4**: For $X \sim \text{Bernoulli}(p)$, find the value of $p$ that maximizes $H(X)$.

### Intermediate ★★

**Exercise 6.1.5**: Let $X$ be uniform on $\{1, 2, \ldots, n\}$ and $Y = X \mod 2$.
(a) Compute $H(X)$, $H(Y)$, and $H(Y|X)$
(b) Verify the chain rule: $H(X,Y) = H(X) + H(Y|X)$

**Exercise 6.1.6**: Prove that if $X$ and $Y$ are independent, then:
$$H(X,Y) = H(X) + H(Y)$$

**Exercise 6.1.7**: Compute the differential entropy of $X \sim \mathcal{N}(\mu, \sigma^2)$ by direct integration.

**Exercise 6.1.8**: For $X \sim \text{Uniform}(a, b)$, compute $h(X)$ and show it increases with $(b-a)$.

**Exercise 6.1.9**: In a decision tree, suppose splitting on feature $X$ divides the data into two subsets with sizes $n_1$ and $n_2$. Express the information gain in terms of entropies before and after the split.

### Advanced ★★★

**Exercise 6.1.10**: Prove Gibbs' inequality: For probability distributions $p$ and $q$ on the same alphabet:
$$-\sum_x p(x) \log p(x) \leq -\sum_x p(x) \log q(x)$$
with equality iff $p = q$. (Hint: Use $\log x \leq x - 1$ with $x = q(x)/p(x)$.)

**Exercise 6.1.11**: Prove that among all distributions on $\{1, 2, \ldots, n\}$ with fixed mean $\mu$, the distribution that maximizes entropy has the form:
$$p(k) = \frac{e^{\lambda k}}{Z}$$
for some $\lambda$ (exponential distribution on integers).

**Exercise 6.1.12**: Show that Rényi entropy $H_\alpha(X)$ is non-increasing in $\alpha$ for $\alpha \geq 0$. (Hint: Use Hölder's inequality.)

**Exercise 6.1.13**: Let $X$ have support on $n$ elements. Show that:
$$H_\infty(X) \leq H(X) \leq H_0(X) = \log n$$
and characterize when equalities hold.

**Exercise 6.1.14**: For $X \sim \text{Exponential}(\lambda)$, compute the Rényi entropy $h_\alpha(X)$ and verify that $\lim_{\alpha \to 1} h_\alpha(X) = h(X)$.

**Exercise 6.1.15**: In reinforcement learning, suppose a policy $\pi(a|s)$ is uniform over $|\mathcal{A}|$ actions. Compute the policy entropy $H(\pi(\cdot|s))$ and explain why this represents maximum exploration.

---

## Related Topics

- **Chapter 6.2**: Mutual Information and Divergence (KL divergence, cross-entropy, Jensen-Shannon divergence)
- **Chapter 6.3**: Information Theory in Practice (coding theory, compression, channel capacity)
- **Chapter 7.4**: Probabilistic Graphical Models (entropy in Bayesian networks, Markov random fields)
- **Chapter 9.2**: Decision Trees and Ensemble Methods (detailed treatment of information gain)
- **Chapter 10.3**: Variational Inference (ELBO, entropy terms in VAEs)
- **Chapter 12.5**: Reinforcement Learning Theory (entropy-regularized RL, maximum entropy RL)

---

## References

- Shannon, C. E. (1948). "A Mathematical Theory of Communication". *Bell System Technical Journal*.
- Cover, T. M., & Thomas, J. A. (2006). *Elements of Information Theory* (2nd ed.). Wiley.
- MacKay, D. J. C. (2003). *Information Theory, Inference, and Learning Algorithms*. Cambridge University Press.
- Rényi, A. (1961). "On measures of entropy and information". *Proceedings of the Fourth Berkeley Symposium on Mathematical Statistics and Probability*.
